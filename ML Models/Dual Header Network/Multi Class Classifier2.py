# ==== Multiclass Dual-Backbone Training Script (nevus optional) ====
# ConvNeXt-Tiny + DeiT-III Tiny -> concatenated features -> MLP -> 8/9 classes
# Weighted sampler + CE, Mixup/CutMix (first 80%), warmup+cosine, AMP optional

import os
import pickle
import time
import math
import json
import pandas as pd
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler, Subset

from torchvision import transforms

import timm
from timm.data import Mixup
from timm.models.vision_transformer import resize_pos_embed

from timm.utils import ModelEmaV2  # optional, not required
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, ConfusionMatrixDisplay, roc_curve
)

# ==================== Paths / Device ====================
original_split_dir = r'C:\Users\shore\Desktop\APS360\Datasets\DataSplit2'
cache_dir = r'C:\Users\shore\Desktop\APS360\Datasets\Cache_Multi\multi320'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
use_cuda = device.type == 'cuda'

# ==================== Labels (nevus ignored?) ====================
# If you truly want to exclude 'nevus', add it here BEFORE caching.
IGNORE_CLASSES = {}  # e.g., {'nevus'}

ALL_CLASSES = [
    'nevus', 'melanoma', 'bcc', 'keratosis',
    'actinic_keratosis', 'scc', 'dermatofibroma',
    'lentigo', 'vascular_lesion'
]

KEPT_CLASSES = [c for c in ALL_CLASSES if c not in IGNORE_CLASSES]
label_mapping = {c: i for i, c in enumerate(KEPT_CLASSES)}
idx_to_class = {i: c for c, i in label_mapping.items()}
num_classes = len(KEPT_CLASSES)

# ==================== Transforms ====================
# Cache at 320 -> faster IO + consistent high-res input
cache_transform = transforms.Compose([
    transforms.Resize((320, 320)),
    transforms.ToTensor()
])

normalize_transform = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)
# For speed, keep PIL augs; you can switch to tensor augs if preferred.
from torchvision.transforms import InterpolationMode

train_transform = transforms.Compose([
    #transforms.Resize((224, 224)),
    transforms.RandomAffine(
        degrees=10, translate=(0.05, 0.05), scale=(0.9, 1.1)
    ),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.2, 0.2, 0.2, 0.05),
    transforms.ToTensor(),
    normalize_transform,   # <--- here
    transforms.RandomErasing(p=0.25, scale=(0.02, 0.08), ratio=(0.3, 3.3))
])

val_transform = transforms.Compose([
    #transforms.Resize((224, 224)),
    transforms.ToTensor(),
    normalize_transform
])


# ==================== Cache Builder ====================
def ensure_dir(path): os.makedirs(path, exist_ok=True)

def cache_split_if_missing(split):
    src_path = os.path.join(original_split_dir, split)
    tgt_path = os.path.join(cache_dir, split)
    if os.path.exists(tgt_path):
        print(f"✅ Cache already exists for split '{split}' → skipping.")
        return

    print(f"⏳ Caching '{split}' split...")
    for class_name in os.listdir(src_path):
        class_src = os.path.join(src_path, class_name)
        class_tgt = os.path.join(tgt_path, class_name)
        if not os.path.isdir(class_src):
            continue

        # skip ignored classes entirely
        if class_name in IGNORE_CLASSES:
            print(f"🚫 Skipping ignored class '{class_name}' during cache.")
            continue

        ensure_dir(class_tgt)
        image_files = [f for f in os.listdir(class_src)
                       if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

        for img_name in tqdm(image_files, desc=f"Caching {split}/{class_name}", unit='img'):
            img_path = os.path.join(class_src, img_name)
            img = Image.open(img_path).convert('RGB')
            tensor = cache_transform(img)
            base = os.path.splitext(img_name)[0]
            torch.save(tensor, os.path.join(class_tgt, f'{base}.pt'))
# ===== Focal loss (multiclass) + class-balanced alpha =====
def make_class_balanced_alpha(class_counts, beta=0.9999):
    """Effective-number weighting (Cui et al.)."""
    import numpy as np, torch
    counts = np.asarray(class_counts, dtype=np.float64).clip(min=1)
    eff_num = 1.0 - np.power(beta, counts)
    alpha = (1.0 - beta) / eff_num
    alpha = alpha / alpha.mean()            # normalize ~ mean=1
    return torch.tensor(alpha, dtype=torch.float32)

def focal_loss_mc(logits, targets, gamma=1.5, alpha=None, reduction="mean"):
    """
    Multiclass focal loss. 'alpha' is per-class weight tensor (C,) or None.
    """
    # CE per-sample
    ce = F.cross_entropy(logits, targets, weight=None, reduction="none")
    # p_t for the true class
    pt = torch.softmax(logits, dim=1).gather(1, targets.view(-1, 1)).squeeze(1).clamp_(1e-6, 1.0)
    # focal factor
    focal = (1.0 - pt).pow(gamma)
    loss = focal * ce
    if alpha is not None:
        loss = loss * alpha.to(logits.device)[targets]
    return loss.mean() if reduction == "mean" else loss

# ==================== Dataset ====================
class CachedMultiDataset(Dataset):
    def __init__(self, cache_dir, augment=False):
        self.data = []
        self.augment = augment
        print(f"📦 Loading dataset from {cache_dir}...")
        class_names = [d for d in os.listdir(cache_dir) if os.path.isdir(os.path.join(cache_dir, d))]

        for class_name in tqdm(class_names, desc="Indexing classes"):
            if class_name in IGNORE_CLASSES:
                print(f"🚫 Ignoring class folder: {class_name}")
                continue
            if class_name not in label_mapping:
                print(f"⚠️ Skipping unknown class folder: {class_name}")
                continue
            label = label_mapping[class_name]
            class_dir = os.path.join(cache_dir, class_name)
            for file in os.listdir(class_dir):
                if file.endswith('.pt'):
                    self.data.append((os.path.join(class_dir, file), label))

        print(f"✅ Indexed {len(self.data)} total samples (classes kept: {KEPT_CLASSES}).")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        path, label = self.data[idx]
        image = torch.load(path)                      # [3,320,320]
        image = transforms.ToPILImage()(image)
        image = train_transform(image) if self.augment else val_transform(image)
        return image, label                           # <- no second normalize


# ==================== Data Loaders ====================
def get_multi_loaders(train_dir, val_dir, batch_size=32, num_workers=4):
    train_dataset = CachedMultiDataset(train_dir, augment=True)
    val_dataset = CachedMultiDataset(val_dir, augment=False)

    # labels from dataset index (fast)
    train_labels = [lbl for _, lbl in train_dataset.data]
    if len(train_labels) == 0:
        raise RuntimeError("No training samples found after filtering. Check your paths and IGNORE_CLASSES.")

    class_counts = np.bincount(train_labels, minlength=num_classes)
    print(f"📊 Train class counts (after filtering): {dict(zip(range(num_classes), class_counts.tolist()))}")

    # Inverse frequency weights per sample
    weights = 1.0 / torch.tensor(class_counts, dtype=torch.float32).clamp(min=1)
    sample_weights = torch.tensor([weights[l] for l in train_labels], dtype=torch.float32)
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    # IMPORTANT: sampler=True -> shuffle must be False
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, sampler=sampler,
        shuffle=False, num_workers=num_workers, pin_memory=use_cuda
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=use_cuda
    )
    return train_loader, val_loader, class_counts

# ==================== Model ====================
class DualBackbone(nn.Module):
    """
    CNN branch: ConvNeXt-Tiny (features_only=True) + GeM pooling
    ViT branch: DeiT-III Tiny (classifier reset)
    """
    def __init__(self, deit_variant='deit3_small_patch16_224', freeze_all=True, img_size=320):
        super().__init__()
        # ---- ConvNeXt-Tiny ----
        self.cnn = timm.create_model('convnext_tiny.fb_in22k', pretrained=True, features_only=True)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))  # simple GAP; GeM optional (see comment below)
        cnn_ch = self.cnn.feature_info[-1]['num_chs']  # usually 768 for convnext_tiny
        self.cnn_feat_dim = cnn_ch

        # ---- DeiT-III Tiny ----
        self.deit = timm.create_model(deit_variant, pretrained=True, img_size=img_size)
        self.deit.reset_classifier(0)
        self.deit_feat_dim = self.deit.num_features  # ~192 for deit3_tiny

        if freeze_all:
            for p in self.cnn.parameters(): p.requires_grad = False
            for p in self.deit.parameters(): p.requires_grad = False

    def forward(self, x):
        # ConvNeXt branch
        feats = self.cnn(x)[-1]        # [B, C, H, W]
        r = self.pool(feats).flatten(1)  # [B, C]

        # DeiT branch
        d = self.deit.forward_features(x)  # [B, tokens, C] or [B, C]
        if d.ndim == 3:
            d = d[:, 0, :]  # CLS token

        return torch.cat([r, d], dim=1)  # [B, cnn + deit]

class MultiClassifier(nn.Module):
    def __init__(self, in_features, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 768), nn.BatchNorm1d(768), nn.GELU(), nn.Dropout(0.4),
            nn.Linear(768, 256), nn.BatchNorm1d(256), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.GELU(),
            nn.Linear(128, num_classes)
        )
    def forward(self, x):
        return self.net(x)

# ==================== Plotting Helpers ====================
def plot_confusion_matrix_multiclass(true_labels, pred_labels, class_names, title="Confusion Matrix"):
    cm = confusion_matrix(true_labels, pred_labels, labels=list(range(len(class_names))))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap='Blues', values_format='d', xticks_rotation=45)
    plt.title(title)
    plt.grid(False)
    plt.tight_layout()
    plt.show()
def _interpolate_deit_pos_embed_to_320(backbone, state_dict_keyed_to_backbone):
    """
    Resize DeiT position embedding from whatever was saved (e.g. 224: 1x197xC)
    to the current model's size (e.g. 320: 1x401xC) by bicubic interpolation.
    """
    key = 'deit.pos_embed'
    if key not in state_dict_keyed_to_backbone:
        return state_dict_keyed_to_backbone  # nothing to do

    saved_pe = state_dict_keyed_to_backbone[key]          # [1, old_tokens, C]
    current_pe = backbone.deit.pos_embed                  # [1, new_tokens, C]
    if saved_pe.shape == current_pe.shape:
        return state_dict_keyed_to_backbone  # already matching

    # Separate CLS + grid
    num_prefix_tokens = 1  # DeiT has CLS
    saved_cls, saved_grid = saved_pe[:, :num_prefix_tokens], saved_pe[:, num_prefix_tokens:]
    new_tokens = current_pe.shape[1] - num_prefix_tokens

    old_size = int(math.sqrt(saved_grid.shape[1]))
    new_size = int(math.sqrt(new_tokens))

    # [1, HW, C] -> [1, C, H, W]
    saved_grid = saved_grid.reshape(1, old_size, old_size, -1).permute(0, 3, 1, 2)
    # interpolate
    saved_grid = F.interpolate(saved_grid, size=(new_size, new_size), mode='bicubic', align_corners=False)
    # back to [1, HW, C]
    saved_grid = saved_grid.permute(0, 2, 3, 1).reshape(1, new_size * new_size, -1)

    new_pe = torch.cat([saved_cls, saved_grid], dim=1)
    state_dict_keyed_to_backbone[key] = new_pe
    return state_dict_keyed_to_backbone

def plot_training_metrics(csv_path='multi_metrics.csv'):
    if not os.path.exists(csv_path):
        print("❌ CSV file not found.")
        return
    df = pd.read_csv(csv_path)
    if df.empty:
        print("⚠️ CSV is empty — nothing to plot.")
        return

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    metrics = [
        ('train_loss', 'Training Loss'),
        ('val_loss', 'Validation Loss'),
        ('val_acc', 'Validation Accuracy'),
        ('val_precision_macro', 'Validation Precision (macro)'),
        ('val_recall_macro', 'Validation Recall (macro)'),
        ('val_f1_macro', 'Validation F1 (macro)'),
        ('val_auc_ovr', 'Validation AUROC (OvR)')
    ]
    for ax, (col, title) in zip(axes.flat, metrics):
        sns.lineplot(data=df, x='epoch', y=col, ax=ax, marker='o')
        ax.set_title(title)
        ax.set_xlabel('Epoch')
        ax.set_ylabel(col)
        ax.grid(True)
    plt.suptitle("Training Metrics Over Time", fontsize=16)
    plt.tight_layout()
    plt.show()

# ==================== Training Utils ====================
from torch.amp import autocast, GradScaler

def unfreeze_and_attach(backbone, optimizer, lr_backbone=1e-5, weight_decay=1e-4):
    """
    Unfreeze both backbones and add them to the optimizer with a gentle LR.
    Safe to call multiple times: it won't duplicate params in the optimizer.
    """
    for p in backbone.cnn.parameters():  p.requires_grad = True
    for p in backbone.deit.parameters(): p.requires_grad = True

    existing = set()
    for g in optimizer.param_groups:
        for p in g['params']:
            existing.add(id(p))

    cnn_params  = [p for p in backbone.cnn.parameters()  if p.requires_grad and id(p) not in existing]
    deit_params = [p for p in backbone.deit.parameters() if p.requires_grad and id(p) not in existing]

    if cnn_params:
        optimizer.add_param_group({
            'params': cnn_params,
            'lr': lr_backbone,
            'weight_decay': weight_decay,
            'is_backbone': True,   # <-- tag
        })
    if deit_params:
        optimizer.add_param_group({
            'params': deit_params,
            'lr': lr_backbone,
            'weight_decay': weight_decay,
            'is_backbone': True,   # <-- tag
        })
def sync_backbone_lr_to_head(optimizer, head_base_lr, backbone_base_lr):
    """
    Make backbone LR follow the same schedule multiplier as the head.
    multiplier = (current_head_lr / head_base_lr)
    backbone_lr = backbone_base_lr * multiplier
    """
    # assume the first param group is the classifier/head group created at optimizer construction
    head_current = optimizer.param_groups[0]['lr']
    mult = max(head_current / max(head_base_lr, 1e-12), 0.0)
    target_backbone_lr = max(backbone_base_lr * mult, 1e-7)
    for pg in optimizer.param_groups[1:]:
        if pg.get('is_backbone', False):
            pg['lr'] = target_backbone_lr

# Soft-target CE for Mixup/CutMix
def soft_target_cross_entropy(logits, target_soft):
    logprob = F.log_softmax(logits, dim=-1)
    return -(target_soft * logprob).sum(dim=-1).mean()
# -------- top-K checkpoint saver (by macro-F1, tiebreak macro-recall) --------
import heapq, os

class TopKSaver:
    def __init__(self, k=3, folder="checkpoints"):
        self.k = k
        self.heap = []  # min-heap of (f1, recall, path)
        os.makedirs(folder, exist_ok=True)
        self.folder = folder

    def save(self, f1_macro, recall_macro, epoch, state_dict, prefix="multi"):
        path = os.path.join(self.folder, f"{prefix}_ep{epoch:03d}_f1{f1_macro:.4f}_rec{recall_macro:.4f}.pt")
        torch.save(state_dict, path)
        heapq.heappush(self.heap, (f1_macro, recall_macro, path))
        while len(self.heap) > self.k:
            _f1, _rec, old = heapq.heappop(self.heap)
            try: os.remove(old)
            except FileNotFoundError: pass

# -------- simple moving-average early stopping on macro-F1 --------
from collections import deque

class MAEarlyStop:
    def __init__(self, window=3, patience=6, eps=1e-5):
        self.q = deque(maxlen=window)
        self.best_ma = -1.0
        self.since = 0
        self.patience = patience
        self.eps = eps

    def step(self, value):
        self.q.append(value)
        if len(self.q) < self.q.maxlen:
            return False  # not enough history yet
        ma = sum(self.q) / len(self.q)
        if ma > self.best_ma + self.eps:
            self.best_ma = ma
            self.since = 0
        else:
            self.since += 1
        return self.since >= self.patience
def _resize_deit_pos_embed_for_resume(backbone, state_dict):
    """
    Safely resize DeiT positional embeddings from a 224px checkpoint to the
    current img_size (e.g., 320px). Handles the CLS token automatically.
    """
    key = 'deit.pos_embed'
    if key not in state_dict:
        return state_dict

    saved_pe = state_dict[key]                  # [1, N_old, C]
    model_pe = backbone.deit.pos_embed          # [1, N_new, C]
    if saved_pe.shape == model_pe.shape:
        return state_dict

    # how many special tokens (CLS) the model uses
    num_tokens = 1 if getattr(backbone.deit, 'cls_token', None) is not None else 0

    N_old = saved_pe.shape[1]
    N_new = model_pe.shape[1]
    old_grid = N_old - num_tokens
    new_grid = N_new - num_tokens

    old_size = int(round(math.sqrt(old_grid)))
    new_size = int(round(math.sqrt(new_grid)))
    if old_size * old_size != old_grid or new_size * new_size != new_grid:
        # shapes weird? fall back to model's current pos_embed to avoid crashing
        state_dict[key] = model_pe
        return state_dict

    # split cls+grid
    if num_tokens > 0:
        saved_cls  = saved_pe[:, :num_tokens, :]          # [1,1,C]
        saved_grid = saved_pe[:, num_tokens:, :]          # [1,old_grid,C]
    else:
        saved_cls  = saved_pe[:, :0, :]
        saved_grid = saved_pe

    # [1, old_size*old_size, C] -> [1, C, old_size, old_size]
    saved_grid = saved_grid.reshape(1, old_size, old_size, -1).permute(0, 3, 1, 2)
    # bicubic resize to new grid size
    resized_grid = F.interpolate(saved_grid, size=(new_size, new_size), mode='bicubic', align_corners=False)
    # back to [1, new_size*new_size, C]
    resized_grid = resized_grid.permute(0, 2, 3, 1).reshape(1, new_size * new_size, -1)

    resized = torch.cat([saved_cls, resized_grid], dim=1) # [1, N_new, C]
    resized = resized.to(model_pe.dtype).to(model_pe.device)

    state_dict[key] = resized
    return state_dict


# ==================== Train ====================
def train_multiclass_model(
    backbone, classifier, train_loader, val_loader, class_counts,
    epochs=100, patience=15, lr=1e-4,
    csv_path='multi_metrics.csv', model_path='multi_best_checkpoint.pt',
    use_amp=True, warmup_epochs=3, lr_backbone=3e-6,
    mixup_alpha=0.2, cutmix_alpha=0.5, mixup_off_pct=0.2,
    mixup_burst_len_after_resume=2, finetune_lr_drop=0.1, topk_checkpoints=3,
    use_focal=True, focal_gamma=1.5, freeze_backbone=True    # <— NEW
):

    from torch.optim.lr_scheduler import CosineAnnealingLR, SequentialLR, LinearLR

    def _scale_scheduler_base_lrs(sched, factor: float):
        # make LR drop persist for SequentialLR + its children
        if hasattr(sched, 'base_lrs'):
            sched.base_lrs = [b * factor for b in sched.base_lrs]
        if hasattr(sched, '_schedulers'):
            for s in sched._schedulers:
                if hasattr(s, 'base_lrs'):
                    s.base_lrs = [b * factor for b in s.base_lrs]

    backbone = backbone.to(device)
    classifier = classifier.to(device)
    alpha_cb = None
    if use_focal:
        alpha_cb = make_class_balanced_alpha(class_counts).to(device)


    params = list(filter(lambda p: p.requires_grad, backbone.parameters())) + list(classifier.parameters())
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=0.05)

    warmup = LinearLR(optimizer, start_factor=0.1, total_iters=max(1, warmup_epochs))
    cosine = CosineAnnealingLR(optimizer, T_max=max(1, epochs - warmup_epochs), eta_min=1e-6)
    scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs])

    scaler = GradScaler(enabled=use_amp and use_cuda)

    mixup_fn = Mixup(
        mixup_alpha=mixup_alpha, cutmix_alpha=cutmix_alpha,
        label_smoothing=0.0, num_classes=num_classes
    )

    topk_saver = TopKSaver(k=topk_checkpoints, folder="checkpoints")
    ma_stop = MAEarlyStop(window=3, patience=6)

    # CSV
    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path)
        except Exception:
            df = pd.DataFrame(columns=[
                'epoch', 'train_loss', 'val_loss', 'val_acc',
                'val_precision_macro', 'val_recall_macro', 'val_f1_macro', 'val_auc_ovr', 'lr'
            ])
    else:
        df = pd.DataFrame(columns=[
            'epoch', 'train_loss', 'val_loss', 'val_acc',
            'val_precision_macro', 'val_recall_macro', 'val_f1_macro', 'val_auc_ovr', 'lr'
        ])

    # ----- resume -----
    start_epoch = 0
    best_val_f1 = -float('inf')
    unfrozen = False
    if os.path.exists(model_path):
        print(f"🔁 Resuming from checkpoint: {model_path}")
        ckpt = torch.load(model_path, map_location=device)
        if isinstance(ckpt, dict) and ('classifier_state' in ckpt or 'backbone_state' in ckpt):
            if 'backbone_state' in ckpt:
                bb = ckpt['backbone_state']
                bb = _resize_deit_pos_embed_for_resume(backbone, bb)  # <-- use the safe resizer
                backbone.load_state_dict(bb, strict=False)


            if 'classifier_state' in ckpt:
                classifier.load_state_dict(ckpt['classifier_state'], strict=False)
            try:
                optimizer.load_state_dict(ckpt['optimizer_state'])
            except Exception as e:
                print(f"⚠️ Could not load optimizer state: {e}")
            try:
                if 'scheduler_state' in ckpt:
                    scheduler.load_state_dict(ckpt['scheduler_state'])
            except Exception as e:
                print(f"⚠️ Could not load scheduler state: {e}")
            best_val_f1 = float(ckpt.get('best_val_f1', best_val_f1))
            start_epoch = int(ckpt.get('epoch', -1)) + 1
            unfrozen = bool(ckpt.get('unfrozen', unfrozen))
        else:
            print("ℹ️ Legacy checkpoint detected. Loading classifier only.")
            try:
                classifier.load_state_dict(ckpt, strict=False)
            except Exception:
                pass
        print(f"📈 Resume epoch {start_epoch}, best val F1 {best_val_f1:.4f}, unfrozen={unfrozen}")

    # post‑resume mixup burst + fine‑tune phase
    burst_start = start_epoch
    burst_end   = start_epoch + mixup_burst_len_after_resume
    finetune_start = burst_end
    did_lr_drop = False

    if (start_epoch >= warmup_epochs or unfrozen) and not freeze_backbone:
        print("🔓 Ensuring backbones are attached to optimizer after resume.")
        unfreeze_and_attach(backbone, optimizer, lr_backbone=lr_backbone)
        unfrozen = True

    # base LR used for syncing backbone each epoch (will change after drop)
    effective_head_base_lr = lr

    last_val_labels, last_val_preds = [], []

    for epoch in range(start_epoch, epochs):
        if (not unfrozen) and (epoch >= warmup_epochs) and not freeze_backbone:
            print(f"🔓 Unfreezing backbones at epoch {epoch} ...")
            unfreeze_and_attach(backbone, optimizer, lr_backbone=lr_backbone)
            unfrozen = True

        # mixup ON only during burst after resume
        mixup_active = (burst_start <= epoch < burst_end)

        # enter fine‑tune: LR drop that PERSISTS (optimizer + scheduler + sync base)
        if (not did_lr_drop) and (epoch >= finetune_start):
            for pg in optimizer.param_groups:
                pg['lr'] = max(pg['lr'] * finetune_lr_drop, 1e-6)
                if 'initial_lr' in pg:
                    pg['initial_lr'] *= finetune_lr_drop
            _scale_scheduler_base_lrs(scheduler, finetune_lr_drop)
            effective_head_base_lr *= finetune_lr_drop
            did_lr_drop = True
            print(f"↓ Entering fine‑tune phase: MixUp OFF, LR ×{finetune_lr_drop}")

        print(f"\nEpoch {epoch+1}/{epochs} | Mixup/CutMix: {'ON' if mixup_active else 'OFF'}")
        backbone.train(); classifier.train()
        total_loss = 0.0

        train_loop = tqdm(train_loader, desc=f"Training [{epoch+1}]")
        for inputs, targets in train_loop:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            if mixup_active:
                inputs, targets_mixed = mixup_fn(inputs, targets)
            else:
                targets_mixed = None

            with autocast(device_type='cuda', enabled=use_amp and use_cuda):
                feats  = backbone(inputs)
                logits = classifier(feats)

                if mixup_active:
                    # you had Mixup OFF in your current runs; if ON, keep CE with soft targets
                    loss = soft_target_cross_entropy(logits, targets_mixed)
                else:
                    if use_focal:
                        loss = focal_loss_mc(logits, targets, gamma=focal_gamma, alpha=alpha_cb)
                    elif mixup_active:
                        loss = soft_target_cross_entropy(logits, targets_mixed)
                    else:
                        loss = F.cross_entropy(logits, targets)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            train_loop.set_postfix(loss=loss.item())

        avg_train_loss = total_loss / max(1, len(train_loader))

        # step scheduler and then sync backbone LR to head's schedule multiplier
        scheduler.step()
        if not freeze_backbone:
            sync_backbone_lr_to_head(
                optimizer,
                head_base_lr=effective_head_base_lr,
                backbone_base_lr=lr_backbone
    )

        

        # ----- validation -----
        backbone.eval(); classifier.eval()
        val_loss = 0.0
        val_labels, val_preds, val_probs = [], [], []
        val_loop = tqdm(val_loader, desc=f"Validation [{epoch+1}]")
        with torch.no_grad():
            for inputs, targets in val_loop:
                inputs = inputs.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                with autocast(device_type='cuda', enabled=use_amp and use_cuda):
                    feats = backbone(inputs)
                    logits = classifier(feats)
                    loss = F.cross_entropy(logits, targets)
                    probs = F.softmax(logits, dim=1)

                val_loss += loss.item()
                preds = torch.argmax(probs, dim=1)
                val_labels.extend(targets.cpu().numpy())
                val_preds.extend(preds.cpu().numpy())
                val_probs.extend(probs.cpu().numpy())
                val_loop.set_postfix(loss=loss.item())

        avg_val_loss = val_loss / max(1, len(val_loader))
        acc = accuracy_score(val_labels, val_preds)
        prec_macro = precision_score(val_labels, val_preds, average='macro', zero_division=0)
        rec_macro  = recall_score(val_labels, val_preds, average='macro', zero_division=0)
        f1_macro   = f1_score(val_labels, val_preds, average='macro', zero_division=0)
        try:
            y_true_ovr = np.eye(num_classes)[val_labels]
            auc_ovr = roc_auc_score(y_true_ovr, np.array(val_probs), multi_class='ovr')
        except Exception:
            auc_ovr = float('nan')

        current_lrs = [pg['lr'] for pg in optimizer.param_groups]
        print(f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        print(f"Val Acc: {acc:.4f} | Prec(m): {prec_macro:.4f} | Rec(m): {rec_macro:.4f} | "
              f"F1(m): {f1_macro:.4f} | AUC(OvR): {auc_ovr:.4f} | LRs: {current_lrs}")

        # per‑class (optional verbose)
        val_labels_np, val_preds_np = np.array(val_labels), np.array(val_preds)
        print("\n📊 Per-class metrics:")
        for class_idx in range(num_classes):
            cls_name = idx_to_class[class_idx]
            true_binary = (val_labels_np == class_idx)
            pred_binary = (val_preds_np == class_idx)
            cls_prec = precision_score(true_binary, pred_binary, zero_division=0)
            cls_rec  = recall_score(true_binary, pred_binary, zero_division=0)
            cls_acc  = np.mean(true_binary == pred_binary)
            print(f"  {cls_name:<20} Prec: {cls_prec:.3f}  Rec: {cls_rec:.3f}  Acc: {cls_acc:.3f}")

        # CSV
        new_row = {
            'epoch': epoch,
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
            'val_acc': acc,
            'val_precision_macro': prec_macro,
            'val_recall_macro': rec_macro,
            'val_f1_macro': f1_macro,
            'val_auc_ovr': auc_ovr,
            'lr': current_lrs[0]
        }
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        df.to_csv(csv_path, index=False)

        # save best + top‑K
        if f1_macro > best_val_f1 + 1e-6:
            best_val_f1 = f1_macro

            ckpt = {
                'epoch': epoch,
                'backbone_state': backbone.state_dict(),
                'classifier_state': classifier.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'scheduler_state': scheduler.state_dict(),
                'best_val_f1': best_val_f1,
                'unfrozen': unfrozen,
                'label_mapping': label_mapping,
            }
            torch.save(ckpt, model_path)
            print("💾 Saved best checkpoint (by macro-F1).")
            topk_saver.save(f1_macro, rec_macro, epoch, ckpt, prefix="multi")

            # Save raw validation outputs so we can tune thresholds
            last_val_labels = np.array(val_labels, dtype=np.int64)
            last_val_probs  = np.array(val_probs, dtype=np.float32)
            with open('multi_val_outputs.pkl', 'wb') as f:
                pickle.dump({'labels': last_val_labels, 'probs': last_val_probs}, f)

            # 👉 Tune per-class thresholds on-the-fly and report gains
            thresholds, tuned_report = tune_per_class_thresholds(last_val_probs, last_val_labels, metric="f1")
            print(f"🎯 Threshold-tuned (val): acc={tuned_report['accuracy']:.4f} "
                f"prec={tuned_report['precision']:.4f} rec={tuned_report['recall']:.4f} "
                f"f1={tuned_report['f1']:.4f}")

            # Save thresholds alongside the checkpoint
            thr_path = os.path.splitext(model_path)[0] + "_thresholds.json"
            save_thresholds(thresholds, thr_path)
            print(f"📝 Saved per-class thresholds → {thr_path}")

    return last_val_labels, last_val_preds


# ==================== Overfit Sanity (optional) ====================
def disable_dropout(module: nn.Module):
    for m in module.modules():
        if isinstance(m, nn.Dropout):
            m.p = 0.0

def unfreeze_all_multiclass(backbone: nn.Module):
    for p in backbone.cnn.parameters():  p.requires_grad = True
    for p in backbone.deit.parameters(): p.requires_grad = True

def make_overfit_subset_loader_multiclass(cache_dir, n_per_class=25, batch_size=16):
    ds = CachedMultiDataset(os.path.join(cache_dir, "train"), augment=False)

    per_class_idxs = {c: [] for c in range(num_classes)}
    for i, (_, y) in enumerate(ds.data):
        per_class_idxs[y].append(i)

    chosen, counts = [], {}
    for c in range(num_classes):
        take = min(n_per_class, len(per_class_idxs[c]))
        counts[c] = take
        chosen.extend(per_class_idxs[c][:take])

    if not chosen:
        raise RuntimeError("Overfit subset is empty — check cache_dir and class folders.")

    print("🔎 Overfit subset per-class counts:",
          {idx_to_class[c]: counts[c] for c in range(num_classes)})

    small_ds = Subset(ds, chosen)
    loader = DataLoader(
        small_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=use_cuda,
        persistent_workers=True,
        prefetch_factor=2
    )
    return loader
# ===== Per-class threshold helpers =====
import json
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def save_thresholds(thresholds, path):
    with open(path, "w") as f:
        json.dump({"thresholds": list(map(float, thresholds))}, f)

def load_thresholds(path):
    with open(path, "r") as f:
        obj = json.load(f)
    return np.asarray(obj["thresholds"], dtype=np.float32)

def adjusted_argmax(probs: np.ndarray, thresholds: np.ndarray, rule="subtract"):
    """
    probs: (N, C) softmax probabilities
    thresholds: (C,) per-class thresholds in [0,1]
    rule="subtract": argmax over (p_c - t_c)
    """
    if rule != "subtract":
        raise ValueError("Only rule='subtract' is implemented here.")
    adj = probs - thresholds.reshape(1, -1)
    return np.argmax(adj, axis=1)

def _metric_from_preds(y_true, y_pred, metric="f1"):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec  = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1   = f1_score(y_true, y_pred, average="macro", zero_division=0)
    score = {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}[metric]
    return score, {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}

def tune_per_class_thresholds(probs: np.ndarray,
                              labels: np.ndarray,
                              metric: str = "f1",
                              grid = None,
                              passes: int = 2):
    """
    Coordinate-ascent threshold search:
      - Start from 0.5 for every class
      - For each class, scan thresholds in 'grid' keeping others fixed
      - Repeat 'passes' times
    Returns: thresholds (C,), best_report (dict of metrics)
    """
    n_samples, n_classes = probs.shape
    if grid is None:
        # denser around 0.4–0.7 which is often useful
        grid = np.unique(np.concatenate([
            np.linspace(0.05, 0.95, 19),
            np.array([0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65])
        ]))
    thr = np.full((n_classes,), 0.5, dtype=np.float32)

    # evaluate start
    preds = adjusted_argmax(probs, thr, rule="subtract")
    best_score, best_report = _metric_from_preds(labels, preds, metric=metric)

    for _ in range(passes):
        improved = False
        for c in range(n_classes):
            best_t_c = thr[c]
            for t in grid:
                thr[c] = t
                preds = adjusted_argmax(probs, thr, rule="subtract")
                score, report = _metric_from_preds(labels, preds, metric=metric)
                if score > best_score + 1e-6:
                    best_score, best_report = score, report
                    best_t_c = t
                    improved = True
            thr[c] = best_t_c
        if not improved:
            break

    return thr, best_report

@torch.no_grad()
def collect_val_outputs(backbone, classifier, val_loader, device="cuda"):
    import torch.nn.functional as F
    backbone.eval(); classifier.eval()
    all_probs, all_labels = [], []
    for x, y in val_loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        feats  = backbone(x)
        logits = classifier(feats)
        probs  = F.softmax(logits, dim=1)
        all_probs.append(probs.cpu().numpy())
        all_labels.append(y.cpu().numpy())
    probs = np.concatenate(all_probs, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    return probs, labels

def preds_from_thresholds(probs: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    """Convert softmax probs (N,C) to predicted labels using per-class thresholds."""
    adj = probs - thresholds.reshape(1, -1)  # subtract rule
    return np.argmax(adj, axis=1)

def plot_confusion_with_thresholds(y_true: np.ndarray,
                                   y_pred: np.ndarray,
                                   class_names,
                                   title="Confusion Matrix (thresholded)"):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap='Blues', values_format='d', xticks_rotation=45)
    plt.title(title)
    plt.grid(False)
    plt.tight_layout()
    plt.show()


# ==================== Main ====================
if __name__ == "__main__":
    # Build cache if missing
    for split in ['train', 'val']:
        cache_split_if_missing(split)

    # Loaders (smaller batch by default due to backbone choices)
    train_loader, val_loader, class_counts = get_multi_loaders(
        os.path.join(cache_dir, "train"),
        os.path.join(cache_dir, "val"),
        batch_size=32,   # adjust to your GPU (try 16 if OOM, 32 if plenty)
        num_workers=4
    )

    print("✅ Dataloaders ready.")

    # Model
    backbone = DualBackbone(deit_variant='deit3_small_patch16_224', freeze_all=True, img_size=320)
    in_features = backbone.cnn_feat_dim + backbone.deit_feat_dim
    classifier = MultiClassifier(in_features, num_classes)
    print(f"✅ Model initialized. Classes kept: {KEPT_CLASSES}")
    print(f"   ConvNeXt-Tiny feat dim: {backbone.cnn_feat_dim}, DeiT-III Tiny feat dim: {backbone.deit_feat_dim}")
    


    # Train
    
    labels_best, preds_best = train_multiclass_model(
        backbone, classifier, train_loader, val_loader, class_counts,
        epochs=150, patience=15,
        lr=2e-4,
        csv_path='multi_metrics.csv',
        model_path='multi_best_classifier.pt',
        use_amp=True, warmup_epochs=2,
        lr_backbone=1e-7,
        mixup_alpha=0.0, cutmix_alpha=0.0, mixup_off_pct=1.0,
        mixup_burst_len_after_resume=0, finetune_lr_drop=1.0, topk_checkpoints=3,
        use_focal=True, focal_gamma=2.2, freeze_backbone=True    # <— here
)

    
    # Threshold Adjustments
    # ===== Per-class threshold tuning on current model (no reload) =====
    print("\n🔧 Tuning per-class decision thresholds on validation set...")
    val_probs, val_labels = collect_val_outputs(backbone, classifier, val_loader, device=device)

    # Choose metric to optimize: "f1" (macro) is usually best for your setup
    thresholds, report = tune_per_class_thresholds(val_probs, val_labels, metric="f1")
    print("Per-class thresholds:", thresholds.tolist())
    print(f"With tuned thresholds → "
        f"acc={report['accuracy']:.4f}  prec={report['precision']:.4f}  "
        f"rec={report['recall']:.4f}  f1={report['f1']:.4f}")

    # Save thresholds next to your best checkpoint
    thr_path = "multi_best_classifier_thresholds.json"
    save_thresholds(thresholds, thr_path)
    print(f"💾 Saved thresholds to {thr_path}")


    # Plots
    # Without Thresholds 
    # Without Thresholds
    if len(val_labels) and len(val_probs):
        y_pred = np.argmax(val_probs, axis=1).astype(int)
        plot_confusion_matrix_multiclass(
            val_labels, y_pred,
            class_names=[idx_to_class[i] for i in range(num_classes)],
            title="Best-Epoch Validation Confusion Matrix"
    )


    #With Thresholds 
    y_pred_thr = preds_from_thresholds(val_probs, thresholds)
    plot_confusion_with_thresholds(
        val_labels, y_pred_thr,
        class_names=[idx_to_class[i] for i in range(num_classes)],
        title="Validation Confusion Matrix (per-class thresholds)"
    )

    plot_training_metrics('multi_metrics.csv')
