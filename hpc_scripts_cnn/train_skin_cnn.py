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
cache_dir = r'C:\Users\shore\Desktop\APS360\Datasets\Cache_Multi\multi'
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
def get_multi_loaders(train_dir, val_dir, batch_size=24, num_workers=4):
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
        optimizer.add_param_group({'params': cnn_params, 'lr': lr_backbone, 'weight_decay': weight_decay})
    if deit_params:
        optimizer.add_param_group({'params': deit_params, 'lr': lr_backbone, 'weight_decay': weight_decay})

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

# ==================== Train ====================
def train_multiclass_model(
    backbone, classifier, train_loader, val_loader, class_counts,
    epochs=100, patience=15, lr=2e-4,
    csv_path='multi_metrics.csv', model_path='multi_best_checkpoint.pt',
    use_amp=True, warmup_epochs=3, lr_backbone=5e-6,
    mixup_alpha=0.4, cutmix_alpha=1.0, mixup_off_pct=0.15,
    mixup_burst_len_after_resume=16,     # NEW: how many epochs to keep MixUp ON after resume
    finetune_lr_drop=0.25,              # NEW: when MixUp turns OFF, drop LR by this factor
    topk_checkpoints=3                  # NEW: keep top-K by F1 (tie-break recall)
):

    """
    WeightedRandomSampler + CrossEntropy (no class weights).
    Mixup/CutMix ON for first (1 - mixup_off_pct) of epochs, OFF for last mixup_off_pct.
    Scheduler: Linear warmup + Cosine.
    """
    backbone = backbone.to(device)
    classifier = classifier.to(device)

    # optimizer: AdamW (fast + reliable)
    params = list(filter(lambda p: p.requires_grad, backbone.parameters())) + list(classifier.parameters())
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=0.05)

    # scheduler: warmup + cosine
    from torch.optim.lr_scheduler import CosineAnnealingLR, SequentialLR, LinearLR

    warmup = LinearLR(optimizer, start_factor=0.1, total_iters=max(1, warmup_epochs))
    cosine = CosineAnnealingLR(optimizer, T_max=max(1, epochs - warmup_epochs), eta_min=1e-6)
    scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs])

    scaler = GradScaler(enabled=use_amp and use_cuda)

    # Mixup/CutMix
    mixup_fn = Mixup(
        mixup_alpha=mixup_alpha, cutmix_alpha=cutmix_alpha,
        label_smoothing=0.0, num_classes=num_classes
    )
        # Small utilities: top-K saver and MA early stop
    topk_saver = TopKSaver(k=topk_checkpoints, folder="checkpoints")
    ma_stop = MAEarlyStop(window=3, patience=6)

    # CSV setup
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

    # Resume logic
    start_epoch = 0
    best_val_f1 = -float('inf')
    unfrozen = False
    if os.path.exists(model_path):
        print(f"🔁 Resuming from checkpoint: {model_path}")
        ckpt = torch.load(model_path, map_location=device)

        if isinstance(ckpt, dict) and ('classifier_state' in ckpt or 'backbone_state' in ckpt):
            if 'backbone_state' in ckpt:
                backbone.load_state_dict(ckpt['backbone_state'], strict=False)
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

    # Ensure unfreeze after resume if needed
        # ---------- MixUp/CutMix schedule around the RESUME ----------
    # We resume at start_epoch (e.g., ~47 from your logs).
    # Keep MixUp/CutMix ON for mixup_burst_len_after_resume epochs,
    # then turn it OFF and drop LR for fine‑tuning.
    burst_start = start_epoch
    burst_end   = start_epoch + mixup_burst_len_after_resume  # MixUp ON for [burst_start, burst_end)
    finetune_start = burst_end
    did_lr_drop = False

    if (start_epoch >= warmup_epochs or unfrozen):
        print("🔓 Ensuring backbones are attached to optimizer after resume.")
        unfreeze_and_attach(backbone, optimizer, lr_backbone=lr_backbone)
        unfrozen = True

    patience_counter = 0
    last_val_labels, last_val_preds = [], []

    # ===== EPOCH LOOP =====
    for epoch in range(start_epoch, epochs):
        # Timed unfreeze
        if (not unfrozen) and (epoch >= warmup_epochs):
            print(f"🔓 Unfreezing backbones at epoch {epoch} (after {warmup_epochs} warmup epochs).")
            unfreeze_and_attach(backbone, optimizer, lr_backbone=lr_backbone)
            unfrozen = True

        # Mixup phase control
                # MixUp/CutMix ON only during the post-resume "burst"
        mixup_active = (burst_start <= epoch < burst_end)

        # When we cross into fine-tune phase: turn MixUp OFF and drop LR once
        if (not did_lr_drop) and (epoch >= finetune_start):
            for i, pg in enumerate(optimizer.param_groups):
                pg['lr'] = max(pg['lr'] * finetune_lr_drop, 1e-6)
            did_lr_drop = True
            print(f"↓ Entering fine‑tune phase: MixUp OFF, LR ×{finetune_lr_drop}")


        print(f"\nEpoch {epoch+1}/{epochs} | Mixup/CutMix: {'ON' if mixup_active else 'OFF'}")
        backbone.train()
        classifier.train()
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
                    loss = soft_target_cross_entropy(logits, targets_mixed)
                else:
                    loss = F.cross_entropy(logits, targets)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            

            total_loss += loss.item()
            train_loop.set_postfix(loss=loss.item())

        avg_train_loss = total_loss / max(1, len(train_loader))
        scheduler.step()  # Step scheduler after each epoch

        # ===== Validation =====
        backbone.eval()
        classifier.eval()
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
                    loss = F.cross_entropy(logits, targets)  # CE at eval
                    probs = F.softmax(logits, dim=1)

                val_loss += loss.item()
                preds = torch.argmax(probs, dim=1)
                val_labels.extend(targets.cpu().numpy())
                val_preds.extend(preds.cpu().numpy())
                val_probs.extend(probs.cpu().numpy())
                val_loop.set_postfix(loss=loss.item())

        avg_val_loss = val_loss / max(1, len(val_loader))

        # ===== Macro Metrics =====
        acc = accuracy_score(val_labels, val_preds)
        prec_macro = precision_score(val_labels, val_preds, average='macro', zero_division=0)
        rec_macro  = recall_score(val_labels, val_preds, average='macro', zero_division=0)
        f1_macro   = f1_score(val_labels, val_preds, average='macro', zero_division=0)
        try:
            y_true_ovr = np.eye(num_classes)[val_labels]
            auc_ovr = roc_auc_score(y_true_ovr, np.array(val_probs), multi_class='ovr')
        except Exception:
            auc_ovr = float('nan')

        # Debug LRs
        current_lrs = [pg['lr'] for pg in optimizer.param_groups]
        print(f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        print(f"Val Acc: {acc:.4f} | Prec(m): {prec_macro:.4f} | Rec(m): {rec_macro:.4f} | "
              f"F1(m): {f1_macro:.4f} | AUC(OvR): {auc_ovr:.4f} | LRs: {current_lrs}")

        # ===== Per-class Metrics =====
        val_labels_np = np.array(val_labels)
        val_preds_np  = np.array(val_preds)
        print("\n📊 Per-class metrics:")
        for class_idx in range(num_classes):
            cls_name = idx_to_class[class_idx]
            true_binary = (val_labels_np == class_idx)
            pred_binary = (val_preds_np == class_idx)

            cls_prec = precision_score(true_binary, pred_binary, zero_division=0)
            cls_rec  = recall_score(true_binary, pred_binary, zero_division=0)
            cls_acc  = np.mean(true_binary == pred_binary)
            print(f"  {cls_name:<20} Prec: {cls_prec:.3f}  Rec: {cls_rec:.3f}  Acc: {cls_acc:.3f}")

        # ===== CSV Logging =====
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

        # ===== Save Best by F1 =====
                # ===== Save Best by F1 (canonical) + also keep Top-K by (F1, recall) =====
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

            # also save into the Top‑K pool
            topk_saver.save(f1_macro, rec_macro, epoch, ckpt, prefix="multi")

            last_val_labels, last_val_preds = val_labels[:], val_preds[:]
            with open('multi_val_predictions.pkl', 'wb') as f:
                pickle.dump({'labels': last_val_labels, 'preds': last_val_preds}, f)

        # ===== Early stop on 3-epoch moving-average of macro-F1 =====
        if ma_stop.step(f1_macro):
            print("⏹️ Early stopping (no improvement in MA macro‑F1).")
            break


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

# ==================== Main ====================
if __name__ == "__main__":
    # Build cache if missing
    for split in ['train', 'val']:
        cache_split_if_missing(split)

    # Loaders (smaller batch by default due to backbone choices)
    train_loader, val_loader, class_counts = get_multi_loaders(
        os.path.join(cache_dir, "train"),
        os.path.join(cache_dir, "val"),
        batch_size=24,   # adjust to your GPU (try 16 if OOM, 32 if plenty)
        num_workers=4
    )

    print("✅ Dataloaders ready.")

    # Model
    backbone = DualBackbone(deit_variant='deit3_small_patch16_224', freeze_all=True, img_size=224)
    in_features = backbone.cnn_feat_dim + backbone.deit_feat_dim
    classifier = MultiClassifier(in_features, num_classes)
    print(f"✅ Model initialized. Classes kept: {KEPT_CLASSES}")
    print(f"   ConvNeXt-Tiny feat dim: {backbone.cnn_feat_dim}, DeiT-III Tiny feat dim: {backbone.deit_feat_dim}")

    # Train
    labels_best, preds_best = train_multiclass_model(
        backbone, classifier, train_loader, val_loader, class_counts,
        epochs=100, patience=10, lr=3e-4,
        csv_path='multi_metrics.csv', model_path='multi_best_classifier.pt',
        use_amp=True, warmup_epochs=5,
        lr_backbone=1e-5,
        mixup_alpha=0.4, cutmix_alpha=1.0, mixup_off_pct=0.2,
        mixup_burst_len_after_resume=8,     # <- change here if you want
        finetune_lr_drop=0.25,              # <- change here if you want
        topk_checkpoints=3
    )


    # Plots
    if len(labels_best) and len(preds_best):
        plot_confusion_matrix_multiclass(
            labels_best, preds_best,
            class_names=[idx_to_class[i] for i in range(num_classes)],
            title="Best-Epoch Validation Confusion Matrix"
        )

    plot_training_metrics('multi_metrics.csv')
