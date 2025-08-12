# ==== Multiclass Dual-Backbone Training Script (nevus ignored) ====
# ResNet50 + DeiT (tiny/base) -> concatenated features -> MLP -> 8 classes
# Weighted sampler, class-weighted CE, CSV logging + resume, scheduler, AMP optional

import os
import pickle
from sched import scheduler
import time
import math
import pandas as pd
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from torchvision import models, transforms
from torchvision.models import ResNet50_Weights

import timm
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
)

# ==================== Paths / Device ====================
original_split_dir = r'C:\Users\shore\Desktop\APS360\Datasets\DataSplit2'
cache_dir = r'C:\Users\shore\Desktop\APS360\Datasets\Cache_Multi\multi320'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
use_cuda = device.type == 'cuda'

# ==================== Labels (nevus ignored) ====================
IGNORE_CLASSES = {} #{'nevus'}  # <- anything in here is excluded entirely

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
# Cache at 224 -> faster IO + consistent model input size
cache_transform = transforms.Compose([
    transforms.Resize((320, 320)),
    transforms.ToTensor()
])

train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor()
])

val_transform = transforms.Compose([
    transforms.ToTensor()
])
# normalize_transform stays the same


normalize_transform = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)

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
        image = torch.load(path)  # tensor [3,224,224]
        image = transforms.ToPILImage()(image)

        if self.augment:
            image = train_transform(image)
        else:
            image = val_transform(image)

        image = normalize_transform(image)
        return image, label

# ==================== Data Loaders ====================
def get_multi_loaders(train_dir, val_dir, batch_size=120, num_workers=4):
    train_dataset = CachedMultiDataset(train_dir, augment=True)
    val_dataset = CachedMultiDataset(val_dir, augment=False)

    # Use labels from dataset index (no image loads)
    train_labels = [lbl for _, lbl in train_dataset.data]
    if len(train_labels) == 0:
        raise RuntimeError("No training samples found after filtering. Check your paths and IGNORE_CLASSES.")

    class_counts = np.bincount(train_labels, minlength=num_classes)
    print(f"📊 Train class counts (after filtering): {dict(zip(range(num_classes), class_counts.tolist()))}")

    # Inverse frequency weights per sample
    weights = 1.0 / torch.tensor(class_counts, dtype=torch.float32).clamp(min=1)
    sample_weights = torch.tensor([weights[l] for l in train_labels], dtype=torch.float32)
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, #sampler=sampler,
                              shuffle=True ,num_workers=num_workers, pin_memory=use_cuda)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=use_cuda)
    return train_loader, val_loader, weights

# ==================== Model ====================
class DualBackbone(nn.Module):
    def __init__(self, deit_variant='deit_base_patch16_224', freeze_all=True, img_size=224):
        super().__init__()
        # ---- ResNet50 ----
        self.resnet = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        if freeze_all:
            for p in self.resnet.parameters():
                p.requires_grad = False
        self.resnet_backbone = nn.Sequential(
            self.resnet.conv1, self.resnet.bn1, self.resnet.relu, self.resnet.maxpool,
            self.resnet.layer1, self.resnet.layer2, self.resnet.layer3, self.resnet.layer4,
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.resnet_feat_dim = self.resnet.fc.in_features  # 2048

        # ---- DeiT ----
        # IMPORTANT: pass img_size so timm builds positional embeddings for 320
        self.deit = timm.create_model(deit_variant, pretrained=True, img_size=img_size)
        if freeze_all:
            for p in self.deit.parameters():
                p.requires_grad = False
        # remove classifier head; keep features only
        self.deit.reset_classifier(0)
        self.deit_feat_dim = self.deit.num_features  # 768 for deit_base

    def forward(self, x):
        # ResNet branch
        r = self.resnet_backbone(x)          # [B, 2048, 1, 1]
        r = r.view(r.size(0), -1)            # [B, 2048]

        # DeiT/Vision Transformer branch
        d = self.deit.forward_features(x)    # [B, tokens, C] or [B, C]
        if d.ndim == 3:                      # some timm models return tokens
            d = d[:, 0, :]                   # CLS token
        # concat
        return torch.cat([r, d], dim=1)      # [B, 2048 + deit_feat_dim]



class MultiClassifier(nn.Module):
    def __init__(self, in_features, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 768), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(768, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.ReLU(),
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

# ==================== Training ====================
from torch.amp import autocast, GradScaler

import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        """
        alpha: class weighting factor (float or tensor of shape [num_classes])
               If None, no weighting is applied.
        gamma: focusing parameter (higher -> more focus on hard examples)
        reduction: 'none' | 'mean' | 'sum'
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

        if isinstance(alpha, (list, torch.Tensor)):
            self.alpha = torch.tensor(alpha, dtype=torch.float32)
        elif isinstance(alpha, (float, int)):
            self.alpha = torch.tensor([alpha], dtype=torch.float32)

    def forward(self, inputs, targets):
        """
        inputs: logits of shape [B, C]
        targets: ground truth labels of shape [B]
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction='none',
                                  weight=self.alpha.to(inputs.device) if self.alpha is not None else None)
        pt = torch.exp(-ce_loss)  # probability of the true class
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss

def unfreeze_and_attach(backbone, optimizer, lr_backbone=1e-5, weight_decay=1e-4):
    """
    Unfreeze both backbones and add them to the optimizer with a gentle LR.
    Safe to call multiple times: it won't duplicate params in the optimizer.
    """
    # 1) Unfreeze
    for p in backbone.resnet.parameters(): p.requires_grad = True
    for p in backbone.deit.parameters():  p.requires_grad = True

    # 2) Avoid adding duplicates to the optimizer
    existing = set()
    for g in optimizer.param_groups:
        for p in g['params']:
            existing.add(id(p))

    resnet_params = [p for p in backbone.resnet.parameters()
                     if p.requires_grad and id(p) not in existing]
    deit_params   = [p for p in backbone.deit.parameters()
                     if p.requires_grad and id(p) not in existing]

    # 3) Attach as new param groups (with lower LR)
    if resnet_params:
        optimizer.add_param_group({
            'params': resnet_params,
            'lr': lr_backbone,
            'weight_decay': weight_decay
        })
    if deit_params:
        optimizer.add_param_group({
            'params': deit_params,
            'lr': lr_backbone,
            'weight_decay': weight_decay
        })

def train_multiclass_model(
    backbone, classifier, train_loader, val_loader, class_weights,
    epochs=50, patience=20, lr=5e-4,
    csv_path='multi_metrics.csv', model_path='multi_best_checkpoint.pt',
    use_amp=True, warmup_epochs=3, lr_backbone=1e-5
):
    backbone = backbone.to(device)
    classifier = classifier.to(device)

    # Loss (no class weights if using WeightedRandomSampler)
    # ---- Loss ----
    # Example: if you want to use class weights with focal loss
    class_weights = class_weights.to(device=device, dtype=torch.float32)  # per-class weights
    criterion = FocalLoss(gamma=2.0, alpha=None, reduction="mean")


    '''
    criterion = nn.CrossEntropyLoss()
    if class_weights is not None:
        class_weights = class_weights.to(device=device, dtype=torch.float32)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    '''


    # Optimizer with only trainable params
    params = list(filter(lambda p: p.requires_grad, backbone.parameters())) + list(classifier.parameters())
    optimizer = torch.optim.Adam(params, lr=lr, weight_decay=1e-4)

    # Scheduler: Reduce LR when F1 stops improving
    from torch.optim.lr_scheduler import ReduceLROnPlateau
    scheduler = ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=7,
        threshold=1e-3, cooldown=0, min_lr=1e-7, verbose=True
    )

    scaler = GradScaler(enabled=use_amp and use_cuda)

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

    # If we resume past warmup, ensure backbones are unfrozen
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

        print(f"\nEpoch {epoch+1}/{epochs}")
        backbone.train()
        classifier.train()
        total_loss = 0.0

        train_loop = tqdm(train_loader, desc=f"Training [{epoch+1}]")
        for inputs, targets in train_loop:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad(set_to_none=True)
            with autocast(device_type='cuda', enabled=use_amp and use_cuda):
                feats = backbone(inputs)
                logits = classifier(feats)
                loss = criterion(logits, targets)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            train_loop.set_postfix(loss=loss.item())

        avg_train_loss = total_loss / max(1, len(train_loader))

        # ===== Validation =====
        backbone.eval()
        classifier.eval()
        val_loss = 0.0
        val_labels, val_preds, val_probs = [], [], []

        val_loop = tqdm(val_loader, desc=f"Validation [{epoch+1}]")
        with torch.no_grad():
            for inputs, targets in val_loop:
                inputs, targets = inputs.to(device), targets.to(device)
                with autocast(device_type='cuda', enabled=use_amp and use_cuda):
                    feats = backbone(inputs)
                    logits = classifier(feats)
                    loss = criterion(logits, targets)
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

        # Step scheduler by F1
        scheduler.step(f1_macro)

        # Debug LRs
        current_lrs = [pg['lr'] for pg in optimizer.param_groups]
        print(f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        print(f"Val Acc: {acc:.4f} | Prec(m): {prec_macro:.4f} | Rec(m): {rec_macro:.4f} | "
              f"F1(m): {f1_macro:.4f} | AUC(OvR): {auc_ovr:.4f} | LRs: {current_lrs} | Best F1: {best_val_f1:.4f}")

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
        if f1_macro > best_val_f1 + 1e-6:
            best_val_f1 = f1_macro
            patience_counter = 0

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

            last_val_labels, last_val_preds = val_labels[:], val_preds[:]
            with open('multi_val_predictions.pkl', 'wb') as f:
                pickle.dump({'labels': last_val_labels, 'preds': last_val_preds}, f)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("⏹️ Early stopping triggered (no F1 improvement).")
                break

    return last_val_labels, last_val_preds




from torch.utils.data import Subset

def disable_dropout(module: nn.Module):
    for m in module.modules():
        if isinstance(m, nn.Dropout):
            m.p = 0.0

def unfreeze_all_multiclass(backbone: nn.Module):
    for p in backbone.resnet.parameters(): p.requires_grad = True
    for p in backbone.deit.parameters():  p.requires_grad = True

def make_overfit_subset_loader_multiclass(cache_dir, n_per_class=25, batch_size=16):
    """
    Build a small stratified subset loader (~n_per_class per class, no aug, plain shuffle).
    """
    ds = CachedMultiDataset(os.path.join(cache_dir, "train"), augment=False)

    per_class_idxs = {c: [] for c in range(num_classes)}
    for i, (_, y) in enumerate(ds.data):  # use the index list, no image loads here
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
from sklearn.metrics import f1_score, roc_curve
import numpy as np

# ========= THRESHOLD SWEEP (fixed) =========
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
def find_f1_thresholds_ovr(y_true_int, probs_2d, num_classes):
    """Return per-class thresholds that maximize F1 (OvR) on validation."""
    y_true = np.eye(num_classes)[y_true_int]  # [N, C] one-hot
    thresholds = np.zeros(num_classes, dtype=np.float32)
    for k in range(num_classes):
        yk = y_true[:, k]
        pk = probs_2d[:, k]
        # candidate thresholds = unique probabilities (plus 0/1 guards)
        cand = np.unique(pk)
        cand = np.concatenate(([0.0], cand, [1.0]))
        best_t, best_f1 = 0.5, -1.0
        for t in cand:
            pred_k = (pk >= t).astype(int)
            f1 = f1_score(yk, pred_k, zero_division=0)
            if f1 > best_f1:
                best_f1, best_t = f1, t
        thresholds[k] = best_t
    return thresholds

def find_youden_thresholds_ovr(y_true_int, probs_2d, num_classes):
    """Return per-class thresholds using Youden's J = TPR - FPR."""
    y_true = np.eye(num_classes)[y_true_int]
    thresholds = np.zeros(num_classes, dtype=np.float32)
    for k in range(num_classes):
        yk = y_true[:, k]
        pk = probs_2d[:, k]
        fpr, tpr, thr = roc_curve(yk, pk)
        j = tpr - fpr
        thresholds[k] = thr[np.argmax(j)]
    # roc_curve can return thresholds beyond [0,1]; clip for safety
    return np.clip(thresholds, 0.0, 1.0)

def predict_with_thresholds_ovr(probs_2d, thresholds):
    """
    Choose class = argmax(p_k - τ_k) if max > 0, else plain argmax.
    """
    score = probs_2d - thresholds[None, :]
    idx = np.argmax(score, axis=1)
    safe = (score[np.arange(score.shape[0]), idx] > 0.0)
    fallback = np.argmax(probs_2d, axis=1)
    return np.where(safe, idx, fallback)
@torch.no_grad()
def _load_ckpt_to_device(backbone, classifier, ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    if isinstance(ckpt, dict) and ('classifier_state' in ckpt or 'backbone_state' in ckpt):
        if 'backbone_state' in ckpt:
            backbone.load_state_dict(ckpt['backbone_state'], strict=False)
        if 'classifier_state' in ckpt:
            classifier.load_state_dict(ckpt['classifier_state'], strict=False)
    else:
        classifier.load_state_dict(ckpt, strict=False)
    backbone.to(device).eval()
    classifier.to(device).eval()

@torch.no_grad()
def collect_val_probs_labels(backbone, classifier, val_loader, device, use_amp=True):
    all_probs, all_labels = [], []
    # guard AMP strictly to CUDA
    use_cuda_amp = (use_amp and device.type == 'cuda')
    for x, y in tqdm(val_loader, desc="Val inference for sweep"):
        x = x.to(device, non_blocking=True)
        with autocast(device_type='cuda', enabled=use_cuda_amp):
            feats  = backbone(x)
            logits = classifier(feats)
            probs  = F.softmax(logits, dim=1)
        all_probs.append(probs.detach().cpu().numpy())
        all_labels.append(y.detach().cpu().numpy())
    probs_2d   = np.concatenate(all_probs, axis=0)
    labels_1d  = np.concatenate(all_labels, axis=0).astype(int)
    return probs_2d, labels_1d

def report_metrics(name, y_true, y_pred, class_names=None):
    acc   = accuracy_score(y_true, y_pred)
    precm = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recm  = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1m   = f1_score(y_true, y_pred, average='macro', zero_division=0)
    print(f"\n== {name} ==")
    print(f"Acc: {acc:.4f} | Prec(m): {precm:.4f} | Rec(m): {recm:.4f} | F1(m): {f1m:.4f}")


def overfit_200_train_multiclass(backbone, classifier, train_loader_small,
                                 max_epochs=200, target_acc=0.99, patience_success=3,
                                 lr_head=1e-3, lr_backbone=1e-4, use_amp=True):
    """
    Train on the tiny stratified subset until ~100% train accuracy.
    Disables dropout, removes weight decay, and unfreezes all backbone params.
    """
    backbone = backbone.to(device)
    classifier = classifier.to(device)
    backbone.train(); classifier.train()

    unfreeze_all_multiclass(backbone)
    disable_dropout(classifier)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optim = torch.optim.Adam([
        {'params': classifier.parameters(), 'lr': lr_head,     'weight_decay': 0.0},
        {'params': backbone.parameters(),   'lr': lr_backbone, 'weight_decay': 0.0},
    ])

    scaler = GradScaler(enabled=use_amp and use_cuda)
    hits = 0

    for epoch in range(1, max_epochs + 1):
        total, correct, running_loss = 0, 0, 0.0

        for x, y in train_loader_small:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optim.zero_grad(set_to_none=True)
            with autocast(device_type='cuda', enabled=use_amp and use_cuda):
                feats  = backbone(x)
                logits = classifier(feats)
                loss   = criterion(logits, y)

            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()

            running_loss += loss.item() * x.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total   += x.size(0)

        train_loss = running_loss / max(1, total)
        train_acc  = correct / max(1, total)
        print(f"[Overfit-8cls] Epoch {epoch:03d} | loss {train_loss:.4f} | acc {train_acc:.4f}")

        hits = hits + 1 if train_acc >= target_acc else 0
        if hits >= patience_success:
            print(f"✅ Reached ≥{int(target_acc*100)}% train accuracy for {patience_success} epochs. Multiclass sanity check passed.")
            break

    if hits < patience_success:
        print("⚠️ Did not reach near-100% on the tiny multiclass subset. Re-check labels/augment/normalization.")


# ==================== Main ====================
if __name__ == "__main__":
    # Build cache if missing
    for split in ['train', 'val']:
        cache_split_if_missing(split)

    # Loaders
    train_loader, val_loader, class_weights = get_multi_loaders(
    os.path.join(cache_dir, "train"),
    os.path.join(cache_dir, "val"),
    batch_size=32,      # smaller for 320px
    num_workers=4
)

    print("✅ Dataloaders ready.")

    # Model
    backbone = DualBackbone(deit_variant='deit_tiny_patch16_224', freeze_all=True, img_size=320)
    in_features = backbone.resnet_feat_dim + backbone.deit_feat_dim
    classifier = MultiClassifier(in_features, num_classes)
    print(f"✅ Model initialized. Classes kept: {KEPT_CLASSES}")
    # ---- Threshold sweep on validation (no training) ----
    import json

    DO_SWEEP = False
    if DO_SWEEP:
        ckpt_path = 'multi_best_classifier.pt'
        print(f"🔎 Loading checkpoint for sweep: {ckpt_path}")
        _load_ckpt_to_device(backbone, classifier, ckpt_path, device)

        probs_2d, y_true_int = collect_val_probs_labels(backbone, classifier, val_loader, device, use_amp=True)

        # Baseline
        y_pred_argmax = probs_2d.argmax(axis=1)
        report_metrics("Baseline Argmax", y_true_int, y_pred_argmax, class_names=[idx_to_class[i] for i in range(num_classes)])

        # Per-class F1-opt thresholds
        tau_f1   = find_f1_thresholds_ovr(y_true_int, probs_2d, num_classes)
        y_pred_f1 = predict_with_thresholds_ovr(probs_2d, tau_f1)
        report_metrics("Per-class F1-max", y_true_int, y_pred_f1)

        # Per-class Youden thresholds
        tau_j   = find_youden_thresholds_ovr(y_true_int, probs_2d, num_classes)
        y_pred_j = predict_with_thresholds_ovr(probs_2d, tau_j)
        report_metrics("Per-class Youden", y_true_int, y_pred_j)

        # Save thresholds
        np.save('thresholds_f1.npy', tau_f1)
        np.save('thresholds_youden.npy', tau_j)
        print("✅ Saved thresholds: thresholds_f1.npy, thresholds_youden.npy")


        # ---- Overfit sanity check (set True to run, False to skip) ----
    RUN_OVERFIT_SANITY = False
    if RUN_OVERFIT_SANITY:
        small_loader = make_overfit_subset_loader_multiclass(cache_dir, n_per_class=25, batch_size=64)
        overfit_200_train_multiclass(
            backbone, classifier, small_loader,
            max_epochs=200, target_acc=0.99, patience_success=3,
            lr_head=1e-3, lr_backbone=1e-4, use_amp=True
        )
        # exit here so it doesn't proceed to full training in the same run
        import sys; sys.exit(0)

    labels_best, preds_best = train_multiclass_model(
        backbone, classifier, train_loader, val_loader, class_weights,
        epochs=70, patience=10, lr=3e-4,                 # head LR
        csv_path='multi_metrics.csv', model_path='multi_best_classifier.pt',
        use_amp=True,
        warmup_epochs=5,                                 # longer frozen warmup
        lr_backbone=1e-5                                 # gentler for backbones
    )


    # Plots
    if len(labels_best) and len(preds_best):
        plot_confusion_matrix_multiclass(
            labels_best, preds_best,
            class_names=[idx_to_class[i] for i in range(num_classes)],
            title="Best-Epoch Validation Confusion Matrix (nevus ignored)"
        )

    plot_training_metrics('multi_metrics.csv')
