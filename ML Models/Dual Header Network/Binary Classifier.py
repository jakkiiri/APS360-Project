import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import models, transforms
from torchvision.models import ResNet50_Weights
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from tqdm import tqdm
import timm

# === Paths and Device ===
original_split_dir = r'C:\Users\shore\Desktop\APS360\Datasets\DataSplit2'
cache_dir = r'C:\Users\shore\Desktop\APS360\Datasets\Cache_Multi\multi'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# === Label Mapping ===
label_mapping = {
    'nevus': 0, 'melanoma': 1, 'bcc': 2, 'keratosis': 3,
    'actinic_keratosis': 4, 'scc': 5, 'dermatofibroma': 6,
    'lentigo': 7, 'vascular_lesion': 8
}

# === Transforms ===
# Used during caching — all images will be saved at 224×224
cache_transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Consistent size for all .pt files
    transforms.ToTensor()
])

# Used during training — keeps size consistent, adds augmentation
augmentation_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor()
])

normalize_transform = transforms.Normalize([0.485, 0.456, 0.406],
                                           [0.229, 0.224, 0.225])

# === Caching if missing ===
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
        ensure_dir(class_tgt)

        image_files = [img_name for img_name in os.listdir(class_src)
                       if img_name.lower().endswith(('.jpg', '.jpeg', '.png'))]

        for img_name in tqdm(image_files, desc=f"Caching {split}/{class_name}", unit='img'):
            img_path = os.path.join(class_src, img_name)
            img = Image.open(img_path).convert("RGB")
            tensor = cache_transform(img)
            base = os.path.splitext(img_name)[0]
            torch.save(tensor, os.path.join(class_tgt, f"{base}.pt"))

# === Dataset (binary labels) ===
class CachedBinaryDataset(Dataset):
    def __init__(self, cache_dir, augment=False):
        self.data = []
        self.augment = augment
        print(f"📦 Loading dataset from {cache_dir}...")
        class_names = os.listdir(cache_dir)

        for class_name in tqdm(class_names, desc="Indexing classes"):
            class_dir = os.path.join(cache_dir, class_name)
            label = label_mapping[class_name]
            binary_label = 1 if label == 0 else 0
            for file in os.listdir(class_dir):
                if file.endswith('.pt'):
                    self.data.append((os.path.join(class_dir, file), binary_label))

        print(f"✅ Indexed {len(self.data)} total samples.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        path, label = self.data[idx]
        image = torch.load(path)
        image = transforms.ToPILImage()(image)
        if self.augment:
            image = augmentation_transform(image)
        else:
            image = transforms.ToTensor()(image)
        image = normalize_transform(image)
        return image, label

# === Data Loaders ===
def get_binary_loaders(train_dir, val_dir, batch_size=64):
    train_dataset = CachedBinaryDataset(train_dir, augment=True)
    val_dataset = CachedBinaryDataset(val_dir, augment=False)

    train_labels = [label for _, label in train_dataset]
    class_counts = np.bincount(train_labels)
    weights = 1. / torch.tensor(class_counts, dtype=torch.float)
    samples_weight = np.array([weights[label] for label in train_labels])
    sampler = WeightedRandomSampler(samples_weight, len(samples_weight), replacement=True)

    return (
        DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, num_workers=4),
        DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    )

# === Dual Backbone Model ===
class DualBackbone(nn.Module):
    def __init__(self):
        super(DualBackbone, self).__init__()

        self.resnet = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        for param in list(self.resnet.parameters())[:-10]:
            param.requires_grad = False
        self.resnet_backbone = nn.Sequential(
            self.resnet.conv1,
            self.resnet.bn1,
            self.resnet.relu,
            self.resnet.maxpool,
            self.resnet.layer1,
            self.resnet.layer2,
            self.resnet.layer3,
            self.resnet.layer4,
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.resnet_feat_dim = self.resnet.fc.in_features

        self.deit = timm.create_model('deit_base_patch16_224', pretrained=True)
        #self.deit = timm.create_model('deit_tiny_patch16_224', pretrained=True)
        for param in list(self.deit.parameters())[:-10]:
            param.requires_grad = False
        self.deit.reset_classifier(0)
        self.deit_feat_dim = self.deit.num_features

    def forward(self, x):
        r = self.resnet_backbone(x)
        r = r.view(r.size(0), -1)  # shape: [B, resnet_feat_dim]

        d = self.deit.forward_features(x)  # shape: [B, num_tokens, emb_dim] or [B, emb_dim]
        if d.ndim == 3:  # ViT output is [B, tokens, dim]
            d = d[:, 0, :]  # Use CLS token

        return torch.cat([r, d], dim=1)  # now both are [B, feat_dim]


# === Binary Classifier ===
class BinaryClassifier(nn.Module):
    def __init__(self, in_features):
        super(BinaryClassifier, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, 128), nn.ReLU(),
            nn.Linear(128, 1)  # Binary
        )

    def forward(self, x):
        return self.net(x)

# === Training ===
from tqdm import tqdm

# === Training ===
import os
import pandas as pd
def freeze_backbone(backbone, train_last_n=0):
    """Freeze all backbone params (optionally keep last N trainable)."""
    # ResNet
    res = list(backbone.resnet.parameters())
    for p in res[:-train_last_n] if train_last_n > 0 else res:
        p.requires_grad = False
    # DeiT
    vit = list(backbone.deit.parameters())
    for p in vit[:-train_last_n] if train_last_n > 0 else vit:
        p.requires_grad = False

def unfreeze_and_attach_binary(backbone, optimizer, lr_backbone=1e-5, weight_decay=1e-4):
    """Unfreeze all backbone params and add them to optimizer with a gentle LR."""
    for p in backbone.resnet.parameters(): p.requires_grad = True
    for p in backbone.deit.parameters():  p.requires_grad = True
    resnet_params = [p for p in backbone.resnet.parameters() if p.requires_grad]
    deit_params   = [p for p in backbone.deit.parameters()   if p.requires_grad]
    if resnet_params:
        optimizer.add_param_group({'params': resnet_params, 'lr': lr_backbone, 'weight_decay': weight_decay})
    if deit_params:
        optimizer.add_param_group({'params': deit_params,   'lr': lr_backbone, 'weight_decay': weight_decay})

def train_binary_model(
    backbone, classifier, train_loader, val_loader,
    epochs=50, patience=10,
    csv_path='binary_metrics.csv', model_path='binary_best_classifier.pt',
    warmup_epochs=3,          # <- NEW: freeze for N epochs
    lr_head=1e-4,             # head LR
    lr_backbone=1e-5,         # LR after unfreezing backbone
    use_sampler=True,         # keep your current sampler by default
    use_pos_weight=False      # alternative imbalance handling (see notes)
):
    backbone = backbone.to(device)
    classifier = classifier.to(device)

    # ---- Freeze everything for warmup ----
    freeze_backbone(backbone, train_last_n=0)

    # ---- Loss (optionally with pos_weight) ----
    if use_pos_weight:
        # Build pos_weight from CURRENT training labels:
        # pos = count of label==1, neg = count of label==0
        # (We can compute from the dataset indices without loading images.)
        train_labels = [lbl for _, lbl in train_loader.dataset]
        pos = max(1, sum(train_labels))
        neg = max(1, len(train_labels) - pos)
        pos_weight = torch.tensor([neg / pos], dtype=torch.float32, device=device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        criterion = nn.BCEWithLogitsLoss()

    # ---- Optimizer: start with HEAD ONLY during warmup ----
    head_params = list(classifier.parameters())
    optimizer = torch.optim.Adam(head_params, lr=lr_head, weight_decay=1e-4)

    from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2)

    # ---- Resume ----
    start_epoch = 0
    best_val_loss = float('inf')
    import pandas as pd, os
    df = pd.DataFrame(columns=[
        'epoch','train_loss','val_loss','val_acc','val_precision','val_recall','val_f1','val_auc','lr_head','lr_backbone'
    ])
    if os.path.exists(model_path):
        print(f"🔁 Resuming from saved model: {model_path}")
        classifier.load_state_dict(torch.load(model_path, map_location='cpu'))
        if os.path.exists(csv_path):
            df_loaded = pd.read_csv(csv_path)
            if not df_loaded.empty:
                df = df_loaded
                best_val_loss = float(df['val_loss'].min())
                start_epoch = int(df['epoch'].max()) + 1
                print(f"📈 Resume epoch {start_epoch}, best val loss {best_val_loss:.4f}")

    # If resuming beyond warmup, unfreeze immediately
    unfrozen = False
    if start_epoch >= warmup_epochs and not unfrozen:
        print(f"🔓 Resuming past warmup (epoch {start_epoch}) — unfreezing backbone now.")
        unfreeze_and_attach_binary(backbone, optimizer, lr_backbone=lr_backbone)
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2)
        unfrozen = True

    patience_counter = 0

    for epoch in range(start_epoch, epochs):
        # Timed unfreeze
        if (not unfrozen) and (epoch >= warmup_epochs):
            print(f"🔓 Unfreezing backbone at epoch {epoch} (after {warmup_epochs} warmup epochs).")
            unfreeze_and_attach_binary(backbone, optimizer, lr_backbone=lr_backbone)
            scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2)
            unfrozen = True

        print(f"\nEpoch {epoch+1}/{epochs}")
        backbone.train(); classifier.train()
        total_loss = 0.0

        train_loop = tqdm(train_loader, desc=f"Training [{epoch+1}]")
        for inputs, labels in train_loop:
            inputs, labels = inputs.to(device), labels.float().to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=False):  # keep it simple; can switch to AMP if you want
                feats = backbone(inputs)
                outputs = classifier(feats).squeeze(1)
                loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            scheduler.step(epoch + train_loop.n / max(1, len(train_loader)))
            total_loss += loss.item()
            train_loop.set_postfix(loss=loss.item())

        avg_train_loss = total_loss / max(1, len(train_loader))

        # ---- Validation ----
        backbone.eval(); classifier.eval()
        val_loss, val_preds, val_labels = 0.0, [], []
        val_loop = tqdm(val_loader, desc=f"Validation [{epoch+1}]")
        with torch.no_grad():
            for inputs, labels in val_loop:
                inputs, labels = inputs.to(device), labels.float().to(device)
                feats = backbone(inputs)
                outputs = classifier(feats).squeeze(1)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                probs = torch.sigmoid(outputs)
                preds = (probs > 0.5).long()
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
                val_loop.set_postfix(loss=loss.item())

        avg_val_loss = val_loss / max(1, len(val_loader))

        # ---- Metrics ----
        acc  = accuracy_score(val_labels, val_preds)
        prec = precision_score(val_labels, val_preds, zero_division=0)
        rec  = recall_score(val_labels, val_preds, zero_division=0)
        f1   = f1_score(val_labels, val_preds, zero_division=0)
        # IMPORTANT: AUC should use probabilities, not 0/1 preds
        try:
            auc = roc_auc_score(val_labels, (np.array(val_preds)).astype(float))
        except Exception:
            auc = float('nan')

        # show LRs (first group is head; groups 2/3 appear after unfreeze)
        lrs = [pg['lr'] for pg in optimizer.param_groups]
        print(f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        print(f"Val Acc: {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f} | AUC: {auc:.4f} | LRs: {lrs}")

        # ---- Save CSV row ----
        row = {
            'epoch': epoch, 'train_loss': avg_train_loss, 'val_loss': avg_val_loss,
            'val_acc': acc, 'val_precision': prec, 'val_recall': rec, 'val_f1': f1, 'val_auc': auc,
            'lr_head': lrs[0], 'lr_backbone': (lrs[1] if len(lrs) > 1 else 0.0)
        }
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        df.to_csv(csv_path, index=False)

        # ---- Checkpoint on best val loss ----
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(classifier.state_dict(), model_path)
            print("💾 Saved Best Model!")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("⏹️ Early stopping triggered.")
                break

    # Optional: confusion matrix
    plot_confusion_matrix(val_labels, val_preds, title="Final Validation Confusion Matrix")


import seaborn as sns

def plot_confusion_matrix(true_labels, pred_labels, title="Confusion Matrix"):
    cm = confusion_matrix(true_labels, pred_labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap='Blues', values_format='d')
    plt.title(title)
    plt.grid(False)
    plt.tight_layout()
    plt.show()

def plot_training_metrics(csv_path='binary_metrics.csv'):
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
        ('val_precision', 'Validation Precision'),
        ('val_f1', 'Validation F1 Score'),
        ('val_auc', 'Validation AUC')
    ]

    for ax, (col, title) in zip(axes.flat, metrics):
        sns.lineplot(data=df, x='epoch', y=col, ax=ax, marker='o')
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(col)
        ax.grid(True)

    plt.suptitle("Training Metrics Over Time", fontsize=16)
    plt.tight_layout()
    plt.show()
from torch.utils.data import Subset

def make_overfit_subset_loaders(cache_dir, batch_size=32, n_per_class=100):
    """
    Build small train loader on ~200 samples (balanced if possible) with NO augmentation.
    Returns: train_loader_small
    """
    # Use your existing dataset class, but no augmentation for overfit test
    full_ds = CachedBinaryDataset(os.path.join(cache_dir, "train"), augment=False)

    # Collect indices by label from the dataset index (no __getitem__ calls)
    idx_pos, idx_neg = [], []
    for i, (_, lbl) in enumerate(full_ds.data):
        (idx_pos if lbl == 1 else idx_neg).append(i)

    # Sample up to n_per_class from each
    n_pos = min(n_per_class, len(idx_pos))
    n_neg = min(n_per_class, len(idx_neg))
    sel = idx_pos[:n_pos] + idx_neg[:n_neg]
    if len(sel) == 0:
        raise RuntimeError("Overfit subset is empty. Check your cache_dir path and dataset.")

    small_ds = Subset(full_ds, sel)
    # Plain shuffle (no sampler), pin_memory if CUDA
    train_loader_small = DataLoader(
        small_ds, batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=(device.type == 'cuda'),
        persistent_workers=True, prefetch_factor=2
    )
    print(f"🔎 Overfit subset: pos={n_pos}, neg={n_neg}, total={len(sel)}")
    return train_loader_small

def unfreeze_all(backbone):
    for p in backbone.resnet.parameters(): p.requires_grad = True
    for p in backbone.deit.parameters():  p.requires_grad = True

from torch.amp import autocast, GradScaler

def overfit_200_train(backbone, classifier, train_loader_small,
                      max_epochs=200, target_acc=0.99, patience_success=3,
                      lr_head=1e-3, lr_backbone=1e-4, use_amp=True):
    """
    Train on a tiny subset until it hits ~100% training accuracy.
    """
    backbone = backbone.to(device)
    classifier = classifier.to(device)
    backbone.train(); classifier.train()

    # Make sure EVERYTHING learns
    unfreeze_all(backbone)

    # BCE loss (binary)
    criterion = nn.BCEWithLogitsLoss()

    # Separate LRs for head vs backbone
    optim = torch.optim.Adam([
        {'params': classifier.parameters(), 'lr': lr_head, 'weight_decay': 0.0},
        {'params': backbone.parameters(),   'lr': lr_backbone, 'weight_decay': 0.0},
    ])

    scaler = GradScaler(enabled=use_amp and (device.type == 'cuda'))

    hits_in_row = 0
    for epoch in range(1, max_epochs + 1):
        total, correct, running_loss = 0, 0, 0.0

        for x, y in train_loader_small:
            x = x.to(device, non_blocking=True)
            y = y.float().to(device, non_blocking=True)

            optim.zero_grad(set_to_none=True)
            with autocast(device_type='cuda', enabled=use_amp and (device.type=='cuda')):
                feats = backbone(x)
                logits = classifier(feats).squeeze(1)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()

            running_loss += loss.item() * x.size(0)
            preds = (torch.sigmoid(logits) > 0.5).long()
            correct += (preds == y.long()).sum().item()
            total += x.size(0)

        train_loss = running_loss / max(1, total)
        train_acc  = correct / max(1, total)

        print(f"[Overfit] Epoch {epoch:03d} | loss {train_loss:.4f} | acc {train_acc:.4f}")

        if train_acc >= target_acc:
            hits_in_row += 1
            if hits_in_row >= patience_success:
                print(f"✅ Reached ≥{int(target_acc*100)}% train accuracy for {patience_success} epochs. Sanity check passed.")
                break
        else:
            hits_in_row = 0

    if hits_in_row < patience_success:
        print("⚠️ Did not reach near-100% accuracy on the tiny subset. Check data/labels/augmentations/normalization.")


# === Run ===
if __name__ == "__main__":
    for split in ['train', 'val']:
        cache_split_if_missing(split)
    print("Loading Completed")
    backbone = DualBackbone()
    print("initialized Model")
    feature_dim = backbone.resnet_feat_dim + backbone.deit_feat_dim
    classifier = BinaryClassifier(feature_dim)
    '''
    small_loader = make_overfit_subset_loaders(cache_dir, batch_size=32, n_per_class=100)

    # Important: disable ALL augmentation for the overfit test (done by dataset above)

    overfit_200_train(
        backbone, classifier, small_loader,
        max_epochs=200, target_acc=0.99, patience_success=3,
        lr_head=1e-3, lr_backbone=1e-4, use_amp=True
    )
    '''
    train_loader, val_loader = get_binary_loaders(
        os.path.join(cache_dir, "train"),
        os.path.join(cache_dir, "val")
    )
    print("Started Training")

    train_binary_model(
        backbone, classifier, train_loader, val_loader,
        epochs=50, patience=5,
        warmup_epochs=3,      # freeze first 3 epochs
        lr_head=1e-3,         # head LR
        lr_backbone=5e-4,     # gentle LR when unfreezing
        use_sampler=True,     # keep your WeightedRandomSampler
        use_pos_weight=False  # or switch to True (see below)   
    )   

# After training
    plot_training_metrics('binary_metrics.csv')
