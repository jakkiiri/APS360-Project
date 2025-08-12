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

# === Paths and Device ===
original_split_dir = r'C:\Users\shore\Desktop\APS360\Datasets\DataSplit2'
cache_dir = r'C:\Users\shore\Desktop\APS360\Datasets\Cache_Multi\multi'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Label Mapping ===
label_mapping = {
    'nevus': 0, 'melanoma': 1, 'bcc': 2, 'seborrheic_keratosis': 3,
    'actinic_keratosis': 4, 'scc': 5, 'dermatofibroma': 6,
    'lentigo': 7, 'vascular_lesion': 8
}

# === Cache transform (resize + center crop to remove border) ===
cache_transform = transforms.Compose([
    transforms.Resize((350, 350)),
    #transforms.CenterCrop(224),
    transforms.ToTensor()
])

# === Augmentation for training ===
augmentation_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.ToTensor()
])

# === Normalize for model input ===
normalize_transform = transforms.Normalize([0.485, 0.456, 0.406],
                                           [0.229, 0.224, 0.225])

# === Cache if missing ===
def ensure_dir(path): os.makedirs(path, exist_ok=True)

def cache_split_if_missing(split):
    src_path = os.path.join(original_split_dir, split)
    tgt_path = os.path.join(cache_dir, split)
    if os.path.exists(tgt_path):
        print(f"✅ Cache already exists for split '{split}' → skipping.")
        return

    print(f"⏳ Caching '{split}' set...")
    for class_name in os.listdir(src_path):
        class_src = os.path.join(src_path, class_name)
        class_tgt = os.path.join(tgt_path, class_name)
        ensure_dir(class_tgt)

        for img_name in tqdm(os.listdir(class_src), desc=f"Caching {split}/{class_name}"):
            if not img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
            img_path = os.path.join(class_src, img_name)
            img = Image.open(img_path).convert("RGB")
            tensor = cache_transform(img)
            base = os.path.splitext(img_name)[0]
            torch.save(tensor, os.path.join(class_tgt, f"{base}.pt"))

# Cache train and val splits
for split in ['train', 'val']:
    cache_split_if_missing(split)

# === Dataset with optional augment and undersample ===
class CachedDataset(Dataset):
    def __init__(self, cache_dir, augment=False, target_per_class=1200):
        self.data = []
        self.augment = augment
        self.samples_by_class = {v: [] for v in label_mapping.values()}

        for class_name in os.listdir(cache_dir):
            class_dir = os.path.join(cache_dir, class_name)
            if not os.path.isdir(class_dir): continue
            label = label_mapping[class_name]
            pt_files = [f for f in os.listdir(class_dir) if f.endswith(".pt")]
            full_paths = [os.path.join(class_dir, f) for f in pt_files]
            self.samples_by_class[label].extend(full_paths)

        # Balancing logic
        for label, files in self.samples_by_class.items():
            n = len(files)

            if target_per_class is None:
                # Use all original files, no balancing
                for path in files:
                    self.data.append((path, label, False))
            elif n >= target_per_class:
                selected = files[:target_per_class]
                for path in selected:
                    self.data.append((path, label, False))
            else:
                for path in files:
                    self.data.append((path, label, False))
                aug_needed = target_per_class - n
                for i in range(aug_needed):
                    path = files[i % n]
                    self.data.append((path, label, True))


        class_counts = {k: sum(1 for _, l, _ in self.data if l == k) for k in label_mapping.values()}
        print(f"📦 Loaded {len(self.data)} balanced samples from {cache_dir}")
        print(f"📊 Class counts (after balancing): {class_counts}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        path, label, do_augment = self.data[idx]
        image = torch.load(path)
        image = transforms.ToPILImage()(image)

        if do_augment and self.augment:
            image = augmentation_transform(image)
        else:
            image = transforms.ToTensor()(image)

        image = normalize_transform(image)
        return image, label


# === Data Loaders with Weighted Sampling ===
def get_loaders(train_dir, val_dir, batch_size=32):
    train_dataset = CachedDataset(train_dir, augment=True, target_per_class=1000)
    val_dataset = CachedDataset(val_dir, augment=False, target_per_class=None)  # no balancing in val

    train_labels = [lbl for _, lbl in train_dataset]
    class_counts = np.bincount(train_labels)
    weights = 1. / torch.tensor(class_counts, dtype=torch.float)
    global multi_class_weights
    multi_class_weights = weights / weights.sum() * len(class_counts)

    samples_weight = np.array([weights[lbl] for lbl in train_labels])
    sampler = WeightedRandomSampler(samples_weight, len(samples_weight), replacement=True)

    return (
        DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, num_workers=1, pin_memory=True),
        DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=1, pin_memory=True)
    )


# === Backbone ===
class BackboneExtractor(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone

    def forward(self, x):
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        return x.view(x.size(0), -1)

# === Classifier ===
class MultiClassifier(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 9)
        )

    def forward(self, x):
        return self.net(x)

# === Focal Loss ===
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, weight=self.alpha, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean() if self.reduction == 'mean' else focal_loss.sum()

# === Training Function ===
def train_model(backbone, classifier, train_loader, val_loader, num_epochs=100, patience=10, backbone_name='resnet'):
    backbone = backbone.to(device)
    classifier = classifier.to(device)
    criterion = FocalLoss(alpha=multi_class_weights.to(device), gamma=2)
    optimizer = torch.optim.Adam([
        {'params': classifier.parameters(), 'lr': 1e-4},
        {'params': backbone.parameters(), 'lr': 1e-4}
    ], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=[1e-4, 1e-4], steps_per_epoch=len(train_loader), epochs=num_epochs
    )

    best_val_loss = float('inf')
    patience_counter = 0
    train_loss_history, val_loss_history = [], []

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs} [multi]")
        classifier.train()
        backbone.train()
        train_loss = 0
        train_preds, train_labels = [], []

        for inputs, targets in tqdm(train_loader, desc="Training"):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            features = backbone(inputs)
            outputs = classifier(features)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            scheduler.step()
            train_loss += loss.item()
            train_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
            train_labels.extend(targets.cpu().numpy())

        avg_train_loss = train_loss / len(train_loader)
        train_loss_history.append(avg_train_loss)

        classifier.eval()
        backbone.eval()
        val_loss = 0
        val_preds, val_labels, val_probs = [], [], []

        with torch.no_grad():
            for inputs, targets in tqdm(val_loader, desc="Validation"):
                inputs, targets = inputs.to(device), targets.to(device)
                features = backbone(inputs)
                outputs = classifier(features)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
                probs = F.softmax(outputs, dim=1)
                val_preds.extend(torch.argmax(probs, dim=1).cpu().numpy())
                val_labels.extend(targets.cpu().numpy())
                val_probs.extend(probs.cpu().numpy())

        avg_val_loss = val_loss / len(val_loader)
        val_loss_history.append(avg_val_loss)

        acc = accuracy_score(val_labels, val_preds)
        prec = precision_score(val_labels, val_preds, average='macro', zero_division=0)
        rec = recall_score(val_labels, val_preds, average='macro', zero_division=0)
        f1 = f1_score(val_labels, val_preds, average='macro', zero_division=0)
        auc = roc_auc_score(np.eye(9)[val_labels], np.array(val_probs), multi_class='ovr')

        print(f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        print(f"Val Acc: {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f} | AUC: {auc:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(classifier.state_dict(), f'{backbone_name}_multi_best_model.pt')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

    cm = confusion_matrix(val_labels, val_preds)
    ConfusionMatrixDisplay(confusion_matrix=cm).plot(cmap='Blues')
    plt.title(f'{backbone_name.upper()} MULTI Confusion Matrix')
    plt.show()

    plt.plot(train_loss_history, label='Train Loss')
    plt.plot(val_loss_history, label='Val Loss')
    plt.title('Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

# === Main Entry ===
if __name__ == "__main__":
    train_loader, val_loader = get_loaders(
        os.path.join(cache_dir, "train"),
        os.path.join(cache_dir, "val")
    )

    model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    for param in list(model.parameters())[:-10]:
        param.requires_grad = False
    backbone = BackboneExtractor(model)
    classifier = MultiClassifier(model.fc.in_features)

    train_model(backbone, classifier, train_loader, val_loader)


