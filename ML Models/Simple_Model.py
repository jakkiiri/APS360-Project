import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import models, transforms
from torchvision.models import MobileNet_V2_Weights, ResNet50_Weights, EfficientNet_B0_Weights
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay, accuracy_score
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import numpy as np
from PIL import Image

# --- Configuration ---
split_dir = r'C:\Users\shore\Desktop\APS360\Datasets\DataSplit2'
cache_root = r'C:\Users\shore\Desktop\APS360\Datasets\Cache'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def cache_dataset(split_dir, cache_dir, transform, label_mapping):
    os.makedirs(cache_dir, exist_ok=True)
    for label_name in os.listdir(split_dir):
        class_path = os.path.join(split_dir, label_name)
        if not os.path.isdir(class_path):
            continue

        target_label = label_mapping[label_name]
        cache_class_dir = os.path.join(cache_dir, label_name)
        os.makedirs(cache_class_dir, exist_ok=True)

        for fname in os.listdir(class_path):
            img_path = os.path.join(class_path, fname)
            try:
                image = Image.open(img_path).convert("RGB")
                tensor = transform(image)
                save_path = os.path.join(cache_class_dir, fname.replace('.jpg', '.pt').replace('.png', '.pt'))
                torch.save({'image': tensor, 'label': target_label}, save_path)
            except Exception as e:
                print(f"❌ Failed to process {img_path}: {e}")

# --- Label Mapping ---
label_mapping = {
    'nevus': 0,
    'melanoma': 1,
    'bcc': 2,
    'keratosis': 3,
    'actinic_keratosis': 4,
    'scc': 5,
    'dermatofibroma': 6,
    'lentigo': 7,
    'vascular_lesion': 8
}

# --- Transformations ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    #transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# --- Cached Dataset ---
class CachedDataset(Dataset):
    def __init__(self, cache_dir):
        from glob import glob
        self.files = glob(os.path.join(cache_dir, '*', '*.pt'))
        self.files.sort()

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = torch.load(self.files[idx], weights_only=True)
        image = data['image']
        label = data['label']
        return image, label

# --- Data Loader with Class Weights ---
def get_loaders(train_dir, val_dir, batch_size=32):
    train_dataset = CachedDataset(train_dir)
    val_dataset = CachedDataset(val_dir)

    train_labels = [torch.load(f, weights_only=True)['label'] for f in train_dataset.files]

    class_counts = np.bincount(train_labels)
    weights = 1.0 / torch.tensor(class_counts, dtype=torch.float)
    global multi_class_weights
    multi_class_weights = weights / weights.sum() * len(class_counts)

    samples_weight = np.array([multi_class_weights[label] for label in train_labels])
    sampler = WeightedRandomSampler(samples_weight, len(samples_weight), replacement=True)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    return train_loader, val_loader

# --- Model Builder ---
class SingleClassifier(nn.Module):
    def __init__(self, backbone):
        super(SingleClassifier, self).__init__()
        self.backbone = backbone

        if isinstance(backbone, models.MobileNetV2):
            feature_dim = backbone.last_channel
        elif isinstance(backbone, models.ResNet):
            feature_dim = backbone.fc.in_features
        elif isinstance(backbone, models.EfficientNet):
            feature_dim = backbone.classifier[1].in_features
        else:
            raise ValueError("Unsupported backbone")

        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 9)
        )

    def forward(self, x):
        if isinstance(self.backbone, models.MobileNetV2):
            features = self.backbone.features(x)
        elif isinstance(self.backbone, models.ResNet):
            features = self.backbone.conv1(x)
            features = self.backbone.bn1(features)
            features = self.backbone.relu(features)
            features = self.backbone.maxpool(features)
            features = self.backbone.layer1(features)
            features = self.backbone.layer2(features)
            features = self.backbone.layer3(features)
            features = self.backbone.layer4(features)
        elif isinstance(self.backbone, models.EfficientNet):
            features = self.backbone.features(x)

        pooled = F.adaptive_avg_pool2d(features, (1, 1))
        pooled = pooled.view(pooled.size(0), -1)

        return self.classifier(pooled)
    
class SkinLesionClassifier(nn.Module):
    def __init__(self, num_classes=9):
        super().__init__()
        self.backbone = models.resnet34(pretrained=True)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.BatchNorm1d(in_features),
            nn.Dropout(0.5),
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)

# ---- Data Preparation ----
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
# --- Training Loop ---
from tqdm import tqdm  # make sure this is imported

def train_model(model, train_loader, val_loader, num_epochs=15, patience=3, backbone_name='model'):
    model = model.to(device)

    for param in model.backbone.parameters():
        param.requires_grad = False

    criterion = nn.CrossEntropyLoss(weight=multi_class_weights.to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=patience, factor=0.5, verbose=True)

    best_val_loss = float('inf')
    patience_counter = 0

    train_losses, val_losses = [], []
    total_start_time = time.time()

    for epoch in range(num_epochs):
        print(f"\n{'=' * 50}")
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print(f"{'=' * 50}")

        epoch_start_time = time.time()
        model.train()
        running_train_loss = 0

        train_bar = tqdm(train_loader, desc=f"[Epoch {epoch+1}] Training", leave=False)
        for inputs, targets in train_bar:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            logits = model(inputs)
            loss = criterion(logits, targets)

            loss.backward()
            optimizer.step()

            running_train_loss += loss.item()
            train_bar.set_postfix(loss=loss.item())

        avg_train_loss = running_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        model.eval()
        running_val_loss = 0
        val_labels, val_preds = [], []

        val_bar = tqdm(val_loader, desc=f"[Epoch {epoch+1}] Validation", leave=False)
        with torch.no_grad():
            for inputs, targets in val_bar:
                inputs, targets = inputs.to(device), targets.to(device)

                logits = model(inputs)
                loss = criterion(logits, targets)

                running_val_loss += loss.item()

                preds = torch.argmax(logits, dim=1).cpu().numpy()
                val_labels.extend(targets.cpu().numpy())
                val_preds.extend(preds)

                val_bar.set_postfix(loss=loss.item())

        avg_val_loss = running_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        scheduler.step(avg_val_loss)

        precision = precision_score(val_labels, val_preds, average='macro', zero_division=0)
        recall = recall_score(val_labels, val_preds, average='macro', zero_division=0)
        f1 = f1_score(val_labels, val_preds, average='macro', zero_division=0)
        acc = accuracy_score(val_labels, val_preds)

        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        elapsed_total_time = epoch_end_time - total_start_time
        estimated_total_time = (elapsed_total_time / (epoch + 1)) * num_epochs
        estimated_remaining_time = estimated_total_time - elapsed_total_time

        print(f"\nEpoch {epoch + 1} Summary")
        print(f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        print(f"Multi-Class -> Acc: {acc:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}")
        print(f"🕒 Epoch duration: {epoch_duration:.2f} seconds")
        print(f"⏳ Estimated remaining time: {estimated_remaining_time / 60:.2f} minutes")
        print(f"{'=' * 50}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), f'{backbone_name}_best_simple_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("⏹️ Early stopping triggered.")
                break

        torch.cuda.empty_cache()

    # Final confusion matrix
    ConfusionMatrixDisplay(confusion_matrix(val_labels, val_preds)).plot()
    plt.title(f'{backbone_name} Final Confusion Matrix')
    plt.show()

    # Loss curves
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'{backbone_name} Training Curve')
    plt.legend()
    plt.show()


# --- Run Training ---
if __name__ == "__main__":
    train_cache = os.path.join(cache_root, 'train')
    val_cache = os.path.join(cache_root, 'val')
    train_split = os.path.join(split_dir, 'train')
    val_split = os.path.join(split_dir, 'val')

    # Check if cache is empty and populate it if needed
    def is_cache_empty(path):
        return not os.path.exists(path) or all(len(files) == 0 for _, _, files in os.walk(path))

    if is_cache_empty(train_cache):
        print("⚙️ Caching training dataset...")
        cache_dataset(train_split, train_cache, transform, label_mapping)

    if is_cache_empty(val_cache):
        print("⚙️ Caching validation dataset...")
        cache_dataset(val_split, val_cache, transform, label_mapping)

    # Load from cache
    train_loader, val_loader = get_loaders(train_cache, val_cache)


    for backbone_name in ['efficientnet', 'mobilenet', 'resnet']:
        print(f"\n🚀 Starting training for {backbone_name} backbone...")

        if backbone_name == 'mobilenet':
            backbone = models.mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
        elif backbone_name == 'resnet':
            backbone = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        elif backbone_name == 'efficientnet':
            backbone = models.efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)

        model = SingleClassifier(backbone)

        for param in model.backbone.parameters():
            param.requires_grad = False

        train_model(model, train_loader, val_loader, num_epochs=15, patience=3, backbone_name=backbone_name)
