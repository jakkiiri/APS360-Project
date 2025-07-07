import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import models, transforms
from torchvision.models import MobileNet_V2_Weights, ResNet50_Weights, EfficientNet_B0_Weights
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay, accuracy_score
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

torch.backends.cudnn.benchmark = True

# --- Configuration ---
split_dir = r'C:\Users\shore\Desktop\APS360\Datasets\DataSplit'
cache_root = r'C:\Users\shore\Desktop\APS360\Datasets\Cache'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- Label Mapping ---
label_mapping = {
    'nevus': 0,
    'melanoma': 1,
    'bcc': 2,
    'seborrheic_keratosis': 3,
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
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
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
        binary_label = 0 if label == 0 else 1
        return image, binary_label, label

# --- Filtered Dataset for Multi-Class (non-nevus only) ---
class FilteredDataset(Dataset):
    def __init__(self, files):
        self.files = files

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = torch.load(self.files[idx], weights_only=True)
        image = data['image']
        label = data['label']
        return image, label

# --- Data Loaders ---
def get_loaders(train_dir, val_dir, batch_size=32):
    train_dataset = CachedDataset(train_dir)
    val_dataset = CachedDataset(val_dir)

    train_labels = []
    binary_labels = []

    for file in train_dataset.files:
        label = torch.load(file, weights_only=True)['label']
        train_labels.append(label)
        binary_labels.append(0 if label == 0 else 1)

    class_sample_count = np.array([binary_labels.count(0), binary_labels.count(1)])
    class_weights = np.array([1. / c if c > 0 else 0.0 for c in class_sample_count])
    samples_weight = np.array([class_weights[t] for t in binary_labels])
    sampler = WeightedRandomSampler(samples_weight, len(samples_weight), replacement=True)

    class_counts = np.bincount(train_labels)
    weights = 1.0 / torch.tensor(class_counts, dtype=torch.float)
    global multi_class_weights
    multi_class_weights = weights / weights.sum() * len(class_counts)

    num_nevus = train_labels.count(0)
    num_non_nevus = len(train_labels) - num_nevus
    binary_class_weight_value = num_non_nevus / num_nevus if num_nevus > 0 else 1.0
    global binary_class_weight
    binary_class_weight = torch.tensor([binary_class_weight_value])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # ➜ Create multi-class loaders (non-nevus only)
    multi_train_files = [file for file in train_dataset.files if torch.load(file, weights_only=True)['label'] != 0]
    multi_val_files = [file for file in val_dataset.files if torch.load(file, weights_only=True)['label'] != 0]

    multi_train_loader = DataLoader(FilteredDataset(multi_train_files), batch_size=batch_size, shuffle=True, num_workers=0)
    multi_val_loader = DataLoader(FilteredDataset(multi_val_files), batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, val_loader, multi_train_loader, multi_val_loader

# --- Backbone Feature Extractor ---
class BackboneExtractor(nn.Module):
    def __init__(self, backbone):
        super(BackboneExtractor, self).__init__()
        self.backbone = backbone

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
        else:
            raise ValueError("Unsupported backbone")

        pooled = F.adaptive_avg_pool2d(features, (1, 1))
        return pooled.view(pooled.size(0), -1)

# --- Training Function ---
def train_classifier(backbone, train_loader, val_loader, num_epochs=15, patience=3, mode='binary', backbone_name='model'):
    feature_dim = None
    if isinstance(backbone, models.MobileNetV2):
        feature_dim = backbone.last_channel
    elif isinstance(backbone, models.ResNet):
        feature_dim = backbone.fc.in_features
    elif isinstance(backbone, models.EfficientNet):
        feature_dim = backbone.classifier[1].in_features

    backbone_extractor = BackboneExtractor(backbone).to(device)
    for param in backbone_extractor.parameters():
        param.requires_grad = False

    if mode == 'binary':
        classifier = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        ).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=binary_class_weight.to(device))
    else:
        classifier = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 9)
        ).to(device)
        criterion = nn.CrossEntropyLoss(weight=multi_class_weights.to(device))

    optimizer = torch.optim.Adam(classifier.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=patience, factor=0.5, verbose=True)

    best_val_loss = float('inf')
    patience_counter = 0
    train_losses, val_losses = [], []

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs} - {mode.capitalize()} Classifier")
        classifier.train()
        running_train_loss = 0

        train_labels, train_preds = [], []
        train_pbar = tqdm(train_loader, desc=f'Training ({mode})', ncols=150)

        for batch in train_pbar:
            if mode == 'binary':
                inputs, binary_targets, _ = batch
                targets = binary_targets.to(device).float()
            else:
                inputs, targets = batch

            inputs = inputs.to(device)

            optimizer.zero_grad()
            features = backbone_extractor(inputs)
            outputs = classifier(features)

            if mode == 'binary':
                loss = criterion(outputs.squeeze(), targets)
                preds = (torch.sigmoid(outputs) > 0.3).int().cpu().numpy()
            else:
                loss = criterion(outputs, targets)
                preds = torch.argmax(outputs, dim=1).cpu().numpy()

            loss.backward()
            optimizer.step()

            running_train_loss += loss.item()

            train_labels.extend(targets.cpu().numpy())
            train_preds.extend(preds)

            acc = accuracy_score(train_labels, train_preds)
            train_pbar.set_postfix({'Loss': f'{loss.item():.4f}', 'Acc': f'{acc:.4f}'})

        avg_train_loss = running_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # --- Validation ---
        classifier.eval()
        running_val_loss = 0
        val_labels, val_preds, val_logits = [], [], []

        val_pbar = tqdm(val_loader, desc=f'Validation ({mode})', ncols=150)
        with torch.no_grad():
            for batch in val_pbar:
                if mode == 'binary':
                    inputs, binary_targets, _ = batch
                    targets = binary_targets.to(device).float()
                else:
                    inputs, targets = batch

                inputs = inputs.to(device)

                features = backbone_extractor(inputs)
                outputs = classifier(features)

                if mode == 'binary':
                    loss = criterion(outputs.squeeze(), targets)
                    probs = torch.sigmoid(outputs).cpu().numpy()
                    preds = (probs > 0.3).astype(int)
                    val_logits.extend(probs)
                else:
                    loss = criterion(outputs, targets)
                    preds = torch.argmax(outputs, dim=1).cpu().numpy()

                running_val_loss += loss.item()
                val_labels.extend(targets.cpu().numpy())
                val_preds.extend(preds)

        avg_val_loss = running_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        scheduler.step(avg_val_loss)

        acc = accuracy_score(val_labels, val_preds)
        precision = precision_score(val_labels, val_preds, average='binary' if mode == 'binary' else 'macro', zero_division=0)
        recall = recall_score(val_labels, val_preds, average='binary' if mode == 'binary' else 'macro', zero_division=0)
        f1 = f1_score(val_labels, val_preds, average='binary' if mode == 'binary' else 'macro', zero_division=0)

        if mode == 'binary':
            auc = roc_auc_score(val_labels, val_logits)
            print(f"Epoch {epoch + 1} -> Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Acc: {acc:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f} | AUC: {auc:.4f}")
        else:
            print(f"Epoch {epoch + 1} -> Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Acc: {acc:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(classifier.state_dict(), f'{backbone_name}_{mode}_best_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

        torch.cuda.empty_cache()

    ConfusionMatrixDisplay(confusion_matrix(val_labels, val_preds)).plot()
    plt.title(f'{backbone_name} {mode.capitalize()} Classifier Confusion Matrix')
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'{backbone_name} {mode.capitalize()} Classifier Training Curve')
    plt.legend()
    plt.show()

# --- Run Pipeline ---
if __name__ == "__main__":
    train_loader, val_loader, multi_train_loader, multi_val_loader = get_loaders(os.path.join(cache_root, 'train'), os.path.join(cache_root, 'val'))

    for backbone_name in ['mobilenet', 'resnet', 'efficientnet']:
        print(f"\n🚀 Starting training for {backbone_name} backbone...")

        if backbone_name == 'mobilenet':
            backbone = models.mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
        elif backbone_name == 'resnet':
            backbone = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        elif backbone_name == 'efficientnet':
            backbone = models.efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)

        for param in backbone.parameters():
            param.requires_grad = False

        train_classifier(backbone, train_loader, val_loader, num_epochs=15, patience=3, mode='binary', backbone_name=backbone_name)

        train_classifier(backbone, multi_train_loader, multi_val_loader, num_epochs=15, patience=3, mode='multi', backbone_name=backbone_name)
