# Full APS360 Skin Disease Detection Script: Binary + Multi-Class + Combined Evaluation with Progress Bar (Non-Line-Updating)

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import models, transforms
from torchvision.models import MobileNet_V2_Weights, ResNet50_Weights, EfficientNet_B0_Weights
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from tqdm import tqdm
import time

# === Standard & ML Imports ===
import cv2
from glob import glob


# === Dataset Paths ===
split_dir = r'C:\Users\shore\Desktop\APS360\Datasets\DataSplit'
cache_root = r'C:\Users\shore\Desktop\APS360\Datasets\Cache'

# === Label Mapping ===
label_mapping = {
    'nevus': 0, 'melanoma': 1, 'bcc': 2, 'seborrheic_keratosis': 3,
    'actinic_keratosis': 4, 'scc': 5, 'dermatofibroma': 6,
    'lentigo': 7, 'vascular_lesion': 8
}

# === Hair Removal Function ===
def apply_hair_removal(image_bgr, threshold=10, kernel_size=17):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    _, mask = cv2.threshold(blackhat, threshold, 255, cv2.THRESH_BINARY)
    inpainted = cv2.inpaint(image_bgr, mask, 1, cv2.INPAINT_TELEA)
    return inpainted

# === PIL ↔ OpenCV Conversion ===
def pil_to_cv2(pil_img):
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

def cv2_to_pil(cv_img):
    return Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))

# === Image Transformations ===
transform = transforms.Compose([
    transforms.Resize((320, 320)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# === Cache Builder ===
def build_cache(split_dir, cache_root, transform):
    os.makedirs(cache_root, exist_ok=True)

    for phase in ['train', 'val', 'test']:
        split_path = os.path.join(split_dir, phase)
        cache_path = os.path.join(cache_root, phase)
        os.makedirs(cache_path, exist_ok=True)

        print(f"\n📦 Caching {phase} data...")

        for class_name in os.listdir(split_path):
            class_dir = os.path.join(split_path, class_name)
            if not os.path.isdir(class_dir):
                continue

            class_cache_dir = os.path.join(cache_path, class_name)
            os.makedirs(class_cache_dir, exist_ok=True)

            label = label_mapping[class_name]

            for fname in tqdm(os.listdir(class_dir), desc=f"{phase}/{class_name}"):
                if not fname.lower().endswith(('.jpg', '.png', '.jpeg')):
                    continue
                img_path = os.path.join(class_dir, fname)
                try:
                    img = Image.open(img_path).convert("RGB")
                    img_cv = pil_to_cv2(img)
                    img_no_hair = apply_hair_removal(img_cv)
                    img_pil = cv2_to_pil(img_no_hair)
                    img_tensor = transform(img_pil)
                    torch.save({'image': img_tensor, 'label': label},
                               os.path.join(class_cache_dir, fname + '.pt'))
                except Exception as e:
                    print(f"❌ Failed on {img_path}: {e}")

# === Cached Dataset ===
class CachedDataset(Dataset):
    def __init__(self, cache_dir, mode='all'):
        label_cache_path = os.path.join(cache_dir, f'labels_{mode}.npy')
        print(f"\n📂 Loading dataset from: {cache_dir} (mode: {mode})")

        self.files = sorted(glob(os.path.join(cache_dir, '*', '*.pt')))
        self.labels = []

        if os.path.exists(label_cache_path):
            print(f"✅ Using cached labels from {label_cache_path}")
            self.labels = np.load(label_cache_path).tolist()
        else:
            print("🔄 Caching labels (first time)...")
            for f in tqdm(self.files, desc="Loading Labels"):
                label = torch.load(f, map_location='cpu')['label']
                self.labels.append(label)
            np.save(label_cache_path, np.array(self.labels))
            print(f"✅ Labels cached at {label_cache_path}")

        if mode == 'non_nevus':
            self.files = [f for f, label in zip(self.files, self.labels) if label != 0]
            self.labels = [label - 1 for label in self.labels if label != 0]

        self.mode = mode
        print(f"✅ Dataset loaded: {len(self.files)} samples")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = torch.load(self.files[idx], weights_only=True)
        image = data['image']
        label = self.labels[idx]
        binary_label = 0 if label == 0 else 1
        return image, binary_label, label

# === DataLoader Builder ===
def get_loaders(train_dir, val_dir, batch_size=32, mode='all'):
    print(f"\n📂 Preparing {mode} DataLoaders...")
    train_dataset = CachedDataset(train_dir, mode=mode)
    val_dataset = CachedDataset(val_dir, mode=mode)

    train_labels = train_dataset.labels
    binary_labels = [0 if label == 0 else 1 for label in train_labels]

    # --- Multiclass Weights ---
    class_counts = np.bincount(train_labels)
    weights = 1.0 / torch.tensor(class_counts, dtype=torch.float)
    global multi_class_weights
    multi_class_weights = weights / weights.sum() * len(class_counts)

    # --- Binary Sampling ---
    class_sample_count = np.array([binary_labels.count(0), binary_labels.count(1)])
    class_weights = np.array([1. / c if c > 0 else 0.0 for c in class_sample_count])
    samples_weight = np.array([class_weights[t] for t in binary_labels])
    sampler = WeightedRandomSampler(samples_weight, len(samples_weight), replacement=True)

    num_nevus = binary_labels.count(0)
    num_non_nevus = len(binary_labels) - num_nevus
    binary_class_weight_value = num_non_nevus / num_nevus if num_nevus > 0 else 1.0
    global binary_class_weight
    binary_class_weight = torch.tensor([binary_class_weight_value])

    print(f"✅ DataLoaders ready: {len(train_dataset)} train, {len(val_dataset)} val")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler,
                              num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=4, pin_memory=True)

    return train_loader, val_loader

# === Optional: Run Everything ===
if __name__ == "__main__":
    build_cache(split_dir, cache_root, transform)

    train_loader, val_loader = get_loaders(
        os.path.join(cache_root, 'train'),
        os.path.join(cache_root, 'val'),
        batch_size=32,
        mode='all'
    )
# === Device Configuration ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --- Models ---
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

class BinaryClassifier(nn.Module):
    def __init__(self, feature_dim):
        super(BinaryClassifier, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.classifier(x)

class MultiClassifier(nn.Module):
    def __init__(self, feature_dim):
        super(MultiClassifier, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 8)
        )

    def forward(self, x):
        return self.classifier(x)

# --- Training Loop with Progress Bar ---
def train_model(backbone, classifier, train_loader, val_loader, mode='binary', num_epochs=50, patience=7, backbone_name='model'):
    from tqdm import tqdm

    backbone.eval()
    classifier = classifier.to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=binary_class_weight.to(device)) if mode == 'binary' else nn.CrossEntropyLoss(weight=multi_class_weights.to(device))
    optimizer = torch.optim.Adam(classifier.parameters(), lr=0.0002)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=patience, factor=0.5, verbose=True)

    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(num_epochs):
        print(f"\n{'=' * 50}\nEpoch {epoch + 1}/{num_epochs} - {mode.capitalize()} Classifier\n{'=' * 50}")

        classifier.train()
        running_train_loss = 0
        train_labels, train_preds = [], []

        for inputs, binary_targets, multi_targets in tqdm(train_loader, leave=True, desc="Training Batches"):
            inputs = inputs.to(device)
            targets = binary_targets.to(device).float() if mode == 'binary' else multi_targets.to(device)

            optimizer.zero_grad()
            features = backbone(inputs)
            outputs = classifier(features)

            if mode == 'binary':
                loss = criterion(outputs.squeeze(), targets)
                preds = (torch.sigmoid(outputs) > 0.5).int().cpu().numpy()
            else:
                loss = criterion(outputs, targets)
                preds = torch.argmax(outputs, dim=1).cpu().numpy()

            if torch.isnan(loss):
                print("NaN loss detected! Stopping training.")
                return

            loss.backward()
            optimizer.step()

            running_train_loss += loss.item()
            train_labels.extend(targets.cpu().numpy())
            train_preds.extend(preds)

        avg_train_loss = running_train_loss / len(train_loader)
        train_acc = accuracy_score(train_labels, train_preds)
        train_prec = precision_score(train_labels, train_preds, average='binary' if mode == 'binary' else 'macro', zero_division=0)
        train_rec = recall_score(train_labels, train_preds, average='binary' if mode == 'binary' else 'macro', zero_division=0)

        classifier.eval()
        running_val_loss = 0
        val_labels, val_preds = [], []

        for inputs, binary_targets, multi_targets in tqdm(val_loader, leave=True, desc="Validation Batches"):
            inputs = inputs.to(device)
            targets = binary_targets.to(device).float() if mode == 'binary' else multi_targets.to(device)

            features = backbone(inputs)
            outputs = classifier(features)

            if mode == 'binary':
                loss = criterion(outputs.squeeze(), targets)
                preds = (torch.sigmoid(outputs) > 0.5).int().cpu().numpy()
            else:
                loss = criterion(outputs, targets)
                preds = torch.argmax(outputs, dim=1).cpu().numpy()

            running_val_loss += loss.item()
            val_labels.extend(targets.cpu().numpy())
            val_preds.extend(preds)

        avg_val_loss = running_val_loss / len(val_loader)
        scheduler.step(avg_val_loss)

        val_acc = accuracy_score(val_labels, val_preds)
        val_prec = precision_score(val_labels, val_preds, average='binary' if mode == 'binary' else 'macro', zero_division=0)
        val_rec = recall_score(val_labels, val_preds, average='binary' if mode == 'binary' else 'macro', zero_division=0)

        print(f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        print(f"Train -> Acc: {train_acc:.4f} | Precision: {train_prec:.4f} | Recall: {train_rec:.4f}")
        print(f"Val   -> Acc: {val_acc:.4f} | Precision: {val_prec:.4f} | Recall: {val_rec:.4f}")

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

    return classifier

# --- Combined Model Evaluation ---
def evaluate_combined_model(backbone, binary_classifier, multi_classifier, val_loader, backbone_name='model'):
    print(f"\n{'=' * 50}\nEvaluating Combined Model on Validation Set\n{'=' * 50}")
    binary_classifier.eval()
    multi_classifier.eval()

    all_labels = []
    combined_preds = []

    with torch.no_grad():
        for inputs, _, multi_targets in tqdm(val_loader, leave=True, desc="Evaluating Combined Model"):
            inputs = inputs.to(device)
            multi_targets = multi_targets.cpu().numpy()

            features = backbone(inputs)

            binary_outputs = binary_classifier(features)
            binary_probs = torch.sigmoid(binary_outputs).cpu().numpy()

            multi_outputs = multi_classifier(features)
            multi_preds = torch.argmax(multi_outputs, dim=1).cpu().numpy()

            for bin_prob, multi_pred in zip(binary_probs, multi_preds):
                if bin_prob > 0.5:
                    combined_preds.append(0)
                else:
                    combined_preds.append(multi_pred + 1)

            all_labels.extend(multi_targets)

    acc = accuracy_score(all_labels, combined_preds)
    prec = precision_score(all_labels, combined_preds, average='macro', zero_division=0)
    rec = recall_score(all_labels, combined_preds, average='macro', zero_division=0)
    f1 = f1_score(all_labels, combined_preds, average='macro', zero_division=0)

    print(f"\nCombined Model Performance:")
    print(f"Accuracy: {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | F1 Score: {f1:.4f}")

    ConfusionMatrixDisplay(confusion_matrix(all_labels, combined_preds)).plot()
    plt.title(f'{backbone_name} Combined Model Confusion Matrix')
    plt.show()

# --- Run Pipeline ---
if __name__ == "__main__":
    build_cache(split_dir, cache_root, transform)
    train_loader, val_loader = get_loaders(os.path.join(cache_root, 'train'), os.path.join(cache_root, 'val'))

    for backbone_name in ['mobilenet', 'resnet', 'efficientnet']:
        print(f"\n🚀 Starting training for {backbone_name} backbone...")

        if backbone_name == 'mobilenet':
            backbone_model = models.mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
            feature_dim = backbone_model.last_channel
        elif backbone_name == 'resnet':
            backbone_model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
            feature_dim = backbone_model.fc.in_features
        elif backbone_name == 'efficientnet':
            backbone_model = models.efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
            feature_dim = backbone_model.classifier[1].in_features

        backbone = BackboneExtractor(backbone_model).to(device)
        for param in backbone.parameters():
            param.requires_grad = False

        binary_classifier = BinaryClassifier(feature_dim)
        trained_binary = train_model(backbone, binary_classifier, train_loader, val_loader,
                                     mode='binary', num_epochs=50, patience=7, backbone_name=backbone_name)

        multi_train_loader, multi_val_loader = get_loaders(os.path.join(cache_root, 'train'),
                                                           os.path.join(cache_root, 'val'), mode='non_nevus')

        multi_classifier = MultiClassifier(feature_dim)
        trained_multi = train_model(backbone, multi_classifier, multi_train_loader, multi_val_loader,
                                    mode='multi', num_epochs=30, patience=7, backbone_name=backbone_name)

        trained_binary.load_state_dict(torch.load(f'{backbone_name}_binary_best_model.pth'))
        trained_multi.load_state_dict(torch.load(f'{backbone_name}_multi_best_model.pth'))

        evaluate_combined_model(backbone, trained_binary, trained_multi, val_loader, backbone_name=backbone_name)

