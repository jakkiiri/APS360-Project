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

# --- Transformations to 224x224 ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# --- Caching Function ---
def create_cache(split_name):
    split_path = os.path.join(split_dir, split_name)
    cache_dir = os.path.join(cache_root, split_name)
    os.makedirs(cache_dir, exist_ok=True)

    print(f"\n📂 Caching {split_name} dataset...")
    for label in os.listdir(split_path):
        label_dir = os.path.join(split_path, label)
        cache_label_dir = os.path.join(cache_dir, str(label_mapping[label]))
        os.makedirs(cache_label_dir, exist_ok=True)

        for image_file in tqdm(os.listdir(label_dir), desc=f'Caching {split_name} - {label}'):
            image_path = os.path.join(label_dir, image_file)
            try:
                image = Image.open(image_path).convert('RGB')
                image_tensor = transform(image)
            except Exception as e:
                print(f"Error loading image: {image_path} | {e}")
                continue

            save_path = os.path.join(cache_label_dir, image_file.replace('.jpg', '.pt').replace('.png', '.pt'))
            torch.save({'image': image_tensor, 'label': label_mapping[label]}, save_path)

# --- Check Cache ---
for split in ['train', 'val']:
    if not os.path.exists(os.path.join(cache_root, split)):
        create_cache(split)
    else:
        print(f"✅ Cache for {split} already exists. Skipping...")

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

# --- Data Loader with Class Weights ---
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

    return train_loader, val_loader

# --- Model Builder ---
class DualClassifier(nn.Module):
    def __init__(self, backbone):
        super(DualClassifier, self).__init__()
        self.backbone = backbone

        if isinstance(backbone, models.MobileNetV2):
            feature_dim = backbone.last_channel
        elif isinstance(backbone, models.ResNet):
            feature_dim = backbone.fc.in_features
        elif isinstance(backbone, models.EfficientNet):
            feature_dim = backbone.classifier[1].in_features
        else:
            raise ValueError("Unsupported backbone")

        self.binary_classifier = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

        self.multi_classifier = nn.Sequential(
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

        binary_out = self.binary_classifier(pooled)
        multi_out = self.multi_classifier(pooled)

        return binary_out, multi_out

from sklearn.metrics import accuracy_score

def train_model(model, train_loader, val_loader, num_epochs=15, patience=3, backbone_name='model', alpha=1.0, beta=2.0, binary_threshold=0.3):
    model = model.to(device)

    # Freeze all backbone layers initially
    for param in model.backbone.parameters():
        param.requires_grad = False

    criterion_binary = nn.BCEWithLogitsLoss(pos_weight=binary_class_weight.to(device))
    criterion_multi = nn.CrossEntropyLoss(weight=multi_class_weights.to(device))

    optimizer = torch.optim.Adam([
        {'params': model.backbone.parameters(), 'lr': 1e-5},
        {'params': model.binary_classifier.parameters(), 'lr': 1e-4},
        {'params': model.multi_classifier.parameters(), 'lr': 1e-4}
    ])

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=patience, factor=0.5, verbose=True)

    best_val_loss = float('inf')
    patience_counter = 0

    train_losses, val_losses = [], []
    fine_tuning = False

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        model.train()
        running_train_loss = 0

        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch + 1} Training', ncols=150, mininterval=0.5, leave=False)

        for inputs, binary_targets, multi_targets in train_pbar:
            inputs, binary_targets, multi_targets = inputs.to(device), binary_targets.to(device).float(), multi_targets.to(device)

            optimizer.zero_grad()
            binary_logits, multi_logits = model(inputs)

            loss_binary = criterion_binary(binary_logits.squeeze(), binary_targets)
            loss_multi = criterion_multi(multi_logits, multi_targets)

            loss = alpha * loss_binary + beta * loss_multi
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item()

            binary_probs = torch.sigmoid(binary_logits).detach().cpu().numpy()
            binary_preds_batch = (binary_probs > binary_threshold).astype(int)
            multi_preds_batch = torch.argmax(multi_logits, dim=1).detach().cpu().numpy()

            binary_acc = accuracy_score(binary_targets.cpu().numpy(), binary_preds_batch)
            multi_acc = accuracy_score(multi_targets.cpu().numpy(), multi_preds_batch)

            train_pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Binary Acc': f'{binary_acc:.4f}',
                'Multi Acc': f'{multi_acc:.4f}'
            })

        avg_train_loss = running_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # --- Validation Loop ---
        model.eval()
        running_val_loss = 0

        val_binary_labels, val_binary_preds = [], []
        val_multi_labels, val_multi_preds = [], []
        combined_preds = []

        val_pbar = tqdm(val_loader, desc=f'Epoch {epoch + 1} Validation', ncols=150, mininterval=0.5, leave=False)
        with torch.no_grad():
            for inputs, binary_targets, multi_targets in val_pbar:
                inputs, binary_targets, multi_targets = inputs.to(device), binary_targets.to(device).float(), multi_targets.to(device)

                binary_logits, multi_logits = model(inputs)

                loss_binary = criterion_binary(binary_logits.squeeze(), binary_targets)
                loss_multi = criterion_multi(multi_logits, multi_targets)
                loss = alpha * loss_binary + beta * loss_multi

                running_val_loss += loss.item()

                binary_probs = torch.sigmoid(binary_logits).cpu().numpy()
                binary_preds_batch = (binary_probs > binary_threshold).astype(int)
                multi_preds_batch = torch.argmax(multi_logits, dim=1).cpu().numpy()

                val_binary_labels.extend(binary_targets.cpu().numpy())
                val_binary_preds.extend(binary_probs)
                val_multi_labels.extend(multi_targets.cpu().numpy())
                val_multi_preds.extend(multi_preds_batch)

                # Combined Decision
                for bin_prob, multi_pred in zip(binary_probs, multi_preds_batch):
                    if bin_prob > binary_threshold:
                        combined_preds.append(0)  # Predict nevus
                    else:
                        combined_preds.append(multi_pred)

        avg_val_loss = running_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        scheduler.step(avg_val_loss)

        val_binary_preds_bin = (np.array(val_binary_preds) > binary_threshold).astype(int)

        # Binary Metrics
        binary_precision = precision_score(val_binary_labels, val_binary_preds_bin, zero_division=0)
        binary_recall = recall_score(val_binary_labels, val_binary_preds_bin, zero_division=0)
        binary_f1 = f1_score(val_binary_labels, val_binary_preds_bin, zero_division=0)
        binary_auc = roc_auc_score(val_binary_labels, val_binary_preds)

        # Multi-Class Metrics
        multi_acc = accuracy_score(val_multi_labels, val_multi_preds)
        multi_precision = precision_score(val_multi_labels, val_multi_preds, average='macro', zero_division=0)
        multi_recall = recall_score(val_multi_labels, val_multi_preds, average='macro', zero_division=0)
        multi_f1 = f1_score(val_multi_labels, val_multi_preds, average='macro', zero_division=0)

        # Combined Metrics
        combined_acc = accuracy_score(val_multi_labels, combined_preds)
        combined_precision = precision_score(val_multi_labels, combined_preds, average='macro', zero_division=0)
        combined_recall = recall_score(val_multi_labels, combined_preds, average='macro', zero_division=0)
        combined_f1 = f1_score(val_multi_labels, combined_preds, average='macro', zero_division=0)

        print(f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        print(f"Binary -> Precision: {binary_precision:.4f} | Recall: {binary_recall:.4f} | F1: {binary_f1:.4f} | AUC: {binary_auc:.4f}")
        print(f"Multi   -> Acc: {multi_acc:.4f} | Precision: {multi_precision:.4f} | Recall: {multi_recall:.4f} | F1: {multi_f1:.4f}")
        print(f"Combined -> Acc: {combined_acc:.4f} | Precision: {combined_precision:.4f} | Recall: {combined_recall:.4f} | F1: {combined_f1:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), f'{backbone_name}_best_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience and not fine_tuning:
                print("Early stopping triggered. Starting fine-tuning...")

                for name, param in model.backbone.named_parameters():
                    if isinstance(model.backbone, models.ResNet) and ('layer4' in name or 'fc' in name):
                        param.requires_grad = True
                    elif isinstance(model.backbone, models.MobileNetV2) and ('18' in name):
                        param.requires_grad = True
                    elif isinstance(model.backbone, models.EfficientNet) and ('blocks.6' in name or 'classifier' in name):
                        param.requires_grad = True

                optimizer = torch.optim.Adam([
                    {'params': model.backbone.parameters(), 'lr': 1e-6},
                    {'params': model.binary_classifier.parameters(), 'lr': 1e-4},
                    {'params': model.multi_classifier.parameters(), 'lr': 1e-4}
                ])

                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=patience, factor=0.5, verbose=True)

                patience_counter = 0
                fine_tuning = True
                print("Fine-tuning optimizer configured. Resuming training.")
                continue

            elif patience_counter >= patience and fine_tuning:
                print("Early stopping triggered after fine-tuning.")
                break

        torch.cuda.empty_cache()

    # --- Final Confusion Matrices ---
    ConfusionMatrixDisplay(confusion_matrix(val_binary_labels, val_binary_preds_bin)).plot()
    plt.title(f'{backbone_name} Binary Classifier Confusion Matrix')
    plt.show()

    ConfusionMatrixDisplay(confusion_matrix(val_multi_labels, val_multi_preds)).plot()
    plt.title(f'{backbone_name} Multi-Class Classifier Confusion Matrix')
    plt.show()

    ConfusionMatrixDisplay(confusion_matrix(val_multi_labels, combined_preds)).plot()
    plt.title(f'{backbone_name} Combined Model Confusion Matrix')
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'{backbone_name} Training Curve')
    plt.legend()
    plt.show()


# --- Run Pipeline ---
train_loader, val_loader = get_loaders(os.path.join(cache_root, 'train'), os.path.join(cache_root, 'val'))

for backbone_name in ['mobilenet', 'resnet', 'efficientnet']:
    print(f"\n🚀 Starting training for {backbone_name} backbone...")

    if backbone_name == 'mobilenet':
        print("📥 Downloading MobileNetV2...")
        backbone = models.mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
    elif backbone_name == 'resnet':
        print("📥 Downloading ResNet50...")
        backbone = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    elif backbone_name == 'efficientnet':
        print("📥 Downloading EfficientNet B0...")
        backbone = models.efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)

    model = DualClassifier(backbone)
    # Freeze all backbone layers initially
    for param in model.backbone.parameters():
        param.requires_grad = False

    train_model(model, train_loader, val_loader, num_epochs=15, patience=3, backbone_name=backbone_name)

