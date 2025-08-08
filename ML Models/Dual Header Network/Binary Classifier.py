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
cache_transform = transforms.Compose([
    transforms.Resize((350, 350)),
    transforms.ToTensor()
])

augmentation_transform = transforms.Compose([
    transforms.Resize((224, 224)),
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


for split in ['train', 'val']:
    cache_split_if_missing(split)

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
def get_binary_loaders(train_dir, val_dir, batch_size=32):
    train_dataset = CachedBinaryDataset(train_dir, augment=True)
    val_dataset = CachedBinaryDataset(val_dir, augment=False)

    train_labels = [label for _, label in train_dataset]
    class_counts = np.bincount(train_labels)
    weights = 1. / torch.tensor(class_counts, dtype=torch.float)
    samples_weight = np.array([weights[label] for label in train_labels])
    sampler = WeightedRandomSampler(samples_weight, len(samples_weight), replacement=True)

    return (
        DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, num_workers=1),
        DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=1)
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

def train_binary_model(backbone, classifier, train_loader, val_loader, epochs=50, patience=10,
                       csv_path='binary_metrics.csv', model_path='binary_best_classifier.pt'):

    backbone = backbone.to(device)
    classifier = classifier.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(list(backbone.parameters()) + list(classifier.parameters()), lr=1e-4)

    # === Resume Setup ===
    start_epoch = 0
    best_val_loss = float('inf')
    if os.path.exists(model_path):
        print(f"🔁 Resuming from saved model: {model_path}")
        classifier.load_state_dict(torch.load(model_path))
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            if not df.empty:
                best_val_loss = df['val_loss'].min()
                start_epoch = int(df['epoch'].max()) + 1
                print(f"📈 Resuming from epoch {start_epoch} with best val loss {best_val_loss:.4f}")
    else:
        df = pd.DataFrame(columns=[
            'epoch', 'train_loss', 'val_loss', 'val_acc',
            'val_precision', 'val_recall', 'val_f1', 'val_auc'
        ])

    patience_counter = 0

    for epoch in range(start_epoch, epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        backbone.train()
        classifier.train()
        total_loss = 0

        train_loop = tqdm(train_loader, desc=f"Training [{epoch+1}]")
        for inputs, labels in train_loop:
            inputs, labels = inputs.to(device), labels.float().to(device)
            optimizer.zero_grad()
            feats = backbone(inputs)
            outputs = classifier(feats).squeeze(1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            train_loop.set_postfix(loss=loss.item())

        avg_train_loss = total_loss / len(train_loader)

        # === Validation ===
        backbone.eval()
        classifier.eval()
        val_loss, val_preds, val_labels = 0, [], []

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

        avg_val_loss = val_loss / len(val_loader)

        # === Metrics ===
        acc = accuracy_score(val_labels, val_preds)
        prec = precision_score(val_labels, val_preds)
        rec = recall_score(val_labels, val_preds)
        f1 = f1_score(val_labels, val_preds)
        auc = roc_auc_score(val_labels, val_preds)

        print(f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        print(f"Val Acc: {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f} | AUC: {auc:.4f}")

        # === Save to CSV ===
        new_row = {
            'epoch': epoch,
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
            'val_acc': acc,
            'val_precision': prec,
            'val_recall': rec,
            'val_f1': f1,
            'val_auc': auc
        }
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        df.to_csv(csv_path, index=False)

        # === Save Best Model ===
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(classifier.state_dict(), model_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("⏹️ Early stopping triggered.")
                break



# === Run ===
if __name__ == "__main__":
    train_loader, val_loader = get_binary_loaders(
        os.path.join(cache_dir, "train"),
        os.path.join(cache_dir, "val")
    )
    print("Loading Completed")
    backbone = DualBackbone()
    print("initialized Model")
    feature_dim = backbone.resnet_feat_dim + backbone.deit_feat_dim
    classifier = BinaryClassifier(feature_dim)
    print("Started Training")
    train_binary_model(backbone, classifier, train_loader, val_loader)
