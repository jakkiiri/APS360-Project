# skin_lesion_training.py

import os
import torch
import torch.nn as nn
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

# ---- Settings ----
NUM_CLASSES = 9
BATCH_SIZE = 64
EPOCHS = 60
LEARNING_RATE = 3e-4
PATIENCE = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_PATH = "best_model.pth"
METRICS_PATH = "training_metrics.csv"
CACHE_ROOT = "./cached_dataset"

# ---- Model Definition ----
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

# ---- Caching Utility ----
def cache_dataset(input_dir, output_dir, transform):
    os.makedirs(output_dir, exist_ok=True)
    dataset = datasets.ImageFolder(input_dir, transform=transform)
    for class_name, class_idx in dataset.class_to_idx.items():
        os.makedirs(os.path.join(output_dir, class_name), exist_ok=True)

    print(f"Caching dataset from {input_dir} to {output_dir}...")
    for idx in tqdm(range(len(dataset))):
        img, label = dataset[idx]
        class_name = dataset.classes[label]
        cache_path = os.path.join(output_dir, class_name, f"{idx}.pt")
        torch.save({'image': img, 'label': label}, cache_path)

class CachedDataset(torch.utils.data.Dataset):
    def __init__(self, cache_dir):
        self.samples = []
        for class_dir in os.listdir(cache_dir):
            full_path = os.path.join(cache_dir, class_dir)
            if os.path.isdir(full_path):
                for file in os.listdir(full_path):
                    self.samples.append(os.path.join(full_path, file))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        data = torch.load(self.samples[idx])
        return data['image'], data['label']

# ---- Data Preparation ----
base_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    #transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

train_cache_path = os.path.join(CACHE_ROOT, "train")
val_cache_path = os.path.join(CACHE_ROOT, "val")

if not os.path.exists(train_cache_path) or not os.listdir(train_cache_path):
    cache_dataset("C:\\Users\\shore\\Desktop\\APS360\\Datasets\\DataSplit2\\train", train_cache_path, base_transform)

if not os.path.exists(val_cache_path) or not os.listdir(val_cache_path):
    cache_dataset("C:\\Users\\shore\\Desktop\\APS360\\Datasets\\DataSplit2\\val", val_cache_path, base_transform)


train_dataset = CachedDataset(os.path.join(CACHE_ROOT, "train"))
val_dataset = CachedDataset(os.path.join(CACHE_ROOT, "val"))

labels = [torch.load(sample)['label'] for sample in train_dataset.samples]
class_counts = np.bincount(labels)
weights = 1. / torch.tensor(class_counts, dtype=torch.float)
sample_weights = torch.DoubleTensor([weights[label] for label in labels])
sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ---- Training Function ----
def train_model(model, train_loader, val_loader):
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2, verbose=True)

    best_acc = 0
    epochs_no_improve = 0
    metrics_df = pd.DataFrame(columns=["epoch", "train_loss", "val_loss", "accuracy", "precision", "recall", "f1", "auc"])

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0
        with tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} - Training") as pbar:
            for imgs, labels in pbar:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                optimizer.zero_grad()
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                pbar.set_postfix(loss=loss.item())

        train_loss = running_loss / len(train_loader)

        model.eval()
        val_loss = 0
        preds, targets, probs = [], [], []
        with torch.no_grad():
            for imgs, labels in tqdm(val_loader, desc="Validating"):
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                softmax_outputs = torch.softmax(outputs, dim=1)
                preds.extend(torch.argmax(softmax_outputs, dim=1).cpu().numpy())
                targets.extend(labels.cpu().numpy())
                probs.extend(softmax_outputs.cpu().numpy())

        val_loss /= len(val_loader)
        acc = accuracy_score(targets, preds)
        prec = precision_score(targets, preds, average='macro', zero_division=0)
        rec = recall_score(targets, preds, average='macro', zero_division=0)
        f1 = f1_score(targets, preds, average='macro', zero_division=0)

        try:
            auc = roc_auc_score(targets, probs, multi_class='ovr')
        except:
            auc = 0.0

        print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f} | Val Loss={val_loss:.4f} | Acc={acc:.4f} | Prec={prec:.4f} | Recall={rec:.4f} | F1={f1:.4f} | AUC={auc:.4f}")

        scheduler.step(acc)

        metrics_df.loc[len(metrics_df)] = [epoch+1, train_loss, val_loss, acc, prec, rec, f1, auc]
        metrics_df.to_csv(METRICS_PATH, index=False)

        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), CHECKPOINT_PATH)
            print(f"✅ Saved new best model (Val Acc: {acc:.4f})")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            print(f"No improvement. ({epochs_no_improve}/{PATIENCE} epochs)")

        if epochs_no_improve >= PATIENCE:
            print("⏹️ Early stopping triggered.")
            break

# ---- Run Training ----
model = SkinLesionClassifier(NUM_CLASSES).to(DEVICE)
train_model(model, train_loader, val_loader)

# ---- Visualization ----
metrics_df = pd.read_csv(METRICS_PATH)

plt.figure(figsize=(12, 6))
plt.plot(metrics_df["epoch"], metrics_df["accuracy"], label="Accuracy")
plt.plot(metrics_df["epoch"], metrics_df["precision"], label="Precision")
plt.plot(metrics_df["epoch"], metrics_df["recall"], label="Recall")
plt.plot(metrics_df["epoch"], metrics_df["f1"], label="F1 Score")
plt.plot(metrics_df["epoch"], metrics_df["auc"], label="AUC")
plt.title("Validation Metrics Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Score")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ---- Confusion Matrix ----
class_names = [
    "Actinic Keratosis", "bcc", "Dermatofibroma", "lentigo", "Melanoma",
    "nevus", "scc", "seborrheic_keratosis", "vascular_lesion"
]

model.load_state_dict(torch.load(CHECKPOINT_PATH))
model.eval()
preds, targets = [], []
with torch.no_grad():
    for imgs, labels in val_loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        outputs = model(imgs)
        preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
        targets.extend(labels.cpu().numpy())

cm = confusion_matrix(targets, preds)
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted Class')
plt.ylabel('True Class')
plt.title('Confusion Matrix on Validation Set')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
