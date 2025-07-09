import os
import torch
import torch.nn as nn
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd

# ---- Settings ----
NUM_CLASSES = 9
BATCH_SIZE = 64
EPOCHS = 60
LEARNING_RATE = 3e-4
PATIENCE = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_PATH = "best_model.pth"
METRICS_PATH = "training_metrics.csv"

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

# ---- Data Preparation ----
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

train_data = datasets.ImageFolder("C:\\Users\\shore\\Desktop\\APS360\\Datasets\\DataSplit\\train", transform=transform)
val_data = datasets.ImageFolder("C:\\Users\\shore\\Desktop\\APS360\\Datasets\\DataSplit\\val", transform=transform)

# Class balancing
labels = [label for _, label in train_data]
class_counts = np.bincount(labels)
weights = 1. / torch.tensor(class_counts, dtype=torch.float)
sample_weights = torch.DoubleTensor([weights[label] for label in labels])
sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, sampler=sampler)
val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)

# ---- Model Setup ----
model = SkinLesionClassifier(NUM_CLASSES).to(DEVICE)
#model.load_state_dict(torch.load(CHECKPOINT_PATH))
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2, verbose=True)

# ---- Training Loop with Early Stopping and Checkpointing ----
train_losses, val_losses, val_accuracies = [], [], []
precisions, recalls, f1s = [], [], []
best_acc = 0
epochs_no_improve = 0

metrics_df = pd.DataFrame(columns=["epoch", "train_loss", "val_loss", "accuracy", "precision", "recall", "f1"])
'''
for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0
    with tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}") as pbar:
        for imgs, labels in pbar:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            pbar.set_postfix(loss=loss.item())

    train_losses.append(epoch_loss / len(train_loader))

    # Validation
    model.eval()
    val_loss = 0
    preds, targets = [], []
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
            targets.extend(labels.cpu().numpy())

    val_losses.append(val_loss / len(val_loader))

    acc = accuracy_score(targets, preds)
    prec = precision_score(targets, preds, average='macro', zero_division=0)
    rec = recall_score(targets, preds, average='macro', zero_division=0)
    f1 = f1_score(targets, preds, average='macro', zero_division=0)

    val_accuracies.append(acc)
    precisions.append(prec)
    recalls.append(rec)
    f1s.append(f1)

    scheduler.step(acc)
    for param_group in optimizer.param_groups:
        print(f"Current learning rate: {param_group['lr']}")
    print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_losses[-1]:.4f} | Val Loss: {val_losses[-1]:.4f} | Val Acc: {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f}")

    metrics_df.loc[len(metrics_df)] = [epoch+1, train_losses[-1], val_losses[-1], acc, prec, rec, f1]
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
'''
# ---- Load Best Model ----
model.load_state_dict(torch.load(CHECKPOINT_PATH))
print(f"\n🏆 Best model loaded from '{CHECKPOINT_PATH}' with validation accuracy: {best_acc:.4f}")

# ---- Visualization ----
metrics_df = pd.read_csv(METRICS_PATH)
'''
# ---- Combined Metrics Plot ----
plt.figure(figsize=(12, 6))
plt.plot(metrics_df["epoch"], metrics_df["accuracy"], label="Accuracy")
plt.plot(metrics_df["epoch"], metrics_df["precision"], label="Precision")
plt.plot(metrics_df["epoch"], metrics_df["recall"], label="Recall")
plt.plot(metrics_df["epoch"], metrics_df["f1"], label="F1 Score")
plt.title("Validation Metrics Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Score")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
'''

# ---- Load Metrics and Plot All Validation Metrics in One Figure ----
metrics_df = pd.read_csv(METRICS_PATH)

# Loss Plot (Train and Val)
plt.figure(figsize=(10, 6))
plt.plot(metrics_df["epoch"], metrics_df["train_loss"], label="Train Loss")
plt.plot(metrics_df["epoch"], metrics_df["val_loss"], label="Val Loss")
plt.title("Training and Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Validation Metrics in a Grid
fig, axs = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Validation Metrics Over Epochs", fontsize=16)

axs[0, 0].plot(metrics_df["epoch"], metrics_df["accuracy"], label="Accuracy", color="tab:blue")
axs[0, 0].set_title("Validation Accuracy")
axs[0, 0].set_xlabel("Epoch")
axs[0, 0].set_ylabel("Accuracy")
axs[0, 0].grid(True)

axs[0, 1].plot(metrics_df["epoch"], metrics_df["precision"], label="Precision", color="tab:green")
axs[0, 1].set_title("Validation Precision")
axs[0, 1].set_xlabel("Epoch")
axs[0, 1].set_ylabel("Precision")
axs[0, 1].grid(True)

axs[1, 0].plot(metrics_df["epoch"], metrics_df["recall"], label="Recall", color="tab:orange")
axs[1, 0].set_title("Validation Recall")
axs[1, 0].set_xlabel("Epoch")
axs[1, 0].set_ylabel("Recall")
axs[1, 0].grid(True)

axs[1, 1].plot(metrics_df["epoch"], metrics_df["f1"], label="F1 Score", color="tab:red")
axs[1, 1].set_title("Validation F1 Score")
axs[1, 1].set_xlabel("Epoch")
axs[1, 1].set_ylabel("F1 Score")
axs[1, 1].grid(True)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()


# ---- Confusion Matrix ----
# ---- Confusion Matrix with Class Labels ----
class_names = [
    "Actinic Keratosis", "bcc", "Dermatofibroma", "lentigo", "Melanoma",
    "nevus", "scc", "seborrheic_keratosis", "vascular_lesion"
]

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
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names)
plt.xlabel('Predicted Class')
plt.ylabel('True Class')
plt.title('Confusion Matrix on Validation Set')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()


