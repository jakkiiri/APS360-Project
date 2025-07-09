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

# ---- Settings ----
NUM_CLASSES = 9
BATCH_SIZE = 64
EPOCHS = 50
LEARNING_RATE = 0.0001
PATIENCE = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_PATH = "best_model.pth"

# ---- Model Definition ----
class SkinLesionClassifier(nn.Module):
    def __init__(self, num_classes=9):
        super().__init__()
        #self.backbone = models.resnet50(pretrained=True)
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
model.load_state_dict(torch.load(CHECKPOINT_PATH))
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2, verbose=True)


# ---- Training Loop with Early Stopping and Checkpointing ----
train_losses, val_losses, val_accuracies = [], [], []
precisions, recalls, f1s = [], [], []
best_acc = 0
epochs_no_improve = 0
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
    scheduler.step(acc)  # acc is validation accuracy
    for param_group in optimizer.param_groups:
        print(f"Current learning rate: {param_group['lr']}")
    print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_losses[-1]:.4f} | Val Loss: {val_losses[-1]:.4f} | Val Acc: {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f}")

    # Checkpointing
    if acc > best_acc:
        best_acc = acc
        torch.save(model.state_dict(), CHECKPOINT_PATH)
        print(f"✅ Saved new best model (Val Acc: {acc:.4f})")
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1
        print(f"No improvement. ({epochs_no_improve}/{PATIENCE} epochs)")

    # Early Stopping
    if epochs_no_improve >= PATIENCE:
        print("⏹️ Early stopping triggered.")
        break
'''
# ---- Load Best Model ----
model.load_state_dict(torch.load(CHECKPOINT_PATH))
print(f"\n🏆 Best model loaded from '{CHECKPOINT_PATH}' with validation accuracy: {best_acc:.4f}")

# ---- Visualization ----
epochs_range = range(1, len(train_losses) + 1)
plt.figure(figsize=(12, 6))
plt.plot(epochs_range, train_losses, label='Train Loss')
plt.plot(epochs_range, val_losses, label='Val Loss')
plt.plot(epochs_range, val_accuracies, label='Val Accuracy')
plt.plot(epochs_range, precisions, label='Precision')
plt.plot(epochs_range, recalls, label='Recall')
plt.plot(epochs_range, f1s, label='F1 Score')
plt.title("Training Progress")
plt.xlabel("Epoch")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ---- Confusion Matrix ----
model.eval()
preds, targets = [], []
with torch.no_grad():
    for imgs, labels in val_loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        outputs = model(imgs)
        preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
        targets.extend(labels.cpu().numpy())

cm = confusion_matrix(targets, preds)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.show()
