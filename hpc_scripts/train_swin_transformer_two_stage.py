#!/usr/bin/env python3
"""
Two-Stage Swin Transformer Training Script for Skin Disease Classification
Stage 1: Binary classification (nevus vs other)
Stage 2: Multi-class classification among the 8 non-nevus classes

Notes:
- This is based on train_swin_transformer.py, keeping most behavior the same
- W&B is fully disabled; all metrics are saved as matplotlib plots to disk
- Headless plotting is enforced; no figures are displayed
"""

import os
# Ensure headless operation for HPC
os.environ['MPLBACKEND'] = 'Agg'
if 'DISPLAY' not in os.environ:
    os.environ['DISPLAY'] = ''
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.transforms as transforms  # noqa: F401  (kept to match original imports)
import timm
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
import matplotlib
matplotlib.use('Agg')  # Set non-interactive backend for HPC
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import random
import argparse
from collections import Counter
import warnings
warnings.filterwarnings('ignore')


# Set seeds for reproducibility
def set_seeds(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def log_cuda_environment() -> None:
    """Log CUDA / PyTorch / GPU environment details to help diagnose kernel issues."""
    try:
        print("===== Environment Diagnostics =====")
        print(f"torch.__version__        : {torch.__version__}")
        print(f"torch.version.cuda       : {torch.version.cuda}")
        try:
            print(f"cudnn.version()          : {torch.backends.cudnn.version()}")
        except Exception as e:
            print(f"cudnn.version()          : unavailable ({e})")
        print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            print(f"CUDA device count        : {device_count}")
            for idx in range(device_count):
                name = torch.cuda.get_device_name(idx)
                major, minor = torch.cuda.get_device_capability(idx)
                print(f" - GPU {idx}: {name} | capability sm_{major}{minor}")
        print("===================================")
    except Exception as e:
        print(f"Failed to log CUDA environment: {e}")


# Class mapping (kept identical ordering to original single-stage trainer)
CLASS_NAMES = [
    'nevus', 'melanoma', 'bcc', 'keratosis',
    'actinic_keratosis', 'scc', 'dermatofibroma', 'lentigo', 'vascular_lesion'
]

OTHER_CLASS_NAMES = [name for name in CLASS_NAMES if name != 'nevus']


class AlbumentationsDataset(Dataset):
    """Custom dataset with Albumentations augmentations (unchanged baseline)"""

    def __init__(self, root_dir: str, split: str = 'train', image_size: int = 512):
        self.root_dir = os.path.join(root_dir, split)
        self.split = split
        self.image_size = image_size

        # Collect all image paths and labels
        self.samples = []  # list[(path, label_idx)]
        self.class_counts = Counter()

        for idx, class_name in enumerate(CLASS_NAMES):
            class_dir = os.path.join(self.root_dir, class_name)
            if os.path.exists(class_dir):
                for img_name in os.listdir(class_dir):
                    if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                        self.samples.append((os.path.join(class_dir, img_name), idx))
                        self.class_counts[idx] += 1

        print(f"{split.upper()} - Class distribution: {dict(self.class_counts)}")

        missing_classes = []
        for idx, class_name in enumerate(CLASS_NAMES):
            if idx not in self.class_counts:
                missing_classes.append(f"{idx}:{class_name}")
        if missing_classes:
            print(f"WARNING: Missing classes in {split} set: {missing_classes}")

        print(f"Total {split} samples: {len(self.samples)}")

        # Define augmentations based on split
        if split == 'train':
            self.transform = self._get_train_transforms()
        else:
            self.transform = self._get_val_transforms()

    def _get_train_transforms(self):
        return A.Compose([
            A.Resize(self.image_size, self.image_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            A.RandomRotate90(p=0.5),
            A.Rotate(limit=30, p=0.7),
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.7),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.6),
            A.GaussNoise(var_limit=(10, 50), p=0.3),
            A.GaussianBlur(blur_limit=3, p=0.3),
            A.RandomResizedCrop(self.image_size, self.image_size, scale=(0.7, 1.0), ratio=(0.8, 1.2), p=0.8),
            A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.3),
            A.GridDistortion(p=0.3),
            A.CoarseDropout(max_holes=8, max_height=32, max_width=32, p=0.3),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])

    def _get_val_transforms(self):
        return A.Compose([
            A.Resize(self.image_size, self.image_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])

    def _apply_minority_augmentation(self, image: np.ndarray, label: int) -> np.ndarray:
        """Apply extra aug to minority classes (kept same set as original)."""
        minority_classes = [1, 4, 5, 6, 7, 8]  # melanoma, actinic_keratosis, scc, dermatofibroma, lentigo, vascular_lesion
        if label in minority_classes:
            extra_aug = A.Compose([
                A.RandomGamma(gamma_limit=(80, 120), p=0.5),
                A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.5),
                A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=0.3),
                A.Emboss(alpha=(0.2, 0.5), strength=(0.2, 0.7), p=0.3),
            ])
            augmented = extra_aug(image=image)
            return augmented['image']
        return image

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        image = np.array(image)

        if self.split == 'train':
            image = self._apply_minority_augmentation(image, label)
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        return image, label


class BinaryNevusDataset(AlbumentationsDataset):
    """Dataset mapping original 9 classes into binary labels: nevus vs other.

    Label mapping:
        0 -> 0 (nevus)
        1..8 -> 1 (other)
    Class names: ['nevus', 'other']
    """

    BINARY_CLASS_NAMES = ['nevus', 'other']

    def __init__(self, root_dir: str, split: str = 'train', image_size: int = 512):
        super().__init__(root_dir=root_dir, split=split, image_size=image_size)
        # Remap samples into binary labels and recalc counts
        remapped = []
        counts = Counter()
        for path, orig_label in self.samples:
            bin_label = 0 if orig_label == 0 else 1
            remapped.append((path, bin_label))
            counts[bin_label] += 1
        self.samples = remapped
        self.class_counts = counts
        print(f"{split.upper()} (BINARY nevus vs other) - Class distribution: {dict(self.class_counts)}")


class OthersEightClassDataset(AlbumentationsDataset):
    """Dataset containing only the 8 non-nevus classes, labels remapped to 0..7.

    Mapping:
        original 1..8 -> new 0..7 (order preserved)
    Class names: OTHER_CLASS_NAMES
    """

    EIGHT_CLASS_NAMES = OTHER_CLASS_NAMES

    def __init__(self, root_dir: str, split: str = 'train', image_size: int = 512):
        # Directly build a filtered sample list using same transforms as parent
        self.root_dir = os.path.join(root_dir, split)
        self.split = split
        self.image_size = image_size

        self.samples = []
        self.class_counts = Counter()

        # Build a mapping from original label 1..8 to new 0..7
        # new_label = orig_label - 1
        for orig_idx, class_name in enumerate(CLASS_NAMES):
            if class_name == 'nevus':
                continue
            class_dir = os.path.join(self.root_dir, class_name)
            if os.path.exists(class_dir):
                for img_name in os.listdir(class_dir):
                    if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                        new_label = orig_idx - 1  # shift down by 1
                        self.samples.append((os.path.join(class_dir, img_name), new_label))
                        self.class_counts[new_label] += 1

        print(f"{split.upper()} (OTHERS 8-class) - Class distribution: {dict(self.class_counts)}")

        if split == 'train':
            self.transform = self._get_train_transforms()
        else:
            self.transform = self._get_val_transforms()

    # Reuse parent's transforms and augmentation behavior but adapt minority mapping
    def _apply_minority_augmentation(self, image: np.ndarray, label: int) -> np.ndarray:
        # Original minority classes are indices [1,4,5,6,7,8] in the 9-class setup.
        # After removing nevus (0) and shifting by -1, they become [0,3,4,5,6,7] in 8-class setup.
        minority_classes_8 = [0, 3, 4, 5, 6, 7]
        if label in minority_classes_8:
            extra_aug = A.Compose([
                A.RandomGamma(gamma_limit=(80, 120), p=0.5),
                A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.5),
                A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=0.3),
                A.Emboss(alpha=(0.2, 0.5), strength=(0.2, 0.7), p=0.3),
            ])
            augmented = extra_aug(image=image)
            return augmented['image']
        return image

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        image = np.array(image)
        if self.split == 'train':
            image = self._apply_minority_augmentation(image, label)
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        return image, label


class SwinTransformerClassifier(nn.Module):
    """Swin Transformer model for classification (same as original)."""

    def __init__(self, num_classes: int = 9, model_name: str = 'swin_base_patch4_window7_224', pretrained: bool = True, image_size: int = 224):
        super(SwinTransformerClassifier, self).__init__()
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0, img_size=image_size)
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, image_size, image_size)
            feature_dim = self.backbone(dummy_input).shape[1]
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return self.classifier(features)


def calculate_class_weights(dataset: Dataset) -> dict:
    """Calculate class weights from dataset.samples for imbalance handling."""
    class_counts = Counter()
    for _, label in dataset.samples:
        class_counts[label] += 1
    total_samples = len(dataset.samples)
    num_classes = len(set([lbl for _, lbl in dataset.samples]))
    weights = {}
    for class_idx, count in class_counts.items():
        weights[class_idx] = total_samples / (num_classes * max(count, 1))
    # Ensure contiguous indices present are covered
    for class_idx in range(num_classes):
        if class_idx not in weights:
            weights[class_idx] = 1.0
    return weights


def get_weighted_sampler(dataset: Dataset) -> WeightedRandomSampler:
    class_weights = calculate_class_weights(dataset)
    print(f"Class weights: {class_weights}")
    sample_weights = [class_weights[label] for _, label in dataset.samples]
    return WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)


def train_epoch(model, train_loader, criterion, optimizer, device, scheduler=None):
    model.train()
    running_loss = 0.0
    predictions, targets = [], []
    progress_bar = tqdm(train_loader, desc="Training")
    for images, labels in progress_bar:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        if scheduler:
            scheduler.step()
        running_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        predictions.extend(preds.cpu().numpy())
        targets.extend(labels.cpu().numpy())
        progress_bar.set_postfix({'Loss': f'{loss.item():.4f}'})
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = accuracy_score(targets, predictions)
    return epoch_loss, epoch_acc


def validate_epoch(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    predictions, targets = [], []
    with torch.no_grad():
        progress_bar = tqdm(val_loader, desc="Validation")
        for images, labels in progress_bar:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            predictions.extend(preds.cpu().numpy())
            targets.extend(labels.cpu().numpy())
            progress_bar.set_postfix({'Loss': f'{loss.item():.4f}'})
    epoch_loss = running_loss / len(val_loader)
    epoch_acc = accuracy_score(targets, predictions)
    precision = precision_score(targets, predictions, average='weighted', zero_division=0)
    recall = recall_score(targets, predictions, average='weighted', zero_division=0)
    f1 = f1_score(targets, predictions, average='weighted', zero_division=0)
    return epoch_loss, epoch_acc, precision, recall, f1, predictions, targets


def save_confusion_matrix(targets, predictions, class_names, save_path: str) -> None:
    cm = confusion_matrix(targets, predictions)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_and_save_curves(history: dict, out_dir: str, prefix: str) -> None:
    epochs = range(1, len(history['train_loss']) + 1)

    # Loss
    plt.figure()
    plt.plot(epochs, history['train_loss'], label='Train Loss')
    plt.plot(epochs, history['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss vs Epoch')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f'{prefix}_loss.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # Accuracy
    plt.figure()
    plt.plot(epochs, history['train_acc'], label='Train Acc')
    plt.plot(epochs, history['val_acc'], label='Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs Epoch')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f'{prefix}_accuracy.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # Precision/Recall/F1 (validation only)
    plt.figure()
    plt.plot(epochs, history['val_precision'], label='Val Precision')
    plt.plot(epochs, history['val_recall'], label='Val Recall')
    plt.plot(epochs, history['val_f1'], label='Val F1')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.title('Precision / Recall / F1 vs Epoch (Val)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f'{prefix}_metrics.png'), dpi=300, bbox_inches='tight')
    plt.close()


def train_one_model(model_name: str,
                    num_classes: int,
                    image_size: int,
                    train_dataset: Dataset,
                    val_dataset: Dataset,
                    output_dir: str,
                    file_prefix: str,
                    args) -> None:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SwinTransformerClassifier(num_classes=num_classes, model_name=model_name, pretrained=True, image_size=image_size)
    model = model.to(device)

    # Dataloaders with weighted sampling
    train_sampler = get_weighted_sampler(train_dataset)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    # Loss with class weights
    class_weights = calculate_class_weights(train_dataset)
    # Ensure weight vector length matches model output dimension
    weight_tensor = torch.tensor([class_weights.get(i, 1.0) for i in range(num_classes)], dtype=torch.float).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight_tensor)

    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [],
        'val_precision': [], 'val_recall': [], 'val_f1': []
    }

    best_val_acc = 0.0
    patience_counter = 0
    best_preds, best_targets = None, None

    print("Starting training...")
    for epoch in range(args.num_epochs):
        print(f"\nEpoch {epoch + 1}/{args.num_epochs}")
        print("-" * 50)
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, scheduler)
        val_loss, val_acc, val_precision, val_recall, val_f1, val_preds, val_targets = validate_epoch(model, val_loader, criterion, device)

        # Update history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_precision'].append(val_precision)
        history['val_recall'].append(val_recall)
        history['val_f1'].append(val_f1)

        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val   Loss: {val_loss:.4f}, Val   Acc: {val_acc:.4f}")
        print(f"Val Precision: {val_precision:.4f}, Val Recall: {val_recall:.4f}, Val F1: {val_f1:.4f}")

        # Save best model and its confusion matrix
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_acc': best_val_acc,
                'config': vars(args),
                'num_classes': num_classes,
            }, os.path.join(output_dir, f'{file_prefix}_best_model.pth'))

            best_preds, best_targets = val_preds, val_targets
            cm_path = os.path.join(output_dir, f'{file_prefix}_confusion_matrix.png')
            if num_classes == 2:
                cm_labels = ['nevus', 'other']
            else:
                cm_labels = OTHER_CLASS_NAMES
            save_confusion_matrix(best_targets, best_preds, cm_labels, cm_path)
            print(f"New best validation accuracy: {best_val_acc:.4f}")
        else:
            patience_counter += 1

        if patience_counter >= args.patience:
            print(f"Early stopping triggered after {epoch + 1} epochs")
            break

    # Final: save training curves
    plot_and_save_curves(history, output_dir, file_prefix)
    print(f"Training completed for {file_prefix}. Best val acc: {best_val_acc:.4f}")


def main(args):
    # Set seeds and device
    set_seeds(args.seed)
    # Print CUDA / PyTorch environment info early
    log_cuda_environment()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Prepare datasets once for binary and eight-class cases
    print("Loading datasets...")
    # Binary datasets
    train_dataset_binary = BinaryNevusDataset(args.data_dir, 'train', args.image_size)
    val_dataset_binary = BinaryNevusDataset(args.data_dir, 'val', args.image_size)

    # Others 8-class datasets
    train_dataset_others = OthersEightClassDataset(args.data_dir, 'train', args.image_size)
    val_dataset_others = OthersEightClassDataset(args.data_dir, 'val', args.image_size)

    print(f"Binary Train samples: {len(train_dataset_binary)}, Val samples: {len(val_dataset_binary)}")
    print(f"Others  Train samples: {len(train_dataset_others)}, Val samples: {len(val_dataset_others)}")

    # Ensure output subdirs exist
    os.makedirs(args.output_dir, exist_ok=True)
    out_bin = os.path.join(args.output_dir, 'binary_nevus')
    out_oth = os.path.join(args.output_dir, 'others_8class')
    os.makedirs(out_bin, exist_ok=True)
    os.makedirs(out_oth, exist_ok=True)

    # Train binary model
    train_one_model(
        model_name=args.model_name,
        num_classes=2,
        image_size=args.image_size,
        train_dataset=train_dataset_binary,
        val_dataset=val_dataset_binary,
        output_dir=out_bin,
        file_prefix='binary_nevus_vs_other',
        args=args,
    )

    # Train 8-class model
    train_one_model(
        model_name=args.model_name,
        num_classes=8,
        image_size=args.image_size,
        train_dataset=train_dataset_others,
        val_dataset=val_dataset_others,
        output_dir=out_oth,
        file_prefix='others_8class',
        args=args,
    )

    print("\nTwo-stage training completed successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Two-Stage Swin Transformer Training (Nevus vs Other, and Others 8-class)")
    # Data arguments
    parser.add_argument("--data_dir", type=str, required=True, help="Path to dataset directory")
    parser.add_argument("--output_dir", type=str, default="./outputs_two_stage", help="Output directory for models and plots")
    parser.add_argument("--image_size", type=int, default=224, help="Input image size")
    # Model arguments
    parser.add_argument("--model_name", type=str, default="swin_base_patch4_window7_224",
                        help="Swin Transformer model name (ensure image_size matches model expected size)")
    # Training arguments
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--num_epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of data loader workers")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    main(args)


