#!/usr/bin/env python3
"""
Swin Transformer Training Script for Skin Disease Classification
Designed for HPC (Killarney) with W&B monitoring and class imbalance handling
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
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import timm
import wandb
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
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
def set_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Class mapping
CLASS_NAMES = [
    'nevus', 'melanoma', 'bcc', 'keratosis',
    'actinic_keratosis', 'scc', 'dermatofibroma', 'lentigo', 'vascular_lesion'
]

class AlbumentationsDataset(Dataset):
    """Custom dataset with Albumentations augmentations"""
    
    def __init__(self, root_dir, split='train', image_size=512, class_weights=None):
        self.root_dir = os.path.join(root_dir, split)
        self.split = split
        self.image_size = image_size
        self.class_weights = class_weights if class_weights else {}
        
        # Collect all image paths and labels
        self.samples = []
        self.class_counts = Counter()
        
        for idx, class_name in enumerate(CLASS_NAMES):
            class_dir = os.path.join(self.root_dir, class_name)
            if os.path.exists(class_dir):
                for img_name in os.listdir(class_dir):
                    if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                        self.samples.append((os.path.join(class_dir, img_name), idx))
                        self.class_counts[idx] += 1
        
        print(f"{split.upper()} - Class distribution: {dict(self.class_counts)}")
        
        # Check for missing classes and warn
        missing_classes = []
        for idx, class_name in enumerate(CLASS_NAMES):
            if idx not in self.class_counts:
                missing_classes.append(f"{idx}:{class_name}")
        
        if missing_classes:
            print(f"WARNING: Missing classes in {split} set: {missing_classes}")
        
        print(f"Total {split} samples: {len(self.samples)}")
        
        # Define augmentations based on split and class weights
        if split == 'train':
            self.transform = self._get_train_transforms()
        else:
            self.transform = self._get_val_transforms()
    
    def _get_train_transforms(self):
        """Strong augmentations for training with minority class emphasis"""
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
        """Minimal augmentations for validation/test"""
        return A.Compose([
            A.Resize(self.image_size, self.image_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
    
    def _apply_minority_augmentation(self, image, label):
        """Apply additional augmentations to minority classes"""
        minority_classes = [1, 4, 5, 6, 7, 8]  # melanoma, actinic_keratosis, scc, dermatofibroma, lentigo, vascular_lesion
        
        if label in minority_classes:
            # Additional strong augmentations for minority classes
            extra_aug = A.Compose([
                A.RandomGamma(gamma_limit=(80, 120), p=0.5),
                A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.5),
                A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=0.3),
                A.Emboss(alpha=(0.2, 0.5), strength=(0.2, 0.7), p=0.3),
            ])
            augmented = extra_aug(image=image)
            return augmented['image']
        return image
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        image = np.array(image)
        
        # Apply minority class augmentation if training
        if self.split == 'train':
            image = self._apply_minority_augmentation(image, label)
        
        # Apply main transformations
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        
        return image, label

class SwinTransformerClassifier(nn.Module):
    """Swin Transformer model for skin disease classification"""
    
    def __init__(self, num_classes=9, model_name='swin_base_patch4_window7_224', pretrained=True, image_size=224):
        super(SwinTransformerClassifier, self).__init__()
        
        # Load pre-trained Swin Transformer
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0, img_size=image_size)
        
        # Get feature dimension using the correct image size
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, image_size, image_size)
            feature_dim = self.backbone(dummy_input).shape[1]
        
        # Classification head with dropout
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)

def calculate_class_weights(dataset):
    """Calculate class weights for handling imbalance"""
    class_counts = Counter()
    for _, label in dataset.samples:
        class_counts[label] += 1
    
    total_samples = len(dataset.samples)
    num_classes = len(class_counts)  # Use actual number of classes in dataset
    
    weights = {}
    # Calculate weights for all classes present in the dataset
    for class_idx, count in class_counts.items():
        weights[class_idx] = total_samples / (num_classes * count)
    
    # Also ensure we have weights for all expected classes (0 to len(CLASS_NAMES)-1)
    for class_idx in range(len(CLASS_NAMES)):
        if class_idx not in weights:
            weights[class_idx] = 1.0  # Default weight for missing classes
    
    return weights

def get_weighted_sampler(dataset):
    """Get weighted random sampler for balanced training"""
    class_weights = calculate_class_weights(dataset)
    
    # Debug: Print class weights
    print(f"Class weights: {class_weights}")
    
    # Get all unique labels from dataset to ensure coverage
    unique_labels = set()
    for _, label in dataset.samples:
        unique_labels.add(label)
    
    # Add any missing labels to class_weights
    for label in unique_labels:
        if label not in class_weights:
            print(f"Warning: Adding missing label {label} to class_weights with default weight 1.0")
            class_weights[label] = 1.0
    
    # Create sample weights
    sample_weights = []
    for _, label in dataset.samples:
        sample_weights.append(class_weights[label])
    
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
    return sampler

def train_epoch(model, train_loader, criterion, optimizer, device, scheduler=None):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    predictions, targets = [], []
    
    progress_bar = tqdm(train_loader, desc="Training")
    for batch_idx, (images, labels) in enumerate(progress_bar):
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
        
        # Collect predictions
        _, preds = torch.max(outputs, 1)
        predictions.extend(preds.cpu().numpy())
        targets.extend(labels.cpu().numpy())
        
        # Update progress bar
        progress_bar.set_postfix({'Loss': f'{loss.item():.4f}'})
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = accuracy_score(targets, predictions)
    
    return epoch_loss, epoch_acc

def validate_epoch(model, val_loader, criterion, device):
    """Validate for one epoch"""
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
            
            # Collect predictions
            _, preds = torch.max(outputs, 1)
            predictions.extend(preds.cpu().numpy())
            targets.extend(labels.cpu().numpy())
            
            progress_bar.set_postfix({'Loss': f'{loss.item():.4f}'})
    
    epoch_loss = running_loss / len(val_loader)
    epoch_acc = accuracy_score(targets, predictions)
    
    # Calculate additional metrics
    precision = precision_score(targets, predictions, average='weighted', zero_division=0)
    recall = recall_score(targets, predictions, average='weighted', zero_division=0)
    f1 = f1_score(targets, predictions, average='weighted', zero_division=0)
    
    return epoch_loss, epoch_acc, precision, recall, f1, predictions, targets

def save_confusion_matrix(targets, predictions, class_names, save_path):
    """Save confusion matrix plot to disk (headless operation for HPC)"""
    cm = confusion_matrix(targets, predictions)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()  # Important: close figure to free memory on HPC

def main(args):
    # Initialize W&B with proper authentication handling for HPC
    if not args.disable_wandb:
        # Option 1: Check for environment variable (recommended for HPC)
        if os.getenv('WANDB_API_KEY'):
            print("Using W&B API key from environment variable")
        elif os.getenv('WANDB_MODE') == 'offline':
            print("Running W&B in offline mode - will sync later")
        else:
            print("Warning: No W&B API key found. Set WANDB_API_KEY environment variable or use WANDB_MODE=offline")
            print("Continuing with offline mode...")
            os.environ['WANDB_MODE'] = 'offline'
        
        wandb.init(
            project="skin-disease-swin-transformer",
            name=f"swin_{args.model_name.split('_')[1]}_{args.image_size}px",
            config={
                "model_name": args.model_name,
                "image_size": args.image_size,
                "batch_size": args.batch_size,
                "learning_rate": args.learning_rate,
                "num_epochs": args.num_epochs,
                "weight_decay": args.weight_decay,
                "num_classes": 9
            },
            # Handle potential authentication issues gracefully
            settings=wandb.Settings(start_method="fork")
        )
    else:
        print("Weights & Biases logging disabled")
    
    # Set seeds and device
    set_seeds(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create datasets
    print("Loading datasets...")
    train_dataset = AlbumentationsDataset(args.data_dir, 'train', args.image_size)
    val_dataset = AlbumentationsDataset(args.data_dir, 'val', args.image_size)
    
    # Create data loaders
    train_sampler = get_weighted_sampler(train_dataset)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, 
                             sampler=train_sampler, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, 
                           shuffle=False, num_workers=args.num_workers, pin_memory=True)
    
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    
    # Create model
    print(f"Creating Swin Transformer model: {args.model_name}")
    model = SwinTransformerClassifier(num_classes=9, model_name=args.model_name, pretrained=True, image_size=args.image_size)
    model = model.to(device)
    
    # Loss function with class weights
    class_weights = calculate_class_weights(train_dataset)
    weight_tensor = torch.tensor([class_weights[i] for i in range(9)], dtype=torch.float).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight_tensor)
    
    # Optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    
    # Training loop
    best_val_acc = 0.0
    patience_counter = 0
    
    print("Starting training...")
    for epoch in range(args.num_epochs):
        print(f"\nEpoch {epoch+1}/{args.num_epochs}")
        print("-" * 50)
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, scheduler)
        
        # Validate
        val_loss, val_acc, val_precision, val_recall, val_f1, val_preds, val_targets = validate_epoch(
            model, val_loader, criterion, device)
        
        # Log to W&B (if enabled)
        if not args.disable_wandb:
            wandb.log({
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_accuracy": train_acc,
                "val_loss": val_loss,
                "val_accuracy": val_acc,
                "val_precision": val_precision,
                "val_recall": val_recall,
                "val_f1": val_f1,
                "learning_rate": optimizer.param_groups[0]['lr']
            })
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        print(f"Val Precision: {val_precision:.4f}, Val Recall: {val_recall:.4f}, Val F1: {val_f1:.4f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            
            # Save model
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_acc': best_val_acc,
                'config': vars(args)
            }, os.path.join(args.output_dir, 'best_model.pth'))
            
            # Save confusion matrix
            save_confusion_matrix(val_targets, val_preds, CLASS_NAMES, 
                                os.path.join(args.output_dir, 'confusion_matrix.png'))
            
            print(f"New best validation accuracy: {best_val_acc:.4f}")
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= args.patience:
            print(f"Early stopping triggered after {epoch + 1} epochs")
            break
    
    print(f"\nTraining completed! Best validation accuracy: {best_val_acc:.4f}")
    if not args.disable_wandb:
        wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Swin Transformer for Skin Disease Classification")
    
    # Data arguments
    parser.add_argument("--data_dir", type=str, required=True, help="Path to dataset directory")
    parser.add_argument("--output_dir", type=str, default="./outputs", help="Output directory for models and plots")
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
    parser.add_argument("--disable_wandb", action="store_true", help="Disable Weights & Biases logging")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    main(args)