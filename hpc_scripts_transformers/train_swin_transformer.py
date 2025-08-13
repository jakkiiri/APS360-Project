#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced Swin Transformer Training Script for Skin Disease Classification
Designed for HPC with advanced class imbalance handling and headless operation
"""

import os
# Ensure headless operation for HPC
os.environ['MPLBACKEND'] = 'Agg'
if 'DISPLAY' not in os.environ:
    os.environ['DISPLAY'] = ''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import timm
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib
matplotlib.use('Agg')  # Set non-interactive backend for HPC
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import random
import argparse
from collections import Counter
import warnings
import pandas as pd
import pickle
import math
import time
import json
from timm.data import Mixup
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

# Minority classes for targeted augmentation and sampling (updated to include keratosis)
MINORITY_CLASSES = {1, 3, 4, 5, 6, 7, 8}  # melanoma, keratosis, actinic_keratosis, scc, dermatofibroma, lentigo, vascular_lesion

# Medical importance weights for each class (enhanced for underperforming classes)
CLASS_IMPORTANCE = {
    0: 1.0,   # nevus - base (most common)
    1: 3.0,   # melanoma - highest (life threatening)
    2: 2.0,   # bcc - moderate (malignant but less aggressive)
    3: 2.2,   # keratosis - increased (challenging to distinguish from actinic_keratosis)
    4: 2.5,   # actinic_keratosis - high (pre-cancerous)
    5: 2.5,   # scc - high (malignant)
    6: 1.8,   # dermatofibroma - moderate (differential diagnosis)
    7: 2.3,   # lentigo - increased (age-related, challenging morphology)
    8: 1.8    # vascular_lesion - moderate (specific diagnosis)
}

# ==================== Advanced Loss Functions ====================
class FocalLoss(nn.Module):
    """Focal Loss implementation for handling class imbalance"""
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class ClassBalancedFocalLoss(nn.Module):
    """Class-Balanced Focal Loss for long-tail distribution"""
    def __init__(self, class_counts, beta=0.9999, gamma=2.0):
        super(ClassBalancedFocalLoss, self).__init__()
        # Clip counts to prevent division by zero
        counts = np.clip(np.array(class_counts, dtype=np.float32), 1.0, None)
        # Calculate effective number of samples
        effective_num = 1.0 - np.power(beta, counts)
        weights = (1.0 - beta) / np.array(effective_num)
        weights = weights / np.sum(weights) * len(weights)
        
        self.weights = torch.tensor(weights, dtype=torch.float32)
        self.gamma = gamma
        
    def forward(self, inputs, targets):
        device = inputs.device
        weights = self.weights.to(device)
        
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=weights)
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        return focal_loss.mean()

class LDAMLoss(nn.Module):
    """Label-Distribution-Aware Margin Loss"""
    def __init__(self, class_counts, max_m=0.5, s=30):
        super(LDAMLoss, self).__init__()
        # Clip counts to prevent division by zero and inf values
        counts = np.clip(np.array(class_counts, dtype=np.float32), 1.0, None)
        m_list = 1.0 / np.sqrt(np.sqrt(counts))
        m_list = m_list * (max_m / np.max(m_list))
        m_list = torch.tensor(m_list, dtype=torch.float32)
        self.m_list = m_list
        self.s = s
        
    def forward(self, x, target):
        device = x.device
        m_list = self.m_list.to(device)
        
        # Use bool dtype instead of deprecated uint8
        index = torch.zeros_like(x, dtype=torch.bool)
        index.scatter_(1, target.data.view(-1, 1), True)
        
        index_float = index.float()
        batch_m = torch.matmul(m_list[None, :], index_float.transpose(0, 1))
        batch_m = batch_m.view((-1, 1))
        x_m = x - batch_m
        
        output = torch.where(index, x_m, x)
        return F.cross_entropy(self.s * output, target)

def soft_target_cross_entropy(logits, target_soft):
    """Soft target cross entropy for Mixup/CutMix"""
    logprob = F.log_softmax(logits, dim=-1)
    return -(target_soft * logprob).sum(dim=-1).mean()

class AlbumentationsDataset(Dataset):
    """Enhanced dataset with aggressive minority class augmentation and oversampling"""
    
    def __init__(self, root_dir, split='train', image_size=512, minority_boost_factor=3.0, 
                 dermatology_aug_prob=0.8, challenging_boost_factor=4.0):
        self.root_dir = os.path.join(root_dir, split)
        self.split = split
        self.image_size = image_size
        self.minority_boost_factor = minority_boost_factor
        self.dermatology_aug_prob = dermatology_aug_prob
        self.challenging_boost_factor = challenging_boost_factor
        
        # Collect all image paths and labels with minority class oversampling
        self.samples = []
        self.class_counts = Counter()  # Will contain oversampled counts
        self.original_class_counts = Counter()  # Contains original counts for loss functions
        
        # First pass: collect all samples
        original_samples = {i: [] for i in range(len(CLASS_NAMES))}
        
        for idx, class_name in enumerate(CLASS_NAMES):
            class_dir = os.path.join(self.root_dir, class_name)
            if os.path.exists(class_dir):
                for img_name in os.listdir(class_dir):
                    if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                        sample = (os.path.join(class_dir, img_name), idx)
                        original_samples[idx].append(sample)
                        self.original_class_counts[idx] += 1
        
        print(f"{split.upper()} - Original class distribution: {dict(self.original_class_counts)}")
        
        # Second pass: oversample minority classes during training
        for class_idx, samples in original_samples.items():
            self.samples.extend(samples)  # Add original samples
            
            # Oversample minority classes
            if split == 'train' and class_idx in MINORITY_CLASSES and len(samples) > 0:
                # Special boost for challenging classes (lentigo=7, keratosis=3)
                if class_idx in [3, 7]:  # keratosis, lentigo
                    boost_count = int(len(samples) * (self.challenging_boost_factor - 1))
                    print(f"?? Extra boost for challenging class '{CLASS_NAMES[class_idx]}': {self.challenging_boost_factor}x")
                else:
                    boost_count = int(len(samples) * (self.minority_boost_factor - 1))
                
                if boost_count > 0 and len(samples) > 0:
                    boost_samples = np.random.choice(len(samples), boost_count, replace=True)
                    for sample_idx in boost_samples:
                        self.samples.append(samples[sample_idx])
                    print(f"?? Boosted minority class '{CLASS_NAMES[class_idx]}' by {boost_count} samples")
                else:
                    print(f"?? Skipping boost for '{CLASS_NAMES[class_idx]}' (samples: {len(samples)}, boost: {boost_count})")
        
        # Update counts after oversampling
        self.class_counts = Counter()
        for _, label in self.samples:
            self.class_counts[label] += 1
        
        print(f"{split.upper()} - Final class distribution: {dict(self.class_counts)}")
        print(f"Total {split} samples after oversampling: {len(self.samples)}")
        
        # Check for missing classes and warn
        missing_classes = []
        for idx, class_name in enumerate(CLASS_NAMES):
            if idx not in self.original_class_counts:
                missing_classes.append(f"{idx}:{class_name}")
        
        if missing_classes:
            print(f"WARNING: Missing classes in {split} set: {missing_classes}")
        
        # Define augmentations based on split
        if split == 'train':
            self.base_transform = self._get_train_transforms()
            self.minority_transform = self._get_minority_transforms()
            self.challenging_transform = self._get_challenging_class_transforms()
            self.dermatology_transform = self._get_dermatology_transforms()
        else:
            self.transform = self._get_val_transforms()
    
    def _get_train_transforms(self):
        """Base augmentations for all classes"""
        return A.Compose([
            A.Resize(self.image_size, self.image_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            A.RandomRotate90(p=0.5),
            A.Rotate(limit=20, p=0.6),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.6),
            A.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.08, p=0.5),
            A.GaussNoise(var_limit=(10, 30), p=0.25),
            A.GaussianBlur(blur_limit=3, p=0.2),
            A.RandomResizedCrop(self.image_size, self.image_size, scale=(0.8, 1.0), ratio=(0.9, 1.1), p=0.7),
            A.CoarseDropout(max_holes=6, max_height=24, max_width=24, p=0.25),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
    
    def _get_minority_transforms(self):
        """Aggressive augmentations specifically for minority classes"""
        return A.Compose([
            A.Resize(self.image_size, self.image_size),
            # More aggressive geometric transforms
            A.HorizontalFlip(p=0.6),
            A.VerticalFlip(p=0.4),
            A.RandomRotate90(p=0.6),
            A.Rotate(limit=35, p=0.8),
            A.ShiftScaleRotate(shift_limit=0.08, scale_limit=0.15, rotate_limit=15, p=0.7),
            A.ElasticTransform(alpha=2, sigma=50, alpha_affine=50, p=0.4),
            A.GridDistortion(p=0.35),
            A.OpticalDistortion(p=0.3),
            A.Perspective(scale=(0.05, 0.1), p=0.25),
            
            # Intensive color augmentations
            A.RandomBrightnessContrast(brightness_limit=0.35, contrast_limit=0.35, p=0.8),
            A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.15, p=0.7),
            A.HueSaturationValue(hue_shift_limit=25, sat_shift_limit=40, val_shift_limit=25, p=0.6),
            A.RandomGamma(gamma_limit=(70, 130), p=0.5),
            
            # Advanced noise and artifacts
            A.OneOf([
                A.GaussNoise(var_limit=(15, 60), p=0.6),
                A.ISONoise(intensity=(0.1, 0.6), p=0.4),
                A.MultiplicativeNoise(multiplier=[0.8, 1.2], p=0.3),
            ], p=0.5),
            
            # More aggressive dropout
            A.CoarseDropout(max_holes=12, max_height=36, max_width=36, p=0.4),
            A.Cutout(num_holes=8, max_h_size=20, max_w_size=20, p=0.3),
            
            # Random crop with more variation
            A.RandomResizedCrop(self.image_size, self.image_size, scale=(0.6, 1.0), ratio=(0.7, 1.3), p=0.8),
            
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
    
    def _get_challenging_class_transforms(self):
        """Ultra-aggressive augmentations for challenging classes (lentigo, keratosis)"""
        return A.Compose([
            A.Resize(self.image_size, self.image_size),
            
            # Ultra-aggressive geometric transforms
            A.HorizontalFlip(p=0.7),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.7),
            A.Rotate(limit=45, p=0.9),
            A.ShiftScaleRotate(shift_limit=0.12, scale_limit=0.2, rotate_limit=20, p=0.8),
            A.ElasticTransform(alpha=3, sigma=50, alpha_affine=60, p=0.5),
            A.GridDistortion(p=0.4),
            A.OpticalDistortion(p=0.4),
            A.Perspective(scale=(0.05, 0.15), p=0.35),
            
            # Extreme color augmentations for better feature learning
            A.RandomBrightnessContrast(brightness_limit=0.4, contrast_limit=0.4, p=0.9),
            A.ColorJitter(brightness=0.35, contrast=0.35, saturation=0.35, hue=0.2, p=0.8),
            A.HueSaturationValue(hue_shift_limit=30, sat_shift_limit=50, val_shift_limit=30, p=0.7),
            A.RandomGamma(gamma_limit=(60, 140), p=0.6),
            A.CLAHE(clip_limit=6.0, tile_grid_size=(8, 8), p=0.7),  # Enhanced contrast
            A.Sharpen(alpha=(0.3, 0.7), lightness=(0.5, 1.0), p=0.5),
            A.Emboss(alpha=(0.3, 0.7), strength=(0.3, 0.8), p=0.4),
            
            # Medical imaging specific augmentations
            A.OneOf([
                A.Downscale(scale_min=0.5, scale_max=0.8, p=0.5),  # Simulate different resolutions
                A.ImageCompression(quality_lower=50, quality_upper=95, p=0.4),  # JPEG artifacts
                A.Blur(blur_limit=5, p=0.3),  # Motion blur
            ], p=0.5),
            
            # Enhanced noise for robustness
            A.OneOf([
                A.GaussNoise(var_limit=(20, 80), p=0.7),
                A.ISONoise(intensity=(0.2, 0.8), p=0.5),
                A.MultiplicativeNoise(multiplier=[0.7, 1.3], p=0.4),
            ], p=0.6),
            
            # Ultra-aggressive dropout
            A.CoarseDropout(max_holes=16, max_height=48, max_width=48, p=0.5),
            A.Cutout(num_holes=12, max_h_size=28, max_w_size=28, p=0.4),
            
            # Extreme crop variation for scale invariance
            A.RandomResizedCrop(self.image_size, self.image_size, scale=(0.5, 1.0), ratio=(0.6, 1.4), p=0.9),
            
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
    
    def _get_dermatology_transforms(self):
        """Dermatology-specific augmentations (histogram equalization, skin tone variation)"""
        return A.Compose([
            A.Resize(self.image_size, self.image_size),
            
            # Dermatology-specific augmentations
            A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.6),  # Histogram equalization
            A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=0.4),
            A.Emboss(alpha=(0.2, 0.5), strength=(0.2, 0.7), p=0.3),
            
            # Simulate different lighting conditions
            A.RandomBrightnessContrast(brightness_limit=0.4, contrast_limit=0.4, p=0.8),
            A.RandomGamma(gamma_limit=(60, 140), p=0.6),
            
            # Simulate different skin tones and camera settings
            A.HueSaturationValue(hue_shift_limit=30, sat_shift_limit=50, val_shift_limit=30, p=0.7),
            A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.4, hue=0.2, p=0.8),
            
            # Medical imaging artifacts
            A.OneOf([
                A.Downscale(scale_min=0.6, scale_max=0.9, p=0.4),  # Different camera qualities
                A.ImageCompression(quality_lower=60, quality_upper=100, p=0.3),  # JPEG artifacts
                A.Blur(blur_limit=3, p=0.2),  # Motion blur from handheld devices
            ], p=0.4),
            
            # Advanced geometric transforms
            A.HorizontalFlip(p=0.6),
            A.VerticalFlip(p=0.4),
            A.Rotate(limit=30, p=0.7),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=20, p=0.6),
            
            # Noise simulation
            A.GaussNoise(var_limit=(10, 50), p=0.4),
            A.CoarseDropout(max_holes=10, max_height=32, max_width=32, p=0.4),
            
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
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        image = np.array(image)
        
        if self.split == 'train':
            # Choose augmentation strategy based on class and random selection
            if label in MINORITY_CLASSES:
                # Special handling for challenging classes (lentigo=7, keratosis=3)
                if label in [3, 7]:  # keratosis, lentigo - most challenging classes
                    rand_val = np.random.random()
                    if rand_val < 0.5:  # 50% ultra-aggressive challenging transforms
                        augmented = self.challenging_transform(image=image)
                    elif rand_val < 0.75:  # 25% minority-specific augmentations
                        augmented = self.minority_transform(image=image)
                    else:  # 25% dermatology-specific augmentations
                        augmented = self.dermatology_transform(image=image)
                else:
                    # For other minority classes: use different augmentation strategies
                    rand_val = np.random.random()
                    if rand_val < 0.4:  # 40% minority-specific augmentations
                        augmented = self.minority_transform(image=image)
                    elif rand_val < 0.7:  # 30% dermatology-specific augmentations
                        augmented = self.dermatology_transform(image=image)
                    else:  # 30% base augmentations
                        augmented = self.base_transform(image=image)
            else:
                # For majority classes: lighter augmentation strategy
                if np.random.random() < self.dermatology_aug_prob:  # 80% dermatology-specific
                    augmented = self.dermatology_transform(image=image)
                else:  # 20% base augmentations
                    augmented = self.base_transform(image=image)
            
            image = augmented['image']
        else:
            # Validation: simple transform
            augmented = self.transform(image=image)
            image = augmented['image']
        
        return image, label

class SwinTransformerClassifier(nn.Module):
    """Swin Transformer model for skin disease classification"""
    
    def __init__(self, num_classes=9, model_name='swin_base_patch4_window7_224', pretrained=True, image_size=224):
        super(SwinTransformerClassifier, self).__init__()
        
        # Load pre-trained Swin Transformer
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0, img_size=image_size)
        
        # Get feature dimension - prefer timm's num_features if available
        if hasattr(self.backbone, 'num_features'):
            feature_dim = self.backbone.num_features
        else:
            # Fallback to forward pass probing in eval mode
            self.backbone.eval()
            with torch.no_grad():
                dummy_input = torch.randn(1, 3, image_size, image_size)
                feature_dim = self.backbone(dummy_input).shape[1]
            self.backbone.train()
        
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

def calculate_enhanced_class_weights(dataset):
    """Calculate enhanced class weights combining frequency and medical importance"""
    class_counts = Counter()
    for _, label in dataset.samples:
        class_counts[label] += 1
    
    total_samples = len(dataset.samples)
    num_classes = len(class_counts)
    
    weights = {}
    # Calculate base inverse frequency weights
    for class_idx, count in class_counts.items():
        base_weight = total_samples / (num_classes * max(count, 1))  # Prevent division by zero
        importance_factor = CLASS_IMPORTANCE.get(class_idx, 1.0)
        # Combine frequency and importance
        weights[class_idx] = base_weight * importance_factor
    
    # Ensure we have weights for all expected classes
    for class_idx in range(len(CLASS_NAMES)):
        if class_idx not in weights:
            weights[class_idx] = CLASS_IMPORTANCE.get(class_idx, 1.0)
    
    # Normalize weights
    total_weight = sum(weights.values())
    weights = {k: v / total_weight * len(weights) for k, v in weights.items()}
    
    return weights, class_counts

def get_enhanced_weighted_sampler(dataset):
    """Get enhanced weighted random sampler with medical importance"""
    class_weights, class_counts = calculate_enhanced_class_weights(dataset)
    
    print(f"?? Class counts: {dict(class_counts)}")
    print(f"?? Enhanced class weights (freq + importance): {class_weights}")
    print(f"?? Medical importance factors: {CLASS_IMPORTANCE}")
    
    # Create sample weights
    sample_weights = []
    for _, label in dataset.samples:
        sample_weights.append(class_weights[label])
    
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
    return sampler, class_counts

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
    """Validate for one epoch with comprehensive metrics"""
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
    
    # Calculate comprehensive metrics
    precision_macro = precision_score(targets, predictions, average='macro', zero_division=0)
    recall_macro = recall_score(targets, predictions, average='macro', zero_division=0)
    f1_macro = f1_score(targets, predictions, average='macro', zero_division=0)
    
    precision_weighted = precision_score(targets, predictions, average='weighted', zero_division=0)
    recall_weighted = recall_score(targets, predictions, average='weighted', zero_division=0)
    f1_weighted = f1_score(targets, predictions, average='weighted', zero_division=0)
    
    return (epoch_loss, epoch_acc, precision_macro, recall_macro, f1_macro, 
            precision_weighted, recall_weighted, f1_weighted, predictions, targets)

def save_confusion_matrix(targets, predictions, class_names, save_path):
    """Save confusion matrix plot to disk (headless operation for HPC)"""
    cm = confusion_matrix(targets, predictions)
    
    # Create confusion matrix with better formatting
    fig, ax = plt.subplots(figsize=(12, 10))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap='Blues', values_format='d', xticks_rotation=45, ax=ax)
    
    plt.title('Confusion Matrix - Best Epoch', fontsize=16)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()  # Important: close figure to free memory on HPC
    print(f"?? Confusion matrix saved to {save_path}")

def plot_training_metrics(metrics_df, save_path):
    """Plot and save training metrics to disk"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Define metrics to plot
    metrics_to_plot = [
        ('train_loss', 'Training Loss'),
        ('val_loss', 'Validation Loss'),
        ('val_acc', 'Validation Accuracy'),
        ('val_f1_macro', 'Validation F1 (Macro)'),
        ('val_precision_macro', 'Validation Precision (Macro)'),
        ('val_recall_macro', 'Validation Recall (Macro)')
    ]
    
    for idx, (metric, title) in enumerate(metrics_to_plot):
        row, col = idx // 3, idx % 3
        ax = axes[row, col]
        
        if metric in metrics_df.columns:
            ax.plot(metrics_df['epoch'], metrics_df[metric], marker='o', linewidth=2)
            ax.set_title(title, fontsize=12)
            ax.set_xlabel('Epoch')
            ax.set_ylabel(metric)
            ax.grid(True, alpha=0.3)
            
            # Highlight best value
            if 'loss' in metric:
                best_idx = metrics_df[metric].idxmin()
                best_val = metrics_df[metric].min()
            else:
                best_idx = metrics_df[metric].idxmax()
                best_val = metrics_df[metric].max()
            
            ax.scatter(metrics_df.iloc[best_idx]['epoch'], best_val, 
                      color='red', s=100, zorder=5, label=f'Best: {best_val:.4f}')
            ax.legend()
    
    plt.suptitle('Training Metrics Over Time', fontsize=16)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"?? Training metrics plot saved to {save_path}")

def save_per_class_metrics(targets, predictions, class_names, save_path):
    """Save detailed per-class metrics"""
    # Calculate per-class metrics
    per_class_precision = precision_score(targets, predictions, average=None, zero_division=0)
    per_class_recall = recall_score(targets, predictions, average=None, zero_division=0)
    per_class_f1 = f1_score(targets, predictions, average=None, zero_division=0)
    
    # Create DataFrame
    metrics_data = {
        'Class': class_names,
        'Precision': per_class_precision,
        'Recall': per_class_recall,
        'F1-Score': per_class_f1
    }
    
    df = pd.DataFrame(metrics_data)
    
    # Save to CSV
    csv_path = save_path.replace('.png', '.csv')
    df.to_csv(csv_path, index=False)
    
    # Create bar plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    x_pos = np.arange(len(class_names))
    
    # Precision
    axes[0].bar(x_pos, per_class_precision, alpha=0.8, color='skyblue')
    axes[0].set_title('Per-Class Precision')
    axes[0].set_xlabel('Class')
    axes[0].set_ylabel('Precision')
    axes[0].set_xticks(x_pos)
    axes[0].set_xticklabels(class_names, rotation=45, ha='right')
    axes[0].grid(True, alpha=0.3)
    
    # Recall
    axes[1].bar(x_pos, per_class_recall, alpha=0.8, color='lightcoral')
    axes[1].set_title('Per-Class Recall')
    axes[1].set_xlabel('Class')
    axes[1].set_ylabel('Recall')
    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels(class_names, rotation=45, ha='right')
    axes[1].grid(True, alpha=0.3)
    
    # F1-Score
    axes[2].bar(x_pos, per_class_f1, alpha=0.8, color='lightgreen')
    axes[2].set_title('Per-Class F1-Score')
    axes[2].set_xlabel('Class')
    axes[2].set_ylabel('F1-Score')
    axes[2].set_xticks(x_pos)
    axes[2].set_xticklabels(class_names, rotation=45, ha='right')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"?? Per-class metrics saved to {save_path} and {csv_path}")
    
    return df

def main(args):
    print("?? Starting Enhanced Swin Transformer Training for Skin Disease Classification")
    print("=" * 80)
    
    # Set seeds and device
    set_seeds(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"?? Using device: {device}")
    if torch.cuda.is_available():
        print(f"?? GPU: {torch.cuda.get_device_name(0)}")
        print(f"?? CUDA version: {torch.version.cuda}")
    
    # Create output directory and subdirectories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'plots'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'checkpoints'), exist_ok=True)
    
    # Create datasets with enhanced augmentation
    print("?? Loading datasets with enhanced augmentation...")
    train_dataset = AlbumentationsDataset(
        args.data_dir, 'train', args.image_size, 
        minority_boost_factor=args.minority_boost_factor,
        dermatology_aug_prob=args.dermatology_aug_prob,
        challenging_boost_factor=args.challenging_class_boost
    )
    val_dataset = AlbumentationsDataset(args.data_dir, 'val', args.image_size)
    
    # Create enhanced data loaders
    train_sampler, class_counts = get_enhanced_weighted_sampler(train_dataset)
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, 
        sampler=train_sampler, num_workers=args.num_workers, 
        pin_memory=torch.cuda.is_available(), 
        persistent_workers=(args.num_workers > 0), drop_last=True  # Drop last incomplete batch
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, 
        shuffle=False, num_workers=args.num_workers, 
        pin_memory=torch.cuda.is_available(), 
        persistent_workers=(args.num_workers > 0)
    )
    
    print(f"?? Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    
    # Create model
    print(f"??? Creating Swin Transformer model: {args.model_name}")
    model = SwinTransformerClassifier(
        num_classes=9, model_name=args.model_name, 
        pretrained=True, image_size=args.image_size
    )
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"?? Total parameters: {total_params:,}")
    print(f"?? Trainable parameters: {trainable_params:,}")
    
    # Enhanced loss function selection - use original counts for CB-Focal/LDAM to prevent double rebalancing
    original_class_counts_list = [train_dataset.original_class_counts.get(i, 0) for i in range(9)]
    print(f"?? Original class counts for loss functions: {dict(train_dataset.original_class_counts)}")
    print(f"?? Oversampled class counts from sampler: {dict(class_counts)}")
    
    if args.loss_function == 'focal':
        class_weights, _ = calculate_enhanced_class_weights(train_dataset)
        weight_tensor = torch.tensor([class_weights[i] for i in range(9)], dtype=torch.float).to(device)
        criterion = FocalLoss(alpha=weight_tensor, gamma=args.focal_gamma)
        print(f"?? Using Focal Loss (gamma={args.focal_gamma})")
    elif args.loss_function == 'cb_focal':
        criterion = ClassBalancedFocalLoss(original_class_counts_list, beta=args.cb_beta, gamma=args.focal_gamma)
        print(f"?? Using Class-Balanced Focal Loss (beta={args.cb_beta}, gamma={args.focal_gamma}) with original counts")
    elif args.loss_function == 'ldam':
        criterion = LDAMLoss(original_class_counts_list, max_m=args.ldam_max_m, s=args.ldam_s)
        print(f"?? Using LDAM Loss (max_m={args.ldam_max_m}, s={args.ldam_s}) with original counts")
    else:  # weighted_ce
        class_weights, _ = calculate_enhanced_class_weights(train_dataset)
        weight_tensor = torch.tensor([class_weights[i] for i in range(9)], dtype=torch.float).to(device)
        criterion = nn.CrossEntropyLoss(weight=weight_tensor)
        print("?? Using Weighted Cross-Entropy Loss")
    
    # Advanced optimizer and scheduler
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=args.learning_rate, 
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8
    )
    
    # Multi-phase scheduler: warmup + cosine annealing with restarts + plateau reduction
    from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, SequentialLR, LinearLR, ReduceLROnPlateau
    
    # Warmup phase (epoch-based stepping)
    warmup_epochs = max(5, args.num_epochs // 20)  # 5% of total epochs for warmup
    warmup_scheduler = LinearLR(optimizer, start_factor=0.1, total_iters=warmup_epochs)
    
    # Main training phase with warm restarts (epoch-based)
    main_epochs = args.num_epochs - warmup_epochs
    cosine_scheduler = CosineAnnealingWarmRestarts(
        optimizer, 
        T_0=max(10, main_epochs // 8),  # Restart every ~12.5% of training
        T_mult=2, 
        eta_min=args.learning_rate * 0.001
    )
    
    scheduler = SequentialLR(
        optimizer, 
        schedulers=[warmup_scheduler, cosine_scheduler], 
        milestones=[warmup_epochs]
    )
    
    # Plateau scheduler for fine-tuning
    plateau_scheduler = ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=8, 
        verbose=True, min_lr=args.learning_rate * 0.0001
    )
    
    print(f"?? Scheduler: Warmup ({warmup_epochs} epochs) + Cosine Annealing with Restarts + Plateau Reduction")
    
    # Training state management
    best_val_f1_macro = 0.0
    best_val_acc = 0.0
    patience_counter = 0
    start_epoch = 0
    
    # Initialize metrics tracking
    metrics_data = []
    
    # Resume from checkpoint if exists
    checkpoint_path = os.path.join(args.output_dir, 'checkpoint.pth')
    if os.path.exists(checkpoint_path):
        print(f"?? Resuming from checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        if 'plateau_scheduler_state_dict' in checkpoint:
            plateau_scheduler.load_state_dict(checkpoint['plateau_scheduler_state_dict'])
            
        start_epoch = checkpoint['epoch']
        best_val_f1_macro = checkpoint.get('best_val_f1_macro', 0.0)
        best_val_acc = checkpoint.get('best_val_acc', 0.0)
        patience_counter = checkpoint.get('patience_counter', 0)
        
        # Load metrics if available
        metrics_path = os.path.join(args.output_dir, 'training_metrics.csv')
        if os.path.exists(metrics_path):
            metrics_df = pd.read_csv(metrics_path)
            metrics_data = metrics_df.to_dict('records')
        
        print(f"?? Resumed from epoch {start_epoch}, best macro F1: {best_val_f1_macro:.4f}")
    
    # Initialize Mixup for data augmentation
    mixup_fn = Mixup(
        mixup_alpha=args.mixup_alpha, cutmix_alpha=args.cutmix_alpha,
        label_smoothing=0.1, num_classes=9
    )
    print(f"?? Using Mixup/CutMix augmentation (a={args.mixup_alpha}, CutMix a={args.cutmix_alpha})")
    
    # Initialize AMP scaler for mixed precision training
    try:
        from torch.amp import autocast, GradScaler  # PyTorch 1.10+
    except ImportError:
        from torch.cuda.amp import autocast, GradScaler  # PyTorch < 1.10
    scaler = GradScaler() if args.use_amp and torch.cuda.is_available() else None
    if scaler:
        print("? Using Automatic Mixed Precision training")
    
    print("?? Starting enhanced training loop...")
    print(f"?? Optimizing for: Macro F1-Score")
    print(f"?? Total epochs: {args.num_epochs}, Patience: {args.patience}")
    
    # Training loop
    for epoch in range(start_epoch, args.num_epochs):
        epoch_start_time = time.time()
        
        print(f"\n?? Epoch {epoch+1}/{args.num_epochs}")
        print("-" * 60)
        
        # Training phase
        model.train()
        running_loss = 0.0
        train_predictions, train_targets = [], []
        
        train_progress = tqdm(train_loader, desc=f"Training [{epoch+1}]")
        for batch_idx, (images, labels) in enumerate(train_progress):
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            
            optimizer.zero_grad(set_to_none=True)
            
            # Apply Mixup/CutMix (only during first 80% of training for better convergence)
            use_mixup = epoch < int(args.num_epochs * 0.8)
            if use_mixup:
                images, targets_mixed = mixup_fn(images, labels)
            
            # Mixed precision training
            if scaler is not None:
                with autocast(enabled=True):
                    outputs = model(images)
                    if use_mixup:
                        loss = soft_target_cross_entropy(outputs, targets_mixed)
                    else:
                        loss = criterion(outputs, labels)
                
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(images)
                if use_mixup:
                    loss = soft_target_cross_entropy(outputs, targets_mixed)
                else:
                    loss = criterion(outputs, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            
            # Note: scheduler stepping moved to end of epoch for proper epoch-based scheduling
            
            running_loss += loss.item()
            
            # Collect predictions for training metrics
            _, preds = torch.max(outputs, 1)
            train_predictions.extend(preds.cpu().numpy())
            train_targets.extend(labels.cpu().numpy())
            
            # Update progress bar
            current_lr = optimizer.param_groups[0]['lr']
            train_progress.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'LR': f'{current_lr:.2e}'
            })
        
        # Calculate training metrics
        train_loss = running_loss / len(train_loader)
        train_acc = accuracy_score(train_targets, train_predictions)
        train_f1_macro = f1_score(train_targets, train_predictions, average='macro', zero_division=0)
        
        # Validation phase
        (val_loss, val_acc, val_precision_macro, val_recall_macro, val_f1_macro, 
         val_precision_weighted, val_recall_weighted, val_f1_weighted, 
         val_predictions, val_targets) = validate_epoch(model, val_loader, criterion, device)
        
        # Step main scheduler once per epoch (epoch-based scheduling)
        scheduler.step()
        
        # Step plateau scheduler based on macro F1
        plateau_scheduler.step(val_f1_macro)
        
        # Calculate epoch time
        epoch_time = time.time() - epoch_start_time
        current_lr = optimizer.param_groups[0]['lr']
        
        # Print comprehensive metrics
        print(f"??  Epoch Time: {epoch_time:.1f}s | LR: {current_lr:.2e}")
        print(f"?? Train - Loss: {train_loss:.4f} | Acc: {train_acc:.4f} | F1(macro): {train_f1_macro:.4f}")
        print(f"?? Val   - Loss: {val_loss:.4f} | Acc: {val_acc:.4f} | F1(macro): {val_f1_macro:.4f}")
        print(f"?? Val   - Prec(m): {val_precision_macro:.4f} | Rec(m): {val_recall_macro:.4f}")
        
        # Per-class validation metrics (brief summary)
        val_f1_per_class = f1_score(val_targets, val_predictions, average=None, zero_division=0)
        print("?? Per-class F1:", end=" ")
        for i, (cls_name, f1_val) in enumerate(zip(CLASS_NAMES, val_f1_per_class)):
            if i < 4:  # Show first 4 classes
                print(f"{cls_name[:8]}:{f1_val:.3f}", end=" ")
        print("...")
        
        # Save metrics
        metrics_row = {
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'train_f1_macro': train_f1_macro,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'val_precision_macro': val_precision_macro,
            'val_recall_macro': val_recall_macro,
            'val_f1_macro': val_f1_macro,
            'val_precision_weighted': val_precision_weighted,
            'val_recall_weighted': val_recall_weighted,
            'val_f1_weighted': val_f1_weighted,
            'learning_rate': current_lr,
            'epoch_time': epoch_time
        }
        metrics_data.append(metrics_row)
        
        # Save metrics to CSV
        metrics_df = pd.DataFrame(metrics_data)
        metrics_csv_path = os.path.join(args.output_dir, 'training_metrics.csv')
        metrics_df.to_csv(metrics_csv_path, index=False)
        
        # Model checkpoint saving - optimize for macro F1
        is_best_f1 = val_f1_macro > best_val_f1_macro
        is_best_acc = val_acc > best_val_acc
        
        if is_best_f1:
            best_val_f1_macro = val_f1_macro
            patience_counter = 0
            
            # Save best F1 model
            best_f1_checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'plateau_scheduler_state_dict': plateau_scheduler.state_dict(),
                'best_val_f1_macro': best_val_f1_macro,
                'best_val_acc': best_val_acc,
                'patience_counter': patience_counter,
                'metrics_data': metrics_data,
                'config': vars(args)
            }
            
            torch.save(best_f1_checkpoint, os.path.join(args.output_dir, 'best_f1_model.pth'))
            torch.save(best_f1_checkpoint, checkpoint_path)  # For resuming
            
            # Save visualizations for best F1 model
            save_confusion_matrix(
                val_targets, val_predictions, CLASS_NAMES,
                os.path.join(args.output_dir, 'plots', 'confusion_matrix_best_f1.png')
            )
            
            # Save per-class metrics
            save_per_class_metrics(
                val_targets, val_predictions, CLASS_NAMES,
                os.path.join(args.output_dir, 'plots', 'per_class_metrics_best_f1.png')
            )
            
            # Save predictions for analysis
            with open(os.path.join(args.output_dir, 'best_f1_predictions.pkl'), 'wb') as f:
                pickle.dump({
                    'targets': val_targets,
                    'predictions': val_predictions,
                    'epoch': epoch + 1,
                    'f1_macro': val_f1_macro
                }, f)
            
            print(f"?? New best macro F1: {best_val_f1_macro:.4f} (saved)")
        else:
            patience_counter += 1
        
        # Also save best accuracy model separately
        if is_best_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'best_val_acc': best_val_acc,
                'val_f1_macro': val_f1_macro,
                'config': vars(args)
            }, os.path.join(args.output_dir, 'best_acc_model.pth'))
            print(f"?? New best accuracy: {best_val_acc:.4f}")
        
        # Early stopping based on macro F1
        if patience_counter >= args.patience:
            print(f"?? Early stopping triggered after {epoch + 1} epochs")
            print(f"   No improvement in macro F1 for {args.patience} epochs")
            break
        
        # Save training plots every 10 epochs or at the end
        if (epoch + 1) % 10 == 0 or epoch == args.num_epochs - 1:
            plot_training_metrics(
                metrics_df, 
                os.path.join(args.output_dir, 'plots', 'training_metrics.png')
            )
    
    # Final results
    print("\n" + "=" * 80)
    print("?? Training completed!")
    print(f"?? Best Macro F1: {best_val_f1_macro:.4f}")
    print(f"?? Best Accuracy: {best_val_acc:.4f}")
    print(f"??  Total epochs trained: {epoch + 1}")
    
    # Generate final plots and analysis
    print("\n?? Generating final analysis...")
    
    # Load best F1 model for final evaluation
    best_model_path = os.path.join(args.output_dir, 'best_f1_model.pth')
    if os.path.exists(best_model_path):
        checkpoint = torch.load(best_model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Final validation run
        final_results = validate_epoch(model, val_loader, criterion, device)
        (final_val_loss, final_val_acc, final_precision_macro, final_recall_macro, 
         final_f1_macro, final_precision_weighted, final_recall_weighted, 
         final_f1_weighted, final_predictions, final_targets) = final_results
        
        # Generate comprehensive final plots
        save_confusion_matrix(
            final_targets, final_predictions, CLASS_NAMES,
            os.path.join(args.output_dir, 'plots', 'final_confusion_matrix.png')
        )
        
        final_per_class_df = save_per_class_metrics(
            final_targets, final_predictions, CLASS_NAMES,
            os.path.join(args.output_dir, 'plots', 'final_per_class_metrics.png')
        )
        
        plot_training_metrics(
            metrics_df, 
            os.path.join(args.output_dir, 'plots', 'final_training_metrics.png')
        )
        
        # Save final summary
        final_summary = {
            'best_epoch': checkpoint['epoch'],
            'final_val_f1_macro': final_f1_macro,
            'final_val_acc': final_val_acc,
            'final_val_precision_macro': final_precision_macro,
            'final_val_recall_macro': final_recall_macro,
            'per_class_metrics': final_per_class_df.to_dict('records'),
            'training_config': vars(args)
        }
        
        with open(os.path.join(args.output_dir, 'final_summary.json'), 'w') as f:
            json.dump(final_summary, f, indent=2, default=str)
        
        print(f"?? Final Macro F1: {final_f1_macro:.4f}")
        print(f"?? Final Accuracy: {final_val_acc:.4f}")
        print(f"?? All results saved to: {args.output_dir}")
    
    print("? Training pipeline completed successfully!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Enhanced Swin Transformer for Skin Disease Classification with Severe Class Imbalance Handling")
    
    # Data arguments
    parser.add_argument("--data_dir", type=str, required=True, help="Path to dataset directory")
    parser.add_argument("--output_dir", type=str, default="./outputs", help="Output directory for models and plots")
    parser.add_argument("--image_size", type=int, default=224, help="Input image size")
    
    # Model arguments
    parser.add_argument("--model_name", type=str, default="swin_base_patch4_window7_224", 
                       help="Swin Transformer model name")
    
    # Training arguments (extended for 200 epochs with 20 patience)
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--num_epochs", type=int, default=200, help="Number of epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--patience", type=int, default=35, help="Early stopping patience (increased for better convergence)")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of data loader workers")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    # Enhanced augmentation arguments
    parser.add_argument("--minority_boost_factor", type=float, default=3.0,
                       help="Factor to oversample minority classes")
    parser.add_argument("--dermatology_aug_prob", type=float, default=0.8,
                       help="Probability of applying dermatology-specific augmentations")
    
    # Advanced loss function arguments
    parser.add_argument("--loss_function", type=str, default="cb_focal", 
                       choices=["weighted_ce", "focal", "cb_focal", "ldam"],
                       help="Loss function: weighted_ce, focal, cb_focal (Class-Balanced Focal), ldam")
    parser.add_argument("--focal_gamma", type=float, default=2.0,
                       help="Gamma parameter for Focal Loss")
    parser.add_argument("--cb_beta", type=float, default=0.9999,
                       help="Beta parameter for Class-Balanced Focal Loss")
    parser.add_argument("--ldam_max_m", type=float, default=0.5,
                       help="Maximum margin for LDAM Loss")
    parser.add_argument("--ldam_s", type=float, default=30,
                       help="Scale parameter for LDAM Loss")
    
    # Training enhancements
    amp_group = parser.add_mutually_exclusive_group()
    amp_group.add_argument("--use_amp", dest="use_amp", action="store_true",
                          help="Enable Automatic Mixed Precision training")
    amp_group.add_argument("--no_amp", dest="use_amp", action="store_false",
                          help="Disable Automatic Mixed Precision training")
    parser.set_defaults(use_amp=True)
    parser.add_argument("--mixup_alpha", type=float, default=0.3,
                       help="Mixup alpha parameter")
    parser.add_argument("--cutmix_alpha", type=float, default=1.0,
                       help="CutMix alpha parameter")
    parser.add_argument("--challenging_class_boost", type=float, default=4.0,
                       help="Additional boost factor for challenging classes (lentigo, keratosis)")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("?? Enhanced Configuration for Challenging Classes:")
    print(f"   ?? Loss Function: {args.loss_function}")
    print(f"   ?? Optimizing for: Macro F1-Score")
    print(f"   ?? Epochs: {args.num_epochs}, Patience: {args.patience} (increased for convergence)")
    print(f"   ?? Minority Boost: {args.minority_boost_factor}x")
    print(f"   ?? Challenging Class Boost: {args.challenging_class_boost}x (keratosis & lentigo)")
    print(f"   ?? Dermatology Aug Prob: {args.dermatology_aug_prob}")
    print(f"   ?? Mixup/CutMix: a={args.mixup_alpha}, CutMix a={args.cutmix_alpha}")
    print(f"   ? Mixed Precision: {args.use_amp}")
    print(f"   ?? Special handling for lentigo (class 7) and keratosis (class 3)")
    print(f"   ?? Enhanced class weights: lentigo={CLASS_IMPORTANCE[7]}, keratosis={CLASS_IMPORTANCE[3]}")
    
    main(args)