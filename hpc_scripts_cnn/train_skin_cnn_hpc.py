#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multiclass Dual-Backbone Training Script for HPC
ConvNeXt-Tiny + DeiT-III Tiny -> concatenated features -> MLP -> 8/9 classes
Designed for HPC (Killarney) with headless operation and proper argument parsing
"""

import os
# Ensure headless operation for HPC
os.environ['MPLBACKEND'] = 'Agg'
if 'DISPLAY' not in os.environ:
    os.environ['DISPLAY'] = ''

import pickle
import time
import math
import json
import pandas as pd
import numpy as np
from PIL import Image
import argparse
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler, Subset

from torchvision import transforms

import timm
from timm.data import Mixup
from timm.utils import ModelEmaV2  # optional, not required
from tqdm import tqdm

# Advanced augmentations
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Set non-interactive backend for HPC
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, ConfusionMatrixDisplay, roc_curve
)

# Set seeds for reproducibility
def set_seeds(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ==================== Labels Configuration ====================
# If you truly want to exclude 'nevus', add it here BEFORE caching.
IGNORE_CLASSES = {}  # e.g., {'nevus'}

ALL_CLASSES = [
    'nevus', 'melanoma', 'bcc', 'keratosis',
    'actinic_keratosis', 'scc', 'dermatofibroma',
    'lentigo', 'vascular_lesion'
]

KEPT_CLASSES = [c for c in ALL_CLASSES if c not in IGNORE_CLASSES]
label_mapping = {c: i for i, c in enumerate(KEPT_CLASSES)}
idx_to_class = {i: c for c, i in label_mapping.items()}
num_classes = len(KEPT_CLASSES)

# Minority classes that need extra augmentation (critical for medical diagnosis)
MINORITY_CLASSES = {
    label_mapping['melanoma'],           # Most critical - malignant
    label_mapping['actinic_keratosis'],  # Pre-cancerous
    label_mapping['scc'],                # Squamous cell carcinoma
    label_mapping['dermatofibroma'],     # Rare benign
    label_mapping['lentigo'],            # Solar lentigo
    label_mapping['vascular_lesion']     # Vascular lesions
}

# Class importance weights (higher = more important for medical diagnosis)
CLASS_IMPORTANCE = {
    label_mapping['melanoma']: 3.0,           # Highest - life threatening
    label_mapping['scc']: 2.5,                # High - malignant
    label_mapping['bcc']: 2.0,                # Moderate - malignant but less aggressive
    label_mapping['actinic_keratosis']: 2.5,  # High - pre-cancerous
    label_mapping['nevus']: 1.0,              # Base - most common
    label_mapping['keratosis']: 1.2,          # Slightly higher - need differentiation
    label_mapping['dermatofibroma']: 1.8,     # Moderate - differential diagnosis
    label_mapping['lentigo']: 1.5,            # Moderate - age-related
    label_mapping['vascular_lesion']: 1.8     # Moderate - specific diagnosis
}

# ==================== Advanced Transforms with Albumentations ====================
# Cache at 320 -> faster IO + consistent high-res input (keep simple for caching)
cache_transform = transforms.Compose([
    transforms.Resize((320, 320)),
    transforms.ToTensor()
])

# Standard normalization for ImageNet pretrained models
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def get_base_augmentations(image_size=224, intensity=1.0):
    """Base augmentations for all classes"""
    return A.Compose([
        # Geometric transforms
        A.Resize(image_size, image_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.RandomRotate90(p=0.5),
        A.Rotate(limit=int(15 * intensity), p=0.7),
        A.ShiftScaleRotate(
            shift_limit=0.05 * intensity,
            scale_limit=0.1 * intensity,
            rotate_limit=10 * intensity,
            p=0.6
        ),
        
        # Distortions for robustness
        A.ElasticTransform(
            alpha=1 * intensity,
            sigma=50,
            alpha_affine=30 * intensity,
            p=0.3
        ),
        A.GridDistortion(p=0.25),
        A.OpticalDistortion(p=0.25),
        
        # Color augmentations
        A.RandomBrightnessContrast(
            brightness_limit=0.2 * intensity,
            contrast_limit=0.2 * intensity,
            p=0.7
        ),
        A.ColorJitter(
            brightness=0.2 * intensity,
            contrast=0.2 * intensity,
            saturation=0.2 * intensity,
            hue=0.1 * intensity,
            p=0.6
        ),
        A.HueSaturationValue(
            hue_shift_limit=int(20 * intensity),
            sat_shift_limit=int(30 * intensity),
            val_shift_limit=int(20 * intensity),
            p=0.5
        ),
        
        # Noise and blur
        A.OneOf([
            A.GaussNoise(var_limit=(10, 50 * intensity), p=0.5),
            A.ISONoise(intensity=(0.1, 0.5 * intensity), p=0.3),
            A.MultiplicativeNoise(multiplier=[0.9, 1.1], p=0.2),
        ], p=0.4),
        
        A.OneOf([
            A.GaussianBlur(blur_limit=3, p=0.3),
            A.MotionBlur(blur_limit=3, p=0.2),
            A.MedianBlur(blur_limit=3, p=0.1),
        ], p=0.3),
        
        # Dropout augmentations
        A.CoarseDropout(
            max_holes=8,
            max_height=int(32 * intensity),
            max_width=int(32 * intensity),
            p=0.3
        ),
        A.Cutout(
            num_holes=8,
            max_h_size=int(16 * intensity),
            max_w_size=int(16 * intensity),
            p=0.2
        ),
        
        # Final normalization
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2()
    ])

def get_minority_augmentations(image_size=224):
    """Intensive augmentations for minority classes"""
    return A.Compose([
        # Start with base augmentations at higher intensity
        A.Resize(image_size, image_size),
        
        # More aggressive geometric transforms
        A.HorizontalFlip(p=0.6),
        A.VerticalFlip(p=0.4),
        A.RandomRotate90(p=0.6),
        A.Rotate(limit=25, p=0.8),
        A.ShiftScaleRotate(
            shift_limit=0.08,
            scale_limit=0.15,
            rotate_limit=15,
            p=0.7
        ),
        
        # Advanced distortions
        A.ElasticTransform(
            alpha=2,
            sigma=50,
            alpha_affine=50,
            p=0.4
        ),
        A.GridDistortion(p=0.35),
        A.OpticalDistortion(p=0.35),
        A.Perspective(scale=(0.05, 0.1), p=0.3),
        
        # Intensive color augmentations
        A.RandomBrightnessContrast(
            brightness_limit=0.3,
            contrast_limit=0.3,
            p=0.8
        ),
        A.ColorJitter(
            brightness=0.3,
            contrast=0.3,
            saturation=0.3,
            hue=0.15,
            p=0.7
        ),
        A.HueSaturationValue(
            hue_shift_limit=25,
            sat_shift_limit=40,
            val_shift_limit=25,
            p=0.6
        ),
        A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.5),
        A.RandomGamma(gamma_limit=(80, 120), p=0.5),
        A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=0.3),
        A.Emboss(alpha=(0.2, 0.5), strength=(0.2, 0.7), p=0.3),
        
        # Advanced noise and artifacts
        A.OneOf([
            A.GaussNoise(var_limit=(10, 80), p=0.6),
            A.ISONoise(intensity=(0.1, 0.7), p=0.4),
            A.MultiplicativeNoise(multiplier=[0.8, 1.2], p=0.3),
        ], p=0.5),
        
        A.OneOf([
            A.GaussianBlur(blur_limit=5, p=0.4),
            A.MotionBlur(blur_limit=5, p=0.3),
            A.MedianBlur(blur_limit=5, p=0.2),
            A.Defocus(radius=(1, 3), alias_blur=(0.1, 0.5), p=0.1),
        ], p=0.4),
        
        # Medical-specific augmentations
        A.OneOf([
            A.Downscale(scale_min=0.7, scale_max=0.9, p=0.3),  # Simulate different camera qualities
            A.ImageCompression(quality_lower=70, quality_upper=100, p=0.2),  # JPEG artifacts
        ], p=0.3),
        
        # More aggressive dropout
        A.CoarseDropout(
            max_holes=12,
            max_height=40,
            max_width=40,
            p=0.4
        ),
        A.Cutout(
            num_holes=12,
            max_h_size=20,
            max_w_size=20,
            p=0.3
        ),
        
        # Final normalization
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2()
    ])

def get_validation_transform(image_size=224):
    """Simple validation transform"""
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2()
    ])

# Fallback torch transforms for compatibility
normalize_transform = transforms.Normalize(
    mean=IMAGENET_MEAN,
    std=IMAGENET_STD
)

val_transform = transforms.Compose([
    transforms.ToTensor(),
    normalize_transform
])

# ==================== Cache Builder ====================
def ensure_dir(path): 
    os.makedirs(path, exist_ok=True)

def cache_split_if_missing(original_split_dir, cache_dir, split):
    src_path = os.path.join(original_split_dir, split)
    tgt_path = os.path.join(cache_dir, split)
    if os.path.exists(tgt_path):
        print(f"✅ Cache already exists for split '{split}' → skipping.")
        return

    print(f"⏳ Caching '{split}' split...")
    for class_name in os.listdir(src_path):
        class_src = os.path.join(src_path, class_name)
        class_tgt = os.path.join(tgt_path, class_name)
        if not os.path.isdir(class_src):
            continue

        # skip ignored classes entirely
        if class_name in IGNORE_CLASSES:
            print(f"🚫 Skipping ignored class '{class_name}' during cache.")
            continue

        ensure_dir(class_tgt)
        image_files = [f for f in os.listdir(class_src)
                       if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

        for img_name in tqdm(image_files, desc=f"Caching {split}/{class_name}", unit='img'):
            img_path = os.path.join(class_src, img_name)
            img = Image.open(img_path).convert('RGB')
            tensor = cache_transform(img)
            base = os.path.splitext(img_name)[0]
            torch.save(tensor, os.path.join(class_tgt, f'{base}.pt'))

# ==================== Enhanced Dataset with Targeted Augmentation ====================
class CachedMultiDataset(Dataset):
    def __init__(self, cache_dir, augment=False, image_size=224, minority_boost_factor=2.0):
        self.data = []
        self.augment = augment
        self.image_size = image_size
        self.minority_boost_factor = minority_boost_factor
        
        print(f"📦 Loading dataset from {cache_dir}...")
        class_names = [d for d in os.listdir(cache_dir) if os.path.isdir(os.path.join(cache_dir, d))]

        # Build index and boost minority classes
        class_samples = {i: [] for i in range(num_classes)}
        
        for class_name in tqdm(class_names, desc="Indexing classes"):
            if class_name in IGNORE_CLASSES:
                print(f"🚫 Ignoring class folder: {class_name}")
                continue
            if class_name not in label_mapping:
                print(f"⚠️ Skipping unknown class folder: {class_name}")
                continue
            label = label_mapping[class_name]
            class_dir = os.path.join(cache_dir, class_name)
            for file in os.listdir(class_dir):
                if file.endswith('.pt'):
                    class_samples[label].append((os.path.join(class_dir, file), label))

        # Add samples with minority class boosting during training
        for label, samples in class_samples.items():
            self.data.extend(samples)
            
            # Boost minority classes by repeating samples (only during training)
            if self.augment and label in MINORITY_CLASSES:
                boost_count = int(len(samples) * (self.minority_boost_factor - 1))
                boost_samples = np.random.choice(len(samples), boost_count, replace=True)
                for idx in boost_samples:
                    self.data.append(samples[idx])
                print(f"🔄 Boosted minority class '{idx_to_class[label]}' by {boost_count} samples")

        print(f"✅ Indexed {len(self.data)} total samples (classes kept: {KEPT_CLASSES}).")
        if self.augment:
            print(f"🎯 Minority classes boosted by factor {self.minority_boost_factor}")
            
        # Initialize transforms
        if self.augment:
            self.base_transform = get_base_augmentations(image_size)
            self.minority_transform = get_minority_augmentations(image_size)
        else:
            self.val_transform = get_validation_transform(image_size)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        path, label = self.data[idx]
        
        # Load cached tensor and convert to numpy for albumentations
        image_tensor = torch.load(path)  # [3,320,320]
        # Convert tensor to numpy array for albumentations [H,W,C] format
        image_np = image_tensor.permute(1, 2, 0).numpy()  # [320,320,3]
        # Denormalize from [0,1] to [0,255] for albumentations
        image_np = (image_np * 255).astype(np.uint8)
        
        if self.augment:
            # Use intensive augmentation for minority classes
            if label in MINORITY_CLASSES:
                # Random chance to use minority-specific augmentations
                if np.random.random() < 0.7:  # 70% chance for intensive augmentation
                    transformed = self.minority_transform(image=image_np)
                else:
                    transformed = self.base_transform(image=image_np)
            else:
                transformed = self.base_transform(image=image_np)
            image = transformed['image']
        else:
            transformed = self.val_transform(image=image_np)
            image = transformed['image']
        
        return image, label

# ==================== Enhanced Data Loaders ====================
def get_multi_loaders(train_dir, val_dir, batch_size=24, num_workers=4, image_size=224, minority_boost_factor=2.0):
    train_dataset = CachedMultiDataset(train_dir, augment=True, image_size=image_size, minority_boost_factor=minority_boost_factor)
    val_dataset = CachedMultiDataset(val_dir, augment=False, image_size=image_size)

    # labels from dataset index (fast)
    train_labels = [lbl for _, lbl in train_dataset.data]
    if len(train_labels) == 0:
        raise RuntimeError("No training samples found after filtering. Check your paths and IGNORE_CLASSES.")

    class_counts = np.bincount(train_labels, minlength=num_classes)
    print(f"📊 Train class counts (after filtering): {dict(zip(range(num_classes), class_counts.tolist()))}")

    # Enhanced weighted sampling with class importance
    # Combine inverse frequency with medical importance
    base_weights = 1.0 / torch.tensor(class_counts, dtype=torch.float32).clamp(min=1)
    importance_weights = torch.tensor([CLASS_IMPORTANCE.get(i, 1.0) for i in range(num_classes)], dtype=torch.float32)
    
    # Combine both weights: base_weight * importance_weight
    combined_weights = base_weights * importance_weights
    
    # Normalize weights
    combined_weights = combined_weights / combined_weights.sum() * num_classes
    
    print(f"🎯 Class sampling weights: {dict(zip(range(num_classes), combined_weights.tolist()))}")
    print(f"📋 Class importance factors: {dict(zip(range(num_classes), importance_weights.tolist()))}")
    
    sample_weights = torch.tensor([combined_weights[l] for l in train_labels], dtype=torch.float32)
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    # IMPORTANT: sampler=True -> shuffle must be False
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, sampler=sampler,
        shuffle=False, num_workers=num_workers, pin_memory=torch.cuda.is_available()
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=torch.cuda.is_available()
    )
    return train_loader, val_loader, class_counts

# ==================== Model ====================
class DualBackbone(nn.Module):
    """
    CNN branch: ConvNeXt-Tiny (features_only=True) + GeM pooling
    ViT branch: DeiT-III Tiny (classifier reset)
    """
    def __init__(self, deit_variant='deit3_small_patch16_224', freeze_all=True, img_size=320):
        super().__init__()
        # ---- ConvNeXt-Tiny ----
        self.cnn = timm.create_model('convnext_tiny.fb_in22k', pretrained=True, features_only=True)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))  # simple GAP; GeM optional
        cnn_ch = self.cnn.feature_info[-1]['num_chs']  # usually 768 for convnext_tiny
        self.cnn_feat_dim = cnn_ch

        # ---- DeiT-III Tiny ----
        self.deit = timm.create_model(deit_variant, pretrained=True, img_size=img_size)
        self.deit.reset_classifier(0)
        self.deit_feat_dim = self.deit.num_features  # ~192 for deit3_tiny

        if freeze_all:
            for p in self.cnn.parameters(): p.requires_grad = False
            for p in self.deit.parameters(): p.requires_grad = False

    def forward(self, x):
        # ConvNeXt branch
        feats = self.cnn(x)[-1]        # [B, C, H, W]
        r = self.pool(feats).flatten(1)  # [B, C]

        # DeiT branch
        d = self.deit.forward_features(x)  # [B, tokens, C] or [B, C]
        if d.ndim == 3:
            d = d[:, 0, :]  # CLS token

        return torch.cat([r, d], dim=1)  # [B, cnn + deit]

class MultiClassifier(nn.Module):
    def __init__(self, in_features, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 768), nn.BatchNorm1d(768), nn.GELU(), nn.Dropout(0.4),
            nn.Linear(768, 256), nn.BatchNorm1d(256), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.GELU(),
            nn.Linear(128, num_classes)
        )
    def forward(self, x):
        return self.net(x)

# ==================== Plotting Helpers (Headless) ====================
def plot_confusion_matrix_multiclass(true_labels, pred_labels, class_names, title="Confusion Matrix", save_path=None):
    cm = confusion_matrix(true_labels, pred_labels, labels=list(range(len(class_names))))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    fig, ax = plt.subplots(figsize=(12, 10))
    disp.plot(cmap='Blues', values_format='d', xticks_rotation=45, ax=ax)
    plt.title(title)
    plt.grid(False)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"💾 Confusion matrix saved to {save_path}")
    plt.close()  # Important: close figure to free memory

def plot_training_metrics(csv_path='multi_metrics.csv', save_path=None):
    if not os.path.exists(csv_path):
        print("ERROR: CSV file not found.")
        return
    df = pd.read_csv(csv_path)
    if df.empty:
        print("WARNING: CSV is empty - nothing to plot.")
        return

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    metrics = [
        ('train_loss', 'Training Loss'),
        ('val_loss', 'Validation Loss'),
        ('val_acc', 'Validation Accuracy'),
        ('val_precision_macro', 'Validation Precision (macro)'),
        ('val_recall_macro', 'Validation Recall (macro)'),
        ('val_f1_macro', 'Validation F1 (macro)')
    ]
    # Handle case where we have fewer metrics than subplots
    for i, (ax, (col, title)) in enumerate(zip(axes.flat, metrics)):
        if col in df.columns:
            sns.lineplot(data=df, x='epoch', y=col, ax=ax, marker='o')
            ax.set_title(title)
            ax.set_xlabel('Epoch')
            ax.set_ylabel(col)
            ax.grid(True)
    
    # Hide unused subplots
    for i in range(len(metrics), len(axes.flat)):
        axes.flat[i].set_visible(False)
    
    plt.suptitle("Training Metrics Over Time", fontsize=16)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"💾 Training metrics plot saved to {save_path}")
    plt.close()

# ==================== Training Utils ====================
try:
    from torch.amp import autocast, GradScaler  # PyTorch 1.10+
except ImportError:
    from torch.cuda.amp import autocast, GradScaler  # PyTorch < 1.10

def unfreeze_and_attach(backbone, optimizer, lr_backbone=1e-5, weight_decay=1e-4):
    """
    Unfreeze both backbones and add them to the optimizer with a gentle LR.
    Safe to call multiple times: it won't duplicate params in the optimizer.
    """
    for p in backbone.cnn.parameters():  p.requires_grad = True
    for p in backbone.deit.parameters(): p.requires_grad = True

    existing = set()
    for g in optimizer.param_groups:
        for p in g['params']:
            existing.add(id(p))

    cnn_params  = [p for p in backbone.cnn.parameters()  if p.requires_grad and id(p) not in existing]
    deit_params = [p for p in backbone.deit.parameters() if p.requires_grad and id(p) not in existing]

    if cnn_params:
        optimizer.add_param_group({'params': cnn_params, 'lr': lr_backbone, 'weight_decay': weight_decay})
    if deit_params:
        optimizer.add_param_group({'params': deit_params, 'lr': lr_backbone, 'weight_decay': weight_decay})

# Soft-target CE for Mixup/CutMix
def soft_target_cross_entropy(logits, target_soft):
    logprob = F.log_softmax(logits, dim=-1)
    return -(target_soft * logprob).sum(dim=-1).mean()

# -------- top-K checkpoint saver (by macro-F1, tiebreak macro-recall) --------
import heapq

class TopKSaver:
    def __init__(self, k=3, folder="checkpoints"):
        self.k = k
        self.heap = []  # min-heap of (f1, recall, path)
        os.makedirs(folder, exist_ok=True)
        self.folder = folder

    def save(self, f1_macro, recall_macro, epoch, state_dict, prefix="multi"):
        path = os.path.join(self.folder, f"{prefix}_ep{epoch:03d}_f1{f1_macro:.4f}_rec{recall_macro:.4f}.pt")
        torch.save(state_dict, path)
        heapq.heappush(self.heap, (f1_macro, recall_macro, path))
        while len(self.heap) > self.k:
            _f1, _rec, old = heapq.heappop(self.heap)
            try: os.remove(old)
            except FileNotFoundError: pass

# -------- simple moving-average early stopping on macro-F1 --------
from collections import deque

class MAEarlyStop:
    def __init__(self, window=3, patience=6, eps=1e-5):
        self.q = deque(maxlen=window)
        self.best_ma = -1.0
        self.since = 0
        self.patience = patience
        self.eps = eps

    def step(self, value):
        self.q.append(value)
        if len(self.q) < self.q.maxlen:
            return False  # not enough history yet
        ma = sum(self.q) / len(self.q)
        if ma > self.best_ma + self.eps:
            self.best_ma = ma
            self.since = 0
        else:
            self.since += 1
        return self.since >= self.patience

# ==================== Train ====================
def train_multiclass_model(
    backbone, classifier, train_loader, val_loader, class_counts,
    epochs=100, patience=15, lr=2e-4,
    csv_path='multi_metrics.csv', model_path='multi_best_checkpoint.pt',
    use_amp=True, warmup_epochs=3, lr_backbone=5e-6,
    mixup_alpha=0.4, cutmix_alpha=1.0, mixup_off_pct=0.15,
    mixup_burst_len_after_resume=16,
    finetune_lr_drop=0.25,
    topk_checkpoints=3,
    output_dir='./',
    device=None
):
    """
    WeightedRandomSampler + CrossEntropy (no class weights).
    Mixup/CutMix ON for first (1 - mixup_off_pct) of epochs, OFF for last mixup_off_pct.
    Scheduler: Linear warmup + Cosine.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    use_cuda = device.type == 'cuda'
    backbone = backbone.to(device)
    classifier = classifier.to(device)

    # optimizer: AdamW (fast + reliable)
    params = list(filter(lambda p: p.requires_grad, backbone.parameters())) + list(classifier.parameters())
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=0.05)

    # Enhanced scheduler: warmup + cosine annealing with restarts
    from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, SequentialLR, LinearLR, ReduceLROnPlateau

    # Phase 1: Linear warmup
    warmup_scheduler = LinearLR(optimizer, start_factor=0.1, total_iters=max(1, warmup_epochs))
    
    # Phase 2: Cosine annealing with warm restarts for better convergence
    # This helps escape local minima and find better solutions
    cosine_scheduler = CosineAnnealingWarmRestarts(
        optimizer, 
        T_0=max(10, (epochs - warmup_epochs) // 4),  # Initial restart period
        T_mult=2,  # Multiply restart period by 2 each time
        eta_min=lr * 0.001,  # Minimum learning rate
        last_epoch=-1
    )
    
    scheduler = SequentialLR(
        optimizer, 
        schedulers=[warmup_scheduler, cosine_scheduler], 
        milestones=[warmup_epochs]
    )
    
    # Backup plateau scheduler for fine-tuning phase
    plateau_scheduler = ReduceLROnPlateau(
        optimizer, 
        mode='max', 
        factor=0.5, 
        patience=5, 
        verbose=True, 
        min_lr=lr * 0.0001
    )

    scaler = GradScaler(enabled=use_amp and use_cuda)

    # Mixup/CutMix
    mixup_fn = Mixup(
        mixup_alpha=mixup_alpha, cutmix_alpha=cutmix_alpha,
        label_smoothing=0.0, num_classes=num_classes
    )
    
    # Small utilities: top-K saver and MA early stop
    checkpoint_dir = os.path.join(output_dir, "checkpoints")
    topk_saver = TopKSaver(k=topk_checkpoints, folder=checkpoint_dir)
    ma_stop = MAEarlyStop(window=3, patience=6)

    # CSV setup
    csv_full_path = os.path.join(output_dir, csv_path)
    if os.path.exists(csv_full_path):
        try:
            df = pd.read_csv(csv_full_path)
        except Exception:
            df = pd.DataFrame(columns=[
                'epoch', 'train_loss', 'val_loss', 'val_acc',
                'val_precision_macro', 'val_recall_macro', 'val_f1_macro', 'val_auc_ovr', 'lr'
            ])
    else:
        df = pd.DataFrame(columns=[
            'epoch', 'train_loss', 'val_loss', 'val_acc',
            'val_precision_macro', 'val_recall_macro', 'val_f1_macro', 'val_auc_ovr', 'lr'
        ])

    # Resume logic
    start_epoch = 0
    best_val_f1 = -float('inf')
    unfrozen = False
    model_full_path = os.path.join(output_dir, model_path)
    if os.path.exists(model_full_path):
        print(f"🔁 Resuming from checkpoint: {model_full_path}")
        ckpt = torch.load(model_full_path, map_location=device)

        if isinstance(ckpt, dict) and ('classifier_state' in ckpt or 'backbone_state' in ckpt):
            if 'backbone_state' in ckpt:
                backbone.load_state_dict(ckpt['backbone_state'], strict=False)
            if 'classifier_state' in ckpt:
                classifier.load_state_dict(ckpt['classifier_state'], strict=False)
            try:
                optimizer.load_state_dict(ckpt['optimizer_state'])
            except Exception as e:
                print(f"⚠️ Could not load optimizer state: {e}")
            try:
                if 'scheduler_state' in ckpt:
                    scheduler.load_state_dict(ckpt['scheduler_state'])
                if 'plateau_scheduler_state' in ckpt:
                    plateau_scheduler.load_state_dict(ckpt['plateau_scheduler_state'])
            except Exception as e:
                print(f"⚠️ Could not load scheduler state: {e}")

            best_val_f1 = float(ckpt.get('best_val_f1', best_val_f1))
            start_epoch = int(ckpt.get('epoch', -1)) + 1
            unfrozen = bool(ckpt.get('unfrozen', unfrozen))
        else:
            print("ℹ️ Legacy checkpoint detected. Loading classifier only.")
            try:
                classifier.load_state_dict(ckpt, strict=False)
            except Exception:
                pass

        print(f"📈 Resume epoch {start_epoch}, best val F1 {best_val_f1:.4f}, unfrozen={unfrozen}")

    # MixUp/CutMix schedule around the RESUME
    burst_start = start_epoch
    burst_end   = start_epoch + mixup_burst_len_after_resume
    finetune_start = burst_end
    did_lr_drop = False

    if (start_epoch >= warmup_epochs or unfrozen):
        print("🔓 Ensuring backbones are attached to optimizer after resume.")
        unfreeze_and_attach(backbone, optimizer, lr_backbone=lr_backbone)
        unfrozen = True

    patience_counter = 0
    last_val_labels, last_val_preds = [], []

    # ===== EPOCH LOOP =====
    for epoch in range(start_epoch, epochs):
        # Timed unfreeze
        if (not unfrozen) and (epoch >= warmup_epochs):
            print(f"🔓 Unfreezing backbones at epoch {epoch} (after {warmup_epochs} warmup epochs).")
            unfreeze_and_attach(backbone, optimizer, lr_backbone=lr_backbone)
            unfrozen = True

        # MixUp/CutMix ON only during the post-resume "burst"
        mixup_active = (burst_start <= epoch < burst_end)

        # When we cross into fine-tune phase: turn MixUp OFF and drop LR once
        if (not did_lr_drop) and (epoch >= finetune_start):
            for i, pg in enumerate(optimizer.param_groups):
                pg['lr'] = max(pg['lr'] * finetune_lr_drop, 1e-6)
            did_lr_drop = True
            print(f"Entering fine-tune phase: MixUp OFF, LR x{finetune_lr_drop}")

        print(f"\nEpoch {epoch+1}/{epochs} | Mixup/CutMix: {'ON' if mixup_active else 'OFF'}")
        backbone.train()
        classifier.train()
        total_loss = 0.0

        train_loop = tqdm(train_loader, desc=f"Training [{epoch+1}]")
        for inputs, targets in train_loop:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            if mixup_active:
                inputs, targets_mixed = mixup_fn(inputs, targets)
            else:
                targets_mixed = None

            with autocast(enabled=use_amp and use_cuda):
                feats  = backbone(inputs)
                logits = classifier(feats)
                if mixup_active:
                    loss = soft_target_cross_entropy(logits, targets_mixed)
                else:
                    loss = F.cross_entropy(logits, targets)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            train_loop.set_postfix(loss=loss.item())

        avg_train_loss = total_loss / max(1, len(train_loader))
        
        # Step scheduler - use different strategies for different phases
        if epoch < warmup_epochs:
            scheduler.step()  # Linear warmup
        elif epoch >= finetune_start and did_lr_drop:
            # Use plateau scheduler in fine-tuning phase (step after getting validation metrics)
            pass  # Will step plateau_scheduler later with val_f1
        else:
            scheduler.step()  # Cosine annealing with restarts

        # ===== Validation =====
        backbone.eval()
        classifier.eval()
        val_loss = 0.0
        val_labels, val_preds, val_probs = [], [], []

        val_loop = tqdm(val_loader, desc=f"Validation [{epoch+1}]")
        with torch.no_grad():
            for inputs, targets in val_loop:
                inputs = inputs.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                with autocast(enabled=use_amp and use_cuda):
                    feats = backbone(inputs)
                    logits = classifier(feats)
                    loss = F.cross_entropy(logits, targets)  # CE at eval
                    probs = F.softmax(logits, dim=1)

                val_loss += loss.item()
                preds = torch.argmax(probs, dim=1)
                val_labels.extend(targets.cpu().numpy())
                val_preds.extend(preds.cpu().numpy())
                val_probs.extend(probs.cpu().numpy())
                val_loop.set_postfix(loss=loss.item())

        avg_val_loss = val_loss / max(1, len(val_loader))

        # ===== Macro Metrics =====
        acc = accuracy_score(val_labels, val_preds)
        prec_macro = precision_score(val_labels, val_preds, average='macro', zero_division=0)
        rec_macro  = recall_score(val_labels, val_preds, average='macro', zero_division=0)
        f1_macro   = f1_score(val_labels, val_preds, average='macro', zero_division=0)
        try:
            y_true_ovr = np.eye(num_classes)[val_labels]
            auc_ovr = roc_auc_score(y_true_ovr, np.array(val_probs), multi_class='ovr')
        except Exception:
            auc_ovr = float('nan')

        # Debug LRs
        current_lrs = [pg['lr'] for pg in optimizer.param_groups]
        # Step plateau scheduler in fine-tuning phase
        if epoch >= finetune_start and did_lr_drop:
            plateau_scheduler.step(f1_macro)  # Step based on validation F1
        
        print(f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        print(f"Val Acc: {acc:.4f} | Prec(m): {prec_macro:.4f} | Rec(m): {rec_macro:.4f} | "
              f"F1(m): {f1_macro:.4f} | AUC(OvR): {auc_ovr:.4f} | LRs: {current_lrs}")

        # ===== Per-class Metrics =====
        val_labels_np = np.array(val_labels)
        val_preds_np  = np.array(val_preds)
        print("\n📊 Per-class metrics:")
        for class_idx in range(num_classes):
            cls_name = idx_to_class[class_idx]
            true_binary = (val_labels_np == class_idx)
            pred_binary = (val_preds_np == class_idx)

            cls_prec = precision_score(true_binary, pred_binary, zero_division=0)
            cls_rec  = recall_score(true_binary, pred_binary, zero_division=0)
            cls_acc  = np.mean(true_binary == pred_binary)
            print(f"  {cls_name:<20} Prec: {cls_prec:.3f}  Rec: {cls_rec:.3f}  Acc: {cls_acc:.3f}")

        # ===== CSV Logging =====
        new_row = {
            'epoch': epoch,
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
            'val_acc': acc,
            'val_precision_macro': prec_macro,
            'val_recall_macro': rec_macro,
            'val_f1_macro': f1_macro,
            'val_auc_ovr': auc_ovr,
            'lr': current_lrs[0]
        }
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        df.to_csv(csv_full_path, index=False)

        # ===== Save Best by F1 (canonical) + also keep Top-K by (F1, recall) =====
        if f1_macro > best_val_f1 + 1e-6:
            best_val_f1 = f1_macro

            ckpt = {
                'epoch': epoch,
                'backbone_state': backbone.state_dict(),
                'classifier_state': classifier.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'scheduler_state': scheduler.state_dict(),
                'plateau_scheduler_state': plateau_scheduler.state_dict(),
                'best_val_f1': best_val_f1,
                'unfrozen': unfrozen,
                'label_mapping': label_mapping,
                'class_importance': CLASS_IMPORTANCE,
                'minority_classes': list(MINORITY_CLASSES),
                'training_args': {
                    'lr': lr,
                    'warmup_epochs': warmup_epochs,
                    'mixup_alpha': mixup_alpha,
                    'cutmix_alpha': cutmix_alpha,
                    'mixup_burst_len_after_resume': mixup_burst_len_after_resume,
                    'finetune_lr_drop': finetune_lr_drop
                }
            }
            torch.save(ckpt, model_full_path)
            print("💾 Saved best checkpoint (by macro-F1).")

            # also save into the Top-K pool
            topk_saver.save(f1_macro, rec_macro, epoch, ckpt, prefix="multi")

            last_val_labels, last_val_preds = val_labels[:], val_preds[:]
            pred_path = os.path.join(output_dir, 'multi_val_predictions.pkl')
            with open(pred_path, 'wb') as f:
                pickle.dump({'labels': last_val_labels, 'preds': last_val_preds}, f)

        # ===== Early stop on 3-epoch moving-average of macro-F1 =====
        if ma_stop.step(f1_macro):
            print("Early stopping (no improvement in MA macro-F1).")
            break

    return last_val_labels, last_val_preds

def main(args):
    # Set seeds and device
    set_seeds(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"PyTorch version: {torch.__version__}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name()}")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Build cache if missing
    print("Building cache if missing...")
    for split in ['train', 'val']:
        cache_split_if_missing(args.data_dir, args.cache_dir, split)

    # Loaders
    print("Creating data loaders...")
    train_loader, val_loader, class_counts = get_multi_loaders(
        os.path.join(args.cache_dir, "train"),
        os.path.join(args.cache_dir, "val"),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=args.img_size,
        minority_boost_factor=args.minority_boost_factor
    )

    print("✅ Dataloaders ready.")

    # Model
    print("Creating dual backbone model...")
    backbone = DualBackbone(
        deit_variant=args.deit_variant, 
        freeze_all=True, 
        img_size=args.img_size
    )
    in_features = backbone.cnn_feat_dim + backbone.deit_feat_dim
    classifier = MultiClassifier(in_features, num_classes)
    print(f"✅ Model initialized. Classes kept: {KEPT_CLASSES}")
    print(f"   ConvNeXt-Tiny feat dim: {backbone.cnn_feat_dim}, DeiT-III feat dim: {backbone.deit_feat_dim}")

    # Train
    print("Starting training...")
    labels_best, preds_best = train_multiclass_model(
        backbone, classifier, train_loader, val_loader, class_counts,
        epochs=args.epochs, 
        patience=args.patience, 
        lr=args.learning_rate,
        csv_path=args.csv_path, 
        model_path=args.model_path,
        use_amp=args.use_amp, 
        warmup_epochs=args.warmup_epochs,
        lr_backbone=args.lr_backbone,
        mixup_alpha=args.mixup_alpha, 
        cutmix_alpha=args.cutmix_alpha, 
        mixup_off_pct=args.mixup_off_pct,
        mixup_burst_len_after_resume=args.mixup_burst_len,
        finetune_lr_drop=args.finetune_lr_drop,
        topk_checkpoints=args.topk_checkpoints,
        output_dir=args.output_dir,
        device=device
    )

    # Plots (headless)
    print("Generating plots...")
    if len(labels_best) and len(preds_best):
        confusion_path = os.path.join(args.output_dir, "confusion_matrix_best_epoch.png")
        plot_confusion_matrix_multiclass(
            labels_best, preds_best,
            class_names=[idx_to_class[i] for i in range(num_classes)],
            title="Best-Epoch Validation Confusion Matrix",
            save_path=confusion_path
        )

    metrics_plot_path = os.path.join(args.output_dir, "training_metrics.png")
    plot_training_metrics(
        os.path.join(args.output_dir, args.csv_path), 
        save_path=metrics_plot_path
    )

    print("✅ Training completed successfully!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Dual-Backbone CNN for Skin Disease Classification")
    
    # Data arguments
    parser.add_argument("--data_dir", type=str, required=True, 
                       help="Path to original dataset directory")
    parser.add_argument("--cache_dir", type=str, required=True,
                       help="Path to cache directory")
    parser.add_argument("--output_dir", type=str, default="./outputs", 
                       help="Output directory for models and plots")
    
    # Model arguments
    parser.add_argument("--deit_variant", type=str, default="deit3_small_patch16_224",
                       help="DeiT variant to use")
    parser.add_argument("--img_size", type=int, default=224,
                       help="Input image size for DeiT")
    
    # Training arguments
    parser.add_argument("--batch_size", type=int, default=24, 
                       help="Batch size")
    parser.add_argument("--epochs", type=int, default=200, 
                       help="Number of epochs")
    parser.add_argument("--learning_rate", type=float, default=3e-4, 
                       help="Learning rate")
    parser.add_argument("--lr_backbone", type=float, default=1e-5,
                       help="Learning rate for backbone when unfrozen")
    parser.add_argument("--patience", type=int, default=20, 
                       help="Early stopping patience")
    parser.add_argument("--num_workers", type=int, default=4, 
                       help="Number of data loader workers")
    parser.add_argument("--warmup_epochs", type=int, default=5,
                       help="Number of warmup epochs")
    parser.add_argument("--minority_boost_factor", type=float, default=2.0,
                       help="Factor to boost minority class samples during training")
    
    # Augmentation arguments
    parser.add_argument("--mixup_alpha", type=float, default=0.4,
                       help="Mixup alpha parameter")
    parser.add_argument("--cutmix_alpha", type=float, default=1.0,
                       help="CutMix alpha parameter")
    parser.add_argument("--mixup_off_pct", type=float, default=0.2,
                       help="Percentage of epochs to turn off mixup")
    parser.add_argument("--mixup_burst_len", type=int, default=8,
                       help="Number of epochs to keep mixup on after resume")
    parser.add_argument("--finetune_lr_drop", type=float, default=0.25,
                       help="LR drop factor when entering fine-tune phase")
    
    # Other arguments
    parser.add_argument("--seed", type=int, default=42, 
                       help="Random seed")
    parser.add_argument("--use_amp", action="store_true", default=True,
                       help="Use automatic mixed precision")
    parser.add_argument("--topk_checkpoints", type=int, default=3,
                       help="Number of top checkpoints to keep")
    parser.add_argument("--csv_path", type=str, default="multi_metrics.csv",
                       help="CSV filename for metrics logging")
    parser.add_argument("--model_path", type=str, default="multi_best_classifier.pt",
                       help="Model checkpoint filename")
    
    args = parser.parse_args()
    
    main(args)
