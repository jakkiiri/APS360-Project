#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Swin Transformer Single Image Inference Script for Skin Disease Classification
Evaluates trained model on test set with comprehensive metrics and visualizations
"""

import os
# Ensure headless operation
os.environ['MPLBACKEND'] = 'Agg'
if 'DISPLAY' not in os.environ:
    os.environ['DISPLAY'] = ''

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import timm
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                           confusion_matrix, classification_report, ConfusionMatrixDisplay)
import matplotlib
matplotlib.use('Agg')  # Set non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import pandas as pd
import json
import argparse
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Import utility functions
try:
    from model_utils import safe_torch_load, print_model_info
    from threshold_tuning import ThresholdTuner, evaluate_with_tuning
except ImportError:
    # Fallback if utility modules are not available
    def safe_torch_load(checkpoint_path, map_location=None):
        try:
            return torch.load(checkpoint_path, map_location=map_location, weights_only=False)
        except TypeError:
            return torch.load(checkpoint_path, map_location=map_location)
    
    def print_model_info(model, checkpoint=None):
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Model parameters: {total_params:,}")
    
    # Minimal threshold tuning fallback
    ThresholdTuner = None
    evaluate_with_tuning = None

# Class mapping - MUST match training script exactly
CLASS_NAMES = [
    'nevus', 'melanoma', 'bcc', 'keratosis',
    'actinic_keratosis', 'scc', 'dermatofibroma', 'lentigo', 'vascular_lesion'
]

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

class TestDataset(Dataset):
    """Simple test dataset with minimal preprocessing"""
    
    def __init__(self, root_dir, image_size=224):
        self.root_dir = os.path.join(root_dir, 'test') if not root_dir.endswith('test') else root_dir
        self.image_size = image_size
        
        # Collect all image paths and labels
        self.samples = []
        self.class_counts = Counter()
        
        for idx, class_name in enumerate(CLASS_NAMES):
            class_dir = os.path.join(self.root_dir, class_name)
            if os.path.exists(class_dir):
                for img_name in os.listdir(class_dir):
                    if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                        sample = (os.path.join(class_dir, img_name), idx)
                        self.samples.append(sample)
                        self.class_counts[idx] += 1
        
        print(f"TEST - Class distribution: {dict(self.class_counts)}")
        print(f"Total test samples: {len(self.samples)}")
        
        # Check for missing classes
        missing_classes = []
        for idx, class_name in enumerate(CLASS_NAMES):
            if idx not in self.class_counts:
                missing_classes.append(f"{idx}:{class_name}")
        
        if missing_classes:
            print(f"WARNING: Missing classes in test set: {missing_classes}")
        
        # Test transforms - minimal preprocessing for unbiased evaluation
        self.transform = A.Compose([
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
        
        # Apply transforms
        augmented = self.transform(image=image)
        image = augmented['image']
        
        return image, label, img_path

def load_model(checkpoint_path, device, model_name='swin_base_patch4_window7_224', image_size=224):
    """Load trained model from checkpoint"""
    print(f"Loading model from: {checkpoint_path}")
    
    # Create model
    model = SwinTransformerClassifier(
        num_classes=9, 
        model_name=model_name, 
        pretrained=False,  # Don't load pretrained weights when loading checkpoint
        image_size=image_size
    )
    
    # Load checkpoint using safe loading function
    checkpoint = safe_torch_load(checkpoint_path, map_location=device)
    
    # Handle different checkpoint formats
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")
        if 'best_val_f1_macro' in checkpoint:
            print(f"Model's best validation F1 (macro): {checkpoint['best_val_f1_macro']:.4f}")
        if 'best_val_acc' in checkpoint:
            print(f"Model's best validation accuracy: {checkpoint['best_val_acc']:.4f}")
    else:
        # Direct state dict
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    
    return model, checkpoint

def evaluate_model(model, test_loader, device, val_loader=None, tune_thresholds=False):
    """Comprehensive model evaluation with optional threshold tuning"""
    model.eval()
    
    all_predictions = []
    all_targets = []
    all_probabilities = []
    all_paths = []
    
    print("Running inference on test set...")
    with torch.no_grad():
        for images, labels, paths in tqdm(test_loader, desc="Evaluating"):
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            probabilities = F.softmax(outputs, dim=1)
            _, predictions = torch.max(outputs, 1)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
            all_paths.extend(paths)
    
    # Convert to numpy arrays
    all_probabilities = np.array(all_probabilities)
    all_targets = np.array(all_targets)
    all_predictions = np.array(all_predictions)
    
    # Calculate standard metrics
    accuracy = accuracy_score(all_targets, all_predictions)
    precision_macro = precision_score(all_targets, all_predictions, average='macro', zero_division=0)
    recall_macro = recall_score(all_targets, all_predictions, average='macro', zero_division=0)
    f1_macro = f1_score(all_targets, all_predictions, average='macro', zero_division=0)
    
    precision_weighted = precision_score(all_targets, all_predictions, average='weighted', zero_division=0)
    recall_weighted = recall_score(all_targets, all_predictions, average='weighted', zero_division=0)
    f1_weighted = f1_score(all_targets, all_predictions, average='weighted', zero_division=0)
    
    # Per-class metrics
    per_class_precision = precision_score(all_targets, all_predictions, average=None, zero_division=0)
    per_class_recall = recall_score(all_targets, all_predictions, average=None, zero_division=0)
    per_class_f1 = f1_score(all_targets, all_predictions, average=None, zero_division=0)
    
    results = {
        'accuracy': accuracy,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'f1_macro': f1_macro,
        'precision_weighted': precision_weighted,
        'recall_weighted': recall_weighted,
        'f1_weighted': f1_weighted,
        'per_class_precision': per_class_precision,
        'per_class_recall': per_class_recall,
        'per_class_f1': per_class_f1,
        'predictions': all_predictions,
        'targets': all_targets,
        'probabilities': all_probabilities,
        'paths': all_paths,
        'tuned_predictions': None,
        'tuned_metrics': None,
        'thresholds': None,
        'threshold_tuner': None
    }
    
    # Threshold tuning if requested and available
    if tune_thresholds and ThresholdTuner is not None:
        print("\n🎯 Performing threshold tuning...")
        
        # Get validation data for threshold tuning
        val_probs, val_targets = None, None
        
        if val_loader is not None:
            print("   Using separate validation set for threshold tuning...")
            val_probs, val_targets = get_validation_predictions(model, val_loader, device)
        else:
            print("   ⚠️  Using test set for threshold tuning (not recommended for final results)")
            val_probs, val_targets = all_probabilities, all_targets
        
        # Tune thresholds
        tuner = ThresholdTuner(CLASS_NAMES)
        tuning_report = tuner.tune(val_probs, val_targets, metric="f1", passes=3, verbose=True)
        
        # Apply tuned thresholds to test set
        tuned_predictions = tuner.predict(all_probabilities, use_tuned=True)
        
        # Calculate tuned metrics
        tuned_accuracy = accuracy_score(all_targets, tuned_predictions)
        tuned_precision_macro = precision_score(all_targets, tuned_predictions, average='macro', zero_division=0)
        tuned_recall_macro = recall_score(all_targets, tuned_predictions, average='macro', zero_division=0)
        tuned_f1_macro = f1_score(all_targets, tuned_predictions, average='macro', zero_division=0)
        tuned_precision_weighted = precision_score(all_targets, tuned_predictions, average='weighted', zero_division=0)
        tuned_recall_weighted = recall_score(all_targets, tuned_predictions, average='weighted', zero_division=0)
        tuned_f1_weighted = f1_score(all_targets, tuned_predictions, average='weighted', zero_division=0)
        
        tuned_per_class_precision = precision_score(all_targets, tuned_predictions, average=None, zero_division=0)
        tuned_per_class_recall = recall_score(all_targets, tuned_predictions, average=None, zero_division=0)
        tuned_per_class_f1 = f1_score(all_targets, tuned_predictions, average=None, zero_division=0)
        
        tuned_metrics = {
            'accuracy': tuned_accuracy,
            'precision_macro': tuned_precision_macro,
            'recall_macro': tuned_recall_macro,
            'f1_macro': tuned_f1_macro,
            'precision_weighted': tuned_precision_weighted,
            'recall_weighted': tuned_recall_weighted,
            'f1_weighted': tuned_f1_weighted,
            'per_class_precision': tuned_per_class_precision,
            'per_class_recall': tuned_per_class_recall,
            'per_class_f1': tuned_per_class_f1
        }
        
        # Update results
        results.update({
            'tuned_predictions': tuned_predictions,
            'tuned_metrics': tuned_metrics,
            'thresholds': tuner.thresholds,
            'threshold_tuner': tuner
        })
        
        # Print improvement
        f1_improvement = tuned_f1_macro - f1_macro
        acc_improvement = tuned_accuracy - accuracy
        print(f"\n📈 Threshold Tuning Results:")
        print(f"   F1 (macro): {f1_macro:.4f} → {tuned_f1_macro:.4f} (+{f1_improvement:.4f})")
        print(f"   Accuracy: {accuracy:.4f} → {tuned_accuracy:.4f} (+{acc_improvement:.4f})")
        
        if f1_improvement > 0.001:
            print("   ✅ Significant improvement achieved!")
        else:
            print("   ⚠️  Minimal improvement - thresholds may not help much for this dataset")
    
    return results

def get_validation_predictions(model, val_loader, device):
    """Get predictions on validation set for threshold tuning"""
    model.eval()
    val_probs = []
    val_targets = []
    
    print("   Collecting validation predictions...")
    with torch.no_grad():
        for images, labels, _ in tqdm(val_loader, desc="Val predictions", leave=False):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            probabilities = F.softmax(outputs, dim=1)
            
            val_probs.append(probabilities.cpu().numpy())
            val_targets.append(labels.cpu().numpy())
    
    return np.concatenate(val_probs, axis=0), np.concatenate(val_targets, axis=0)

def save_confusion_matrix(targets, predictions, class_names, save_path):
    """Save confusion matrix plot"""
    cm = confusion_matrix(targets, predictions)
    
    # Create confusion matrix with better formatting
    fig, ax = plt.subplots(figsize=(12, 10))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap='Blues', values_format='d', xticks_rotation=45, ax=ax)
    
    plt.title('Test Set Confusion Matrix', fontsize=16)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix saved to {save_path}")

def save_per_class_metrics(results, class_names, save_path):
    """Save detailed per-class metrics"""
    # Create DataFrame
    metrics_data = {
        'Class': class_names,
        'Precision': results['per_class_precision'],
        'Recall': results['per_class_recall'],
        'F1-Score': results['per_class_f1']
    }
    
    df = pd.DataFrame(metrics_data)
    
    # Save to CSV
    csv_path = save_path.replace('.png', '.csv')
    df.to_csv(csv_path, index=False)
    
    # Create bar plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    x_pos = np.arange(len(class_names))
    
    # Precision
    axes[0].bar(x_pos, results['per_class_precision'], alpha=0.8, color='skyblue')
    axes[0].set_title('Per-Class Precision')
    axes[0].set_xlabel('Class')
    axes[0].set_ylabel('Precision')
    axes[0].set_xticks(x_pos)
    axes[0].set_xticklabels(class_names, rotation=45, ha='right')
    axes[0].grid(True, alpha=0.3)
    
    # Recall
    axes[1].bar(x_pos, results['per_class_recall'], alpha=0.8, color='lightcoral')
    axes[1].set_title('Per-Class Recall')
    axes[1].set_xlabel('Class')
    axes[1].set_ylabel('Recall')
    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels(class_names, rotation=45, ha='right')
    axes[1].grid(True, alpha=0.3)
    
    # F1-Score
    axes[2].bar(x_pos, results['per_class_f1'], alpha=0.8, color='lightgreen')
    axes[2].set_title('Per-Class F1-Score')
    axes[2].set_xlabel('Class')
    axes[2].set_ylabel('F1-Score')
    axes[2].set_xticks(x_pos)
    axes[2].set_xticklabels(class_names, rotation=45, ha='right')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Per-class metrics saved to {save_path} and {csv_path}")
    
    return df

def main(args):
    print("🧪 Starting Swin Transformer Test Set Evaluation")
    print("=" * 80)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'plots'), exist_ok=True)
    
    # Load test dataset
    print("📦 Loading test dataset...")
    test_dataset = TestDataset(args.data_dir, args.image_size)
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.num_workers, 
        pin_memory=torch.cuda.is_available()
    )
    
    # Load validation dataset if threshold tuning is requested
    val_loader = None
    if args.tune_thresholds:
        print("📦 Loading validation dataset for threshold tuning...")
        val_data_dir = args.data_dir.replace('/test', '').replace('\\test', '')  # Remove /test from path
        if not val_data_dir.endswith('val'):
            val_data_dir = os.path.join(val_data_dir, 'val')
        
        try:
            val_dataset = TestDataset(val_data_dir, args.image_size)
            val_loader = DataLoader(
                val_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=torch.cuda.is_available()
            )
            print(f"✅ Validation dataset loaded: {len(val_dataset)} samples")
        except Exception as e:
            print(f"⚠️  Could not load validation dataset: {e}")
            print("   Will use test set for threshold tuning (not recommended)")
            val_loader = None
    
    # Load model
    print("🤖 Loading trained model...")
    model, checkpoint = load_model(args.checkpoint_path, device, args.model_name, args.image_size)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
    
    # Run evaluation
    print("🔍 Evaluating model on test set...")
    results = evaluate_model(model, test_loader, device, val_loader=val_loader, tune_thresholds=args.tune_thresholds)
    
    # Print results
    print("\n" + "=" * 60)
    print("📊 TEST SET RESULTS")
    print("=" * 60)
    
    # Standard results
    print("🔹 Standard (argmax) Results:")
    print(f"   Accuracy:           {results['accuracy']:.4f}")
    print(f"   Precision (macro):  {results['precision_macro']:.4f}")
    print(f"   Recall (macro):     {results['recall_macro']:.4f}")
    print(f"   F1-Score (macro):   {results['f1_macro']:.4f}")
    print(f"   Precision (weight): {results['precision_weighted']:.4f}")
    print(f"   Recall (weight):    {results['recall_weighted']:.4f}")
    print(f"   F1-Score (weight):  {results['f1_weighted']:.4f}")
    
    # Tuned results if available
    if results['tuned_metrics'] is not None:
        tuned = results['tuned_metrics']
        print("\n🔹 Threshold-Tuned Results:")
        print(f"   Accuracy:           {tuned['accuracy']:.4f}")
        print(f"   Precision (macro):  {tuned['precision_macro']:.4f}")
        print(f"   Recall (macro):     {tuned['recall_macro']:.4f}")
        print(f"   F1-Score (macro):   {tuned['f1_macro']:.4f}")
        print(f"   Precision (weight): {tuned['precision_weighted']:.4f}")
        print(f"   Recall (weight):    {tuned['recall_weighted']:.4f}")
        print(f"   F1-Score (weight):  {tuned['f1_weighted']:.4f}")
        
        # Show improvements
        f1_improvement = tuned['f1_macro'] - results['f1_macro']
        acc_improvement = tuned['accuracy'] - results['accuracy']
        print(f"\n🔹 Improvements:")
        print(f"   F1 (macro): +{f1_improvement:.4f}")
        print(f"   Accuracy:   +{acc_improvement:.4f}")
        
        # Show thresholds
        print(f"\n🔹 Optimized Thresholds:")
        for i, (class_name, threshold) in enumerate(zip(CLASS_NAMES, results['thresholds'])):
            print(f"   {class_name:>15}: {threshold:.3f}")
    else:
        print("\n🔹 Threshold tuning: Not performed")
    
    # Per-class results
    print("\n📋 PER-CLASS RESULTS:")
    print("-" * 60)
    for i, class_name in enumerate(CLASS_NAMES):
        if i < len(results['per_class_f1']):
            print(f"{class_name:>15}: Prec={results['per_class_precision'][i]:.3f}, "
                  f"Rec={results['per_class_recall'][i]:.3f}, F1={results['per_class_f1'][i]:.3f}")
    
    # Generate and save visualizations
    print("\n📊 Generating visualizations...")
    
    # Use tuned predictions if available, otherwise standard
    eval_predictions = results['tuned_predictions'] if results['tuned_predictions'] is not None else results['predictions']
    eval_metrics = results['tuned_metrics'] if results['tuned_metrics'] is not None else results
    
    # Confusion matrix
    save_confusion_matrix(
        results['targets'], eval_predictions, CLASS_NAMES,
        os.path.join(args.output_dir, 'plots', 'test_confusion_matrix.png')
    )
    
    # Per-class metrics
    per_class_df = save_per_class_metrics(
        eval_metrics, CLASS_NAMES,
        os.path.join(args.output_dir, 'plots', 'test_per_class_metrics.png')
    )
    
    # Save comprehensive results
    print("💾 Saving comprehensive results...")
    
    # Detailed classification report
    class_report = classification_report(
        results['targets'], eval_predictions, 
        target_names=CLASS_NAMES, output_dict=True
    )
    
    # Compile final results
    final_results = {
        'test_metrics': {
            'accuracy': float(results['accuracy']),
            'precision_macro': float(results['precision_macro']),
            'recall_macro': float(results['recall_macro']),
            'f1_macro': float(results['f1_macro']),
            'precision_weighted': float(results['precision_weighted']),
            'recall_weighted': float(results['recall_weighted']),
            'f1_weighted': float(results['f1_weighted'])
        },
        'per_class_metrics': per_class_df.to_dict('records'),
        'classification_report': class_report,
        'test_set_info': {
            'total_samples': len(test_dataset),
            'class_distribution': dict(test_dataset.class_counts)
        },
        'model_info': {
            'checkpoint_path': args.checkpoint_path,
            'model_name': args.model_name,
            'image_size': args.image_size,
            'total_parameters': total_params
        }
    }
    
    # Add tuned results if available
    if results['tuned_metrics'] is not None:
        final_results['tuned_test_metrics'] = {
            'accuracy': float(results['tuned_metrics']['accuracy']),
            'precision_macro': float(results['tuned_metrics']['precision_macro']),
            'recall_macro': float(results['tuned_metrics']['recall_macro']),
            'f1_macro': float(results['tuned_metrics']['f1_macro']),
            'precision_weighted': float(results['tuned_metrics']['precision_weighted']),
            'recall_weighted': float(results['tuned_metrics']['recall_weighted']),
            'f1_weighted': float(results['tuned_metrics']['f1_weighted'])
        }
        final_results['threshold_info'] = {
            'thresholds': results['thresholds'].tolist(),
            'threshold_by_class': {name: float(thr) for name, thr in zip(CLASS_NAMES, results['thresholds'])}
        }
    
    # Add checkpoint info if available
    if 'epoch' in checkpoint:
        final_results['model_info']['training_epoch'] = checkpoint['epoch']
    if 'best_val_f1_macro' in checkpoint:
        final_results['model_info']['best_val_f1_macro'] = float(checkpoint['best_val_f1_macro'])
    if 'best_val_acc' in checkpoint:
        final_results['model_info']['best_val_acc'] = float(checkpoint['best_val_acc'])
    
    # Save JSON results
    with open(os.path.join(args.output_dir, 'test_results.json'), 'w') as f:
        json.dump(final_results, f, indent=2)
    
    # Save predictions for further analysis
    predictions_data = {
        'targets': results['targets'],
        'predictions': results['predictions'],
        'probabilities': results['probabilities'],
        'image_paths': results['paths'],
        'class_names': CLASS_NAMES
    }
    
    # Add threshold tuning results if available
    if results['tuned_predictions'] is not None:
        predictions_data.update({
            'tuned_predictions': results['tuned_predictions'],
            'thresholds': results['thresholds'].tolist(),
            'tuned_metrics': results['tuned_metrics']
        })
        
        # Save threshold tuner for future use
        if results['threshold_tuner'] is not None:
            threshold_save_path = os.path.join(args.output_dir, 'optimized_thresholds.json')
            results['threshold_tuner'].save(threshold_save_path)
            print(f"💾 Optimized thresholds saved to: {threshold_save_path}")
    
    import pickle
    with open(os.path.join(args.output_dir, 'test_predictions.pkl'), 'wb') as f:
        pickle.dump(predictions_data, f)
    
    print(f"\n✅ Evaluation completed! Results saved to: {args.output_dir}")
    print(f"🏆 Final Test F1 (macro): {results['f1_macro']:.4f}")
    print(f"🎯 Final Test Accuracy: {results['accuracy']:.4f}")
    
    if results['tuned_metrics'] is not None:
        print(f"🎯 Tuned Test F1 (macro): {results['tuned_metrics']['f1_macro']:.4f}")
        print(f"🎯 Tuned Test Accuracy: {results['tuned_metrics']['accuracy']:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Swin Transformer Test Set Evaluation with Threshold Tuning")
    
    # Required arguments
    parser.add_argument("--checkpoint_path", type=str, required=True,
                       help="Path to trained model checkpoint")
    parser.add_argument("--data_dir", type=str, required=True,
                       help="Path to dataset directory (should contain test/ folder)")
    
    # Optional arguments
    parser.add_argument("--output_dir", type=str, default="./test_results",
                       help="Output directory for results and plots")
    parser.add_argument("--model_name", type=str, default="swin_base_patch4_window7_224",
                       help="Swin Transformer model name (must match training)")
    parser.add_argument("--image_size", type=int, default=224,
                       help="Input image size (must match training)")
    parser.add_argument("--batch_size", type=int, default=32,
                       help="Batch size for inference")
    parser.add_argument("--num_workers", type=int, default=4,
                       help="Number of data loader workers")
    
    # Threshold tuning arguments
    parser.add_argument("--tune_thresholds", action="store_true",
                       help="Enable threshold tuning for improved F1 score")
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint_path}")
    
    test_dir = os.path.join(args.data_dir, 'test')
    if not os.path.exists(test_dir):
        raise FileNotFoundError(f"Test directory not found: {test_dir}")
    
    print("🔧 Configuration:")
    print(f"   Checkpoint: {args.checkpoint_path}")
    print(f"   Data dir: {args.data_dir}")
    print(f"   Output dir: {args.output_dir}")
    print(f"   Model: {args.model_name}")
    print(f"   Image size: {args.image_size}")
    print(f"   Batch size: {args.batch_size}")
    print(f"   Threshold tuning: {args.tune_thresholds}")
    print(f"   Classes: {CLASS_NAMES}")
    
    main(args)
