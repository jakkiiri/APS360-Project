#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple Swin Transformer Inference Script
Takes a model .pth file and test folder, runs inference and saves results
Loads model exactly the same way as single_image_gui.py
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import timm
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import pandas as pd
import json
from tqdm import tqdm
from collections import Counter

# Import utility functions - same as GUI
try:
    from model_utils import safe_torch_load, print_model_info
    from threshold_tuning import ThresholdTuner, adjusted_argmax
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
    adjusted_argmax = None

# Class mapping - MUST match training script exactly
CLASS_NAMES = [
    'nevus', 'melanoma', 'bcc', 'keratosis',
    'actinic_keratosis', 'scc', 'dermatofibroma', 'lentigo', 'vascular_lesion'
]

# Class descriptions and risk levels - same as GUI
CLASS_DESCRIPTIONS = {
    'nevus': 'Benign mole (non-cancerous)',
    'melanoma': 'Malignant melanoma (dangerous cancer)',
    'bcc': 'Basal cell carcinoma (common skin cancer)', 
    'keratosis': 'Seborrheic keratosis (benign growth)',
    'actinic_keratosis': 'Actinic keratosis (pre-cancerous)',
    'scc': 'Squamous cell carcinoma (skin cancer)',
    'dermatofibroma': 'Dermatofibroma (benign fibrous nodule)',
    'lentigo': 'Solar lentigo (age spot)',
    'vascular_lesion': 'Vascular lesion (blood vessel related)'
}

RISK_LEVELS = {
    'nevus': 'Low',
    'melanoma': 'Critical',
    'bcc': 'High',
    'keratosis': 'Low',
    'actinic_keratosis': 'Moderate',
    'scc': 'High',
    'dermatofibroma': 'Low',
    'lentigo': 'Low',
    'vascular_lesion': 'Low'
}

class SwinTransformerClassifier(nn.Module):
    """Swin Transformer model for skin disease classification - matches GUI exactly"""
    
    def __init__(self, num_classes=9, model_name='swin_base_patch4_window7_224', pretrained=True, image_size=224):
        super(SwinTransformerClassifier, self).__init__()
        
        # Load pre-trained Swin Transformer
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0, img_size=image_size)
        
        # Get feature dimension - same logic as GUI
        if hasattr(self.backbone, 'num_features'):
            feature_dim = self.backbone.num_features
        else:
            # Fallback to forward pass probing in eval mode
            self.backbone.eval()
            with torch.no_grad():
                dummy_input = torch.randn(1, 3, image_size, image_size)
                feature_dim = self.backbone(dummy_input).shape[1]
            self.backbone.train()
        
        # Classification head with dropout - matches GUI exactly
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
    """Test dataset using Albumentations with validation transforms only - exactly matching training script"""
    
    def __init__(self, test_dir, image_size=512):
        self.test_dir = test_dir
        self.image_size = image_size
        
        # Collect all image paths and labels
        self.samples = []
        self.class_counts = Counter()
        
        print(f"Loading test data from: {test_dir}")
        
        for idx, class_name in enumerate(CLASS_NAMES):
            class_dir = os.path.join(test_dir, class_name)
            if os.path.exists(class_dir):
                class_count = 0
                for img_name in os.listdir(class_dir):
                    if img_name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                        sample = (os.path.join(class_dir, img_name), idx)
                        self.samples.append(sample)
                        class_count += 1
                self.class_counts[idx] = class_count
                print(f"  {class_name}: {class_count} images")
            else:
                print(f"  WARNING: {class_name} directory not found")
        
        print(f"Total test samples: {len(self.samples)}")
        
        # Use exactly the same validation transforms as training script
        self.transform = self._get_val_transforms()
    
    def _get_val_transforms(self):
        """Minimal augmentations for validation/test - exactly matching training script"""
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
        try:
            image = Image.open(img_path).convert('RGB')
            image = np.array(image)
            
            # Apply validation transforms - exactly like training script validation
            augmented = self.transform(image=image)
            image = augmented['image']
            
            return image, label, img_path
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a dummy tensor if image fails to load
            dummy_image = torch.zeros(3, self.image_size, self.image_size)
            return dummy_image, label, img_path

def load_model(checkpoint_path, device, model_name='swin_base_patch4_window7_224', image_size=512):
    """Load trained model from checkpoint - exactly like GUI"""
    print(f"Loading model from: {checkpoint_path}")
    
    # Create model - exactly like GUI
    model = SwinTransformerClassifier(
        num_classes=len(CLASS_NAMES),
        model_name=model_name,
        pretrained=False,  # Don't load pretrained when loading checkpoint
        image_size=image_size
    )
    
    # Load checkpoint using safe loading function - exactly like GUI
    checkpoint = safe_torch_load(checkpoint_path, map_location=device)
    
    # Handle different checkpoint formats - exactly like GUI
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        epoch_info = f" (Epoch {checkpoint.get('epoch', '?')})"
        val_f1 = checkpoint.get('best_val_f1_macro', None)
        if val_f1:
            epoch_info += f"\nVal F1: {val_f1:.3f}"
        print(f"✅ Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")
        if 'best_val_f1_macro' in checkpoint:
            print(f"   Best validation F1 (macro): {checkpoint['best_val_f1_macro']:.4f}")
        if 'best_val_acc' in checkpoint:
            print(f"   Best validation accuracy: {checkpoint['best_val_acc']:.4f}")
    else:
        model.load_state_dict(checkpoint)
        print("✅ Loaded model weights")
    
    model = model.to(device)
    model.eval()
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Model parameters: {total_params:,}")
    
    return model, checkpoint

def run_inference(model, test_loader, device, threshold_tuner=None, use_tuned_thresholds=False):
    """Run inference on test set - exactly like GUI"""
    model.eval()
    
    all_predictions = []
    all_targets = []
    all_probabilities = []
    all_paths = []
    all_tuned_predictions = []
    
    print("\n🔍 Running inference on test set...")
    with torch.no_grad():
        for images, labels, paths in tqdm(test_loader, desc="Processing"):
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            probabilities = F.softmax(outputs, dim=1)
            
            # Standard predictions
            _, predictions = torch.max(outputs, 1)
            all_predictions.extend(predictions.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
            all_paths.extend(paths)
            
            # Tuned predictions if available - exactly like GUI logic
            if use_tuned_thresholds and threshold_tuner is not None:
                batch_probs = probabilities.cpu().numpy()
                tuned_preds = threshold_tuner.predict(batch_probs, use_tuned=True)
                all_tuned_predictions.extend(tuned_preds)
            else:
                all_tuned_predictions.extend(predictions.cpu().numpy())
    
    return {
        'predictions': all_predictions,
        'targets': all_targets,
        'probabilities': all_probabilities,
        'paths': all_paths,
        'tuned_predictions': all_tuned_predictions
    }

def calculate_metrics(targets, predictions):
    """Calculate comprehensive metrics"""
    accuracy = accuracy_score(targets, predictions)
    precision_macro = precision_score(targets, predictions, average='macro', zero_division=0)
    recall_macro = recall_score(targets, predictions, average='macro', zero_division=0)
    f1_macro = f1_score(targets, predictions, average='macro', zero_division=0)
    
    precision_weighted = precision_score(targets, predictions, average='weighted', zero_division=0)
    recall_weighted = recall_score(targets, predictions, average='weighted', zero_division=0)
    f1_weighted = f1_score(targets, predictions, average='weighted', zero_division=0)
    
    # Per-class metrics
    per_class_precision = precision_score(targets, predictions, average=None, zero_division=0)
    per_class_recall = recall_score(targets, predictions, average=None, zero_division=0)
    per_class_f1 = f1_score(targets, predictions, average=None, zero_division=0)
    
    return {
        'accuracy': accuracy,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'f1_macro': f1_macro,
        'precision_weighted': precision_weighted,
        'recall_weighted': recall_weighted,
        'f1_weighted': f1_weighted,
        'per_class_precision': per_class_precision.tolist(),
        'per_class_recall': per_class_recall.tolist(),
        'per_class_f1': per_class_f1.tolist()
    }

def print_results(metrics, test_dataset, tuned_metrics=None, threshold_info=None):
    """Print comprehensive results with optional tuned metrics"""
    print("\n" + "=" * 60)
    print("📊 INFERENCE RESULTS")
    print("=" * 60)
    
    print(f"Test samples: {len(test_dataset)}")
    print(f"Accuracy:           {metrics['accuracy']:.4f}")
    print(f"Precision (macro):  {metrics['precision_macro']:.4f}")
    print(f"Recall (macro):     {metrics['recall_macro']:.4f}")
    print(f"F1-Score (macro):   {metrics['f1_macro']:.4f}")
    print(f"Precision (weight): {metrics['precision_weighted']:.4f}")
    print(f"Recall (weight):    {metrics['recall_weighted']:.4f}")
    print(f"F1-Score (weight):  {metrics['f1_weighted']:.4f}")
    
    # Show tuned results if available
    if tuned_metrics:
        print("\n🎯 TUNED THRESHOLD RESULTS:")
        print("-" * 60)
        print(f"Accuracy:           {tuned_metrics['accuracy']:.4f}")
        print(f"Precision (macro):  {tuned_metrics['precision_macro']:.4f}")
        print(f"Recall (macro):     {tuned_metrics['recall_macro']:.4f}")
        print(f"F1-Score (macro):   {tuned_metrics['f1_macro']:.4f}")
        print(f"Precision (weight): {tuned_metrics['precision_weighted']:.4f}")
        print(f"Recall (weight):    {tuned_metrics['recall_weighted']:.4f}")
        print(f"F1-Score (weight):  {tuned_metrics['f1_weighted']:.4f}")
        
        print("\n📈 IMPROVEMENT:")
        print("-" * 60)
        print(f"F1 (macro) improvement: {tuned_metrics['f1_macro'] - metrics['f1_macro']:+.4f}")
        print(f"Accuracy improvement:   {tuned_metrics['accuracy'] - metrics['accuracy']:+.4f}")
    
    print("\n📋 PER-CLASS RESULTS:")
    print("-" * 60)
    for i, class_name in enumerate(CLASS_NAMES):
        if i < len(metrics['per_class_f1']):
            risk = RISK_LEVELS[class_name]
            print(f"{class_name:>15} ({risk:>8}): Prec={metrics['per_class_precision'][i]:.3f}, "
                  f"Rec={metrics['per_class_recall'][i]:.3f}, F1={metrics['per_class_f1'][i]:.3f}")
    
    if threshold_info:
        print(f"\n🎯 Using optimized thresholds: {threshold_info}")

def save_results(results, metrics, test_dataset, output_file, model_path, test_dir, tuned_metrics=None, threshold_tuner=None):
    """Save results to JSON file with optional tuned metrics"""
    # Create comprehensive results
    output_data = {
        'model_info': {
            'model_path': model_path,
            'test_directory': test_dir,
            'class_names': CLASS_NAMES,
            'class_descriptions': CLASS_DESCRIPTIONS,
            'risk_levels': RISK_LEVELS
        },
        'test_set_info': {
            'total_samples': len(test_dataset),
            'class_distribution': dict(test_dataset.class_counts),
            'class_names': CLASS_NAMES
        },
        'standard_metrics': metrics,
        'detailed_results': {
            'per_class_metrics': [
                {
                    'class': CLASS_NAMES[i],
                    'description': CLASS_DESCRIPTIONS[CLASS_NAMES[i]],
                    'risk_level': RISK_LEVELS[CLASS_NAMES[i]],
                    'precision': metrics['per_class_precision'][i],
                    'recall': metrics['per_class_recall'][i],
                    'f1_score': metrics['per_class_f1'][i]
                }
                for i in range(len(CLASS_NAMES))
            ]
        }
    }
    
    # Add tuned metrics if available
    if tuned_metrics:
        output_data['tuned_metrics'] = tuned_metrics
        output_data['improvements'] = {
            'f1_macro': tuned_metrics['f1_macro'] - metrics['f1_macro'],
            'accuracy': tuned_metrics['accuracy'] - metrics['accuracy']
        }
    
    # Add threshold information if available
    if threshold_tuner:
        output_data['threshold_info'] = {
            'thresholds': threshold_tuner.thresholds.tolist(),
            'rule': threshold_tuner.rule,
            'tuning_report': threshold_tuner.tuning_report
        }
    
    # Save results
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\n💾 Results saved to: {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Simple Swin Transformer Inference - exactly like GUI")
    parser.add_argument("model_path", help="Path to the trained model .pth file")
    parser.add_argument("test_dir", help="Path to the test directory containing class folders")
    parser.add_argument("--output", "-o", default="inference_results.json", 
                       help="Output file for results (default: inference_results.json)")
    parser.add_argument("--batch_size", "-b", type=int, default=32, 
                       help="Batch size for inference (default: 32)")
    parser.add_argument("--image_size", type=int, default=512, 
                       help="Input image size (default: 512, matching training dataset default)")
    parser.add_argument("--model_name", default="swin_base_patch4_window7_224",
                       help="Swin Transformer model name (default: swin_base_patch4_window7_224)")
    parser.add_argument("--thresholds", type=str, default=None,
                       help="Path to optimized thresholds file (.json or .pkl)")
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.model_path):
        print(f"❌ Model file not found: {args.model_path}")
        return 1
    
    if not os.path.exists(args.test_dir):
        print(f"❌ Test directory not found: {args.test_dir}")
        return 1
    
    # Check if test directory has class folders
    class_dirs_found = []
    for class_name in CLASS_NAMES:
        class_dir = os.path.join(args.test_dir, class_name)
        if os.path.exists(class_dir):
            class_dirs_found.append(class_name)
    
    if not class_dirs_found:
        print(f"❌ No class directories found in {args.test_dir}")
        print(f"Expected directories: {CLASS_NAMES}")
        return 1
    
    print("🚀 Starting Swin Transformer Inference (GUI-Compatible)")
    print("=" * 60)
    print(f"Model: {args.model_path}")
    print(f"Test directory: {args.test_dir}")
    print(f"Output: {args.output}")
    print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    if args.thresholds:
        print(f"Thresholds: {args.thresholds}")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    try:
        # Load model - exactly like GUI
        model, checkpoint = load_model(args.model_path, device, args.model_name, args.image_size)
        
        # Load thresholds if provided
        threshold_tuner = None
        use_tuned_thresholds = False
        
        if args.thresholds and ThresholdTuner is not None:
            try:
                print(f"\n🎯 Loading optimized thresholds from: {args.thresholds}")
                threshold_tuner = ThresholdTuner(CLASS_NAMES)
                threshold_tuner.load(args.thresholds)
                use_tuned_thresholds = True
                print(f"✅ Thresholds loaded successfully!")
                print(f"   Tuning report F1: {threshold_tuner.tuning_report.get('f1_macro', 'N/A'):.4f}")
            except Exception as e:
                print(f"⚠️  Failed to load thresholds: {e}")
                print("   Continuing with standard inference...")
        
        # Load test dataset - exactly like GUI
        test_dataset = TestDataset(args.test_dir, args.image_size)
        
        if len(test_dataset) == 0:
            print("❌ No test images found!")
            return 1
        
        test_loader = DataLoader(
            test_dataset, 
            batch_size=args.batch_size, 
            shuffle=False, 
            num_workers=4, 
            pin_memory=torch.cuda.is_available()
        )
        
        # Run inference - exactly like GUI
        results = run_inference(model, test_loader, device, threshold_tuner, use_tuned_thresholds)
        
        # Calculate standard metrics
        metrics = calculate_metrics(results['targets'], results['predictions'])
        
        # Calculate tuned metrics if available
        tuned_metrics = None
        if use_tuned_thresholds:
            tuned_metrics = calculate_metrics(results['targets'], results['tuned_predictions'])
        
        # Print results
        threshold_info = f"Loaded from {args.thresholds}" if use_tuned_thresholds else None
        print_results(metrics, test_dataset, tuned_metrics, threshold_info)
        
        # Save results
        save_results(results, metrics, test_dataset, args.output, args.model_path, args.test_dir, 
                    tuned_metrics, threshold_tuner)
        
        print(f"\n✅ Inference completed successfully!")
        if use_tuned_thresholds and tuned_metrics:
            print(f"🏆 Standard F1 Score: {metrics['f1_macro']:.4f}")
            print(f"🎯 Tuned F1 Score:    {tuned_metrics['f1_macro']:.4f} ({tuned_metrics['f1_macro'] - metrics['f1_macro']:+.4f})")
            print(f"🎯 Standard Accuracy: {metrics['accuracy']:.4f}")
            print(f"🎯 Tuned Accuracy:    {tuned_metrics['accuracy']:.4f} ({tuned_metrics['accuracy'] - metrics['accuracy']:+.4f})")
        else:
            print(f"🏆 Final F1 Score: {metrics['f1_macro']:.4f}")
            print(f"🎯 Final Accuracy: {metrics['accuracy']:.4f}")
        
        return 0
        
    except Exception as e:
        print(f"❌ Error during inference: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
