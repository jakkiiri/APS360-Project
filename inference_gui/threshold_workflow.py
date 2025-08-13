#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Complete Threshold Tuning Workflow
Demonstrates the full process: tune on validation set, apply to test set, save for GUI
"""

import os
import sys
import argparse
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import classification_report

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

try:
    from swin_inference import SwinTransformerClassifier, TestDataset, load_model
    from threshold_tuning import ThresholdTuner, evaluate_with_tuning
    from model_utils import safe_torch_load
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure you're running from the inference_gui directory")
    sys.exit(1)

# Class names - must match training
CLASS_NAMES = [
    'nevus', 'melanoma', 'bcc', 'keratosis',
    'actinic_keratosis', 'scc', 'dermatofibroma', 'lentigo', 'vascular_lesion'
]

def run_complete_workflow(checkpoint_path, data_dir, output_dir, args):
    """
    Run the complete threshold tuning workflow:
    1. Load model and validation set
    2. Tune thresholds on validation set  
    3. Evaluate on test set with both standard and tuned predictions
    4. Save optimized thresholds for GUI use
    5. Generate comprehensive report
    """
    
    print("🔬 Starting Complete Threshold Tuning Workflow")
    print("=" * 60)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"📱 Device: {device}")
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Load model
    print("\n🤖 Loading model...")
    model, checkpoint = load_model(checkpoint_path, device, args.model_name, args.image_size)
    
    # Setup data paths
    data_root = Path(data_dir)
    val_dir = data_root / 'val'
    test_dir = data_root / 'test'
    
    if not val_dir.exists():
        print(f"❌ Validation directory not found: {val_dir}")
        print("   Threshold tuning requires a separate validation set")
        return False
    
    if not test_dir.exists():
        print(f"❌ Test directory not found: {test_dir}")
        return False
    
    # Load datasets
    print("\n📦 Loading datasets...")
    val_dataset = TestDataset(str(val_dir), args.image_size)
    test_dataset = TestDataset(str(test_dir), args.image_size)
    
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, 
                           num_workers=args.num_workers, pin_memory=torch.cuda.is_available())
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=torch.cuda.is_available())
    
    print(f"   Validation samples: {len(val_dataset)}")
    print(f"   Test samples: {len(test_dataset)}")
    
    # Step 1: Get validation predictions for threshold tuning
    print("\n🎯 Step 1: Collecting validation predictions for threshold tuning...")
    val_probs, val_targets = get_predictions(model, val_loader, device)
    
    # Standard validation performance
    val_standard_preds = np.argmax(val_probs, axis=1)
    from sklearn.metrics import f1_score, accuracy_score
    val_standard_f1 = f1_score(val_targets, val_standard_preds, average='macro', zero_division=0)
    val_standard_acc = accuracy_score(val_targets, val_standard_preds)
    
    print(f"   Validation baseline F1: {val_standard_f1:.4f}")
    print(f"   Validation baseline accuracy: {val_standard_acc:.4f}")
    
    # Step 2: Tune thresholds on validation set
    print("\n🎯 Step 2: Tuning thresholds on validation set...")
    tuner = ThresholdTuner(CLASS_NAMES)
    tuning_report = tuner.tune(
        val_probs, val_targets, 
        metric=args.metric, 
        passes=args.passes, 
        verbose=True
    )
    
    # Validation performance with tuned thresholds
    val_tuned_preds = tuner.predict(val_probs, use_tuned=True)
    val_tuned_f1 = f1_score(val_targets, val_tuned_preds, average='macro', zero_division=0)
    val_tuned_acc = accuracy_score(val_targets, val_tuned_preds)
    
    print(f"\n📈 Validation Results:")
    print(f"   Standard F1: {val_standard_f1:.4f}")
    print(f"   Tuned F1: {val_tuned_f1:.4f} (+{val_tuned_f1 - val_standard_f1:.4f})")
    print(f"   Standard Accuracy: {val_standard_acc:.4f}")
    print(f"   Tuned Accuracy: {val_tuned_acc:.4f} (+{val_tuned_acc - val_standard_acc:.4f})")
    
    # Step 3: Apply to test set
    print("\n🎯 Step 3: Applying tuned thresholds to test set...")
    test_probs, test_targets = get_predictions(model, test_loader, device)
    
    # Test performance - standard
    test_standard_preds = np.argmax(test_probs, axis=1)
    test_standard_f1 = f1_score(test_targets, test_standard_preds, average='macro', zero_division=0)
    test_standard_acc = accuracy_score(test_targets, test_standard_preds)
    
    # Test performance - tuned
    test_tuned_preds = tuner.predict(test_probs, use_tuned=True)
    test_tuned_f1 = f1_score(test_targets, test_tuned_preds, average='macro', zero_division=0)
    test_tuned_acc = accuracy_score(test_targets, test_tuned_preds)
    
    print(f"\n📊 Test Results:")
    print(f"   Standard F1: {test_standard_f1:.4f}")
    print(f"   Tuned F1: {test_tuned_f1:.4f} (+{test_tuned_f1 - test_standard_f1:.4f})")
    print(f"   Standard Accuracy: {test_standard_acc:.4f}")
    print(f"   Tuned Accuracy: {test_tuned_acc:.4f} (+{test_tuned_acc - test_standard_acc:.4f})")
    
    # Step 4: Save results and thresholds
    print("\n💾 Step 4: Saving results...")
    
    # Save threshold tuner for GUI use
    threshold_path = output_dir / 'optimized_thresholds.json'
    tuner.save(threshold_path)
    print(f"   Thresholds saved: {threshold_path}")
    
    # Save detailed results
    results = {
        'model_info': {
            'checkpoint_path': str(checkpoint_path),
            'model_name': args.model_name,
            'image_size': args.image_size
        },
        'tuning_config': {
            'metric': args.metric,
            'passes': args.passes,
            'tuning_set': 'validation'
        },
        'validation_results': {
            'standard_f1': float(val_standard_f1),
            'tuned_f1': float(val_tuned_f1),
            'f1_improvement': float(val_tuned_f1 - val_standard_f1),
            'standard_accuracy': float(val_standard_acc),
            'tuned_accuracy': float(val_tuned_acc),
            'accuracy_improvement': float(val_tuned_acc - val_standard_acc)
        },
        'test_results': {
            'standard_f1': float(test_standard_f1),
            'tuned_f1': float(test_tuned_f1),
            'f1_improvement': float(test_tuned_f1 - test_standard_f1),
            'standard_accuracy': float(test_standard_acc),
            'tuned_accuracy': float(test_tuned_acc),
            'accuracy_improvement': float(test_tuned_acc - test_standard_acc)
        },
        'thresholds': {
            'values': tuner.thresholds.tolist(),
            'by_class': {name: float(thr) for name, thr in zip(CLASS_NAMES, tuner.thresholds)}
        }
    }
    
    import json
    results_path = output_dir / 'threshold_tuning_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"   Results saved: {results_path}")
    
    # Generate classification reports
    print("\n📋 Generating detailed reports...")
    
    # Standard report
    standard_report = classification_report(test_targets, test_standard_preds, 
                                          target_names=CLASS_NAMES, output_dict=True)
    
    # Tuned report  
    tuned_report = classification_report(test_targets, test_tuned_preds,
                                       target_names=CLASS_NAMES, output_dict=True)
    
    # Save reports
    report_path = output_dir / 'classification_reports.json'
    with open(report_path, 'w') as f:
        json.dump({
            'standard': standard_report,
            'tuned': tuned_report
        }, f, indent=2)
    print(f"   Classification reports saved: {report_path}")
    
    # Step 5: Generate summary
    print(f"\n✅ Threshold tuning workflow complete!")
    print(f"📁 Results saved to: {output_dir}")
    
    if test_tuned_f1 > test_standard_f1 + 0.001:
        print(f"🎉 Significant improvement achieved! (+{test_tuned_f1 - test_standard_f1:.4f} F1)")
        print(f"💡 Load {threshold_path} in the GUI for optimized predictions")
    else:
        print(f"⚠️  Minimal improvement. Standard argmax may be sufficient for this model/dataset.")
    
    # Print threshold summary
    print(f"\n🎯 Optimized Thresholds:")
    for name, thr in zip(CLASS_NAMES, tuner.thresholds):
        print(f"   {name:>15}: {thr:.3f}")
    
    return True

def get_predictions(model, dataloader, device):
    """Get model predictions and targets from dataloader"""
    model.eval()
    all_probs = []
    all_targets = []
    
    with torch.no_grad():
        for images, labels, _ in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            probs = F.softmax(outputs, dim=1)
            
            all_probs.append(probs.cpu().numpy())
            all_targets.append(labels.cpu().numpy())
    
    return np.concatenate(all_probs, axis=0), np.concatenate(all_targets, axis=0)

def main():
    parser = argparse.ArgumentParser(description="Complete Threshold Tuning Workflow")
    
    # Required arguments
    parser.add_argument("--checkpoint_path", type=str, required=True,
                       help="Path to trained model checkpoint")
    parser.add_argument("--data_dir", type=str, required=True,
                       help="Root data directory (should contain val/ and test/ subdirs)")
    
    # Optional arguments
    parser.add_argument("--output_dir", type=str, default="./threshold_tuning_results",
                       help="Output directory for results")
    parser.add_argument("--model_name", type=str, default="swin_base_patch4_window7_224",
                       help="Swin Transformer model name")
    parser.add_argument("--image_size", type=int, default=224,
                       help="Input image size")
    parser.add_argument("--batch_size", type=int, default=32,
                       help="Batch size for inference")
    parser.add_argument("--num_workers", type=int, default=4,
                       help="Number of data loader workers")
    
    # Threshold tuning parameters
    parser.add_argument("--metric", type=str, default="f1", choices=["f1", "precision", "recall", "accuracy"],
                       help="Metric to optimize during threshold tuning")
    parser.add_argument("--passes", type=int, default=3,
                       help="Number of coordinate ascent passes")
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.checkpoint_path):
        print(f"❌ Checkpoint not found: {args.checkpoint_path}")
        return 1
    
    if not os.path.exists(args.data_dir):
        print(f"❌ Data directory not found: {args.data_dir}")
        return 1
    
    print("🔧 Configuration:")
    print(f"   Checkpoint: {args.checkpoint_path}")
    print(f"   Data directory: {args.data_dir}")
    print(f"   Output directory: {args.output_dir}")
    print(f"   Model: {args.model_name}")
    print(f"   Optimization metric: {args.metric}")
    print(f"   Coordinate ascent passes: {args.passes}")
    
    # Run workflow
    success = run_complete_workflow(args.checkpoint_path, args.data_dir, args.output_dir, args)
    
    if success:
        print(f"\n🎯 Next steps:")
        print(f"   1. Review results in {args.output_dir}")
        print(f"   2. Use threshold file in GUI: Load optimized_thresholds.json")
        print(f"   3. Apply thresholds in batch inference: --tune_thresholds flag")
        return 0
    else:
        return 1

if __name__ == "__main__":
    sys.exit(main())
