#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Threshold Tuning Module for Multi-Class Classification
Implements coordinate ascent threshold optimization for improved F1 scores
"""

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from sklearn.metrics import classification_report
import torch
import torch.nn.functional as F
from typing import Tuple, Dict, Optional, Union
import pickle
import json
from pathlib import Path

def adjusted_argmax(probs: np.ndarray, 
                   thresholds: np.ndarray, 
                   rule: str = "subtract") -> np.ndarray:
    """
    Apply per-class thresholds and return adjusted predictions
    
    Args:
        probs: (N, C) probability matrix
        thresholds: (C,) per-class thresholds
        rule: How to apply thresholds ("subtract" or "divide")
    
    Returns:
        predictions: (N,) predicted class indices
    """
    if rule == "subtract":
        # Subtract threshold from each class probability
        adjusted_probs = probs - thresholds[None, :]
    elif rule == "divide":
        # Divide by threshold (avoid division by zero)
        thresholds_safe = np.maximum(thresholds, 1e-8)
        adjusted_probs = probs / thresholds_safe[None, :]
    else:
        raise ValueError(f"Unknown rule: {rule}")
    
    # Take argmax of adjusted probabilities
    return np.argmax(adjusted_probs, axis=1)

def _metric_from_preds(labels: np.ndarray, 
                      preds: np.ndarray, 
                      metric: str = "f1") -> Tuple[float, Dict]:
    """
    Calculate metric from predictions and labels
    
    Args:
        labels: (N,) true labels
        preds: (N,) predicted labels  
        metric: Metric to optimize ("f1", "precision", "recall", "accuracy")
    
    Returns:
        score: Scalar metric value
        report: Dictionary with detailed metrics
    """
    if metric == "f1":
        score = f1_score(labels, preds, average='macro', zero_division=0)
    elif metric == "precision":
        score = precision_score(labels, preds, average='macro', zero_division=0)
    elif metric == "recall":
        score = recall_score(labels, preds, average='macro', zero_division=0)
    elif metric == "accuracy":
        score = accuracy_score(labels, preds)
    else:
        raise ValueError(f"Unknown metric: {metric}")
    
    # Generate detailed report
    report = {
        'f1_macro': f1_score(labels, preds, average='macro', zero_division=0),
        'f1_weighted': f1_score(labels, preds, average='weighted', zero_division=0),
        'precision_macro': precision_score(labels, preds, average='macro', zero_division=0),
        'recall_macro': recall_score(labels, preds, average='macro', zero_division=0),
        'accuracy': accuracy_score(labels, preds),
        'per_class_f1': f1_score(labels, preds, average=None, zero_division=0).tolist()
    }
    
    return score, report

def tune_per_class_thresholds(probs: np.ndarray,
                              labels: np.ndarray,
                              metric: str = "f1",
                              grid: Optional[np.ndarray] = None,
                              passes: int = 2,
                              rule: str = "subtract",
                              verbose: bool = True) -> Tuple[np.ndarray, Dict]:
    """
    Coordinate-ascent threshold search for multi-class classification
      
    Start from 0.5 for every class
    For each class, scan thresholds in 'grid' keeping others fixed
    Repeat 'passes' times
    
    Args:
        probs: (N, C) probability matrix
        labels: (N,) true labels
        metric: Metric to optimize ("f1", "precision", "recall", "accuracy")
        grid: Threshold values to search (if None, use default)
        passes: Number of coordinate ascent passes
        rule: How to apply thresholds ("subtract" or "divide")
        verbose: Print progress information
    
    Returns:
        thresholds: (C,) optimized per-class thresholds
        best_report: Dictionary with best metrics achieved
    """
    n_samples, n_classes = probs.shape
    
    if grid is None:
        # Denser around 0.4–0.7 which is often useful
        grid = np.unique(np.concatenate([
            np.linspace(0.05, 0.95, 19),
            np.array([0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65])
        ]))
    
    # Initialize thresholds to 0.5
    thr = np.full((n_classes,), 0.5, dtype=np.float32)
    
    # Evaluate starting point
    preds = adjusted_argmax(probs, thr, rule=rule)
    best_score, best_report = _metric_from_preds(labels, preds, metric=metric)
    
    if verbose:
        print(f"🎯 Starting threshold tuning...")
        print(f"   Initial {metric}: {best_score:.4f}")
        print(f"   Grid size: {len(grid)} points")
        print(f"   Classes: {n_classes}")
    
    # Coordinate ascent optimization
    for pass_idx in range(passes):
        improved = False
        if verbose:
            print(f"\n🔄 Pass {pass_idx + 1}/{passes}")
        
        for c in range(n_classes):
            best_t_c = thr[c]
            original_score = best_score
            
            # Try each threshold value for class c
            for t in grid:
                thr[c] = t
                preds = adjusted_argmax(probs, thr, rule=rule)
                score, report = _metric_from_preds(labels, preds, metric=metric)
                
                if score > best_score + 1e-6:  # Small epsilon for numerical stability
                    best_score, best_report = score, report
                    best_t_c = t
                    improved = True
            
            # Update threshold for class c
            thr[c] = best_t_c
            
            if verbose and best_score > original_score + 1e-6:
                print(f"   Class {c}: threshold {best_t_c:.3f} → {metric} {best_score:.4f}")
        
        if not improved:
            if verbose:
                print(f"   No improvement in pass {pass_idx + 1}, stopping early")
            break
    
    if verbose:
        print(f"\n✅ Threshold tuning complete!")
        print(f"   Final {metric}: {best_score:.4f}")
        print(f"   Improvement: {best_score - best_report.get(f'{metric}_macro', best_score):.4f}")
    
    return thr, best_report

class ThresholdTuner:
    """
    Threshold tuning class for easier integration with models
    """
    
    def __init__(self, class_names: Optional[list] = None):
        self.class_names = class_names
        self.thresholds = None
        self.tuning_report = None
        self.is_tuned = False
        
    def tune(self, probs: np.ndarray, labels: np.ndarray, **kwargs) -> Dict:
        """
        Tune thresholds on validation data
        
        Args:
            probs: (N, C) probability matrix from validation set
            labels: (N,) true labels from validation set
            **kwargs: Additional arguments for tune_per_class_thresholds
        
        Returns:
            Tuning report with metrics and thresholds
        """
        self.thresholds, self.tuning_report = tune_per_class_thresholds(
            probs, labels, **kwargs
        )
        self.is_tuned = True
        
        # Add threshold information to report
        self.tuning_report['thresholds'] = self.thresholds.tolist()
        self.tuning_report['default_threshold'] = 0.5
        
        if self.class_names:
            self.tuning_report['threshold_by_class'] = {
                name: float(thr) for name, thr in zip(self.class_names, self.thresholds)
            }
        
        return self.tuning_report
    
    def predict(self, probs: np.ndarray, use_tuned: bool = True) -> np.ndarray:
        """
        Make predictions using tuned or default thresholds
        
        Args:
            probs: (N, C) probability matrix
            use_tuned: Whether to use tuned thresholds (if available)
        
        Returns:
            predictions: (N,) predicted class indices
        """
        if use_tuned and self.is_tuned:
            return adjusted_argmax(probs, self.thresholds)
        else:
            # Use standard argmax (equivalent to 0.5 threshold for all classes)
            return np.argmax(probs, axis=1)
    
    def save(self, filepath: Union[str, Path]):
        """Save tuned thresholds to file"""
        if not self.is_tuned:
            raise ValueError("No thresholds have been tuned yet")
        
        save_data = {
            'thresholds': self.thresholds.tolist(),
            'tuning_report': self.tuning_report,
            'class_names': self.class_names,
            'is_tuned': self.is_tuned
        }
        
        filepath = Path(filepath)
        if filepath.suffix == '.json':
            with open(filepath, 'w') as f:
                json.dump(save_data, f, indent=2)
        else:
            with open(filepath, 'wb') as f:
                pickle.dump(save_data, f)
    
    def load(self, filepath: Union[str, Path]):
        """Load tuned thresholds from file"""
        filepath = Path(filepath)
        
        if filepath.suffix == '.json':
            with open(filepath, 'r') as f:
                save_data = json.load(f)
        else:
            with open(filepath, 'rb') as f:
                save_data = pickle.load(f)
        
        self.thresholds = np.array(save_data['thresholds'])
        self.tuning_report = save_data['tuning_report']
        self.class_names = save_data['class_names']
        self.is_tuned = save_data['is_tuned']

def evaluate_with_tuning(model, dataloader, device, class_names, 
                        tune_on_same_data: bool = False, verbose: bool = True):
    """
    Evaluate model with and without threshold tuning
    
    Args:
        model: PyTorch model
        dataloader: DataLoader for evaluation
        device: torch.device
        class_names: List of class names
        tune_on_same_data: Whether to tune thresholds on the same data (for demo)
        verbose: Print detailed results
    
    Returns:
        Dictionary with results for both standard and tuned predictions
    """
    model.eval()
    
    all_probs = []
    all_labels = []
    
    print("🔍 Collecting predictions for threshold tuning...")
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            probs = F.softmax(outputs, dim=1)
            
            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    
    # Concatenate all results
    all_probs = np.concatenate(all_probs, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    
    # Standard predictions (argmax)
    standard_preds = np.argmax(all_probs, axis=1)
    standard_score, standard_report = _metric_from_preds(all_labels, standard_preds)
    
    if verbose:
        print(f"\n📊 Standard Performance (argmax):")
        print(f"   F1 (macro): {standard_report['f1_macro']:.4f}")
        print(f"   Accuracy: {standard_report['accuracy']:.4f}")
    
    # Tune thresholds
    if tune_on_same_data:
        print("\n⚠️  Warning: Tuning thresholds on same data (for demonstration)")
    
    tuner = ThresholdTuner(class_names)
    tuning_report = tuner.tune(all_probs, all_labels, verbose=verbose)
    
    # Tuned predictions
    tuned_preds = tuner.predict(all_probs, use_tuned=True)
    tuned_score, tuned_report = _metric_from_preds(all_labels, tuned_preds)
    
    if verbose:
        print(f"\n📈 Tuned Performance:")
        print(f"   F1 (macro): {tuned_report['f1_macro']:.4f}")
        print(f"   Accuracy: {tuned_report['accuracy']:.4f}")
        print(f"   Improvement: {tuned_report['f1_macro'] - standard_report['f1_macro']:.4f}")
    
    return {
        'standard': {
            'predictions': standard_preds,
            'metrics': standard_report
        },
        'tuned': {
            'predictions': tuned_preds,
            'metrics': tuned_report,
            'thresholds': tuner.thresholds,
            'tuner': tuner
        },
        'probabilities': all_probs,
        'labels': all_labels
    }

# Example usage and testing
if __name__ == "__main__":
    print("🧪 Threshold Tuning Module Test")
    print("=" * 40)
    
    # Generate synthetic test data
    np.random.seed(42)
    n_samples, n_classes = 1000, 9
    
    # Create synthetic probabilities (some classes harder than others)
    probs = np.random.dirichlet(np.ones(n_classes) * 0.5, n_samples)
    
    # Create synthetic labels with class imbalance
    class_weights = np.array([0.3, 0.05, 0.15, 0.1, 0.08, 0.07, 0.1, 0.05, 0.1])
    labels = np.random.choice(n_classes, n_samples, p=class_weights)
    
    print(f"📊 Test data: {n_samples} samples, {n_classes} classes")
    print(f"   Class distribution: {np.bincount(labels)}")
    
    # Test threshold tuning
    class_names = ['class_' + str(i) for i in range(n_classes)]
    tuner = ThresholdTuner(class_names)
    
    print("\n🎯 Testing threshold tuning...")
    report = tuner.tune(probs, labels, metric="f1", passes=3, verbose=True)
    
    # Compare standard vs tuned predictions
    standard_preds = np.argmax(probs, axis=1)
    tuned_preds = tuner.predict(probs, use_tuned=True)
    
    standard_f1 = f1_score(labels, standard_preds, average='macro', zero_division=0)
    tuned_f1 = f1_score(labels, tuned_preds, average='macro', zero_division=0)
    
    print(f"\n📈 Results:")
    print(f"   Standard F1: {standard_f1:.4f}")
    print(f"   Tuned F1: {tuned_f1:.4f}")
    print(f"   Improvement: {tuned_f1 - standard_f1:.4f}")
    
    # Test save/load
    print(f"\n💾 Testing save/load...")
    tuner.save("test_thresholds.json")
    
    new_tuner = ThresholdTuner()
    new_tuner.load("test_thresholds.json")
    
    # Verify loaded tuner gives same results
    loaded_preds = new_tuner.predict(probs, use_tuned=True)
    assert np.array_equal(tuned_preds, loaded_preds), "Save/load failed!"
    
    print("✅ Save/load test passed!")
    
    # Clean up
    import os
    os.remove("test_thresholds.json")
    
    print("\n✅ All tests passed!")
