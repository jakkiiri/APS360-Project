#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Threshold Tuning Module for Multi-Class Classification
Implements coordinate ascent threshold optimization for improved F1 scores
"""

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
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
        adjusted_probs = probs - thresholds[None, :]
    elif rule == "divide":
        thresholds_safe = np.maximum(thresholds, 1e-8)
        adjusted_probs = probs / thresholds_safe[None, :]
    else:
        raise ValueError(f"Unknown rule: {rule}")
    
    return np.argmax(adjusted_probs, axis=1)


def _metric_from_preds(labels: np.ndarray, 
                      preds: np.ndarray, 
                      metric: str = "f1") -> Tuple[float, Dict]:
    """
    Calculate metric from predictions and labels
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
    """
    n_samples, n_classes = probs.shape
    
    if grid is None:
        grid = np.unique(np.concatenate([
            np.linspace(0.05, 0.95, 19),
            np.array([0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65])
        ]))
    
    thr = np.full((n_classes,), 0.5, dtype=np.float32)
    
    preds = adjusted_argmax(probs, thr, rule=rule)
    best_score, best_report = _metric_from_preds(labels, preds, metric=metric)
    
    if verbose:
        print(f"🎯 Starting threshold tuning...")
        print(f"   Initial {metric}: {best_score:.4f}")
    
    for pass_idx in range(passes):
        improved = False
        if verbose:
            print(f"\n🔄 Pass {pass_idx + 1}/{passes}")
        
        for c in range(n_classes):
            best_t_c = thr[c]
            original_score = best_score
            
            for t in grid:
                thr[c] = t
                preds = adjusted_argmax(probs, thr, rule=rule)
                score, report = _metric_from_preds(labels, preds, metric=metric)
                
                if score > best_score + 1e-6:
                    best_score, best_report = score, report
                    best_t_c = t
                    improved = True
            
            thr[c] = best_t_c
            
            if verbose and best_score > original_score + 1e-6:
                print(f"   Class {c}: threshold {best_t_c:.3f} → {metric} {best_score:.4f}")
        
        if not improved:
            if verbose:
                print(f"   No improvement in pass {pass_idx + 1}, stopping early")
            break
    
    if verbose:
        print(f"\n✅ Threshold tuning complete! Final {metric}: {best_score:.4f}")
    
    return thr, best_report


class ThresholdTuner:
    """Threshold tuning class for easier integration with models"""
    
    def __init__(self, class_names: Optional[list] = None):
        self.class_names = class_names
        self.thresholds = None
        self.tuning_report = None
        self.is_tuned = False
        self.rule = "subtract"
        
    def tune(self, probs: np.ndarray, labels: np.ndarray, **kwargs) -> Dict:
        """Tune thresholds on validation data"""
        self.thresholds, self.tuning_report = tune_per_class_thresholds(
            probs, labels, **kwargs
        )
        self.is_tuned = True
        
        self.tuning_report['thresholds'] = self.thresholds.tolist()
        self.tuning_report['default_threshold'] = 0.5
        
        if self.class_names:
            self.tuning_report['threshold_by_class'] = {
                name: float(thr) for name, thr in zip(self.class_names, self.thresholds)
            }
        
        return self.tuning_report
    
    def predict(self, probs: np.ndarray, use_tuned: bool = True) -> np.ndarray:
        """Make predictions using tuned or default thresholds"""
        if use_tuned and self.is_tuned:
            return adjusted_argmax(probs, self.thresholds, rule=self.rule)
        else:
            return np.argmax(probs, axis=1)
    
    def save(self, filepath: Union[str, Path]):
        """Save tuned thresholds to file"""
        if not self.is_tuned:
            raise ValueError("No thresholds have been tuned yet")
        
        save_data = {
            'thresholds': self.thresholds.tolist(),
            'tuning_report': self.tuning_report,
            'class_names': self.class_names,
            'is_tuned': self.is_tuned,
            'rule': self.rule
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
        self.tuning_report = save_data.get('tuning_report', {})
        self.class_names = save_data.get('class_names')
        self.is_tuned = save_data.get('is_tuned', True)
        self.rule = save_data.get('rule', 'subtract')
