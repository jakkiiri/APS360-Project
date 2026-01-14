#!/usr/bin/env python3
"""
Utility functions for model loading and handling
"""

import torch


def safe_torch_load(checkpoint_path, map_location=None):
    """
    Safely load PyTorch checkpoint with proper handling of weights_only parameter
    
    Args:
        checkpoint_path: Path to the checkpoint file
        map_location: Device to map the checkpoint to
    
    Returns:
        Loaded checkpoint dictionary
    """
    try:
        # Try with weights_only=False for compatibility with saved optimizers/schedulers
        checkpoint = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
        return checkpoint
    except TypeError:
        # Fallback for older PyTorch versions that don't have weights_only parameter
        checkpoint = torch.load(checkpoint_path, map_location=map_location)
        return checkpoint
    except Exception as e:
        if "GLOBAL" in str(e) and "numpy.core" in str(e):
            print("⚠️  Warning: Checkpoint was saved with an older PyTorch version.")
            print("   Attempting to load with pickle protocol compatibility...")
            try:
                import pickle
                checkpoint = torch.load(
                    checkpoint_path, 
                    map_location=map_location, 
                    weights_only=False, 
                    pickle_module=pickle
                )
                return checkpoint
            except:
                print("   Using fallback loading method...")
                checkpoint = torch.load(checkpoint_path, map_location=map_location)
                return checkpoint
        else:
            raise e


def get_pytorch_version_info():
    """Get PyTorch version information for debugging"""
    return {
        'version': torch.__version__,
        'cuda_available': torch.cuda.is_available(),
        'cuda_version': torch.version.cuda if torch.cuda.is_available() else None,
    }


def print_model_info(model, checkpoint=None):
    """Print useful model information"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"📊 Model Information:")
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    
    if checkpoint:
        if 'epoch' in checkpoint:
            print(f"   Training epoch: {checkpoint['epoch']}")
        if 'best_val_f1_macro' in checkpoint:
            print(f"   Best validation F1 (macro): {checkpoint['best_val_f1_macro']:.4f}")
        if 'best_val_acc' in checkpoint:
            print(f"   Best validation accuracy: {checkpoint['best_val_acc']:.4f}")


def count_parameters(model) -> dict:
    """Count model parameters"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total': total_params,
        'trainable': trainable_params,
        'frozen': total_params - trainable_params
    }
