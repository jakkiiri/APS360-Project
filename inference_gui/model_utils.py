#!/usr/bin/env python3
"""
Utility functions for model loading and handling
"""

import torch
import warnings

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
        # This is needed for checkpoints that contain more than just model weights
        checkpoint = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
        return checkpoint
    except TypeError:
        # Fallback for older PyTorch versions that don't have weights_only parameter
        checkpoint = torch.load(checkpoint_path, map_location=map_location)
        return checkpoint
    except Exception as e:
        # Handle other potential loading errors
        if "GLOBAL" in str(e) and "numpy.core" in str(e):
            # This is the specific pickle error mentioned
            print("⚠️  Warning: Checkpoint was saved with an older PyTorch version.")
            print("   Attempting to load with pickle protocol compatibility...")
            try:
                # Try loading with pickle_module specification for older checkpoints
                import pickle
                checkpoint = torch.load(checkpoint_path, map_location=map_location, 
                                      weights_only=False, pickle_module=pickle)
                return checkpoint
            except:
                # Final fallback - load without weights_only
                print("   Using fallback loading method...")
                checkpoint = torch.load(checkpoint_path, map_location=map_location)
                return checkpoint
        else:
            # Re-raise other exceptions
            raise e

def get_pytorch_version_info():
    """Get PyTorch version information for debugging"""
    return {
        'version': torch.__version__,
        'cuda_available': torch.cuda.is_available(),
        'cuda_version': torch.version.cuda if torch.cuda.is_available() else None,
        'supports_weights_only': hasattr(torch, 'load') and 'weights_only' in torch.load.__code__.co_varnames
    }

def print_model_info(model, checkpoint=None):
    """Print useful model information"""
    # Count parameters
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

def validate_checkpoint_format(checkpoint):
    """Validate that checkpoint has expected format"""
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            return 'full_checkpoint'  # Contains optimizer, scheduler, etc.
        else:
            return 'state_dict_only'  # Just model weights
    else:
        raise ValueError("Checkpoint should be a dictionary")

# Example usage and testing
if __name__ == "__main__":
    print("🔧 Model Utilities Test")
    print("=" * 30)
    
    # Print PyTorch version info
    version_info = get_pytorch_version_info()
    print("PyTorch Environment:")
    for key, value in version_info.items():
        print(f"   {key}: {value}")
    
    print("\nThis module provides:")
    print("   - safe_torch_load(): Handles weights_only parameter safely")
    print("   - get_pytorch_version_info(): Version compatibility checking")
    print("   - print_model_info(): Model statistics")
    print("   - validate_checkpoint_format(): Checkpoint validation")
