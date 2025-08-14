#!/usr/bin/env python3
"""
Quick script to check what image size your trained model expects
"""

import torch
import sys
import os

# Add the current directory to path
sys.path.append(os.path.dirname(__file__))

try:
    from model_utils import safe_torch_load
except ImportError:
    def safe_torch_load(checkpoint_path, map_location=None):
        try:
            return torch.load(checkpoint_path, map_location=map_location, weights_only=False)
        except TypeError:
            return torch.load(checkpoint_path, map_location=map_location)

def check_model_input_size(model_path):
    """Check what input size the model was trained with"""
    print(f"Checking model: {model_path}")
    
    try:
        checkpoint = safe_torch_load(model_path, map_location='cpu')
        
        # Check if config is saved in checkpoint
        if 'config' in checkpoint:
            config = checkpoint['config']
            if hasattr(config, 'image_size'):
                print(f"✅ Found image_size in config: {config.image_size}")
            elif 'image_size' in config:
                print(f"✅ Found image_size in config: {config['image_size']}")
            else:
                print("⚠️  No image_size found in config")
                print("Available config keys:", list(config.keys()) if isinstance(config, dict) else "Config is not a dict")
        else:
            print("⚠️  No config found in checkpoint")
        
        # Check checkpoint keys
        print("\nCheckpoint keys:")
        for key in checkpoint.keys():
            if key == 'model_state_dict':
                print(f"  ✅ {key}: {len(checkpoint[key])} parameters")
            else:
                print(f"  📋 {key}: {type(checkpoint[key])}")
        
        # Try to load model and test different input sizes
        print("\n🧪 Testing different input sizes...")
        
        from simple_inference import SwinTransformerClassifier
        
        # Test common sizes
        test_sizes = [224, 384, 512]
        
        for size in test_sizes:
            try:
                model = SwinTransformerClassifier(num_classes=9, pretrained=False, image_size=size)
                
                if 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    model.load_state_dict(checkpoint)
                
                # Test forward pass
                dummy_input = torch.randn(1, 3, size, size)
                with torch.no_grad():
                    output = model(dummy_input)
                    print(f"  ✅ Size {size}: SUCCESS - Output shape: {output.shape}")
                    
            except Exception as e:
                print(f"  ❌ Size {size}: FAILED - {str(e)[:100]}")
                
    except Exception as e:
        print(f"❌ Error loading checkpoint: {e}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python check_model_size.py <model.pth>")
        print("Example: python check_model_size.py ../outputs/best_f1_model.pth")
        sys.exit(1)
    
    model_path = sys.argv[1]
    check_model_input_size(model_path)
