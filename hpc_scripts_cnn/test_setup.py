#!/usr/bin/env python3
"""
Quick test script to verify the HPC setup and dependencies
Run this after setting up the virtual environment to ensure everything works
"""

import sys
import os

def test_imports():
    """Test all required imports"""
    print("Testing imports...")
    
    try:
        import torch
        print(f"✅ PyTorch {torch.__version__}")
        print(f"   CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"   CUDA version: {torch.version.cuda}")
            print(f"   GPU count: {torch.cuda.device_count()}")
            if torch.cuda.device_count() > 0:
                print(f"   GPU 0: {torch.cuda.get_device_name(0)}")
    except ImportError as e:
        print(f"❌ PyTorch import failed: {e}")
        return False
    
    try:
        import torchvision
        print(f"✅ TorchVision {torchvision.__version__}")
    except ImportError as e:
        print(f"❌ TorchVision import failed: {e}")
        return False
    
    try:
        import timm
        print(f"✅ timm {timm.__version__}")
    except ImportError as e:
        print(f"❌ timm import failed: {e}")
        return False
    
    try:
        import numpy as np
        print(f"✅ NumPy {np.__version__}")
    except ImportError as e:
        print(f"❌ NumPy import failed: {e}")
        return False
    
    try:
        import pandas as pd
        print(f"✅ Pandas {pd.__version__}")
    except ImportError as e:
        print(f"❌ Pandas import failed: {e}")
        return False
    
    try:
        import matplotlib
        matplotlib.use('Agg')  # Set non-interactive backend
        import matplotlib.pyplot as plt
        print(f"✅ Matplotlib {matplotlib.__version__} (backend: {matplotlib.get_backend()})")
    except ImportError as e:
        print(f"❌ Matplotlib import failed: {e}")
        return False
    
    try:
        import seaborn as sns
        print(f"✅ Seaborn {sns.__version__}")
    except ImportError as e:
        print(f"❌ Seaborn import failed: {e}")
        return False
    
    try:
        from sklearn.metrics import accuracy_score
        print(f"✅ Scikit-learn")
    except ImportError as e:
        print(f"❌ Scikit-learn import failed: {e}")
        return False
    
    try:
        from tqdm import tqdm
        print(f"✅ tqdm")
    except ImportError as e:
        print(f"❌ tqdm import failed: {e}")
        return False
    
    try:
        from PIL import Image
        print(f"✅ Pillow")
    except ImportError as e:
        print(f"❌ Pillow import failed: {e}")
        return False
    
    try:
        import albumentations as A
        from albumentations.pytorch import ToTensorV2
        print(f"✅ Albumentations {A.__version__}")
    except ImportError as e:
        print(f"❌ Albumentations import failed: {e}")
        return False
    
    try:
        import cv2
        print(f"✅ OpenCV {cv2.__version__}")
    except ImportError as e:
        print(f"❌ OpenCV import failed: {e}")
        return False
    
    return True

def test_model_creation():
    """Test basic model creation"""
    print("\nTesting model creation...")
    
    try:
        import torch
        import timm
        
        # Test ConvNeXt creation
        print("Creating ConvNeXt-Tiny...")
        cnn = timm.create_model('convnext_tiny.fb_in22k', pretrained=False, features_only=True)
        print(f"✅ ConvNeXt-Tiny created, features: {len(cnn.feature_info)} levels")
        
        # Test DeiT creation
        print("Creating DeiT-III Small...")
        deit = timm.create_model('deit3_small_patch16_224', pretrained=False, img_size=224)
        deit.reset_classifier(0)
        print(f"✅ DeiT-III Small created, features: {deit.num_features}")
        
        # Test forward pass with dummy data
        print("Testing forward pass...")
        dummy_input = torch.randn(1, 3, 224, 224)
        
        with torch.no_grad():
            cnn_feats = cnn(dummy_input)[-1]  # Last feature level
            deit_feats = deit.forward_features(dummy_input)
            if deit_feats.ndim == 3:
                deit_feats = deit_feats[:, 0, :]  # CLS token
        
        print(f"✅ Forward pass successful")
        print(f"   ConvNeXt features shape: {cnn_feats.shape}")
        print(f"   DeiT features shape: {deit_feats.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Model creation failed: {e}")
        return False

def test_data_transforms():
    """Test data transformations including Albumentations"""
    print("\nTesting data transforms...")
    
    try:
        import torch
        from torchvision import transforms
        from PIL import Image
        import numpy as np
        import albumentations as A
        from albumentations.pytorch import ToTensorV2
        
        # Create dummy image
        dummy_img = Image.fromarray(np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8))
        
        # Test cache transform
        cache_transform = transforms.Compose([
            transforms.Resize((320, 320)),
            transforms.ToTensor()
        ])
        
        cached = cache_transform(dummy_img)
        print(f"✅ Cache transform: {cached.shape}")
        
        # Test albumentations base augmentations
        base_aug = A.Compose([
            A.Resize(224, 224),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.7),
            A.ColorJitter(p=0.6),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
        
        # Convert PIL to numpy for albumentations
        img_np = np.array(dummy_img)
        base_transformed = base_aug(image=img_np)
        print(f"✅ Albumentations base transform: {base_transformed['image'].shape}")
        
        # Test minority class augmentations
        minority_aug = A.Compose([
            A.Resize(224, 224),
            A.HorizontalFlip(p=0.6),
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.8),
            A.HueSaturationValue(p=0.6),
            A.CLAHE(p=0.5),
            A.GaussNoise(p=0.4),
            A.CoarseDropout(max_holes=12, max_height=40, max_width=40, p=0.4),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
        
        minority_transformed = minority_aug(image=img_np)
        print(f"✅ Albumentations minority transform: {minority_transformed['image'].shape}")
        
        # Test validation transform
        val_aug = A.Compose([
            A.Resize(224, 224),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
        
        val_transformed = val_aug(image=img_np)
        print(f"✅ Albumentations validation transform: {val_transformed['image'].shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Transform test failed: {e}")
        return False

def test_amp():
    """Test Automatic Mixed Precision"""
    print("\nTesting AMP...")
    
    try:
        import torch
        from torch.amp import autocast, GradScaler
        
        if torch.cuda.is_available():
            device = torch.device('cuda')
            print(f"✅ Testing AMP on CUDA")
        else:
            device = torch.device('cpu')
            print(f"✅ Testing AMP on CPU")
        
        # Create simple model and data
        model = torch.nn.Linear(10, 5).to(device)
        data = torch.randn(4, 10).to(device)
        target = torch.randint(0, 5, (4,)).to(device)
        
        # Test AMP
        scaler = GradScaler(enabled=torch.cuda.is_available())
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        
        with autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu', 
                     enabled=torch.cuda.is_available()):
            output = model(data)
            loss = torch.nn.functional.cross_entropy(output, target)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        print(f"✅ AMP test passed")
        return True
        
    except Exception as e:
        print(f"❌ AMP test failed: {e}")
        return False

def main():
    print("HPC CNN Training Setup Test")
    print("=" * 40)
    
    print(f"Python version: {sys.version}")
    print(f"Python executable: {sys.executable}")
    print(f"Current working directory: {os.getcwd()}")
    
    # Test environment variables
    print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')}")
    print(f"MPLBACKEND: {os.environ.get('MPLBACKEND', 'Not set')}")
    
    print("\n" + "=" * 40)
    
    all_passed = True
    
    # Run tests
    if not test_imports():
        all_passed = False
    
    if not test_model_creation():
        all_passed = False
    
    if not test_data_transforms():
        all_passed = False
    
    if not test_amp():
        all_passed = False
    
    print("\n" + "=" * 40)
    
    if all_passed:
        print("🎉 All tests passed! Setup is ready for training.")
        return 0
    else:
        print("❌ Some tests failed. Please check your environment setup.")
        return 1

if __name__ == "__main__":
    exit(main())
