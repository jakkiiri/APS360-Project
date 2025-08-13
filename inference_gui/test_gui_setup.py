#!/usr/bin/env python3
"""
Quick test script to verify GUI setup and dependencies
"""

import sys
import os
from pathlib import Path

def test_basic_imports():
    """Test basic Python packages"""
    print("🔍 Testing basic imports...")
    
    try:
        import tkinter as tk
        print("  ✅ tkinter")
    except ImportError:
        print("  ❌ tkinter - Install Python with tkinter support")
        return False
    
    try:
        import numpy as np
        print("  ✅ numpy")
    except ImportError:
        print("  ❌ numpy - Run: pip install numpy")
        return False
    
    try:
        from PIL import Image, ImageTk
        print("  ✅ PIL (Pillow)")
    except ImportError:
        print("  ❌ PIL - Run: pip install Pillow")
        return False
    
    return True

def test_ml_imports():
    """Test ML-specific packages"""
    print("\n🔍 Testing ML packages...")
    
    try:
        import torch
        print(f"  ✅ PyTorch {torch.__version__}")
        if torch.cuda.is_available():
            print(f"      GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("      CPU only")
    except ImportError:
        print("  ❌ PyTorch - Run: pip install torch torchvision")
        return False
    
    try:
        import timm
        print("  ✅ timm")
    except ImportError:
        print("  ❌ timm - Run: pip install timm")
        return False
    
    try:
        import albumentations as A
        print("  ✅ albumentations")
    except ImportError:
        print("  ❌ albumentations - Run: pip install albumentations")
        return False
    
    return True

def test_viz_imports():
    """Test visualization packages"""
    print("\n🔍 Testing visualization packages...")
    
    try:
        import matplotlib.pyplot as plt
        print("  ✅ matplotlib")
    except ImportError:
        print("  ❌ matplotlib - Run: pip install matplotlib")
        return False
    
    try:
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        print("  ✅ matplotlib tkinter backend")
    except ImportError:
        print("  ❌ matplotlib tkinter backend")
        return False
    
    return True

def test_gui_components():
    """Test GUI component creation"""
    print("\n🔍 Testing GUI components...")
    
    try:
        sys.path.append(str(Path(__file__).parent))
        from single_image_gui import SwinTransformerClassifier, CLASS_NAMES
        
        print(f"  ✅ GUI classes imported")
        print(f"  ✅ {len(CLASS_NAMES)} disease classes defined")
        
        # Test model creation (without pretrained weights)
        model = SwinTransformerClassifier(
            num_classes=len(CLASS_NAMES),
            pretrained=False
        )
        print("  ✅ Model architecture created")
        
        return True
        
    except Exception as e:
        print(f"  ❌ GUI component test failed: {e}")
        return False

def main():
    print("🧪 GUI Setup Test")
    print("=" * 40)
    
    tests = [
        ("Basic Imports", test_basic_imports),
        ("ML Packages", test_ml_imports), 
        ("Visualization", test_viz_imports),
        ("GUI Components", test_gui_components)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        if test_func():
            passed += 1
        else:
            print(f"\n❌ {test_name} failed")
    
    print(f"\n📊 Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n✅ All tests passed! Your GUI is ready to use.")
        print("\n🚀 To launch the GUI:")
        print("   python run_gui.py")
        print("   or")
        print("   python single_image_gui.py")
        
        print("\n📋 What you need:")
        print("   1. A trained model file (.pth)")
        print("   2. Skin lesion images (JPG/PNG)")
        print("   3. Click and classify!")
        
    else:
        print(f"\n❌ {total - passed} test(s) failed.")
        print("\n💡 To fix missing packages:")
        print("   pip install torch torchvision timm albumentations pillow matplotlib numpy")
        print("   or")
        print("   pip install -r requirements.txt")
    
    return 0 if passed == total else 1

if __name__ == "__main__":
    sys.exit(main())
