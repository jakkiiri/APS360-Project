#!/usr/bin/env python3
"""
Simple launcher script for the Swin Transformer GUI
Checks dependencies and launches the GUI application
"""

import sys
import os
from pathlib import Path

def check_dependencies():
    """Check if all required dependencies are installed"""
    required_packages = {
        'torch': 'PyTorch',
        'torchvision': 'TorchVision',
        'timm': 'timm (PyTorch Image Models)',
        'albumentations': 'Albumentations',
        'PIL': 'Pillow',
        'numpy': 'NumPy',
        'matplotlib': 'Matplotlib'
    }
    
    missing = []
    
    for package, name in required_packages.items():
        try:
            __import__(package)
        except ImportError:
            missing.append(name)
    
    # Check tkinter separately (comes with Python but may be missing)
    try:
        import tkinter
    except ImportError:
        missing.append('tkinter (usually comes with Python)')
    
    return missing

def main():
    print("🚀 Swin Transformer GUI Launcher")
    print("=" * 40)
    
    # Check dependencies
    print("📦 Checking dependencies...")
    missing = check_dependencies()
    
    if missing:
        print("❌ Missing required packages:")
        for package in missing:
            print(f"   - {package}")
        print("\n💡 Install missing packages with:")
        print("   pip install torch torchvision timm albumentations pillow numpy matplotlib")
        print("   or: pip install -r requirements.txt")
        return 1
    
    print("✅ All dependencies found!")
    
    # Check if GUI file exists
    gui_file = Path(__file__).parent / "single_image_gui.py"
    if not gui_file.exists():
        print(f"❌ GUI file not found: {gui_file}")
        return 1
    
    print("🎯 Launching GUI...")
    
    try:
        # Import and run GUI
        sys.path.insert(0, str(Path(__file__).parent))
        from single_image_gui import main as gui_main
        gui_main()
        
    except ImportError as e:
        print(f"❌ Failed to import GUI: {e}")
        return 1
    except Exception as e:
        print(f"❌ GUI error: {e}")
        return 1
    
    print("👋 GUI closed")
    return 0

if __name__ == "__main__":
    sys.exit(main())
