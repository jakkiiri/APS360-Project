#!/usr/bin/env python3
"""
Simple wrapper script to run Swin Transformer inference with default paths
This script helps you run inference without having to specify all command line arguments
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

def find_data_directory():
    """Find the data directory automatically"""
    current_dir = Path.cwd()
    possible_paths = [
        current_dir / "DataSplit",
        current_dir / "DataSplit2", 
        current_dir.parent / "DataSplit",
        current_dir.parent / "DataSplit2",
        current_dir.parent / "Data Preprocessing" / "DataSplit",
    ]
    
    for path in possible_paths:
        if path.exists() and (path / "test").exists():
            return str(path)
    
    return None

def find_checkpoint():
    """Find the best checkpoint automatically"""
    current_dir = Path.cwd()
    possible_paths = [
        current_dir / "outputs" / "best_f1_model.pth",
        current_dir / "outputs" / "best_acc_model.pth",
        current_dir / "outputs" / "checkpoints" / "best_f1_model.pth",
        current_dir.parent / "outputs" / "best_f1_model.pth",
        current_dir.parent / "hpc_scripts_transformers" / "outputs" / "best_f1_model.pth",
    ]
    
    for path in possible_paths:
        if path.exists():
            return str(path)
    
    return None

def main():
    parser = argparse.ArgumentParser(description="Easy Swin Transformer Inference Runner")
    parser.add_argument("--checkpoint", type=str, help="Path to model checkpoint (auto-detect if not provided)")
    parser.add_argument("--data_dir", type=str, help="Path to data directory (auto-detect if not provided)")
    parser.add_argument("--output_dir", type=str, default="./inference_results", help="Output directory")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--tune_thresholds", action="store_true", help="Enable threshold tuning")
    parser.add_argument("--gpu", action="store_true", help="Force GPU usage check")
    
    args = parser.parse_args()
    
    print("🔍 Swin Transformer Inference Runner")
    print("=" * 50)
    
    # Find checkpoint
    checkpoint_path = args.checkpoint
    if not checkpoint_path:
        print("🔍 Auto-detecting checkpoint...")
        checkpoint_path = find_checkpoint()
        if not checkpoint_path:
            print("❌ Could not find checkpoint automatically. Please specify with --checkpoint")
            print("   Looking for files like: outputs/best_f1_model.pth")
            return 1
        print(f"✅ Found checkpoint: {checkpoint_path}")
    else:
        if not os.path.exists(checkpoint_path):
            print(f"❌ Checkpoint not found: {checkpoint_path}")
            return 1
    
    # Find data directory
    data_dir = args.data_dir
    if not data_dir:
        print("🔍 Auto-detecting data directory...")
        data_dir = find_data_directory()
        if not data_dir:
            print("❌ Could not find data directory automatically. Please specify with --data_dir")
            print("   Looking for directories like: DataSplit/ (with test/ subfolder)")
            return 1
        print(f"✅ Found data directory: {data_dir}")
    else:
        if not os.path.exists(data_dir):
            print(f"❌ Data directory not found: {data_dir}")
            return 1
        if not os.path.exists(os.path.join(data_dir, "test")):
            print(f"❌ Test subdirectory not found in: {data_dir}")
            return 1
    
    # Check for GPU if requested
    if args.gpu:
        try:
            import torch
            if torch.cuda.is_available():
                print(f"✅ GPU available: {torch.cuda.get_device_name(0)}")
            else:
                print("⚠️  GPU not available, will use CPU")
        except ImportError:
            print("⚠️  PyTorch not available, cannot check GPU")
    
    # Build command
    script_dir = Path(__file__).parent
    inference_script = script_dir / "swin_inference.py"
    
    cmd = [
        sys.executable, str(inference_script),
        "--checkpoint_path", checkpoint_path,
        "--data_dir", data_dir,
        "--output_dir", args.output_dir,
        "--batch_size", str(args.batch_size)
    ]
    
    if args.tune_thresholds:
        cmd.append("--tune_thresholds")
    
    print(f"\n🚀 Running inference...")
    print(f"   Checkpoint: {checkpoint_path}")
    print(f"   Data: {data_dir}")
    print(f"   Output: {args.output_dir}")
    print(f"   Batch size: {args.batch_size}")
    print(f"   Threshold tuning: {args.tune_thresholds}")
    print()
    
    # Run the inference script
    try:
        result = subprocess.run(cmd, check=True)
        print("\n✅ Inference completed successfully!")
        print(f"📁 Results saved to: {args.output_dir}")
        return 0
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Inference failed with exit code: {e.returncode}")
        return e.returncode
    except KeyboardInterrupt:
        print("\n⏹️  Inference interrupted by user")
        return 1

if __name__ == "__main__":
    sys.exit(main())
