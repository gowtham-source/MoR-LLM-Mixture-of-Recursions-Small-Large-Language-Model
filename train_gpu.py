#!/usr/bin/env python3
"""
GPU-optimized training script for MoR-SLM.
This script bypasses UV environment issues and forces GPU usage.
"""

import sys
import os
import subprocess

def check_gpu():
    """Check if GPU is available and force GPU usage."""
    try:
        import torch
        if not torch.cuda.is_available():
            print("❌ No CUDA GPU detected. Installing CUDA PyTorch...")
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", 
                "torch==2.5.1+cu121", "torchvision==0.20.1+cu121", 
                "--index-url", "https://download.pytorch.org/whl/cu121"
            ])
            print("✅ CUDA PyTorch installed. Please restart this script.")
            sys.exit(0)
        else:
            print(f"🚀 GPU detected: {torch.cuda.get_device_name(0)}")
            return True
    except ImportError:
        print("❌ PyTorch not found. Please install PyTorch first.")
        return False

if __name__ == "__main__":
    print("🔥 MoR-SLM GPU Training")
    print("=" * 50)
    
    # Check GPU availability
    if check_gpu():
        print("🚀 Starting GPU-accelerated training...")
        
        # Run the original training script
        script_path = os.path.join(os.path.dirname(__file__), "train.py")
        config_path = os.path.join(os.path.dirname(__file__), "configs", "small_model.yaml")
        
        cmd = [sys.executable, script_path, "--config", config_path]
        print(f"Running: {' '.join(cmd)}")
        subprocess.run(cmd)
