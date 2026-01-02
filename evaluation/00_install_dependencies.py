#!/usr/bin/env python3
"""
🔧 Install Required Dependencies for Evaluation Suite
=====================================================

Installs all required packages for running the evaluation scripts.

Usage:
    conda activate enarmgpu
    python evaluation/00_install_dependencies.py
"""

import subprocess
import sys

def install_package(package):
    """Install a package using pip"""
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

def main():
    print("="*70)
    print("🔧 Installing Evaluation Dependencies")
    print("="*70)
    
    packages = [
        "matplotlib",
        "seaborn",
        "scikit-learn",
        "sentence-transformers",
        "numpy"
    ]
    
    for package in packages:
        print(f"\n📦 Installing {package}...")
        try:
            install_package(package)
            print(f"✅ {package} installed successfully")
        except Exception as e:
            print(f"❌ Failed to install {package}: {e}")
    
    print("\n" + "="*70)
    print("✅ All dependencies installed!")
    print("="*70)

if __name__ == "__main__":
    main()
