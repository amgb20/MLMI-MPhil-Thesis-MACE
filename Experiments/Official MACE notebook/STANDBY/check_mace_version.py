#!/usr/bin/env python3
"""
Script to check which mace version is being used
"""

import sys
import os
from pathlib import Path

def check_mace_usage():
    """Check which mace version is being used"""
    
    print("🔍 Checking MACE usage...")
    
    # Check if local mace folder exists
    local_mace_path = Path('/homes/ab3149/Documents/MLMI-MPhil-Thesis-MACE/mace')
    if local_mace_path.exists():
        print(f"✅ Local mace folder exists: {local_mace_path}")
        
        # Check what's in the local mace folder
        print("📁 Contents of local mace folder:")
        for item in local_mace_path.iterdir():
            if item.is_dir():
                print(f"  📂 {item.name}/")
            else:
                print(f"  📄 {item.name}")
    else:
        print(f"❌ Local mace folder not found: {local_mace_path}")
    
    # Check Python path
    print(f"\n🐍 Python path:")
    for i, path in enumerate(sys.path):
        if 'mace' in path:
            print(f"  [{i}] {path}")
    
    # Try to import mace and check its location
    try:
        import mace
        print(f"\n📦 Imported mace from: {mace.__file__}")
        
        # Check if it's the local version
        if str(local_mace_path) in mace.__file__:
            print("✅ Using LOCAL mace version")
        else:
            print("⚠️  Using INSTALLED mace version")
            
    except ImportError as e:
        print(f"❌ Could not import mace: {e}")
    
    # Check specific modules
    try:
        from mace.cli import run_train
        print(f"📦 run_train module from: {run_train.__file__}")
    except ImportError as e:
        print(f"❌ Could not import run_train: {e}")

if __name__ == "__main__":
    check_mace_usage() 