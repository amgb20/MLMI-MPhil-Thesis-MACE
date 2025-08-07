#!/usr/bin/env python3
"""
Wrapper script to run MACE training with local mace folder
"""

import sys
import os
from pathlib import Path

# Add local mace folder to Python path
project_root = Path('/homes/ab3149/Documents/MLMI-MPhil-Thesis-MACE')
local_mace_path = project_root / 'mace'

if str(local_mace_path) not in sys.path:
    sys.path.insert(0, str(local_mace_path))
    print(f"✅ Added local mace folder to Python path: {local_mace_path}")

# Change to project directory
os.chdir(str(project_root))
print(f"📁 Changed to project directory: {project_root}")

# Now import and run the training script
from test_train import main

if __name__ == "__main__":
    main() 