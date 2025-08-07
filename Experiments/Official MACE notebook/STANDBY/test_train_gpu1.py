import warnings
warnings.filterwarnings("ignore")
import sys
import logging
import torch
import os
from pathlib import Path

# Set GPU 1 as the only visible device
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

# Add local mace folder to Python path BEFORE importing mace
local_mace_path = '/homes/ab3149/Documents/MLMI-MPhil-Thesis-MACE/mace'
if local_mace_path not in sys.path:
    sys.path.insert(0, local_mace_path)
    print(f"✅ Added local mace folder to Python path: {local_mace_path}")

# Now import mace modules (they will use the local version)
from mace.cli.run_train import main as mace_run_train_main

print(f"🔧 CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')}")
print(f"🔧 CUDA available: {torch.cuda.is_available()}")
print(f"🔧 Current device: {torch.cuda.current_device()}")
print(f"🔧 Device name: {torch.cuda.get_device_name()}")

import os
os.chdir('/homes/ab3149/Documents/MLMI-MPhil-Thesis-MACE')

def train_mace(config_file_path):
    logging.getLogger().handlers.clear()
    sys.argv = ["program", "--config", config_file_path]
    print("about to run mace_run_train_main")
    mace_run_train_main()

def _get_config_file_path():

    # Create the directory if it doesn't exist
    config_dir = Path('Experiments/Official MACE notebook/config')
    config_dir.mkdir(parents=True, exist_ok=True)

    # YAML content - explicitly set device to cuda:0 (which will be GPU 1)
    yaml_content = '''model: "MACE"
    num_channels: 32
    max_L: 0
    r_max: 4.0
    name: "mace01"
    model_dir: "Experiments/Official MACE notebook/MACE_models"
    log_dir: "Experiments/Official MACE notebook/MACE_models"
    checkpoints_dir: "Experiments/Official MACE notebook/MACE_models"
    results_dir: "Experiments/Official MACE notebook/MACE_models"
    train_file: "Experiments/Official MACE notebook/data/solvent_xtb_train_200.xyz"
    valid_fraction: 0.10
    test_file: "Experiments/Official MACE notebook/data/solvent_xtb_test.xyz"
    energy_key: "energy_xtb"
    forces_key: "forces_xtb"
    device: cuda:0
    batch_size: 10
    max_num_epochs: 100
    swa: True
    seed: 123
    default_dtype: float64
    '''

    # Write the file
    if not os.path.exists('Experiments/Official MACE notebook/config/config-gpu1.yml'):
        with open('Experiments/Official MACE notebook/config/config-gpu1.yml', 'w') as f:
            f.write(yaml_content)
    else:
        print("Config file already exists!")

    print("Config file created successfully!")


def main():
    _get_config_file_path()
    train_mace('Experiments/Official MACE notebook/config/config-gpu1.yml')

if __name__ == "__main__":
    main() 