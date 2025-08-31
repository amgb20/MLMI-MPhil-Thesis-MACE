# def create_config_file():
# num_interactions: 2
# num_channels: 128
# max_L: 1
# correlation: 3
# r_max: 6.0
# forces_weight: 1000
# energy_weight: 40
# energy_key: "energy"
# forces_key: "forces"
# weight_decay: 5e-10
# clip_grad: 1.0
# batch_size: 128
# valid_batch_size: 128
# max_num_epochs: 180
# scheduler_patience: 20
# patience: 50
# eval_interval: 1
# ema: true
# swa: true
# start_swa: 115
# swa_lr: 0.00025
# swa_forces_weight: 10
# num_workers: 16
# error_table: 'PerAtomMAE'
# default_dtype: "float64"
# device: cuda
# # seed: 1
# restart_latest: true
# save_cpu: true
# distributed: true
# enable_cueq: true
# interaction_first: "RealAgnosticInteractionBlock"
# interaction: "RealAgnosticResidualInteractionBlock"


import warnings
warnings.filterwarnings("ignore")
from mace.cli.run_train import run
from mace.tools.arg_parser import build_default_arg_parser
import argparse

def train_mace(config_file_path):
    # Create argument parser and parse config file
    parser = build_default_arg_parser()
    
    # Parse arguments from config file
    args = parser.parse_args(['--config', config_file_path])
    
    # Run training directly
    run(args)

# pass the scaleshift config file path
train_mace("Experiments/numerical_stability/src/training/config/config-fp64.yml")