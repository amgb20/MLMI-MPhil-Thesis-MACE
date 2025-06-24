import logging
import sys
import warnings

import torch
import yaml
from run_precision import main as mace_run_precision_main

from mace.modules import interaction_classes
from mace.tools.arg_parser import build_default_arg_parser
from mace.tools.model_script_utils import _build_model

warnings.filterwarnings("ignore")


def train_mace(config_file_path):
    logging.getLogger().handlers.clear()
    sys.argv = ["program", "--config", config_file_path]

    # Parse arguments
    parser = build_default_arg_parser()
    args = parser.parse_args()

    # Load config file
    with open(config_file_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Print the interaction block being used
    print("\n===========INTERACTION BLOCKS===========")
    print(f"First layer interaction block: {args.interaction_first}")
    print(f"Subsequent layers interaction block: {args.interaction}")

    # Print the actual class being used
    print("\nActual classes:")
    print(f"First layer: {interaction_classes[args.interaction_first].__name__}")
    print(f"Subsequent layers: {interaction_classes[args.interaction].__name__}")

    mace_run_precision_main()


if __name__ == "__main__":
    print(f"PyTorch default dtype: {torch.get_default_dtype()}")
    # set default dtype to float64
    torch.set_default_dtype(torch.float64)
    print(f"PyTorch new default dtype: {torch.get_default_dtype()}")

    # verify that we are using cuda
    print(f"Is cuda: {torch.cuda.is_available()}")

    train_config_file = "Experiments/Official MACE notebook/config/cg-prec-iter_cuda_noCuEq_conv_fusion.yml"

    # add debug flag for showing block sizes
    debug_block_sizes_flag = False

    #  add argument to the config file to show block sizes
    with open(train_config_file, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    config["show_block_sizes"] = debug_block_sizes_flag
    with open(train_config_file, "w", encoding="utf-8") as f:
        yaml.dump(config, f)

    train_mace(train_config_file)
