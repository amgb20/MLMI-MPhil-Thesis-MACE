import logging
import sys
import warnings

import torch
import yaml

from mace.cli.run_train import main as mace_run_train_main
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

    # Build the model
    model = _build_model(args, config, None, ["Default"])

    # Attach shape logger to all interaction blocks
    print("\n===========ATTACHING SHAPE LOGGER===========")
    for i, interaction in enumerate(model.interactions):
        print(f"\nAttaching shape logger to interaction block {i+1}")
        interaction.attach_shape_debug()

    mace_run_train_main()


if __name__ == "__main__":
    print(f"PyTorch default dtype: {torch.get_default_dtype()}")
    # set default dtype to float64
    torch.set_default_dtype(torch.float64)
    print(f"PyTorch new default dtype: {torch.get_default_dtype()}")

    # verify that we are using cuda
    print(f"Is cuda: {torch.cuda.is_available()}")

    train_mace("Experiments/Official MACE notebook/config/config-02_cpu.yml")
