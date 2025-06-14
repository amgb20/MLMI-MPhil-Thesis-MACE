import warnings
warnings.filterwarnings("ignore")
from mace.cli.run_train import main as mace_run_train_main
import sys
import logging
import torch


def train_mace(config_file_path):
    logging.getLogger().handlers.clear()
    sys.argv = ["program", "--config", config_file_path]
    mace_run_train_main()

if __name__ == "__main__":
    print(f"PyTorch default dtype: {torch.get_default_dtype()}")
    # set default dtype to float64
    torch.set_default_dtype(torch.float64)
    print(f"PyTorch new default dtype: {torch.get_default_dtype()}")

    # verify that we are using cuda
    print(f"Is cuda: {torch.cuda.is_available()}")

    train_mace("Experiments/Official MACE notebook/config/config-02.yml")
