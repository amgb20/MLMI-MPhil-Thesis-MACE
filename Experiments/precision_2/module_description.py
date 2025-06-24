import mace
import torch
import copy
import logging
import numpy as np
from typing import Dict, List, Tuple

# Configure logging to show INFO level messages
logging.basicConfig(level=logging.INFO, format='%(message)s')

# TODO: check that cuEq is enabled
# TODO: check that you can enter conv_tp and conv_fusion

from mace.calculators import mace_off
from ase import build

atoms = build.molecule('H2O')

# check if cuda is available and set the device to cuda
if torch.cuda.is_available():
    device = 'cuda'
else:   
    device = 'cpu'

calc = mace_off(model="medium", device=device)
logging.info(f"Device: {device}")
atoms.calc = calc

ssmace = calc.models
model = ssmace[0]

logging.info("\n==== Model ====")
logging.info(model)