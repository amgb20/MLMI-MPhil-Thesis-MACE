# 1. STANDARD LIBRARY IMPORTS (alphabetical)
import copy
import gc
import logging
import types
import warnings
import sys
import os

# 2. THIRD-PARTY LIBRARY IMPORTS (alphabetical)
import ase.io
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional
from e3nn import o3
from e3nn.o3 import Irreps

# 3. LOCAL/APPLICATION IMPORTS (alphabetical)
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from src.utils.get_logging_profile import logger
from src.utils.config import data_prep, get_default_model_config
from src.utils.get_gpu_details import get_gpu_with_least_memory
from mace import data, modules, tools

# 4. CONDITIONAL IMPORTS (after main imports)
try:
    import cuequivariance as cue
    import cuequivariance_torch as cuet
    from src.utils.get_memory_allocation import start_record_memory_history, stop_record_memory_history, export_memory_snapshot
    cueq_available = True
    logger.info("✓ cuEquivariance library is available")
except ImportError:
    cueq_available = False
    logger.info("✗ cuEquivariance library is not available - cuEq will be disabled")

# 5. CONFIGURATION (after imports)
warnings.filterwarnings("ignore")