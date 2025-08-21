import os
import sys
import logging
import warnings
import functools
warnings.filterwarnings("ignore")
import logging
logging.basicConfig(level=logging.INFO)
import torch
from datetime import datetime
from torch.profiler import profile, ProfilerActivity, tensorboard_trace_handler

from mace.cli.run_train import main as mace_run_train_main

TRACE_DIR = "profiling_traces/mace_run"

def _wrap_forward(Cls, label):
    """Idempotently wrap Cls.forward with a named profiler range."""
    if not hasattr(Cls, "_orig_forward"):
        Cls._orig_forward = Cls.forward

        @functools.wraps(Cls._orig_forward)
        def wrapped(self, *args, **kwargs):
            with torch.profiler.record_function(f"MACE/{label}"):
                return Cls._orig_forward(self, *args, **kwargs)
        Cls.forward = wrapped

def enable_mace_block_ranges():
    # Model
    try:
        from mace.modules.models import ScaleShiftMACE
        _wrap_forward(ScaleShiftMACE, "ModelForward")
    except Exception:
        pass

    # Interaction blocks
    try:
        from mace.modules.blocks import InteractionBlock, RealAgnosticInteractionBlock
        _wrap_forward(InteractionBlock, "InteractionBlock")
        _wrap_forward(RealAgnosticInteractionBlock, "InteractionBlockAgnostic")
    except Exception:
        pass

    # Tensor product wrapper (may or may not be a typical nn.Module)
    try:
        from mace.modules import wrapper_ops
        if hasattr(wrapper_ops, "TensorProduct"):
            _wrap_forward(wrapper_ops.TensorProduct, "TensorProduct")
    except Exception:
        pass

    # Readout / product (names vary by version)
    try:
        from mace.modules.blocks import ProductBlock, ReadoutBlock
        _wrap_forward(ProductBlock, "ProductBlock")
        _wrap_forward(ReadoutBlock, "ReadoutBlock")
    except Exception:
        pass

    # Spherical harmonics may be a function in some versions
    try:
        from mace.modules.spherical_harmonics import SphericalHarmonics
        _wrap_forward(SphericalHarmonics, "SphericalHarmonics")
    except Exception:
        pass
    
def train_mace(config_file_path):
    logging.getLogger().handlers.clear()
    sys.argv = ["program", "--config", config_file_path]
    print("about to run mace_run_train_main")
    mace_run_train_main()

def run_with_profiler(config_path, trace_dir=TRACE_DIR, use_cuda=True):
    activities = [ProfilerActivity.CPU]
    if use_cuda and torch.cuda.is_available():
        activities.append(ProfilerActivity.CUDA)

    # Ensure directory exists
    os.makedirs(trace_dir, exist_ok=True)

    enable_mace_block_ranges()

    with profile(
        activities=activities,
        record_shapes=False,       # start light
        profile_memory=True,
        with_stack=False,
        with_modules=True,
        use_cuda = True,
        on_trace_ready=tensorboard_trace_handler("Experiments/Official MACE notebook/profiling_traces/mace_tb"),
    ) as prof:
        with torch.profiler.record_function("MACE/TopLevel"):
            train_mace(config_path)

    # prof.export_chrome_trace(os.path.join(trace_dir, "trace.json"))

    # Optional: print a concise table right away
    logging.info("\n=== By CUDA time (kernels) ===")
    prof_table_cuda = prof.key_averages().table(sort_by="cuda_time_total", row_limit=50)
    logging.info(prof_table_cuda)
    with open(f"Experiments/Official MACE notebook/profiling_traces/mace_tb/cuda_time_total.txt_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}", "w") as f:
        f.write(prof_table_cuda)

    logging.info("\n=== By CPU time (ops) ===")
    prof_table_cpu = prof.key_averages().table(sort_by="cpu_time_total", row_limit=50)  
    logging.info(prof_table_cpu)
    with open(f"Experiments/Official MACE notebook/profiling_traces/mace_tb/cpu_time_total.txt_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}", "w") as f:
        f.write(prof_table_cpu)

if __name__ == "__main__":
    import os
    os.chdir('/homes/ab3149/Documents/MLMI-MPhil-Thesis-MACE')
    cfg = "Experiments/Official MACE notebook/config/config-02_profiler_setup.yml"

    torch.set_default_dtype(torch.float32)
    if torch.cuda.is_available():
        # Check if TF32 is supported on this GPU
        tf32_supported = False
        try:
            major = torch.cuda.get_device_properties(0).major
            minor = torch.cuda.get_device_properties(0).minor
            # TF32 is supported on Ampere (compute capability 8.0+) and newer
            if major >= 8:
                tf32_supported = True
        except Exception as e:
            logging.info(f"Could not determine CUDA device properties: {e}")

        if tf32_supported:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            logging.info("TF32 is supported and enabled.")
        else:
            logging.info("TF32 is not supported on this GPU (compute capability < 8.0). Skipping TF32 enablement.")
        

    # ---- CUDA run ----
    # Default is CUDA if available
    run_with_profiler(cfg, trace_dir="profiling_traces/mace_cuda", use_cuda=True)
