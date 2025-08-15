import os
import sys
import logging
import warnings
warnings.filterwarnings("ignore")

import torch
from torch.profiler import profile, ProfilerActivity, schedule, tensorboard_trace_handler

from mace.cli.run_train import main as mace_run_train_main

TRACE_DIR = "profiling_traces/mace_run"

def train_mace(config_file_path):
    logging.getLogger().handlers.clear()
    sys.argv = ["program", "--config", config_file_path]
    print("about to run mace_run_train_main")
    mace_run_train_main()

def run_with_profiler(config_path, trace_dir=TRACE_DIR, use_cuda=True):
    activities = [ProfilerActivity.CPU]
    if use_cuda and torch.cuda.is_available():
        activities.append(ProfilerActivity.CUDA)

    # A short schedule so traces don’t get gigantic; tweak to your epoch/step length.
    prof_schedule = schedule(wait=2, warmup=2, active=6, repeat=1)  # total 10 steps captured

    # Ensure directory exists
    os.makedirs(trace_dir, exist_ok=True)

    with profile(
        activities=activities,
        record_shapes=True,       # start light
        profile_memory=True,
        with_stack=False,
        with_modules=True,
        on_trace_ready=tensorboard_trace_handler("Experiments/Official MACE notebook/profiling_traces/mace_tb"),
    ) as prof:
        with torch.profiler.record_function("MACE/TopLevel"):
            train_mace(config_path)

    # prof.export_chrome_trace(os.path.join(trace_dir, "trace.json"))

    # Optional: print a concise table right away
    print("\n=== By CUDA time (kernels) ===")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=50))
    print("\n=== By CPU time (ops) ===")
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=50))

if __name__ == "__main__":
    import os
    os.chdir('/homes/ab3149/Documents/MLMI-MPhil-Thesis-MACE')
    cfg = "Experiments/Official MACE notebook/config/config-02.yml"

    torch.set_default_dtype(torch.float32)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # ---- CUDA run ----
    # Default is CUDA if available
    run_with_profiler(cfg, trace_dir="profiling_traces/mace_cuda", use_cuda=True)

    # ---- CPU run ----
    # Quick way to force CPU without touching the YAML:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    run_with_profiler(cfg, trace_dir="profiling_traces/mace_cpu", use_cuda=False)
