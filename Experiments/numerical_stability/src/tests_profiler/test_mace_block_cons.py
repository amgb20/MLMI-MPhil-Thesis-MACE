import os
import sys
import logging
import warnings
warnings.filterwarnings("ignore")
import logging
logging.basicConfig(level=logging.INFO)
import torch
from datetime import datetime
import tarfile
from torch.profiler import profile, ProfilerActivity, tensorboard_trace_handler

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.utils.get_tracing_table import save_tracing_table_with_openpyxl

from mace.cli.run_train import main as mace_run_train_main

TRACE_DIR = "profiling_traces/mace_run"

def train_mace(config_file_path):
    logging.getLogger().handlers.clear()
    sys.argv = ["program", "--config", config_file_path]
    print("about to run mace_run_train_main")
    mace_run_train_main()

def convert_profiler_to_tar_gz(tar_dir: str) -> None:
    # Find the file with the name ".pt.trace.json" in the tar_dir
    trace_file = next((f for f in os.listdir(tar_dir) if f.endswith('.pt.trace.json')), None)

    if trace_file:
        # Create tar.gz filename with proper extension
        tar_filename = os.path.join(tar_dir, f"{trace_file}.tar.gz")
        
        try:
            with tarfile.open(tar_filename, "w:gz") as tar:
                tar.add(os.path.join(tar_dir, trace_file), arcname=trace_file)
            logging.info(f"Trace file {trace_file} converted to tar.gz file: {tar_filename}")
        except Exception as e:
            logging.error(f"Failed to create tar.gz file: {e}")
    else:
        logging.error(f"No trace file found in {tar_dir}")

def run_with_profiler(config_path, trace_dir=TRACE_DIR, use_cuda=True):
    activities = [ProfilerActivity.CPU]
    if use_cuda and torch.cuda.is_available():
        activities.append(ProfilerActivity.CUDA)

    # Ensure directory exists
    os.makedirs(trace_dir, exist_ok=True)

    # create a directory for story this run profiler results
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    run_dir = f"Experiments/Official MACE notebook/profiling_traces/mace_tb/run_{timestamp}"
    os.makedirs(run_dir, exist_ok=True)

    # Create the xlsx directory for Excel files
    xlsx_dir = f"{run_dir}/xlsx"
    os.makedirs(xlsx_dir, exist_ok=True)

    with profile(
        activities=activities,
        # record_shapes=True,       # start light
        profile_memory=True,
        with_stack=False,
        with_modules=True,
        use_cuda = True,
        on_trace_ready=tensorboard_trace_handler(run_dir),
    ) as prof:
        with torch.profiler.record_function("MACE/TopLevel"):
            train_mace(config_path)

    convert_profiler_to_tar_gz(run_dir)

    # Optional: print a concise table right away
    logging.info("\n=== By CUDA time (kernels) ===")
    prof_table_cuda = prof.key_averages().table(sort_by="cuda_time_total", row_limit=50)
    logging.info(prof_table_cuda)
    with open(f"{xlsx_dir}/cuda_time_total_{timestamp}.txt", "w") as f:
        f.write(prof_table_cuda)
    save_tracing_table_with_openpyxl(prof_table_cuda, f"{run_dir}/cuda_time_total_{timestamp}.xlsx")

    logging.info("\n=== By CPU time (ops) ===")
    prof_table_cpu = prof.key_averages().table(sort_by="cpu_time_total", row_limit=50)  
    logging.info(prof_table_cpu)
    with open(f"{xlsx_dir}/cpu_time_total_{timestamp}.txt", "w") as f:
        f.write(prof_table_cpu)
    save_tracing_table_with_openpyxl(prof_table_cpu, f"{run_dir}/cpu_time_total_{timestamp}.xlsx")
        
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


### SQL Commands --------
# SELECT name, SUM(dur)/1e9 AS sec
# FROM slice
# GROUP BY name
# ORDER BY sec DESC
# LIMIT 50;
### 