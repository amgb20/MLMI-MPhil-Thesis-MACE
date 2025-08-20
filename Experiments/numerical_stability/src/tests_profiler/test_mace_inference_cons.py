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
from ase.io import read
from aseMolec import extAtoms as ea
from torch.profiler import schedule, profile, ProfilerActivity, tensorboard_trace_handler
from torch import inference_mode

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.utils.get_tracing_table import save_tracing_table_with_openpyxl

from mace.calculators import MACECalculator

TRACE_DIR = "profiling_traces/mace_inference"

def inference_mace(atoms, model_path, device):
    atoms.calc = MACECalculator(
        model_paths=[model_path], 
        device=device, 
        default_dtype="float64",
    )
    return atoms.calc


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

def run_with_profiler(init_conf: str, model_path: str,trace_dir: str,  use_cuda=True, device: str = None):
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

    # build once - measure cold start
    with profile(
        activities=activities,
        # record_shapes=True,       # start light
        profile_memory=True,
        with_stack=False,
        with_modules=True,
        use_cuda = True,
        on_trace_ready=tensorboard_trace_handler(run_dir),
    ) as prof:
        with torch.profiler.record_function("Setup/LoadCalculator"):
            atoms_calc = inference_mace(init_conf,model_path, device)

    # Warm-up without measuring
    model_warmups = 5
    with inference_mode():
        for _ in range(model_warmups):
            atoms_calc.calculate(init_conf, properties=["energy", "forces"])

    # 3) Profile steady-state inference (per-call)
    sched = schedule(wait=2, warmup=2, active=20, repeat=1)
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=sched,
        record_shapes=True,
        profile_memory=True,
        on_trace_ready=tensorboard_trace_handler("./log/mace_infer"),
    ) as prof:
        with inference_mode():  # disables autograd & tensor versioning
            for i in range(50):  # your inference calls
                with torch.profiler.record_function("Infer/E+F"):
                    atoms_calc.calculate(init_conf, properties=["energy", "forces"])
                    # If you also need the values:
                    E = atoms_calc.results["energy"]
                    F = atoms_calc.results["forces"]
                prof.step()

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
    init_conf = ea.sel_by_info_val(read('Experiments/Official MACE notebook/data/solvent_molecs.xyz',':'), 'Nmols', 1)[0].copy()
    model_path = 'Experiments/Official MACE notebook/MACE_models/mace01_run-123_stagetwo.model'


    torch.set_default_dtype(torch.float32)
    if torch.cuda.is_available():
        # Check if TF32 is supported on this GPU
        tf32_supported = False
        device = 'cuda'
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
    else:
        device = 'cpu'

    # ---- CUDA run ----
    # Default is CUDA if available
    run_with_profiler(init_conf, model_path, trace_dir=TRACE_DIR, use_cuda=True, device=device)


### SQL Commands --------
# SELECT name, SUM(dur)/1e9 AS sec
# FROM slice
# GROUP BY name
# ORDER BY sec DESC
# LIMIT 50;
### 