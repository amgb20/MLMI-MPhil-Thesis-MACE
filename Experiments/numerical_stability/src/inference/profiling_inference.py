from mace.calculators import MACECalculator
from ase import build
import numpy as np
import torch
from torch.profiler import ProfilerActivity, profile, tensorboard_trace_handler
from torch.profiler import record_function
import torch.nn as nn
from torch.utils.benchmark import Timer
import tarfile
import os
import logging
logging.basicConfig(level=logging.INFO)
import argparse
from datetime import datetime
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.utils.get_tracing_table import save_tracing_table_with_openpyxl

try:
    import cuequivariance as cue  # pylint: disable=unused-import
    CUET_AVAILABLE = True
except ImportError:
    CUET_AVAILABLE = False

TRACE_DIR = "Experiments/numerical_stability/src/inference/results"

def parse_args():
    parser = argparse.ArgumentParser(description="MACE profiling with different calculation modes")
    parser.add_argument("--mode", 
                       choices=["energy_only", "energy_forces", "complete_calc", "inference_only"],
                       default="inference_only",
                       help="Calculation mode to profile")
    parser.add_argument("--supercell_size", type=int, default=2, help="Size of water box")
    parser.add_argument("--warmup", type=int, default=100, help="Number of warmup steps")
    parser.add_argument("--num_iters", type=int, default=100, help="Number of iterations")
    parser.add_argument("--tf32", action="store_true", help="Enable TF32")
    args = parser.parse_args()
    return args

def _monitor_gpu_usage():
    if torch.cuda.is_available():
        logging.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logging.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        logging.info(f"Current GPU Memory: {torch.cuda.memory_allocated(0) / 1e9:.3f} GB")
        logging.info(f"Peak GPU Memory: {torch.cuda.max_memory_allocated(0) / 1e9:.3f} GB")
        logging.info(f"GPU Memory Cached: {torch.cuda.memory_reserved(0) / 1e9:.3f} GB")

def tf32_status(args):
    if torch.cuda.is_available() and args.tf32 == True:
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
        logging.info("FP32 is used")

def make_directory():
    # Ensure directory exists
    os.makedirs(TRACE_DIR, exist_ok=True)

    # create a directory for story this run profiler results
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    run_dir = f"{TRACE_DIR}/run_{timestamp}"
    os.makedirs(run_dir, exist_ok=True)

    # Create the xlsx directory for Excel files
    xlsx_dir = f"{run_dir}/xlsx"
    os.makedirs(xlsx_dir, exist_ok=True)

    return run_dir, xlsx_dir

def _get_atoms_batch(mace_calc, atoms):
    batch = mace_calc._atoms_to_batch(atoms)
    batch = mace_calc._clone_batch(batch)
    return batch

def _get_energy_only(mace_calc, atoms):
    # Energy-only (no backward)
    with record_function("EnergyOnly/forward"):
        batch = mace_calc._atoms_to_batch(atoms)
        batch = mace_calc._clone_batch(batch)
        out_E = mace_calc.models[0](
            batch.to_dict(),
            compute_force=False,
            compute_virials=False,
            compute_stress=False,
            training=False,
            compute_edge_forces=False,
            compute_atomic_stresses=False,
        )
        energy = out_E["energy"]  # total E = e0 + inter_e
    return energy

def _get_energy_forces(mace_calc, atoms):
    # Energy+Forces (forward + backward)
    with record_function("Energy+Forces/forward+backward"):
        batch = mace_calc._atoms_to_batch(atoms)
        batch = mace_calc._clone_batch(batch)
        out_F = mace_calc.models[0](
            batch.to_dict(),
            compute_force=True,
            compute_virials=False,
            compute_stress=False,
            training=False,
            compute_edge_forces=False,
            compute_atomic_stresses=False,
        )
        forces = out_F["forces"]
    return forces

def _build_complete_calculations(mace_calc, atoms, warmup=False):
    atoms.calc = mace_calc
    with record_function("Infer/E+F"):
        atoms.calc.calculate(atoms, properties=["energy", "forces"], system_changes=atoms.calc.implemented_properties)

def _build_inference_only(mace_calc, atoms):
    atoms.calc = mace_calc
    atoms.calc.calculate(atoms, properties=["energy", "forces"], system_changes=atoms.calc.implemented_properties)
    torch.cuda.synchronize()

def _wrap_once(mod, label: str):
    if mod is None or getattr(mod, "_rf_wrapped", False):
        return
    orig = mod.forward
    def wrapped(*a, **k):
        with record_function(label):
            return orig(*a, **k)
    mod.forward = wrapped
    mod._rf_wrapped = True

def _wrap_seq(seq, base_label: str):
    if seq is None:
        return
    if isinstance(seq, (nn.ModuleList, nn.Sequential, list, tuple)):
        for i, m in enumerate(seq):
            _wrap_once(m, f"MACE/{base_label}[{i}]")
    else:
        _wrap_once(seq, f"MACE/{base_label}")

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

def run_inference(args, mace_calc, atoms):
    if args.mode == "energy_only":
        with torch.no_grad():
            _get_energy_only(mace_calc, atoms)
    elif args.mode == "energy_forces":
        # must NOT disable autograd here
        _get_energy_forces(mace_calc, atoms)
    elif args.mode == "complete_calc":
        _build_complete_calculations(mace_calc, atoms)
    elif args.mode == "inference_only":
        _build_inference_only(mace_calc, atoms)
    else:
        raise ValueError(f"Invalid mode: {args.mode}")

def label_blocks(model):
    # Embeddings and SH - wrap independently to avoid folding under one label
    _wrap_seq(getattr(model, "node_embedding", None), "NodeEmbedding")
    _wrap_seq(getattr(model, "radial_embedding", None), "RadialEmbedding")
    _wrap_seq(getattr(model, "spherical_harmonics", None), "SphericalHarmonics")
    _wrap_seq(getattr(model, "atomic_energies_fn", None), "AtomicEnergies")
    _wrap_once(getattr(model, "scale_shift", None), "ScaleShift")

    # Core computational blocks
    for i, blk in enumerate(getattr(model, "interactions", [])):
        _wrap_once(blk, f"MACE/Interaction[{i}]/Main")
        # Go deeper into interaction blocks
        if hasattr(blk, "linear_up"):
            _wrap_once(getattr(blk, "linear_up", None), f"MACE/Interaction[{i}]/LinearUp_Interaction")
        if hasattr(blk, "conv_tp"):
            _wrap_once(getattr(blk, "conv_tp", None), f"MACE/Interaction[{i}]/ConvTensorProduct")
        if hasattr(blk, "conv_tp_weights"):
            _wrap_once(getattr(blk, "conv_tp_weights", None), f"MACE/Interaction[{i}]/ConvTPWeights")
        if hasattr(blk, "linear"):
            _wrap_once(getattr(blk, "linear", None), f"MACE/Interaction[{i}]/Linear_Interaction")
        if hasattr(blk, "skip_tp"):
            _wrap_once(getattr(blk, "skip_tp", None), f"MACE/Interaction[{i}]/SkipTensorProduct")
        if hasattr(blk, "reshape"):
            _wrap_once(getattr(blk, "reshape", None), f"MACE/Interaction[{i}]/Reshape")

    for i, blk in enumerate(getattr(model, "products", [])):
        _wrap_once(blk, f"MACE/Product[{i}]/Main")
        # Go deeper into product blocks
        if hasattr(blk, "symmetric_contractions"):
            _wrap_once(getattr(blk, "symmetric_contractions", None), f"MACE/Product[{i}]/SymmetricContractions")
        if hasattr(blk, "linear"):
            _wrap_once(getattr(blk, "linear", None), f"MACE/Product[{i}]/Linear_Product")

    for i, blk in enumerate(getattr(model, "readouts", [])):
        _wrap_once(blk, f"MACE/Readout[{i}]/Main")
        # Go deeper into readout blocks
        if hasattr(blk, "linear"):
            _wrap_once(getattr(blk, "linear", None), f"MACE/Readout[{i}]/Linear_readout")
        if hasattr(blk, "linear_1"):
            _wrap_once(getattr(blk, "linear_1", None), f"MACE/Readout[{i}]/Linear1")
        if hasattr(blk, "non_linearity"):
            _wrap_once(getattr(blk, "non_linearity", None), f"MACE/Readout[{i}]/NonLinearity")
        if hasattr(blk, "linear_2"):
            _wrap_once(getattr(blk, "linear_2", None), f"MACE/Readout[{i}]/Linear2")

def main():
    args = parse_args()
    # Simple profiling of MACE calculator load and energy/forces evaluation
    activities = [ProfilerActivity.CPU]
    if torch.cuda.is_available():
        logging.info("CUDA is available")
        activities.append(ProfilerActivity.CUDA)
        device = 'cuda'
        torch.cuda.init()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.reset_accumulated_memory_stats()
        torch.cuda.synchronize()
    else:
        logging.info("CUDA is not available")
        device = 'cpu'

    tf32_status(args)

    run_dir, xlsx_dir = make_directory()

    # Setup (not profiled)
    with record_function("Setup/LoadMACECalculator"):
        mace_calc = MACECalculator(
            model_paths="Experiments/numerical_stability/src/inference/model/MACE-OFF24_medium.model",
            default_dtype="float64",
            enable_cueq=True,
            device=device,
        )
    model = mace_calc.models[0]
    atoms = build.bulk("C", "diamond", a=3.567) # unit cell of diamond , a dimension of unit cell
    water_box_size = args.supercell_size
    atoms = atoms.repeat((water_box_size,water_box_size,water_box_size))

    # Warmup (not profiled)
    for _ in range(args.warmup):
        run_inference(args, mace_calc, atoms)

    _monitor_gpu_usage()

    # Profile only the measured section
    with profile(
        activities=activities,
        profile_memory=True,
        with_stack=False,
        with_modules=False,
        on_trace_ready=tensorboard_trace_handler(run_dir),
    ) as prof:
        # label MACE blocks into the tracing
        label_blocks(model)

        # run inference and benchmark time
        timer = Timer(
            stmt="run_inference(args, mace_calc, atoms)",
            globals={
                "run_inference": run_inference,
                "args": args,
                "mace_calc": mace_calc,
                "atoms": atoms,
            },
        )
        measurement = timer.timeit(args.num_iters)
        print(f"Benchmark mean per iter for inference: {measurement.mean * 1e3:.3f} ms")

    _monitor_gpu_usage()

    sort_key = "cuda_time_total" if torch.cuda.is_available() else "cpu_time_total"
    logging.info("\n=== By CUDA time (kernels) ===")
    prof_table_cuda = prof.key_averages().table(sort_by=sort_key, row_limit=500)
    logging.info(prof_table_cuda)
    with open(f"{xlsx_dir}/cuda_time_total.txt", "w") as f:
        f.write(prof_table_cuda)
    save_tracing_table_with_openpyxl(prof_table_cuda, f"{xlsx_dir}/cuda_time_total.xlsx")

if __name__ == "__main__":

    os.chdir('/homes/ab3149/Documents/MLMI-MPhil-Thesis-MACE')

    main()