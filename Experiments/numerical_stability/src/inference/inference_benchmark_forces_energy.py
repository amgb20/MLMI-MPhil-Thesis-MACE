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
import csv
from tqdm import tqdm
from contextlib import nullcontext

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.utils.get_tracing_table import save_tracing_table_with_openpyxl
from src.utils.get_model_audition import normalize_fp64_only, audit_model_dtypes

try:
    import cuequivariance as cue  # pylint: disable=unused-import
    CUET_AVAILABLE = True
except ImportError:
    CUET_AVAILABLE = False

RUN_DIR = "Experiments/numerical_stability/src/inference/results"

def parse_args():
    parser = argparse.ArgumentParser(description="MACE profiling with different calculation modes")
    parser.add_argument("--mode", 
                       choices=["energy_only", "energy_forces", "inference_only"],
                       default="inference_only",
                       help="Calculation mode to profile")
    parser.add_argument("--supercell_size", type=int, default=8, help="Size of water box")
    parser.add_argument("--warmup", type=int, default=100, help="Number of warmup steps")
    parser.add_argument("--num_iters", type=int, default=100, help="Number of iterations")
    parser.add_argument("--tf32", action="store_true", help="Enable TF32")
    parser.add_argument("--default_dtype", type=str, default="float64", help="Default dtype for MACECalculator")
    parser.add_argument("--layer_default_dtype", type=str, default="float64", help="Default dtype for layers")
    parser.add_argument("--enable_cueq", action="store_true", help="Enable CUEQ")
    parser.add_argument("--autocast", action="store_true", help="Enable autocast")
    args = parser.parse_args()
    return args

def make_directory():
    # Ensure directory exists
    os.makedirs(RUN_DIR, exist_ok=True)

    # create a directory for story this run profiler results
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    run_dir = f"{RUN_DIR}/run_{timestamp}"
    os.makedirs(run_dir, exist_ok=True)

    return run_dir

def _get_atoms_batch(mace_calc, atoms):
    batch = mace_calc._atoms_to_batch(atoms)
    batch = mace_calc._clone_batch(batch)
    return batch

def _get_energy_only(mace_calc, atoms):
    # Energy-only (no backward)
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

def eval_energy_forces(mace_calc, atoms):
    """Return total energy (float, eV) and forces (np.ndarray, shape [N,3], eV/Å)."""
    with torch.no_grad():
        atoms = atoms.copy()           # avoid mutating the original
        atoms.calc = mace_calc         # attach calculator
        E = atoms.get_potential_energy()
        F = atoms.get_forces()

    E_per_atom = E/len(atoms)
    return float(E),float(E_per_atom), np.asarray(F)


def _build_inference_only(mace_calc, atoms, autocast_context):
    atoms.calc = mace_calc
    if autocast_context:
        with autocast_context:
            atoms.calc.calculate(atoms, properties=["energy", "forces"], system_changes=atoms.calc.implemented_properties)
    else:
        atoms.calc.calculate(atoms, properties=["energy", "forces"], system_changes=atoms.calc.implemented_properties)
    torch.cuda.synchronize()
    return atoms

def run_inference(args, mace_calc, atoms, autocast_context):
    if args.mode == "energy_only":
        with torch.no_grad():
            _get_energy_only(mace_calc, atoms, autocast_context)
    elif args.mode == "energy_forces":
        # must NOT disable autograd here
        _get_energy_forces(mace_calc, atoms, autocast_context)
    elif args.mode == "inference_only":
        atoms = _build_inference_only(mace_calc, atoms, autocast_context)
        return atoms
    else:
        raise ValueError(f"Invalid mode: {args.mode}")

def get_result_summary(args, device, energy_list, energy_per_atom_list, forces_list, time_list, run_dir):

    if args.enable_cueq:
        backend = "CUEQ"
    else:
        backend = "E3NN"

    logging.info(f'========================================')
    logging.info(f'Result summary')
    logging.info(f'========================================')
    logging.info(f'Device: {device}')
    logging.info(f'Default dtype: {args.default_dtype}')
    logging.info(f'Layer default dtype: {args.layer_default_dtype}')
    logging.info(f'Backend: {backend}')
    logging.info(f'========================================')
    # print result summary
    inference_time = sum(time_list)/len(time_list)
    # inference standard deviation
    inference_time_std = np.std(time_list)
    logging.info(f"Benchmark mean per iter for inference: {inference_time:.3f} ± {inference_time_std:.3f} ms")

    # Compute average Energy, force, and time
    E_avg = np.mean(energy_list)
    E_std = np.std(energy_list)
    E_per_atom_avg = np.mean(energy_per_atom_list)
    E_per_atom_std = np.std(energy_per_atom_list)
    F_avg = np.mean(forces_list, axis=0)
    
    # Compute force magnitude for each iteration
    force_magnitudes = [np.mean(np.linalg.norm(F, axis=1)) for F in forces_list]
    F_mag_std = np.std(force_magnitudes)  # Scalar std of force magnitudes
    
    logging.info(f"Average energy: {E_avg:.6f} ± {E_std:.6f} eV")
    mean_force = np.mean(np.linalg.norm(F_avg, axis=1))
    mean_force_str = f"{mean_force:.6e}"
    logging.info(f"Average force magnitude: {mean_force_str} ± {F_mag_std:.6e} eV/Å")

    # Save the results to a csv file
    with open(f"{run_dir}/result_summary.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(["Backend", "Default dtype", "Layer default dtype", "Avg Inference Time (ms)", "Inference Time Std (ms)", "Avg Energy (eV)", "Energy Std (eV)", "Avg Energy per atom (eV/atom)", "Energy per atom Std (eV/atom)", "Avg Force (eV/Å)", "Force Std (eV/Å)"])
        writer.writerow([backend, args.default_dtype, args.layer_default_dtype, inference_time, inference_time_std, E_avg, E_std, E_per_atom_avg, E_per_atom_std, mean_force_str, F_mag_std])

def summarize_model_dtypes(model):
    from collections import Counter
    c = Counter(p.dtype for p in model.parameters())
    b = Counter(b.dtype for b in model.buffers())
    logging.info(f"Param dtypes: {c}")
    logging.info(f"Buffer dtypes: {b}")

def main():

    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    # torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    args = parse_args()
    # Simple profiling of MACE calculator load and energy/forces evaluation
    if torch.cuda.is_available():
        logging.info("CUDA is available")
        device = 'cuda'
        torch.cuda.init()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.reset_accumulated_memory_stats()
        torch.cuda.synchronize()
    else:
        logging.info("CUDA is not available")
        device = 'cpu'

    logging.info(f"Get normal default dtype: {torch.get_default_dtype()}")

    run_dir = make_directory()

    if args.autocast:
        logging.info("Using autocast")
        autocast_context = torch.amp.autocast(device_type=device)
    else:
        logging.info("Not using autocast")
        autocast_context = nullcontext()

    # Setup (not profiled)
    mace_calc = MACECalculator(
        model_paths="Experiments/numerical_stability/src/inference/model/MACE-OFF24_medium.model",
        default_dtype=args.default_dtype,
        enable_cueq=args.enable_cueq,
        device=device,
        layer_default_dtype=args.layer_default_dtype,
    )

    audit_model_dtypes(mace_calc.models[0])
    if args.default_dtype == "float32":
        normalize_fp64_only(mace_calc.models[0], default_target=torch.float32)  # keep FP16 layers intact
        audit_model_dtypes(mace_calc.models[0])               # should now be pure float32             # should now be pure float64

    summarize_model_dtypes(mace_calc.models[0])

    logging.info(f"Get normal default dtype AFTER: {torch.get_default_dtype()}")


    atoms = build.bulk("C", "diamond", a=3.567) # unit cell of diamond , a dimension of unit cell
    water_box_size = args.supercell_size
    atoms = atoms.repeat((water_box_size,water_box_size,water_box_size))

    # Warmup (not profiled)
    for _ in tqdm(range(args.warmup)):
        run_inference(args, mace_calc, atoms, autocast_context)

    time_list = []
    energy_list = []
    forces_list = []
    energy_per_atom_list = []
    for _ in tqdm(range(args.num_iters)):
        if device.startswith("cuda"):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            atoms = run_inference(args, mace_calc, atoms, autocast_context)
            end.record()
            torch.cuda.synchronize()
            time = start.elapsed_time(end)

        E, E_per_atom, F = eval_energy_forces(mace_calc, atoms)
        energy_list.append(E)
        energy_per_atom_list.append(E_per_atom)
        forces_list.append(F)
        time_list.append(time)


    # print result summary
    get_result_summary(args, device,energy_list, energy_per_atom_list, forces_list, time_list, run_dir)
    
if __name__ == "__main__":

    os.chdir('/homes/ab3149/Documents/MLMI-MPhil-Thesis-MACE')

    main()