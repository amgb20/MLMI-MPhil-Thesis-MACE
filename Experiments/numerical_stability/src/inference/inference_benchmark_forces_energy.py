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

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.utils.get_tracing_table import save_tracing_table_with_openpyxl
from src.utils.get_model_audition import normalize_fp64_only, audit_model_dtypes

try:
    import cuequivariance as cue  # pylint: disable=unused-import
    CUET_AVAILABLE = True
except ImportError:
    CUET_AVAILABLE = False

CUBLAS_WORKSPACE_CONFIG = "4096:8"

RUN_DIR = "Experiments/numerical_stability/src/inference/results"

def parse_args():
    parser = argparse.ArgumentParser(description="MACE profiling with different calculation modes")
    parser.add_argument("--mode", 
                       choices=["energy_only", "energy_forces", "inference_only"],
                       default="inference_only",
                       help="Calculation mode to profile")
    parser.add_argument("--supercell_size", type=int, default=2, help="Size of water box")
    parser.add_argument("--warmup", type=int, default=100, help="Number of warmup steps")
    parser.add_argument("--num_iters", type=int, default=100, help="Number of iterations")
    parser.add_argument("--tf32", action="store_true", help="Enable TF32")
    parser.add_argument("--default_dtype", type=str, default="float32", help="Default dtype for MACECalculator")
    parser.add_argument("--layer_default_dtype", type=str, default="float16", help="Default dtype for layers")
    parser.add_argument("--enable_cueq", action="store_true", help="Enable CUEQ")
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
    return float(E), np.asarray(F)


def _build_inference_only(mace_calc, atoms):
    atoms.calc = mace_calc
    atoms.calc.calculate(atoms, properties=["energy", "forces"], system_changes=atoms.calc.implemented_properties)
    torch.cuda.synchronize()
    return atoms

def run_inference(args, mace_calc, atoms):
    if args.mode == "energy_only":
        with torch.no_grad():
            _get_energy_only(mace_calc, atoms)
    elif args.mode == "energy_forces":
        # must NOT disable autograd here
        _get_energy_forces(mace_calc, atoms)
    elif args.mode == "inference_only":
        atoms = _build_inference_only(mace_calc, atoms)
        return atoms
    else:
        raise ValueError(f"Invalid mode: {args.mode}")

def get_result_summary(args, device, atoms_list, time_list, run_dir, E_last, F_last):

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
    logging.info(f"Benchmark mean per iter for inference: {inference_time:.3f} ms")

    # show the results for the average energy and forces calculated
    logging.info(f"Final energy: {E_last:.6f} eV")
    mean_force = np.mean(np.linalg.norm(F_last, axis=1))
    # convert the forces to eV/A in scientific notation
    mean_force = f"{mean_force:.6e}"
    logging.info(f"Final forces shape: {F_last.shape}  "
             f"mean||F_i||: {mean_force} eV/Å")

    # save the results to a csv file
    with open(f"{run_dir}/result_summary.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(["Backend","Default dtype", "Layer default dtype", "Inference Time","Energy", "Forces"])
        writer.writerow([backend, args.default_dtype, args.layer_default_dtype, inference_time, E_last, mean_force])

def summarize_model_dtypes(model):
    from collections import Counter
    c = Counter(p.dtype for p in model.parameters())
    b = Counter(b.dtype for b in model.buffers())
    logging.info(f"Param dtypes: {c}")
    logging.info(f"Buffer dtypes: {b}")

# import torch

# _DTYPE_MAP = {
#     "float64": torch.float64,
#     "float32": torch.float32,
#     "float16": torch.float16,
#     "bfloat16": torch.bfloat16,
#     torch.float64: torch.float64,
#     torch.float32: torch.float32,
#     torch.float16: torch.float16,
#     torch.bfloat16: torch.bfloat16,
# }

# def _get_parent_and_attr(root, dotted):
#     p = root
#     parts = dotted.split(".")
#     for t in parts[:-1]:
#         p = getattr(p, t)
#     return p, parts[-1]

# def _resolve_target_dtype(module, default_target):
#     """If the module advertises a layer_dtype, use it; else use default_target."""
#     ld = getattr(module, "layer_dtype", None)
#     if ld is None:
#         return default_target
#     return _DTYPE_MAP.get(ld, default_target)

# def normalize_fp64_only(model: torch.nn.Module, default_target=torch.float32):
#     """
#     Convert ONLY float64 params/buffers to a target dtype.
#     If a module has .layer_dtype (e.g., 'float16'), that wins for tensors in that module.
#     Otherwise, use default_target (e.g., torch.float32).
#     """
#     # 1) Parameters
#     for name, p in model.named_parameters(recurse=True):
#         if not p.is_floating_point() or p.dtype != torch.float64:
#             continue
#         parent, attr = _get_parent_and_attr(model, name)
#         tgt = _resolve_target_dtype(parent, default_target)
#         if p.data.dtype != tgt:
#             p.data = p.data.to(tgt)
#             # (params don't need re-registration)

#     # 2) Buffers
#     # Try direct register_buffer; if that fails (ScriptModule), fall back to state_dict reload.
#     to_update_in_sd = {}
#     for name, b in model.named_buffers(recurse=True):
#         if not b.is_floating_point() or b.dtype != torch.float64:
#             continue
#         parent, attr = _get_parent_and_attr(model, name)
#         tgt = _resolve_target_dtype(parent, default_target)
#         new_b = b.to(tgt)
#         try:
#             parent.register_buffer(attr, new_b, persistent=True)
#         except Exception:
#             to_update_in_sd[name] = new_b

#     if to_update_in_sd:
#         sd = model.state_dict()
#         for k, v in to_update_in_sd.items():
#             sd[k] = v
#         model.load_state_dict(sd)

# def audit_model_dtypes(model):
#     from collections import Counter
#     pc = Counter(p.dtype for p in model.parameters())
#     bc = Counter(b.dtype for b in model.buffers())
#     print("Param dtypes:", pc)
#     print("Buffer dtypes:", bc)


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

    # Setup (not profiled)
    mace_calc = MACECalculator(
        model_paths="Experiments/numerical_stability/src/inference/model/MACE-OFF24_medium.model",
        default_dtype=args.default_dtype,
        enable_cueq= True,
        # enable_cueq=args.enable_cueq,
        device=device,
        layer_default_dtype=args.layer_default_dtype,
    )

    # audit_model_dtypes(mace_calc.models[0])
    # if args.default_dtype == "float32":
    #     normalize_fp64_only(mace_calc.models[0], default_target=torch.float32)  # keep FP16 layers intact
    #     audit_model_dtypes(mace_calc.models[0])               # should now be pure float32             # should now be pure float64

    # summarize_model_dtypes(mace_calc.models[0])

    logging.info(f"Get normal default dtype AFTER: {torch.get_default_dtype()}")


    atoms = build.bulk("C", "diamond", a=3.567) # unit cell of diamond , a dimension of unit cell
    water_box_size = args.supercell_size
    atoms = atoms.repeat((water_box_size,water_box_size,water_box_size))

    # batch = mace_calc._atoms_to_batch(atoms)
    # batch = mace_calc._clone_batch(batch)
    # bdict = batch.to_dict()

    # print("\n=== Batch tensor dtypes ===")
    # for k, v in bdict.items():
    #     if torch.is_tensor(v) and v.is_floating_point():
    #         print(f"{k:28s} {v.dtype} {tuple(v.shape)}")

    # Warmup (not profiled)
    for _ in range(args.warmup):
        run_inference(args, mace_calc, atoms)

    time_list = []
    atoms_list = []

    for _ in range(args.num_iters):
        if device.startswith("cuda"):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            atoms = run_inference(args, mace_calc, atoms)
            end.record()
            torch.cuda.synchronize()
            time = start.elapsed_time(end)
        
        atoms_list.append(atoms)
        time_list.append(time)

    E_last, F_last = eval_energy_forces(mace_calc, atoms)

    # print result summary
    get_result_summary(args, device,atoms_list, time_list, run_dir, E_last, F_last)
    
if __name__ == "__main__":

    os.chdir('/homes/ab3149/Documents/MLMI-MPhil-Thesis-MACE')

    main()