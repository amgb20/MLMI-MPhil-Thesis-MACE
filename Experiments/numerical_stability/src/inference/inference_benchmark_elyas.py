import argparse
import logging
from pathlib import Path

import ase.io
import numpy as np
import torch
from mace import data, modules, tools
from mace.cli.convert_e3nn_cueq import run as run_e3nn_to_cueq
from mace.tools import torch_geometric
from torch.utils.benchmark import Timer
import torch.utils.benchmark as benchmark
from mace.calculators import mace_mp
from tqdm import tqdm
import os
os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"

# Then import e3nn
from e3nn import o3

"""
This script is used to benchmark the performance of E3NN and CUET for MACE with simple argument.
Sister and more refine script is inference_e3nn_vs_cueq.py
"""

TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

try:
    import cuequivariance as cue  # pylint: disable=unused-import
    CUET_AVAILABLE = True
except ImportError:
    CUET_AVAILABLE = False

def create_model(hidden_irreps, max_ell, cueq_config=None, layer_dtype=None):
    table = tools.AtomicNumberTable([8, 82, 53, 55])
    model_config = {
        "r_max": 6.0,
        "num_bessel": 8,
        "num_polynomial_cutoff": 6,
        "max_ell": max_ell,
        "interaction_cls": modules.interaction_classes["RealAgnosticResidualInteractionBlock"],
        "interaction_cls_first": modules.interaction_classes["RealAgnosticResidualInteractionBlock"],
        "num_interactions": 2,
        "num_elements": len(table),
        "hidden_irreps": o3.Irreps(hidden_irreps),
        "MLP_irreps": o3.Irreps("16x0e"),
        "gate": torch.nn.functional.silu,
        "atomic_energies": torch.ones(len(table)),
        "avg_num_neighbors": 8,
        "atomic_numbers": table.zs,
        "correlation": 3,
        "radial_type": "bessel",
        "num_elements": 4,
        "cueq_config": cueq_config,
        "atomic_inter_scale": 1.0,
        "atomic_inter_shift": 0.0,
        "layer_default_dtype": layer_dtype,
    }
    return modules.ScaleShiftMACE(**model_config)

def make_batch(atoms_list, table, batch_size, device):
    data_loader = torch_geometric.dataloader.DataLoader(
        dataset=[data.AtomicData.from_config(
            data.config_from_atoms(atoms),
            z_table=table,
            cutoff=6.0
        ) for atoms in atoms_list],
        batch_size=min(len(atoms_list), batch_size),
        shuffle=False,
        drop_last=False,
    )
    batch = next(iter(data_loader)).to(device)
    return batch.to_dict()

def benchmark_model(model, batch, num_iterations=100, warmup=100, label="Inference", sub_label=None, description=None):
    def run_inference():
        if torch.get_default_dtype() == torch.float32:
            print("Running with autocast")
            with torch.autocast("cuda"):  # THIS WAS AN ADDED BIT
                out = model(batch,training=True)
                torch.cuda.synchronize()
        else:
            out = model(batch,training=True)
            torch.cuda.synchronize()
        return out

    # Warmup
    for _ in range(warmup):
        run_inference()

    # Benchmark
    timer = Timer(
        stmt="run_inference()",
        globals={
            "run_inference": run_inference,
        },
        label=label,
        sub_label=sub_label,
        description=description,
    )
    #warmu_up_measurement = timer.timeit(num_iterations)
    measurement = timer.timeit(num_iterations)
    return measurement

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("xyz_file", type=str, help="Path to xyz file")
    parser.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"])
    parser.add_argument("--num_iters", type=int, default=100)
    parser.add_argument("--max_ell", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--hidden_irreps", type=str, default="16x0e + 16x1o")
    parser.add_argument("--layer_dtype", type=str, default="float64")
    parser.add_argument("--warmup", type=int, default=100)
    args = parser.parse_args()
    torch.set_default_dtype(torch.float64)
    device = torch.device(args.device)
    hidden_irreps = o3.Irreps(args.hidden_irreps)

    # Create dataset
    atoms_list = ase.io.read(args.xyz_file, index=":")
    #table = tools.AtomicNumberTable(list(set(np.concatenate([atoms.numbers for atoms in atoms_list]))))
    table = tools.AtomicNumberTable([6, 82, 53, 55])
    batch_dict = make_batch(atoms_list, table, args.batch_size, device)

    print("\nBenchmarking Configuration:")
    print(f"Number of atoms: {len(atoms_list[0])}")
    print(f"Number of edges: {batch_dict['edge_index'].shape[1]}")
    print(f"Batch size: {min(len(atoms_list), args.batch_size)}")
    print(f"Device: {args.device}")
    print(f"Hidden irreps: {hidden_irreps}")
    print(f"Number of iterations: {args.num_iters}\n")

    # Test without CUET
    model_e3nn = create_model(hidden_irreps, args.max_ell, layer_dtype=args.layer_dtype).to(device)
    #model_e3nn = mace_mp(model="large", device="cuda", default_dtype="float64")
    description = f"device={args.device}, batch={min(len(atoms_list), args.batch_size)}, max_ell={args.max_ell}"
    results = []
    measurement_e3nn = benchmark_model(
        model_e3nn,
        batch_dict,
        num_iterations=args.num_iters,
        warmup=args.warmup,
        label="MACE Inference",
        sub_label="E3NN",
        description=description,
    )
    results.append(measurement_e3nn)

    print("measurement_e3nn", measurement_e3nn)

    # Test with CUET if available
    if CUET_AVAILABLE and args.device == "cuda":
        print("Running CUET")
        model_cueq = run_e3nn_to_cueq(model_e3nn)
        model_cueq = model_cueq.to(device)
        measurement_cueq = benchmark_model(
            model_cueq,
            batch_dict,
            num_iterations=args.num_iters,
            warmup=args.warmup,
            label="MACE Inference",
            sub_label="CUET",
            description=description,
        )
        results.append(measurement_cueq)

    # Print comparison
    if len(results) > 1:
        print("\nBenchmark comparison:")
        compare = benchmark.Compare(results)
        compare.print()
        # Also print speedup for convenience if CUET is present
        if CUET_AVAILABLE and args.device == "cuda":
            print(f"\nSpeedup (E3NN -> CUET): {measurement_e3nn.mean / measurement_cueq.mean:.2f}x")
    else:
        print(f"E3NN Measurement:\n{measurement_e3nn}")

if __name__ == "__main__":
    main()