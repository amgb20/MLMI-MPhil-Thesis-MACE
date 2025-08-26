import argparse
import os
import csv
from datetime import datetime
from collections import defaultdict
import itertools

import ase.io
import numpy as np
import torch
import torch.utils.benchmark as benchmark
from torch.utils.benchmark import Timer

os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"

from e3nn import o3
from mace import data, modules, tools
from mace.cli.convert_e3nn_cueq import run as run_e3nn_to_cueq
from mace.tools import torch_geometric

import matplotlib.pyplot as plt
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except Exception:
    TQDM_AVAILABLE = False

try:
    import cuequivariance as cue  # noqa: F401
    CUET_AVAILABLE = True
except Exception:
    CUET_AVAILABLE = False


"""
This script is used to benchmark the performance of E3NN and CUET for MACE with
different batch sizes and hidden irreps.
"""


def create_model(table: tools.AtomicNumberTable, hidden_irreps: str, max_ell: int, cueq_config=None):
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
        "cueq_config": cueq_config,
        "atomic_inter_scale": 1.0,
        "atomic_inter_shift": 0.0,
    }
    return modules.ScaleShiftMACE(**model_config)


def benchmark_model(model, batch, num_iterations=100, warmup=100, label="Inference", sub_label=None, description=None, show_progress=False):
    def run_inference():
        out = model(batch, training=False)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        return out

    if warmup > 0:
        iterator = range(warmup)
        if show_progress and TQDM_AVAILABLE:
            iterator = tqdm(iterator, desc=f"Warmup {sub_label or ''}", leave=False)
        for _ in iterator:
            run_inference()

    timer = Timer(
        stmt="run_inference()",
        globals={
            "run_inference": run_inference,
        },
        label=label,
        sub_label=sub_label,
        description=description,
    )
    measurement = timer.timeit(num_iterations)
    return measurement


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


def run_one(model, batch_dict, num_iters, warmup, label, sub_label, description):
    m = benchmark_model(
        model,
        batch_dict,
        num_iterations=num_iters,
        warmup=warmup,
        label=label,
        sub_label=sub_label,
        description=description,
    )
    return m.mean * 1e3, m


def sweep_and_save(atoms_list, device, table, hidden_list, max_ell_list, batch_sizes, num_iters, warmup, out_dir, show_progress=False):
    os.makedirs(out_dir, exist_ok=True)
    rows = []
    for hidden in hidden_list:
        for max_ell in max_ell_list:
            model_e3nn = create_model(table, hidden, max_ell).to(device)
            model_cueq = None
            if CUET_AVAILABLE and device.type == "cuda":
                model_cueq = run_e3nn_to_cueq(model_e3nn).to(device)

            bs_iter = batch_sizes
            progress_bar = None
            if show_progress and TQDM_AVAILABLE:
                progress_bar = tqdm(bs_iter, total=len(batch_sizes), desc=f"{hidden}, ℓ={max_ell}", leave=False)
                bs_iter = progress_bar
            for bs in bs_iter:
                batch_dict = make_batch(atoms_list, table, bs, device)
                n_edges = batch_dict["edge_index"].shape[1]
                n_atoms = len(atoms_list[0])

                desc = f"device={device.type}, batch={bs}, max_ell={max_ell}, hidden='{hidden}'"

                e3_ms, _ = run_one(model_e3nn, batch_dict, num_iters, warmup, "MACE Inference", "E3NN", desc)
                cue_ms = None
                if model_cueq is not None:
                    cue_ms, _ = run_one(model_cueq, batch_dict, num_iters, warmup, "MACE Inference", "CUET", desc)

                speedup = (e3_ms / cue_ms) if (cue_ms is not None and cue_ms > 0) else None
                rows.append({
                    "device": device.type,
                    "batch_size": bs,
                    "max_ell": max_ell,
                    "hidden_irreps": str(hidden),
                    "num_atoms": n_atoms,
                    "num_edges": n_edges,
                    "e3nn_ms": e3_ms,
                    "cueq_ms": cue_ms,
                    "speedup_e3_to_cueq": speedup,
                })
            if progress_bar is not None:
                progress_bar.close()

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_csv = os.path.join(out_dir, f"e3nn_vs_cueq_{stamp}.csv")
    if rows:
        with open(out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    return rows, out_csv


def plot_speedup(rows, out_dir):
    groups = defaultdict(list)
    for r in rows:
        if r["speedup_e3_to_cueq"] is not None:
            key = (r["hidden_irreps"], r["max_ell"])
            groups[key].append((r["batch_size"], r["speedup_e3_to_cueq"]))

    if not groups:
        return None

    plt.figure(figsize=(6, 4))
    for (hidden, max_ell), pts in groups.items():
        pts = sorted(pts, key=lambda x: x[0])
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        plt.plot(xs, ys, marker="o", label=f"{hidden}, ℓ={max_ell}")
    plt.xlabel("Batch size")
    plt.ylabel("Speedup (E3NN → CUET)")
    plt.title("Cuequivariance speedup vs batch size")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    out_png = os.path.join(out_dir, "speedup_vs_batch.png")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()
    return out_png


def main():
    parser = argparse.ArgumentParser(description="Benchmark E3NN vs CUET for MACE")
    parser.add_argument("xyz_file", type=str, help="Path to xyz file")
    parser.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"])
    parser.add_argument("--num_iters", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--max_ell", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--hidden_irreps", type=str, default="16x0e + 16x1o")
    parser.add_argument("--out_dir", type=str, default="Experiments/numerical_stability/src/inference/results")
    parser.add_argument("--sweep_batches", type=str, default=None, help="Comma-separated batch sizes, e.g. '1,4,8,16,32'")
    parser.add_argument("--sweep_max_ell", type=str, default=None, help="Comma-separated ℓ, e.g. '1,2,3'")
    parser.add_argument("--sweep_hidden", type=str, default=None, help="Semicolon-separated irreps strings, e.g. '64x0e + 64x1o;128x0e + 128x1o + 128x2e'")
    parser.add_argument("--progress", action="store_true", help="Show tqdm progress bars for warmup and sweeps")
    args = parser.parse_args()

    torch.set_default_dtype(torch.float64)
    device = torch.device(args.device)

    atoms_list = ase.io.read(args.xyz_file, index=":")
    unique_numbers = sorted({int(z) for atoms in atoms_list for z in atoms.numbers})
    table = tools.AtomicNumberTable(unique_numbers)

    if any([args.sweep_batches, args.sweep_max_ell, args.sweep_hidden]):
        batch_sizes = [int(x) for x in (args.sweep_batches or str(args.batch_size)).split(",")]
        max_ell_list = [int(x) for x in (args.sweep_max_ell or str(args.max_ell)).split(",")]
        hidden_list = [s.strip() for s in (args.sweep_hidden.split(";") if args.sweep_hidden else [args.hidden_irreps])]

        rows, out_csv = sweep_and_save(
            atoms_list=atoms_list,
            device=device,
            table=table,
            hidden_list=hidden_list,
            max_ell_list=max_ell_list,
            batch_sizes=batch_sizes,
            num_iters=args.num_iters,
            warmup=args.warmup,
            out_dir=args.out_dir,
            show_progress=args.progress,
        )
        print(f"Wrote CSV: {out_csv}")
        out_png = plot_speedup(rows, args.out_dir)
        if out_png:
            print(f"Wrote plot: {out_png}")
        return

    batch_dict = make_batch(atoms_list, table, args.batch_size, device)

    print("\nBenchmarking Configuration:")
    print(f"Number of atoms: {len(atoms_list[0])}")
    print(f"Number of edges: {batch_dict['edge_index'].shape[1]}")
    print(f"Batch size: {min(len(atoms_list), args.batch_size)}")
    print(f"Device: {args.device}")
    print(f"Hidden irreps: {args.hidden_irreps}")
    print(f"max_ell: {args.max_ell}")
    print(f"Number of iterations: {args.num_iters}\n")

    model_e3nn = create_model(table, args.hidden_irreps, args.max_ell).to(device)
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
        show_progress=args.progress,
    )
    results.append(measurement_e3nn)

    if CUET_AVAILABLE and args.device == "cuda":
        model_cueq = run_e3nn_to_cueq(model_e3nn).to(device)
        measurement_cueq = benchmark_model(
            model_cueq,
            batch_dict,
            num_iterations=args.num_iters,
            warmup=args.warmup,
            label="MACE Inference",
            sub_label="CUET",
            description=description,
            show_progress=args.progress,
        )
        results.append(measurement_cueq)

    if len(results) > 1:
        print("\nBenchmark comparison:")
        compare = benchmark.Compare(results)
        compare.print()
        if CUET_AVAILABLE and args.device == "cuda":
            print(f"\nSpeedup (E3NN -> CUET): {measurement_e3nn.mean / measurement_cueq.mean:.2f}x")
    else:
        print(f"E3NN Measurement:\n{measurement_e3nn}")


if __name__ == "__main__":
    main()