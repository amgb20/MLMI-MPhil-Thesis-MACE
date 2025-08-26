"""
Benchmark for Readout Linear (readouts[0].linear) in MACE

This benchmarks the readout linear sub-block across:
- Backends: e3nn (reference) and cuEq (if available)
- Precisions: fp64, fp32, bf16/fp16 where supported
- Batch sizes
- Forward and backward latency

Inputs to readouts[0].linear are captured exactly as used in the model:
- Run the full pipeline: interactions[0] -> products[0] -> updates[0] -> interactions[1] -> products[1] -> updates[1]
- Call readouts[0](...) to trigger readouts[0].linear and capture its exact call arguments

Results are saved as CSV in a local results/ directory. No JSON and no GPU memory metrics.
"""

import argparse
import os
import csv
from datetime import datetime
from collections import defaultdict
import warnings
import sys
import logging

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import ase.io
import numpy as np
import torch

# Suppress warnings
warnings.filterwarnings("ignore")

# Set up logging at the top level
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Set environment variable for MACE
os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"

from e3nn import o3
from mace import data, modules, tools
from mace.tools import torch_geometric


try:
    import cuequivariance as cue  # noqa: F401
    CUET_AVAILABLE = True
    logging.info("✓ Cuequivariance library is available")
except ImportError:
    CUET_AVAILABLE = False
    logging.info("✗ Cuequivariance library is not available - cuEq will be disabled")


def create_model(table: tools.AtomicNumberTable, cueq_config=None):
    model_config = {
        "r_max": 6.0,
        "num_bessel": 8,
        "num_polynomial_cutoff": 6,
        "max_ell": 3,
        "interaction_cls": modules.interaction_classes["RealAgnosticResidualInteractionBlock"],
        "interaction_cls_first": modules.interaction_classes["RealAgnosticResidualInteractionBlock"],
        "num_interactions": 2,
        "num_elements": len(table),
        "hidden_irreps": o3.Irreps("16x0e + 16x1o"),
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


def create_batch_data(atoms_list, table, batch_size, device):
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


def prepare_inputs(model, batch_dict, device, dtype, z_table):
    batch_d = batch_dict.copy()
    for key in ['node_attrs', 'positions']:
        if key in batch_d:
            batch_d[key] = batch_d[key].to(dtype)

    vectors, lengths = modules.utils.get_edge_vectors_and_lengths(
        positions=batch_d["positions"],
        edge_index=batch_d["edge_index"],
        shifts=batch_d["shifts"],
    )

    vectors = vectors.to(dtype)
    lengths = lengths.to(dtype)

    x0 = model.node_embedding(batch_d["node_attrs"]).to(dtype)
    e_feats, _ = model.radial_embedding(lengths, batch_d["node_attrs"], batch_d["edge_index"], z_table)
    e_feats = e_feats.to(dtype)
    e_attrs = model.spherical_harmonics(vectors).to(dtype)

    inputs = {
        "node_feats": x0,
        "node_attrs": batch_d["node_attrs"],
        "edge_feats": e_feats,
        "edge_attrs": e_attrs,
        "edge_index": batch_d["edge_index"].to(torch.long),
    }
    return inputs


def build_product0_output(model, inputs):
    # Compute product[0] output as in models.MACE.forward
    out0, sc0 = model.interactions[0](
        node_attrs=inputs["node_attrs"],
        node_feats=inputs["node_feats"],
        edge_attrs=inputs["edge_attrs"],
        edge_feats=inputs["edge_feats"],
        edge_index=inputs["edge_index"],
        first_layer=True,
    )
    h0 = model.products[0](node_feats=out0, sc=sc0, node_attrs=inputs["node_attrs"])
    return h0


def get_readout_linear_and_input(model, inputs):
    readout_linear = model.readouts[0].linear
    h0 = build_product0_output(model, inputs)
    logging.info(f"  Using product[0] output for readout.linear: shape={tuple(h0.shape)} dtype={h0.dtype} device={h0.device}")
    return readout_linear, h0


def benchmark_readout_linear(readout_linear, input_tensor, num_iterations=100, warmup=50, device="cuda"):
    bench_tensor = input_tensor.detach().clone().requires_grad_(True)
    bench_tensor.retain_grad()

    def run_forward():
        out = readout_linear(bench_tensor)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        return out

    def run_backward(output):
        loss = (output**2).sum() + 0.1 * (bench_tensor**2).sum()
        loss.backward(retain_graph=True)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        grad = bench_tensor.grad.clone() if bench_tensor.grad is not None else torch.zeros_like(bench_tensor)
        return grad

    for _ in range(warmup):
        out = run_forward(); _ = run_backward(out); bench_tensor.grad.zero_()

    bench_tensor.grad.zero_()

    f_times, b_times = [], []
    for _ in range(num_iterations):
        if device.startswith("cuda"):
            sf = torch.cuda.Event(enable_timing=True); ef = torch.cuda.Event(enable_timing=True)
            sf.record(); out = run_forward(); ef.record(); torch.cuda.synchronize(); fwd = sf.elapsed_time(ef)
        else:
            import time
            t0 = time.perf_counter(); out = run_forward(); fwd = (time.perf_counter() - t0) * 1000
        f_times.append(fwd)

        if device.startswith("cuda"):
            sb = torch.cuda.Event(enable_timing=True); eb = torch.cuda.Event(enable_timing=True)
            sb.record(); grad = run_backward(out); eb.record(); torch.cuda.synchronize(); bwd = sb.elapsed_time(eb)
        else:
            import time
            t0 = time.perf_counter(); grad = run_backward(out); bwd = (time.perf_counter() - t0) * 1000
        b_times.append(bwd)
        bench_tensor.grad.zero_()

    return {
        "forward_latency_ms": float(np.mean(f_times)),
        "backward_latency_ms": float(np.mean(b_times)),
        "forward_std_ms": float(np.std(f_times)),
        "backward_std_ms": float(np.std(b_times)),
        "output": out.detach().cpu(),
        "grad": grad.detach().cpu(),
        "num_iterations": num_iterations,
    }


def run_linear_readout_benchmark(device="cuda", batch_sizes=None, num_iterations=100, warmup=50):
    if batch_sizes is None:
        batch_sizes = [1, 4, 8, 16, 32]

    logging.info("=== MACE Linear Readout Benchmark ===")
    logging.info(f"Device: {device}")
    logging.info(f"Batch sizes: {batch_sizes}")
    logging.info(f"Iterations: {num_iterations}, Warmup: {warmup}")

    logging.info("\nLoading test data...")
    xyz_file = 'Experiments/numerical_stability/src/inference/data/carbon.xyz'
    atoms_list = ase.io.read(xyz_file, index=":")

    table = tools.AtomicNumberTable([6, 82, 53, 55])

    first_atoms = atoms_list[0]
    num_nodes = len(first_atoms)
    cutoff = 6.0
    num_edges = 0
    pos = first_atoms.get_positions()
    for i in range(len(pos)):
        for j in range(i+1, len(pos)):
            if np.linalg.norm(pos[i] - pos[j]) <= cutoff:
                num_edges += 2

    logging.info(f"Elements: {table.zs}")
    logging.info(f"Atoms per molecule: {num_nodes}")
    logging.info(f"Approximate edges per molecule (cutoff={cutoff}Å): {num_edges}")
    logging.info(f"Number of molecules: {len(atoms_list)}")
    logging.info(f"Total system size: {num_nodes * len(atoms_list)} atoms")

    precision_configs = [("fp64", torch.float64), ("fp32", torch.float32)]
    if device.startswith("cuda"):
        try:
            torch.zeros(1, device="cuda", dtype=torch.bfloat16)
            precision_configs.append(("bf16", torch.bfloat16))
            logging.info("✓ BF16 supported")
        except Exception:
            logging.info("✗ BF16 not supported")
        try:
            torch.zeros(1, device="cuda", dtype=torch.float16)
            precision_configs.append(("fp16", torch.float16))
            logging.info("✓ FP16 supported")
        except Exception:
            logging.info("✗ FP16 not supported")

    backend_configs = [("e3nn", None)]
    if CUET_AVAILABLE and device.startswith("cuda"):
        cueq_config = modules.wrapper_ops.CuEquivarianceConfig(enabled=True, optimize_all=True)
        backend_configs.append(("cueq", cueq_config))
        logging.info("✓ cuEq backend enabled")
    else:
        logging.info("✗ cuEq backend disabled")

    logging.info("\n--- Creating Master Models ---")
    torch.manual_seed(42)
    if device.startswith("cuda"):
        torch.cuda.manual_seed(42); torch.cuda.manual_seed_all(42)

    master_models = {}
    master_e3nn = create_model(table, None).to(device=device, dtype=torch.float64)
    master_models["e3nn"] = master_e3nn.state_dict()
    if CUET_AVAILABLE and device.startswith("cuda"):
        master_cueq = create_model(table, modules.wrapper_ops.CuEquivarianceConfig(enabled=True, optimize_all=True)).to(device=device, dtype=torch.float64)
        master_models["cueq"] = master_cueq.state_dict()

    all_results = []
    total_configs = len(backend_configs) * len(precision_configs) * len(batch_sizes)
    config_count = 0

    for backend_name, backend_config in backend_configs:
        logging.info(f"\n--- Testing {backend_name.upper()} backend ---")
        for precision_name, dtype in precision_configs:
            logging.info(f"\n  Testing {precision_name.upper()} precision...")
            if backend_name == "cueq" and precision_name in ["bf16", "fp16"]:
                logging.info(f"    Skipping {precision_name} - cuEq doesn't support this precision"); continue

            model = create_model(table, backend_config)
            model.load_state_dict(master_models[backend_name])
            model = model.to(device=device, dtype=dtype)

            for batch_size in batch_sizes:
                config_count += 1
                logging.info(f"    Batch size {batch_size} ({config_count}/{total_configs})")
                try:
                    batch_dict = create_batch_data(atoms_list, table, batch_size, device)
                    inputs = prepare_inputs(model, batch_dict, device, dtype, table)

                    try:
                        readout_linear, h0 = get_readout_linear_and_input(model, inputs)

                        results = benchmark_readout_linear(
                            readout_linear, h0, num_iterations, warmup, device
                        )

                        result_row = {
                            "backend": backend_name,
                            "dtype": precision_name,
                            "batch_size": batch_size,
                            "num_nodes": num_nodes,
                            "num_edges": num_edges,
                            "forward_latency_ms": results["forward_latency_ms"],
                            "backward_latency_ms": results["backward_latency_ms"],
                            "forward_std_ms": results["forward_std_ms"],
                            "backward_std_ms": results["backward_std_ms"],
                            "output": results["output"],
                            "grad": results["grad"],
                        }
                        all_results.append(result_row)
                        logging.info(
                            f"      ReadoutLinear: Fwd: {results['forward_latency_ms']:.2f}ms, "
                            f"Bwd: {results['backward_latency_ms']:.2f}ms"
                        )
                    except Exception as e:
                        logging.info(f"      Error benchmarking readout linear: {e}")
                        continue
                except Exception as e:
                    logging.info(f"      Error with batch size {batch_size}: {e}")
                    continue

    # Compute speedups vs FP64 per backend and batch size
    fp64_refs = {}
    for r in all_results:
        if r["dtype"] == "fp64":
            key = (r["backend"], r["batch_size"])
            fp64_refs[key] = r
    for r in all_results:
        if r["dtype"] == "fp64":
            r.update({
                "forward_speedup_vs_fp64": "N/A",
                "backward_speedup_vs_fp64": "N/A",
            })
            continue
        key = (r["backend"], r["batch_size"])
        if key in fp64_refs:
            ref = fp64_refs[key]
            f_sp = ref["forward_latency_ms"] / r["forward_latency_ms"] if r["forward_latency_ms"] > 0 else 0
            b_sp = ref["backward_latency_ms"] / r["backward_latency_ms"] if r["backward_latency_ms"] > 0 else 0
            r.update({
                "forward_speedup_vs_fp64": f_sp,
                "backward_speedup_vs_fp64": b_sp,
            })
        else:
            r.update({
                "forward_speedup_vs_fp64": "N/A",
                "backward_speedup_vs_fp64": "N/A",
            })

    save_results(all_results)
    generate_summary_report(all_results)
    return all_results


def save_results(results):
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    logging.info(f"  Saving {len(results)} results to CSV...")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(script_dir, "results"); os.makedirs(results_dir, exist_ok=True)

    column_mapping = {
        "backend": "Backend",
        "dtype": "Precision",
        "batch_size": "Batch Size",
        "num_nodes": "Nodes per Molecule",
        "num_edges": "Edges per Molecule",
        "forward_latency_ms": "Forward Latency (ms)",
        "backward_latency_ms": "Backward Latency (ms)",
        "forward_std_ms": "Forward Std Dev (ms)",
        "backward_std_ms": "Backward Std Dev (ms)",
        "forward_speedup_vs_fp64": "Forward Speedup vs FP64",
        "backward_speedup_vs_fp64": "Backward Speedup vs FP64",
        "output": "Output Tensor",
        "grad": "Gradient Tensor",
    }

    csv_data = [{k: v for k, v in r.items() if not isinstance(v, torch.Tensor)} for r in results]
    csv_path = os.path.join(results_dir, f"linear_readout_benchmark_{timestamp}.csv")
    with open(csv_path, 'w', newline='') as f:
        if csv_data:
            fields = sorted({k for row in csv_data for k in row.keys()})
            headers = [column_mapping.get(field, field.replace('_', ' ').title()) for field in fields]
            w = csv.writer(f); w.writerow(headers)
            for row in csv_data:
                w.writerow([row.get(field, "") for field in fields])
            logging.info(f"  Wrote {len(csv_data)} rows to CSV")
        else:
            logging.warning("  No CSV data to write!")
    logging.info(f"Results saved to: {csv_path}")


def generate_summary_report(results):
    logging.info("\n=== Summary Report ===")
    grouped = defaultdict(lambda: defaultdict(list))
    for r in results:
        grouped[r["backend"]][r["dtype"]].append(r)
    for dtype in ["fp64", "fp32", "bf16", "fp16"]:
        logging.info(f"\n--- {dtype.upper()} Precision ---")
        e3 = grouped["e3nn"].get(dtype, [])
        cq = grouped["cueq"].get(dtype, [])
        if not e3:
            logging.info(f"  No e3nn results for {dtype}"); continue
        if not cq:
            logging.info(f"  No cuEq results for {dtype} (unsupported)"); continue
        e3r = e3[0]; cqr = cq[0]
        fwd_speedup = e3r["forward_latency_ms"]/cqr["forward_latency_ms"] if cqr["forward_latency_ms"]>0 else 0
        bwd_speedup = e3r["backward_latency_ms"]/cqr["backward_latency_ms"] if cqr["backward_latency_ms"]>0 else 0
        logging.info(f"  Forward e3nn {e3r['forward_latency_ms']:.2f}ms ± {e3r.get('forward_std_ms',0):.2f}, cuEq {cqr['forward_latency_ms']:.2f}ms ± {cqr.get('forward_std_ms',0):.2f}, speedup {fwd_speedup:.2f}x")
        logging.info(f"  Backward e3nn {e3r['backward_latency_ms']:.2f}ms ± {e3r.get('backward_std_ms',0):.2f}, cuEq {cqr['backward_latency_ms']:.2f}ms ± {cqr.get('backward_std_ms',0):.2f}, speedup {bwd_speedup:.2f}x")


def main():
    parser = argparse.ArgumentParser(description="MACE Linear Readout Benchmark")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use (cuda/cpu)")
    parser.add_argument("--batch-sizes", type=str, default="1,4,8,16,32", help="Comma-separated list of batch sizes")
    parser.add_argument("--iterations", type=int, default=100, help="Number of benchmark iterations")
    parser.add_argument("--warmup", type=int, default=50, help="Number of warmup iterations")
    args = parser.parse_args()

    torch.manual_seed(42)
    batch_sizes = [int(x.strip()) for x in args.batch_sizes.split(",")]
    if args.device == "cuda" and not torch.cuda.is_available():
        logging.info("CUDA not available, falling back to CPU"); args.device = "cpu"
    if args.device.startswith("cuda"):
        torch.cuda.empty_cache()

    results = run_linear_readout_benchmark(
        device=args.device,
        batch_sizes=batch_sizes,
        num_iterations=args.iterations,
        warmup=args.warmup,
    )
    logging.info(f"\nBenchmark completed! Generated {len(results)} result entries.")


if __name__ == "__main__":
    main()


