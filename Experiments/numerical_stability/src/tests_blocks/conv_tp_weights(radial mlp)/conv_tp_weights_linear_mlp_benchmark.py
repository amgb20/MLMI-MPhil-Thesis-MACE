"""
Comprehensive benchmark for conv_tp_weights (radial MLP) in MACE interactions

This script benchmarks the conv_tp_weights operation across:
- Backends: cuEq (cuequivariance) vs e3nn (reference)
- Precisions: fp64, fp32, bf16, fp16 (where supported)
- Batch sizes: 1, 4, 8, 16, 32
- Forward and backward passes
- Accuracy vs reference (fp64 e3nn)
- Performance (latency, memory usage)
"""

import argparse
import os
import csv
from datetime import datetime
from collections import defaultdict
import warnings
import sys
import logging

# Add parent directories to path
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
    """Create MACE model with specified configuration"""
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
    """Create batch data for given batch size"""
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
    """Prepare inputs for conv_tp_weights testing"""
    # Cast batch to target dtype
    batch_d = batch_dict.copy()
    for key in ['node_attrs', 'positions']:
        if key in batch_d:
            batch_d[key] = batch_d[key].to(dtype)

    # Get edge vectors and lengths
    vectors, lengths = modules.utils.get_edge_vectors_and_lengths(
        positions=batch_d["positions"],
        edge_index=batch_d["edge_index"],
        shifts=batch_d["shifts"],
    )

    # Cast to target dtype
    vectors = vectors.to(dtype)
    lengths = lengths.to(dtype)

    # Node embedding (not directly used here but mirrors full model prep)
    x0 = model.node_embedding(batch_d["node_attrs"]).to(dtype)

    # Radial embedding produces edge_feats used by conv_tp_weights
    e_feats, _ = model.radial_embedding(
        lengths, batch_d["node_attrs"], batch_d["edge_index"], z_table
    )
    e_feats = e_feats.to(dtype).requires_grad_()
    e_feats.retain_grad()

    e_attrs = model.spherical_harmonics(vectors)
    e_attrs = e_attrs.to(dtype)

    inputs = {
        "edge_feats": e_feats,
        "edge_attrs": e_attrs,
        "node_attrs": batch_d["node_attrs"],
        "edge_index": batch_d["edge_index"].to(torch.long),
    }

    return inputs, e_feats


def benchmark_conv_tp_weights(block, inputs, num_iterations=100, warmup=50, device="cuda"):
    """Benchmark conv_tp_weights block with multiple iterations for stable timing"""

    def run_forward():
        # conv_tp_weights expects edge_feats
        output = block(inputs["edge_feats"])
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        return output

    def run_backward(output):
        # Create a loss that depends on weights to drive gradients to edge_feats
        if isinstance(output, torch.Tensor):
            loss = (output**2).sum()
        else:
            loss = torch.tensor(0.0, device=device, requires_grad=True)

        loss.backward(retain_graph=True)
        if device.startswith("cuda"):
            torch.cuda.synchronize()

        grad = inputs["edge_feats"].grad.clone() if inputs["edge_feats"].grad is not None else torch.zeros_like(inputs["edge_feats"])
        return grad

    # Warmup
    for _ in range(warmup):
        output = run_forward()
        grad = run_backward(output)
        # Reset gradients
        inputs["edge_feats"].grad.zero_()

    # Ensure clean grads before timing
    inputs["edge_feats"].grad.zero_()

    # No GPU memory stats collection

    forward_times_list = []
    backward_times_list = []

    for _ in range(num_iterations):
        # Forward timing
        if device.startswith("cuda"):
            start_forward = torch.cuda.Event(enable_timing=True)
            end_forward = torch.cuda.Event(enable_timing=True)
            start_forward.record()
            output = run_forward()
            end_forward.record()
            torch.cuda.synchronize()
            forward_time = start_forward.elapsed_time(end_forward)
        else:
            import time
            start_time = time.perf_counter()
            output = run_forward()
            forward_time = (time.perf_counter() - start_time) * 1000
        forward_times_list.append(forward_time)

        # Backward timing
        if device.startswith("cuda"):
            start_backward = torch.cuda.Event(enable_timing=True)
            end_backward = torch.cuda.Event(enable_timing=True)
            start_backward.record()
            grad = run_backward(output)
            end_backward.record()
            torch.cuda.synchronize()
            backward_time = start_backward.elapsed_time(end_backward)
        else:
            import time
            start_time = time.perf_counter()
            grad = run_backward(output)
            backward_time = (time.perf_counter() - start_time) * 1000
        backward_times_list.append(backward_time)

        # Reset grads
        inputs["edge_feats"].grad.zero_()

    # Stats
    forward_mean = np.mean(forward_times_list)
    forward_std = np.std(forward_times_list)
    backward_mean = np.mean(backward_times_list)
    backward_std = np.std(backward_times_list)

    return {
        "forward_latency_ms": forward_mean,
        "backward_latency_ms": backward_mean,
        "forward_std_ms": forward_std,
        "backward_std_ms": backward_std,
        "output": output.detach().cpu(),
        "grad": grad.detach().cpu(),
        "num_iterations": num_iterations,
    }


def compute_accuracy_metrics(reference_output, reference_grad, test_output, test_grad):
    """Compute accuracy metrics relative to reference"""
    ref_out = reference_output.double().numpy()
    ref_grad = reference_grad.double().numpy()
    test_out = test_output.double().numpy()
    test_grad = test_grad.double().numpy()

    fwd_abs_error = np.abs(ref_out - test_out)
    fwd_rel_error = np.abs(fwd_abs_error / (np.abs(ref_out) + 1e-6))

    bwd_abs_error = np.abs(ref_grad - test_grad)
    bwd_rel_error = np.abs(bwd_abs_error / (np.abs(ref_grad) + 1e-6))

    return {
        "max_abs_error_fwd": float(fwd_abs_error.max()),
        "max_rel_error_fwd": float(fwd_rel_error.max()),
        "max_abs_error_bwd": float(bwd_abs_error.max()),
        "max_rel_error_bwd": float(bwd_rel_error.max()),
    }


def run_conv_tp_weights_benchmark(device="cuda", batch_sizes=None, num_iterations=100, warmup=50):
    """Run comprehensive conv_tp_weights benchmark"""

    if batch_sizes is None:
        batch_sizes = [1, 4, 8, 16, 32]

    logging.info(f"=== MACE conv_tp_weights Benchmark ===")
    logging.info(f"Device: {device}")
    logging.info(f"Batch sizes: {batch_sizes}")
    logging.info(f"Iterations: {num_iterations}, Warmup: {warmup}")

    # Load test data
    logging.info("\nLoading test data...")
    xyz_file = 'Experiments/numerical_stability/src/inference/data/carbon.xyz'
    atoms_list = ase.io.read(xyz_file, index=":")

    table = tools.AtomicNumberTable([6, 82, 53, 55])  # C, Pb, I, Cs

    # System info
    first_atoms = atoms_list[0]
    num_nodes = len(first_atoms)
    num_edges = 0
    cutoff = 6.0
    positions = first_atoms.get_positions()
    for i in range(len(positions)):
        for j in range(i+1, len(positions)):
            dist = np.linalg.norm(positions[i] - positions[j])
            if dist <= cutoff:
                num_edges += 2

    logging.info(f"Elements: {table.zs}")
    logging.info(f"Atoms per molecule: {num_nodes}")
    logging.info(f"Approximate edges per molecule (cutoff={cutoff}Å): {num_edges}")
    logging.info(f"Number of molecules: {len(atoms_list)}")
    logging.info(f"Total system size: {num_nodes * len(atoms_list)} atoms")

    # Precisions
    precision_configs = [
        ("fp64", torch.float64),
        ("fp32", torch.float32),
    ]
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

    # Backends
    backend_configs = [("e3nn", None)]
    if CUET_AVAILABLE and device.startswith("cuda"):
        cueq_config = modules.wrapper_ops.CuEquivarianceConfig(
            enabled=True,
            optimize_all=True,
        )
        backend_configs.append(("cueq", cueq_config))
        logging.info("✓ cuEq backend enabled")
    else:
        logging.info("✗ cuEq backend disabled")

    # Master models
    logging.info("\n--- Creating Master Models ---")
    torch.manual_seed(42)
    if device.startswith("cuda"):
        torch.cuda.manual_seed(42)
        torch.cuda.manual_seed_all(42)

    master_models = {}
    master_e3nn = create_model(table, None).to(device=device, dtype=torch.float64)
    master_models["e3nn"] = master_e3nn.state_dict()
    if CUET_AVAILABLE and device.startswith("cuda"):
        master_cueq = create_model(table, modules.wrapper_ops.CuEquivarianceConfig(
            enabled=True,
            optimize_all=True,
        )).to(device=device, dtype=torch.float64)
        master_models["cueq"] = master_cueq.state_dict()

    all_results = []
    total_configs = len(backend_configs) * len(precision_configs) * len(batch_sizes)
    config_count = 0

    for backend_name, backend_config in backend_configs:
        logging.info(f"\n--- Testing {backend_name.upper()} backend ---")
        for precision_name, dtype in precision_configs:
            logging.info(f"\n  Testing {precision_name.upper()} precision...")
            if backend_name == "cueq" and precision_name in ["bf16", "fp16"]:
                logging.info(f"    Skipping {precision_name} - cuEq doesn't support this precision")
                continue

            # Instantiate model from master weights per backend
            if backend_name == "e3nn":
                model = create_model(table, backend_config)
                model.load_state_dict(master_models["e3nn"])
                model = model.to(device=device, dtype=dtype)
            else:
                model = create_model(table, backend_config)
                model.load_state_dict(master_models["cueq"])
                model = model.to(device=device, dtype=dtype)

            # Log component presence
            logging.info(f"    Model has {len(model.interactions)} interactions")
            for i, interaction in enumerate(model.interactions):
                has = hasattr(interaction, "conv_tp_weights")
                logging.info(f"      Interaction {i}: conv_tp_weights present: {has}")

            for batch_size in batch_sizes:
                config_count += 1
                logging.info(f"    Batch size {batch_size} ({config_count}/{total_configs})")
                try:
                    batch_dict = create_batch_data(atoms_list, table, batch_size, device)
                    inputs, e_feats = prepare_inputs(model, batch_dict, device, dtype, table)

                    for interaction_idx in [0, 1]:
                        interaction_block = model.interactions[interaction_idx]
                        if not hasattr(interaction_block, "conv_tp_weights"):
                            logging.info(f"      Block {interaction_idx}: No conv_tp_weights found")
                            continue

                        conv_tp_weights_component = interaction_block.conv_tp_weights
                        logging.info(f"      Block {interaction_idx}: conv_tp_weights type: {type(conv_tp_weights_component).__name__}")

                        # Inputs for conv_tp_weights: edge_feats
                        tpw_inputs = {
                            "edge_feats": inputs["edge_feats"]
                        }

                        try:
                            results = benchmark_conv_tp_weights(
                                conv_tp_weights_component, tpw_inputs, num_iterations, warmup, device
                            )

                            result_row = {
                                "backend": backend_name,
                                "dtype": precision_name,
                                "batch_size": batch_size,
                                "interaction": interaction_idx,
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
                                f"      Block {interaction_idx} conv_tp_weights: "
                                f"Fwd: {results['forward_latency_ms']:.2f}ms, "
                                f"Bwd: {results['backward_latency_ms']:.2f}ms"
                            )
                        except Exception as e:
                            logging.info(f"      Error benchmarking block {interaction_idx}: {e}")
                            continue

                except Exception as e:
                    logging.info(f"      Error with batch size {batch_size}: {e}")
                    continue

    # Accuracy vs FP64 reference (per backend, batch, interaction)
    logging.info("\n--- Computing Accuracy Metrics ---")
    fp64_references = {}
    for result in all_results:
        if result["dtype"] == "fp64":
            key = (result["backend"], result["batch_size"], result["interaction"])
            fp64_references[key] = result
            logging.info(f"  FP64 Reference stored: {key}")

    for result in all_results:
        if result["dtype"] == "fp64":
            continue
        key = (result["backend"], result["batch_size"], result["interaction"])
        if key in fp64_references:
            ref = fp64_references[key]
            accuracy = compute_accuracy_metrics(ref["output"], ref["grad"], result["output"], result["grad"])
            result.update(accuracy)
        else:
            result.update({
                "max_abs_error_fwd": "N/A",
                "max_rel_error_fwd": "N/A",
                "max_abs_error_bwd": "N/A",
                "max_rel_error_bwd": "N/A",
            })

    # Speedups vs FP64 per backend
    logging.info("\n--- Computing Speedups ---")
    for result in all_results:
        if result["dtype"] == "fp64":
            continue
        key = (result["backend"], result["batch_size"], result["interaction"])
        if key in fp64_references:
            ref = fp64_references[key]
            fwd_speedup = ref["forward_latency_ms"] / result["forward_latency_ms"] if result["forward_latency_ms"] > 0 else 0
            bwd_speedup = ref["backward_latency_ms"] / result["backward_latency_ms"] if result["backward_latency_ms"] > 0 else 0
            result.update({
                "forward_speedup_vs_fp64": fwd_speedup,
                "backward_speedup_vs_fp64": bwd_speedup,
            })
        else:
            result.update({
                "forward_speedup_vs_fp64": "N/A",
                "backward_speedup_vs_fp64": "N/A",
            })

    save_results(all_results)
    generate_summary_report(all_results)
    return all_results


def save_results(results):
    """Save results to CSV file (next to this script in results/)"""
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    logging.info(f"  Saving {len(results)} results to CSV...")

    # Output directory: same dir as this script, in 'results'
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(script_dir, "results")
    os.makedirs(results_dir, exist_ok=True)

    column_mapping = {
        "backend": "Backend",
        "dtype": "Precision",
        "batch_size": "Batch Size",
        "interaction": "Interaction Block",
        "num_nodes": "Nodes per Molecule",
        "num_edges": "Edges per Molecule",
        "forward_latency_ms": "Forward Latency (ms)",
        "backward_latency_ms": "Backward Latency (ms)",
        "forward_std_ms": "Forward Std Dev (ms)",
        "backward_std_ms": "Backward Std Dev (ms)",
        "max_abs_error_fwd": "Max Abs Error Forward",
        "max_rel_error_fwd": "Max Rel Error Forward",
        "max_abs_error_bwd": "Max Abs Error Backward",
        "max_rel_error_bwd": "Max Rel Error Backward",
        "forward_speedup_vs_fp64": "Forward Speedup vs FP64",
        "backward_speedup_vs_fp64": "Backward Speedup vs FP64",
        "output": "Output Tensor",
        "grad": "Gradient Tensor",
    }

    # CSV rows (exclude tensors)
    csv_data = []
    for result in results:
        csv_row = {k: v for k, v in result.items() if not isinstance(v, torch.Tensor)}
        csv_data.append(csv_row)

    csv_path = os.path.join(results_dir, f"conv_tp_weights_benchmark_{timestamp}.csv")
    with open(csv_path, 'w', newline='') as csvfile:
        if csv_data:
            all_fieldnames = set()
            for row in csv_data:
                all_fieldnames.update(row.keys())
            sorted_fieldnames = sorted(all_fieldnames)
            proper_headers = [column_mapping.get(field, field.replace('_', ' ').title()) for field in sorted_fieldnames]
            writer = csv.writer(csvfile)
            writer.writerow(proper_headers)
            for row in csv_data:
                csv_row = []
                for fieldname in sorted_fieldnames:
                    csv_row.append(row.get(fieldname, ""))
                writer.writerow(csv_row)
            logging.info(f"  Wrote {len(csv_data)} rows to CSV")
        else:
            logging.warning("  No CSV data to write!")
    logging.info(f"Results saved to: {csv_path}")

    # JSON export removed per user request


def generate_summary_report(results):
    """Generate summary report comparing backends and precisions"""
    logging.info("\n=== Summary Report ===")

    grouped = defaultdict(lambda: defaultdict(list))
    for result in results:
        backend = result["backend"]
        dtype = result["dtype"]
        grouped[backend][dtype].append(result)

    for dtype in ["fp64", "fp32", "bf16", "fp16"]:
        logging.info(f"\n--- {dtype.upper()} Precision ---")
        e3nn_results = grouped["e3nn"].get(dtype, [])
        cueq_results = grouped["cueq"].get(dtype, [])
        if not e3nn_results:
            logging.info(f"  No e3nn results for {dtype}")
            continue
        if not cueq_results:
            logging.info(f"  No cuEq results for {dtype} (unsupported)")
            continue

        e3nn_by_batch = defaultdict(list)
        cueq_by_batch = defaultdict(list)
        for r in e3nn_results:
            e3nn_by_batch[r["batch_size"]].append(r)
        for r in cueq_results:
            cueq_by_batch[r["batch_size"]].append(r)

        for batch_size in sorted(set(e3nn_by_batch.keys()) & set(cueq_by_batch.keys())):
            logging.info(f"\n  Batch size {batch_size}:")
            for interaction in [0, 1]:
                logging.info(f"    --- Interaction Block {interaction} ---")
                e3nn_inter = [r for r in e3nn_results if r["batch_size"] == batch_size and r["interaction"] == interaction]
                cueq_inter = [r for r in cueq_results if r["batch_size"] == batch_size and r["interaction"] == interaction]
                if not e3nn_inter or not cueq_inter:
                    logging.info(f"      No results for interaction {interaction}")
                    continue
                e3nn_r = e3nn_inter[0]
                cueq_r = cueq_inter[0]
                e3nn_fwd = e3nn_r["forward_latency_ms"]
                e3nn_bwd = e3nn_r["backward_latency_ms"]
                cueq_fwd = cueq_r["forward_latency_ms"]
                cueq_bwd = cueq_r["backward_latency_ms"]
                fwd_speedup = e3nn_fwd / cueq_fwd if cueq_fwd > 0 else 0
                bwd_speedup = e3nn_bwd / cueq_bwd if cueq_bwd > 0 else 0
                logging.info(
                    f"      Forward:  e3nn {e3nn_fwd:.2f}ms ± {e3nn_r.get('forward_std_ms', 0):.2f}ms, "
                    f"cuEq {cueq_fwd:.2f}ms ± {cueq_r.get('forward_std_ms', 0):.2f}ms, speedup {fwd_speedup:.2f}x"
                )
                logging.info(
                    f"      Backward: e3nn {e3nn_bwd:.2f}ms ± {e3nn_r.get('backward_std_ms', 0):.2f}ms, "
                    f"cuEq {cueq_bwd:.2f}ms ± {cueq_r.get('backward_std_ms', 0):.2f}ms, speedup {bwd_speedup:.2f}x"
                )
                if "max_abs_error_fwd" in cueq_r and cueq_r["max_abs_error_fwd"] != "N/A":
                    logging.info(f"      Accuracy (cuEq vs e3nn FP64):")
                    logging.info(f"        Fwd abs error: {cueq_r['max_abs_error_fwd']:.2e}")
                    logging.info(f"        Bwd abs error: {cueq_r['max_abs_error_bwd']:.2e}")
                    logging.info(f"        Fwd rel error: {cueq_r['max_rel_error_fwd']:.2e}")
                    logging.info(f"        Bwd rel error: {cueq_r['max_rel_error_bwd']:.2e}")
                else:
                    logging.info(f"      Accuracy: N/A (no FP64 reference)")

    # FP64 references summary
    logging.info(f"\n--- FP64 Reference Storage ---")
    fp64_results = [r for r in results if r["dtype"] == "fp64"]
    if fp64_results:
        logging.info(
            f"  Stored FP64 references for {len(set((r['backend'], r['batch_size'], r['interaction']) for r in fp64_results))} configurations:"
        )
        for r in fp64_results:
            logging.info(f"    {r['backend']} backend, batch_size={r['batch_size']}, interaction={r['interaction']}")
    else:
        logging.info("  No FP64 references stored")


def main():
    parser = argparse.ArgumentParser(description="MACE conv_tp_weights Benchmark")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use (cuda/cpu)")
    parser.add_argument("--batch-sizes", type=str, default="1,4,8,16,32",
                        help="Comma-separated list of batch sizes")
    parser.add_argument("--iterations", type=int, default=100, help="Number of benchmark iterations")
    parser.add_argument("--warmup", type=int, default=50, help="Number of warmup iterations")

    args = parser.parse_args()

    torch.manual_seed(42)

    batch_sizes = [int(x.strip()) for x in args.batch_sizes.split(",")]

    if args.device == "cuda" and not torch.cuda.is_available():
        logging.info("CUDA not available, falling back to CPU")
        args.device = "cpu"

    if args.device.startswith("cuda"):
        torch.cuda.empty_cache()

    results = run_conv_tp_weights_benchmark(
        device=args.device,
        batch_sizes=batch_sizes,
        num_iterations=args.iterations,
        warmup=args.warmup
    )

    logging.info(f"\nBenchmark completed! Generated {len(results)} result entries.")


if __name__ == "__main__":
    main()


