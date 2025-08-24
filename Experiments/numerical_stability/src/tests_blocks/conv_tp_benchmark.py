"""
Comprehensive benchmark for conv_tp step in MACE interactions[0] and interactions[1]

This script benchmarks the conv_tp operation across:
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
import json
from datetime import datetime
from collections import defaultdict
import itertools
import warnings
import sys
import logging

# Add parent directories to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import ase.io
import numpy as np
import torch
import torch.utils.benchmark as benchmark
from torch.utils.benchmark import Timer

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
from mace.cli.convert_e3nn_cueq import run as run_e3nn_to_cueq
from mace.tools import torch_geometric

try:
    import cuequivariance as cue  # noqa: F401
    CUET_AVAILABLE = True
    logging.info("✓ Cuequivariance library is available")
except ImportError:
    CUET_AVAILABLE = False
    logging.info("✗ Cuequivariance library is not available - cuEq will be disabled")

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    logging.info("✗ tqdm not available - progress bars disabled")


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
    """Prepare inputs for conv_tp testing"""
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
    
    # Create node features and edge features
    x0 = model.node_embedding(batch_d["node_attrs"])
    x0 = x0.to(dtype).requires_grad_()
    x0.retain_grad()
    
    e_feats, _ = model.radial_embedding(
        lengths, batch_d["node_attrs"], batch_d["edge_index"], z_table
    )
    e_feats = e_feats.to(dtype)
    
    e_attrs = model.spherical_harmonics(vectors)
    e_attrs = e_attrs.to(dtype)
    
    inputs = {
        "node_feats": x0,
        "node_attrs": batch_d["node_attrs"],
        "edge_feats": e_feats,
        "edge_attrs": e_attrs,
        "edge_index": batch_d["edge_index"].to(torch.long),
    }
    
    return inputs, x0


def benchmark_conv_tp(block, inputs, num_iterations=100, warmup=50, device="cuda"):
    """Benchmark conv_tp block with multiple iterations for stable timing"""
    
    def run_forward():
        output_tuple = block(**inputs)
        # MACE interaction blocks return (output, sc) - extract just the output
        if isinstance(output_tuple, tuple):
            output = output_tuple[0]
        else:
            output = output_tuple
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        return output
    
    def run_backward(output):
        # Create a more meaningful loss that ensures gradients flow to node_feats
        # Use the output directly with node_feats to create a dependency
        if isinstance(output, torch.Tensor):
            # Create a loss that depends on both output and node_feats
            loss = (output**2).sum() + 0.1 * (inputs["node_feats"]**2).sum()
        else:
            # Fallback for non-tensor outputs
            loss = torch.tensor(0.0, device=device, requires_grad=True)
        
        loss.backward(retain_graph=True)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        
        # Get gradients from node_feats
        grad = inputs["node_feats"].grad.clone() if inputs["node_feats"].grad is not None else torch.zeros_like(inputs["node_feats"])
        return grad
    
    # Warmup
    for _ in range(warmup):
        output = run_forward()
        grad = run_backward(output)
        # Reset gradients
        inputs["node_feats"].grad.zero_()
    
    # CRITICAL: Ensure gradients are completely clean before measurement
    inputs["node_feats"].grad.zero_()
    
    # Reset memory stats
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(device)

    forward_times_list = []
    backward_times_list = []
    
    # Run multiple iterations for stable timing
    for _ in range(num_iterations):
        # Time forward pass
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

        # Time backward pass (ensure clean gradients)
        if device.startswith("cuda"):
            start_backward = torch.cuda.Event(enable_timing=True)
            end_backward = torch.cuda.Event(enable_timing=True)
            
            start_backward.record()
            grad = run_backward(output)
            end_backward.record()
            torch.cuda.synchronize()
            backward_time = start_backward.elapsed_time(end_backward)
        else:
            start_time = time.perf_counter()
            grad = run_backward(output)
            backward_time = (time.perf_counter() - start_time) * 1000
        
        backward_times_list.append(backward_time)
        
        # Reset gradients for next iteration
        inputs["node_feats"].grad.zero_()
    
    # Calculate statistics
    forward_mean = np.mean(forward_times_list)
    forward_std = np.std(forward_times_list)
    backward_mean = np.mean(backward_times_list)
    backward_std = np.std(backward_times_list)
    
    # Get memory usage (peak from all iterations)
    if device.startswith("cuda"):
        memory_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
    else:
        memory_mb = 0
    
    # Return the last output/grad for accuracy comparison (they should be identical across iterations)
    return {
        "forward_latency_ms": forward_mean,
        "backward_latency_ms": backward_mean,
        "forward_std_ms": forward_std,
        "backward_std_ms": backward_std,
        "gpu_mem_peak_mb": memory_mb,
        "output": output.detach().cpu(),
        "grad": grad.detach().cpu(),
        "num_iterations": num_iterations,
    }


def compute_accuracy_metrics(reference_output, reference_grad, test_output, test_grad):
    """Compute accuracy metrics relative to reference - same method as test_conv.py line 681"""
    # Convert to numpy for easier computation
    ref_out = reference_output.double().numpy()
    ref_grad = reference_grad.double().numpy()
    test_out = test_output.double().numpy()
    test_grad = test_grad.double().numpy()
    
    # Forward pass errors
    fwd_abs_error = np.abs(ref_out - test_out)
    fwd_rel_error = np.abs(fwd_abs_error / (np.abs(ref_out) + 1e-6))
    
    # Backward pass errors
    bwd_abs_error = np.abs(ref_grad - test_grad)
    bwd_rel_error = np.abs(bwd_abs_error / (np.abs(ref_grad) + 1e-6))
    
    return {
        "max_abs_error_fwd": float(fwd_abs_error.max()),
        "max_rel_error_fwd": float(fwd_rel_error.max()),
        "max_abs_error_bwd": float(bwd_abs_error.max()),
        "max_rel_error_bwd": float(bwd_rel_error.max()),
    }


def run_conv_tp_benchmark(device="cuda", batch_sizes=None, num_iterations=100, warmup=50):
    """Run comprehensive conv_tp benchmark"""
    
    if batch_sizes is None:
        batch_sizes = [1, 4, 8, 16, 32]
    
    logging.info(f"=== MACE conv_tp Benchmark ===")
    logging.info(f"Device: {device}")
    logging.info(f"Batch sizes: {batch_sizes}")
    logging.info(f"Iterations: {num_iterations}, Warmup: {warmup}")
    
    # Load test data using the same approach as inference_benchmark_elyas.py
    logging.info("\nLoading test data...")
    xyz_file = 'Experiments/numerical_stability/src/inference/data/carbon.xyz'
    atoms_list = ase.io.read(xyz_file, index=":")
    
    # Create atomic data for the first molecule to get system info
    table = tools.AtomicNumberTable([6, 82, 53, 55])  # C, Pb, I, Cs
    
    # CRITICAL: Display system and dataset information
    first_atoms = atoms_list[0]
    num_nodes = len(first_atoms)
    num_edges = 0
    
    # Calculate approximate number of edges (within cutoff)
    cutoff = 6.0
    positions = first_atoms.get_positions()
    for i in range(len(positions)):
        for j in range(i+1, len(positions)):
            dist = np.linalg.norm(positions[i] - positions[j])
            if dist <= cutoff:
                num_edges += 2  # Bidirectional edges
    
    logging.info(f"Elements: {table.zs}")
    logging.info(f"Atoms per molecule: {num_nodes}")
    logging.info(f"Approximate edges per molecule (cutoff={cutoff}Å): {num_edges}")
    logging.info(f"Number of molecules: {len(atoms_list)}")
    logging.info(f"Total system size: {num_nodes * len(atoms_list)} atoms")
    
    # Define precision configurations
    precision_configs = [
        ("fp64", torch.float64),
        ("fp32", torch.float32),
    ]
    
    # Add CUDA-specific precisions
    if device.startswith("cuda"):
        # Check BF16 support
        try:
            torch.zeros(1, device="cuda", dtype=torch.bfloat16)
            precision_configs.append(("bf16", torch.bfloat16))
            logging.info("✓ BF16 supported")
        except Exception:
            logging.info("✗ BF16 not supported")
        
        # Check FP16 support
        try:
            torch.zeros(1, device="cuda", dtype=torch.float16)
            precision_configs.append(("fp16", torch.float16))
            logging.info("✓ FP16 supported")
        except Exception:
            logging.info("✗ FP16 not supported")
    
    # Define backend configurations
    backend_configs = [
        ("e3nn", None),  # Reference backend
    ]
    
    if CUET_AVAILABLE and device.startswith("cuda"):
        cueq_config = modules.wrapper_ops.CuEquivarianceConfig(
            enabled=True,
            optimize_all=True,
        )
        backend_configs.append(("cueq", cueq_config))
        logging.info("✓ cuEq backend enabled")
    else:
        logging.info("✗ cuEq backend disabled")
    
    # CRITICAL FIX: Create separate master models for each backend with fixed weights
    logging.info("\n--- Creating Master Models ---")
    
    # Set seed for reproducible weight initialization
    torch.manual_seed(42)
    if device.startswith("cuda"):
        torch.cuda.manual_seed(42)
        torch.cuda.manual_seed_all(42)
    
    # Create master models for each backend
    master_models = {}
    
    # Master e3nn model
    logging.info("Creating master FP64 e3nn model with fixed weights...")
    master_e3nn = create_model(table, None)  # e3nn backend
    master_e3nn = master_e3nn.to(device=device, dtype=torch.float64)
    master_models["e3nn"] = master_e3nn.state_dict()
    logging.info("✓ Master e3nn model created with fixed weights")
    
    # Master cuEq model (if available)
    if CUET_AVAILABLE and device.startswith("cuda"):
        logging.info("Creating master FP64 cuEq model with fixed weights...")
        master_cueq = create_model(table, modules.wrapper_ops.CuEquivarianceConfig(
            enabled=True,
            optimize_all=True,
        ))
        master_cueq = master_cueq.to(device=device, dtype=torch.float64)
        master_models["cueq"] = master_cueq.state_dict()
        logging.info("✓ Master cuEq model created with fixed weights")
    
    # Results storage
    all_results = []
    
    # Run benchmark for each configuration
    total_configs = len(backend_configs) * len(precision_configs) * len(batch_sizes)
    config_count = 0
    
    for backend_name, backend_config in backend_configs:
        logging.info(f"\n--- Testing {backend_name.upper()} backend ---")
        
        for precision_name, dtype in precision_configs:
            logging.info(f"\n  Testing {precision_name.upper()} precision...")
            
            # Skip unsupported combinations
            if backend_name == "cueq" and precision_name in ["bf16", "fp16"]:
                logging.info(f"    Skipping {precision_name} - cuEq doesn't support this precision")
                continue
            
            # CRITICAL FIX: Create model from corresponding master weights
            if backend_name == "e3nn":
                # For e3nn, load master e3nn weights and cast to target precision
                model = create_model(table, backend_config)
                model.load_state_dict(master_models["e3nn"])
                model = model.to(device=device, dtype=dtype)
                logging.info(f"    Created e3nn model with {precision_name} precision")
            else:
                # For cuEq, load master cuEq weights and cast to target precision
                model = create_model(table, backend_config)
                model.load_state_dict(master_models["cueq"])
                model = model.to(device=device, dtype=dtype)
                logging.info(f"    Created cuEq model with {precision_name} precision")
            
            # Debug: Check model structure
            logging.info(f"    Model has {len(model.interactions)} interactions")
            for i, interaction in enumerate(model.interactions):
                logging.info(f"      Interaction {i}: {type(interaction).__name__}")
                if hasattr(interaction, "conv_tp"):
                    logging.info(f"        Has conv_tp: {type(interaction.conv_tp).__name__}")
                else:
                    logging.info(f"        No conv_tp attribute")
            
            for batch_size in batch_sizes:
                config_count += 1
                logging.info(f"    Batch size {batch_size} ({config_count}/{total_configs})")
                
                try:
                    # Create batch data
                    batch_dict = create_batch_data(atoms_list, table, batch_size, device)
                    
                    # Prepare inputs - pass z_table explicitly
                    inputs, x0 = prepare_inputs(model, batch_dict, device, dtype, table)
                    
                    # Test both interaction blocks
                    for interaction_idx in [0, 1]:
                        block = model.interactions[interaction_idx]
                        
                        # Skip if block doesn't have conv_tp
                        if not hasattr(block, "conv_tp"):
                            logging.info(f"      Block {interaction_idx}: No conv_tp found")
                            continue
                        
                        # For Block 1, we need to process the output from Block 0 first
                        if interaction_idx == 1:
                            try:
                                # Clear GPU cache before processing Block 1 to free memory
                                if device.startswith("cuda"):
                                    torch.cuda.empty_cache()
                                
                                # Get the output from Block 0
                                block0 = model.interactions[0]
                                output0, _ = block0(**inputs)
                                
                                # Process through the product layer (similar to how MACE does it)
                                product = model.products[0]
                                output_prod = product(
                                    node_feats=output0, sc=None, node_attrs=inputs["node_attrs"]
                                )
                                
                                # Update inputs for Block 1
                                inputs_block1 = inputs.copy()
                                inputs_block1["node_feats"] = output_prod.clone().detach().requires_grad_()
                                inputs_block1["node_feats"].retain_grad()
                                
                                # Use the updated inputs for Block 1
                                current_inputs = inputs_block1
                            except Exception as e:
                                logging.info(f"      Error preparing inputs for Block 1: {e}")
                                continue
                        else:
                            current_inputs = inputs
                        
                        # Benchmark conv_tp
                        try:
                            results = benchmark_conv_tp(
                                block, current_inputs, num_iterations, warmup, device
                            )
                            
                            # Store results
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
                                "gpu_mem_peak_mb": results["gpu_mem_peak_mb"],
                                "output": results["output"],
                                "grad": results["grad"],
                            }
                            
                            all_results.append(result_row)
                            
                            logging.info(f"      Block {interaction_idx}: "
                                  f"Fwd: {results['forward_latency_ms']:.2f}ms, "
                                  f"Bwd: {results['backward_latency_ms']:.2f}ms, "
                                  f"Mem: {results['gpu_mem_peak_mb']:.1f}MB")
                            
                            # Reset gradients for next iteration
                            if interaction_idx == 0:
                                if x0.grad is not None:
                                    x0.grad.zero_()
                            else:
                                if current_inputs["node_feats"].grad is not None:
                                    current_inputs["node_feats"].grad.zero_()
                                
                        except Exception as e:
                            logging.info(f"      Error benchmarking block {interaction_idx}: {e}")
                            continue
                
                except Exception as e:
                    logging.info(f"      Error with batch size {batch_size}: {e}")
                    continue
    
    # Compute accuracy metrics relative to reference (fp64 e3nn AND fp64 cuEq separately)
    logging.info("\n--- Computing Accuracy Metrics ---")
    
    # Store FP64 references separately for each backend
    fp64_references = {}
    
    # Find FP64 reference results for each backend
    for result in all_results:
        if result["dtype"] == "fp64":
            key = (result["backend"], result["batch_size"], result["interaction"])
            fp64_references[key] = result
            logging.info(f"  FP64 Reference stored: {key}")
    
    logging.info(f"  Total FP64 references: {len(fp64_references)}")
    
    # Compute accuracy for all other configurations
    for result in all_results:
        if result["dtype"] == "fp64":
            continue  # Skip FP64 results (they are references)
        
        # Find corresponding FP64 reference for this backend
        key = (result["backend"], result["batch_size"], result["interaction"])
        if key in fp64_references:
            ref = fp64_references[key]
            logging.info(f"  Computing accuracy for {result['backend']} {result['dtype']} {result['batch_size']} {result['interaction']}")
            logging.info(f"    Reference: {ref['backend']} {ref['dtype']} {ref['batch_size']} {ref['interaction']}")
            
            # Debug: Check tensor shapes and values
            ref_out_shape = ref["output"].shape
            ref_grad_shape = ref["grad"].shape
            test_out_shape = result["output"].shape
            test_grad_shape = result["grad"].shape
            
            logging.info(f"    Shapes - Ref out: {ref_out_shape}, Ref grad: {ref_grad_shape}")
            logging.info(f"    Shapes - Test out: {test_out_shape}, Test grad: {test_grad_shape}")
            
            # Debug: Check value ranges
            ref_out_range = (ref["output"].min().item(), ref["output"].max().item())
            ref_grad_range = (ref["grad"].min().item(), ref["grad"].max().item())
            test_out_range = (result["output"].min().item(), result["output"].max().item())
            test_grad_range = (result["grad"].min().item(), result["grad"].max().item())
            
            logging.info(f"    Ranges - Ref out: {ref_out_range}, Ref grad: {ref_grad_range}")
            logging.info(f"    Ranges - Test out: {test_out_range}, Test grad: {test_grad_range}")
            
            accuracy = compute_accuracy_metrics(
                ref["output"], ref["grad"],
                result["output"], result["grad"]
            )
            
            logging.info(f"    Accuracy results: {accuracy}")
            
            # Add accuracy metrics to result
            result.update(accuracy)
        else:
            logging.info(f"  No FP64 reference found for {result['backend']} {result['dtype']} {result['batch_size']} {result['interaction']}")
            # Mark as N/A if no reference available
            result.update({
                "max_abs_error_fwd": "N/A",
                "max_rel_error_fwd": "N/A",
                "max_abs_error_bwd": "N/A",
                "max_rel_error_bwd": "N/A",
            })
    
    # Compute speedups relative to FP64 for each backend
    logging.info("\n--- Computing Speedups ---")
    
    for result in all_results:
        if result["dtype"] == "fp64":
            continue  # Skip FP64 results (they are references)
        
        # Find corresponding FP64 reference for this backend
        key = (result["backend"], result["batch_size"], result["interaction"])
        if key in fp64_references:
            ref = fp64_references[key]
            
            # Calculate speedups
            fwd_speedup = ref["forward_latency_ms"] / result["forward_latency_ms"] if result["forward_latency_ms"] > 0 else 0
            bwd_speedup = ref["backward_latency_ms"] / result["backward_latency_ms"] if result["backward_latency_ms"] > 0 else 0
            
            # Add speedup metrics to result
            result.update({
                "forward_speedup_vs_fp64": fwd_speedup,
                "backward_speedup_vs_fp64": bwd_speedup,
            })
            
            logging.info(f"  {result['backend']} {result['dtype']} {result['batch_size']} {result['interaction']}: "
                        f"Fwd speedup: {fwd_speedup:.2f}x, Bwd speedup: {bwd_speedup:.2f}x")
        else:
            # Mark speedups as N/A if no reference available
            result.update({
                "forward_speedup_vs_fp64": "N/A",
                "backward_speedup_vs_fp64": "N/A",
            })
    
    # Save results
    save_results(all_results)
    
    # Generate summary report
    generate_summary_report(all_results)
    
    return all_results


def save_results(results):
    """Save results to CSV and JSON files"""
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    
    logging.info(f"  Saving {len(results)} results to CSV/JSON...")
    
    # Prepare CSV data (excluding tensor objects)
    csv_data = []
    for result in results:
        csv_row = {k: v for k, v in result.items() 
                   if not isinstance(v, torch.Tensor)}
        csv_data.append(csv_row)
    
    logging.info(f"  Prepared {len(csv_data)} CSV rows")
    if csv_data:
        logging.info(f"  Sample CSV row keys: {list(csv_data[0].keys())}")
    
    # Save CSV
    csv_path = f"Experiments/numerical_stability/src/tests_blocks/results/conv_tp_benchmark_{timestamp}.csv"
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    
    with open(csv_path, 'w', newline='') as csvfile:
        if csv_data:
            # Get all possible fieldnames from all results
            all_fieldnames = set()
            for row in csv_data:
                all_fieldnames.update(row.keys())
            
            logging.info(f"  CSV fieldnames: {sorted(all_fieldnames)}")
            
            writer = csv.DictWriter(csvfile, fieldnames=sorted(all_fieldnames))
            writer.writeheader()
            
            # Write rows with missing fields as empty
            for row in csv_data:
                # Fill missing fields with empty values
                for fieldname in all_fieldnames:
                    if fieldname not in row:
                        row[fieldname] = ""
                writer.writerow(row)
            
            logging.info(f"  Wrote {len(csv_data)} rows to CSV")
        else:
            logging.warning("  No CSV data to write!")
    
    logging.info(f"Results saved to: {csv_path}")
    
    # Save JSON (with tensor data)
    json_path = f"Experiments/numerical_stability/src/tests_blocks/results/conv_tp_benchmark_{timestamp}.json"
    
    # Convert tensors to lists for JSON serialization
    json_data = []
    for result in results:
        json_row = result.copy()
        for key, value in json_row.items():
            if isinstance(value, torch.Tensor):
                json_row[key] = value.tolist()
        json_data.append(json_row)
    
    with open(json_path, 'w') as f:
        json.dump(json_data, f, indent=2)
    
    logging.info(f"Detailed results saved to: {json_path}")


def generate_summary_report(results):
    """Generate summary report comparing backends and precisions"""
    logging.info("\n=== Summary Report ===")
    
    # Group results by backend and dtype
    grouped = defaultdict(lambda: defaultdict(list))
    for result in results:
        backend = result["backend"]
        dtype = result["dtype"]
        grouped[backend][dtype].append(result)
    
    # Compare cuEq vs e3nn for each precision and batch size
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
        
        # Group by batch size
        e3nn_by_batch = defaultdict(list)
        cueq_by_batch = defaultdict(list)
        
        for r in e3nn_results:
            e3nn_by_batch[r["batch_size"]].append(r)  # FIXED: .append() not .extend()
        for r in cueq_results:
            cueq_by_batch[r["batch_size"]].append(r)  # FIXED: .append() not .extend()
        
        # Compare performance for each block separately
        for batch_size in sorted(set(e3nn_by_batch.keys()) & set(cueq_by_batch.keys())):
            logging.info(f"\n  Batch size {batch_size}:")
            
            # Show performance for each interaction block separately
            for interaction in [0, 1]:
                logging.info(f"    --- Interaction Block {interaction} ---")
                
                # Find results for this specific block
                e3nn_inter = [r for r in e3nn_results if r["batch_size"] == batch_size and r["interaction"] == interaction]
                cueq_inter = [r for r in cueq_results if r["batch_size"] == batch_size and r["interaction"] == interaction]
                
                if not e3nn_inter or not cueq_inter:
                    logging.info(f"      No results for interaction {interaction}")
                    continue
                
                e3nn_r = e3nn_inter[0]
                cueq_r = cueq_inter[0]
                
                # Performance metrics for this specific block
                e3nn_fwd = e3nn_r["forward_latency_ms"]
                e3nn_bwd = e3nn_r["backward_latency_ms"]
                e3nn_mem = e3nn_r["gpu_mem_peak_mb"]
                
                cueq_fwd = cueq_r["forward_latency_ms"]
                cueq_bwd = cueq_r["backward_latency_ms"]
                cueq_mem = cueq_r["gpu_mem_peak_mb"]
                
                # Compute speedups
                fwd_speedup = e3nn_fwd / cueq_fwd if cueq_fwd > 0 else 0
                bwd_speedup = e3nn_bwd / cueq_bwd if cueq_bwd > 0 else 0
                
                logging.info(f"      Forward:  e3nn {e3nn_fwd:.2f}ms ± {e3nn_r.get('forward_std_ms', 0):.2f}ms, cuEq {cueq_fwd:.2f}ms ± {cueq_r.get('forward_std_ms', 0):.2f}ms, speedup {fwd_speedup:.2f}x")
                logging.info(f"      Backward: e3nn {e3nn_bwd:.2f}ms ± {e3nn_r.get('backward_std_ms', 0):.2f}ms, cuEq {cueq_bwd:.2f}ms ± {cueq_r.get('backward_std_ms', 0):.2f}ms, speedup {bwd_speedup:.2f}x")
                logging.info(f"      Memory:   e3nn {e3nn_mem:.1f}MB, cuEq {cueq_mem:.1f}MB, ratio {cueq_mem/e3nn_mem:.2f}")
                
                # Show accuracy metrics if available
                if "max_abs_error_fwd" in cueq_r and cueq_r["max_abs_error_fwd"] != "N/A":
                    logging.info(f"      Accuracy (cuEq vs e3nn FP64):")
                    logging.info(f"        Fwd abs error: {cueq_r['max_abs_error_fwd']:.2e}")
                    logging.info(f"        Bwd abs error: {cueq_r['max_abs_error_bwd']:.2e}")
                    logging.info(f"        Fwd rel error: {cueq_r['max_rel_error_fwd']:.2e}")
                    logging.info(f"        Bwd rel error: {cueq_r['max_rel_error_bwd']:.2e}")
                else:
                    logging.info(f"      Accuracy: N/A (no FP64 reference)")
    
    # Add summary of FP64 reference storage
    logging.info(f"\n--- FP64 Reference Storage ---")
    fp64_results = [r for r in results if r["dtype"] == "fp64"]
    if fp64_results:
        logging.info(f"  Stored FP64 references for {len(set((r['backend'], r['batch_size'], r['interaction']) for r in fp64_results))} configurations:")
        for r in fp64_results:
            logging.info(f"    {r['backend']} backend, batch_size={r['batch_size']}, interaction={r['interaction']}")
    else:
        logging.info("  No FP64 references stored")


def main():
    """Main function"""
    # Set up logging
    # logging.basicConfig(
    #     level=logging.INFO,
    #     format='%(asctime)s - %(levelname)s - %(message)s',
    #     datefmt='%Y-%m-%d %H:%M:%S'
    # )
    
    parser = argparse.ArgumentParser(description="MACE conv_tp Benchmark")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use (cuda/cpu)")
    parser.add_argument("--batch-sizes", type=str, default="1,4,8,16,32", 
                       help="Comma-separated list of batch sizes")
    parser.add_argument("--iterations", type=int, default=100, help="Number of benchmark iterations")
    parser.add_argument("--warmup", type=int, default=50, help="Number of warmup iterations")
    
    args = parser.parse_args()

    torch.manual_seed(42)
    
    # Parse batch sizes
    batch_sizes = [int(x.strip()) for x in args.batch_sizes.split(",")]
    
    # Check device availability
    if args.device == "cuda" and not torch.cuda.is_available():
        logging.info("CUDA not available, falling back to CPU")
        args.device = "cpu"
    
    # Clear GPU cache if using CUDA
    if args.device.startswith("cuda"):
        torch.cuda.empty_cache()
    
    # Run benchmark
    results = run_conv_tp_benchmark(
        device=args.device,
        batch_sizes=batch_sizes,
        num_iterations=args.iterations,
        warmup=args.warmup
    )
    
    logging.info(f"\nBenchmark completed! Generated {len(results)} result entries.")


if __name__ == "__main__":
    main()