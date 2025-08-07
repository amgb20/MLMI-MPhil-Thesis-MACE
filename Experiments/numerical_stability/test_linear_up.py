import torch
from mace import data, modules, tools
import numpy as np
import torch.nn.functional
from e3nn import o3
import ase.io
import matplotlib.pyplot as plt
import warnings
import copy
import pandas as pd
warnings.filterwarnings("ignore")
import logging
import types
import copy
from e3nn.o3 import Irreps
import cuequivariance   as cue
import cuequivariance_torch as cuet
import memory_test.get_memory_allocation as get_memory_allocation

from utils.get_logging_profile import logger
from utils.config import get_default_model_config, data_prep
from utils.get_gpu_details import get_gpu_with_least_memory

try:
    import cuequivariance as cue
    cueq_available = True
    logger.info("✓ cuEquivariance library is available")
except ImportError:
    cueq_available = False
    logger.info("✗ cuEquivariance library is not available - cuEq will be disabled")

import torch
from e3nn import o3
import gc

# Flag: is cuEq available?
try:
    import cuequivariance        as cue
    import cuequivariance_torch  as cuet
    CUET_AVAILABLE = True
except ImportError:
    CUET_AVAILABLE = False


def ensure_torch_device(device_like) -> torch.device:
    """Return a valid torch.device from various inputs."""
    if isinstance(device_like, torch.device):
        return device_like
    if isinstance(device_like, int):
        return torch.device(f"cuda:{device_like}")
    if isinstance(device_like, str):
        return torch.device(device_like)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def is_cuda_device(device: torch.device) -> bool:
    try:
        return device is not None and device.type == "cuda" and torch.cuda.is_available()
    except Exception:
        return False


def copy_state_dict_cast(
    src_module: torch.nn.Module,
    dst_module: torch.nn.Module,
    dtype: torch.dtype,
    device: torch.device,
):
    """
    Copy parameters/buffers from src to dst, casting to dtype and moving to device.
    Falls back silently if unexpected keys exist.
    """
    try:
        src_state = src_module.impl.state_dict()
        cast_state = {}
        for key, value in src_state.items():
            if torch.is_tensor(value):
                cast_state[key] = value.detach().to(device=device, dtype=dtype)
            else:
                cast_state[key] = value
        # Load into destination; ignore unexpected/missing to be robust across backends
        dst_module.impl.load_state_dict(cast_state, strict=False)
    except Exception as exc:
        logger.warning(f"Could not copy state dict between modules: {exc}")


class LinearLayer(torch.nn.Module):
    """
    Thin wrapper over cuet.Linear or e3nn.o3.Linear with a unified constructor
    and a single-forward signature: y = W x.
    """

    def __init__(
        self,
        irreps_in: o3.Irreps,
        irreps_out: o3.Irreps,
        *,
        cueq_config=None,          # your CuEquivarianceConfig
        shared_weights: bool = True,
        internal_weights: bool = True,
        use_cueq: bool = True,
        math_dtype: torch.dtype,
        device: torch.device
    ):
        super().__init__()
        if use_cueq:
            self.impl = cuet.Linear(
                cue.Irreps("O3", irreps_in),
                cue.Irreps("O3", irreps_out),
                layout="mul_ir",
                shared_weights=shared_weights,
                math_dtype=math_dtype,
                use_fallback=False,
            ).to(device)
        else:
            self.impl = o3.Linear(
                irreps_in,
                irreps_out,
                shared_weights=shared_weights,
                internal_weights=internal_weights,
            ).to(device)
            # Ensure parameters are in the correct dtype
            for param in self.impl.parameters():
                param.data = param.data.to(math_dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : Tensor of shape [..., irreps_in.dim]
        returns y : Tensor of shape [..., irreps_out.dim]
        """
        output = self.impl(x)
        # Ensure output is in the same dtype as input
        if output.dtype != x.dtype:
            output = output.to(x.dtype)
        return output

def test_precision_accuracy(layer, input_tensor, reference_output, precision_name):
    """
    Test accuracy degradation for a given precision compared to FP64 reference.
    
    Args:
        input_tensor: Input tensor
        reference_output: FP64 reference output (keep as FP64 for true comparison)
        precision_name: Name of the precision being tested
    
    Returns:
        relative_error: Relative error compared to FP64 reference
        absolute_error: Absolute error compared to FP64 reference
    """
    with torch.inference_mode():
        
        logger.info(f"input_tensor.dtype: {input_tensor.dtype}")
        logger.info(f"reference_output.dtype: {reference_output.dtype}")

        current_output = layer(input_tensor)
        
        # Calculate errors against FP64 reference
        absolute_error = torch.abs(current_output - reference_output)
        relative_error = absolute_error / (torch.abs(reference_output).clamp_min(1e-6))  # Add small epsilon to avoid division by zero
        
        # Compute statistics
        max_abs_error = torch.max(absolute_error).item()
        mean_abs_error = torch.mean(absolute_error).item()
        max_rel_error = torch.max(relative_error).item()
        mean_rel_error = torch.mean(relative_error).item()
        
        logger.info(f"  {precision_name} Accuracy vs FP64 Reference:")
        logger.info(f"    Max Absolute Error: {max_abs_error:.2e}")
        logger.info(f"    Mean Absolute Error: {mean_abs_error:.2e}")
        logger.info(f"    Max Relative Error: {max_rel_error:.2e}")
        logger.info(f"    Mean Relative Error: {mean_rel_error:.2e}")
        
        return {
            'max_abs_error': max_abs_error,
            'mean_abs_error': mean_abs_error,
            'max_rel_error': max_rel_error,
            'mean_rel_error': mean_rel_error
        }

def benchmark_precision_effects(irreps_in, irreps_out, input_tensor, device, warmup=10, runs=50, use_cueq=False, layer_number=None):
    """
    Benchmark e3nn linear layers at different precisions and measure accuracy degradation.
    
    Args:
        irreps_in: Input irreps
        irreps_out: Output irreps
        input_tensor: Input tensor (FP64)
        device: Device to run on
        warmup: Number of warmup runs
        runs: Number of benchmark runs
    
    Returns:
        results: Dictionary with benchmark and accuracy results
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"PRECISION EFFECTS TESTING (e3nn Linear Layers)")
    logger.info(f"{'='*60}")
    
    # Create layers at different precisions - only FP64 and FP32
    precisions = [
        (torch.float64, "FP64"),
        (torch.float32, "FP32")
    ]
    
    results = {}
    
    # First, get FP64 reference output with a single parameter set to be reused
    logger.info(f"\nGenerating FP64 reference output with fixed weights...")
    fp64_layer = LinearLayer(
        irreps_in,
        irreps_out,
        use_cueq=use_cueq,
        math_dtype=torch.float64,
        device=device,
    )
    with torch.inference_mode():
        reference_output = fp64_layer(input_tensor)
    
    # Test each precision
    for math_dtype, precision_name in precisions:
        logger.info(f"\n{'-'*40}")
        logger.info(f"Testing {precision_name} precision...")
        logger.info(f"{'-'*40}")
        
        # Create or reuse layer with specific precision, ensuring identical weights
        if precision_name == "FP64":
            layer = fp64_layer
            input_precision = input_tensor
        else:
            layer = LinearLayer(
                irreps_in,
                irreps_out,
                use_cueq=use_cueq,
                math_dtype=math_dtype,
                device=device,
            )
            copy_state_dict_cast(fp64_layer, layer, dtype=math_dtype, device=device)
            # Convert input to same precision for fair comparison
            input_precision = input_tensor.to(math_dtype)
        
        # Benchmark performance
        time_ms, peak_mem_bytes, peak_mem_reserved, output_dtype = benchmark_linear_up(layer, input_precision, warmup, runs, precision_name, layer_number, device)
        
        # Test accuracy degradation
        accuracy_results = test_precision_accuracy(layer, input_precision, reference_output, precision_name)
        
        # Store results
        results[precision_name] = {
            'time_ms': time_ms,
            'peak_mem_mb': peak_mem_bytes / 1024**2,
            'peak_mem_reserved_mb': peak_mem_reserved / 1024**2,
            'output_dtype': output_dtype,
            'accuracy': accuracy_results
        }
        
        logger.info(f"  Performance: {time_ms:.2f} ms, Peak Mem Allocated: {peak_mem_bytes/1024**2:.2f} MB, Peak Mem Reserved: {peak_mem_reserved/1024**2:.2f} MB")
        logger.info(f"  Output dtype: {output_dtype}")
    
    return results

def analyze_precision_results(results):
    """
    Analyze and display precision effects results.
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"PRECISION EFFECTS ANALYSIS (FP32 vs FP64)")
    logger.info(f"{'='*60}")
    
    # Performance comparison
    logger.info(f"\nPerformance Comparison:")
    fp64_time = results['FP64']['time_ms']
    fp32_time = results['FP32']['time_ms']
    
    logger.info(f"  FP64: {fp64_time:.2f} ms (baseline)")
    logger.info(f"  FP32: {fp32_time:.2f} ms ({fp64_time/fp32_time:.2f}x speedup)")
    
    # Memory comparison
    logger.info(f"\nMemory Usage:")
    fp64_mem = results['FP64']['peak_mem_mb']
    fp32_mem = results['FP32']['peak_mem_mb']
    
    logger.info(f"  FP64: {fp64_mem:.2f} MB (baseline)")
    logger.info(f"  FP32: {fp32_mem:.2f} MB ({fp64_mem/fp32_mem:.2f}x memory reduction)")
    
    # Accuracy degradation analysis
    logger.info(f"\nAccuracy Degradation Analysis:")
    fp64_acc = results['FP64']['accuracy']
    fp32_acc = results['FP32']['accuracy']
    
    logger.info(f"  FP32 vs FP64:")
    logger.info(f"    Max Absolute Error: {fp32_acc['max_abs_error']:.2e}")
    logger.info(f"    Mean Absolute Error: {fp32_acc['mean_abs_error']:.2e}")
    logger.info(f"    Max Relative Error: {fp32_acc['max_rel_error']:.2e}")
    logger.info(f"    Mean Relative Error: {fp32_acc['mean_rel_error']:.2e}")
    
    # Summary
    logger.info(f"\nSummary:")
    logger.info(f"  ✓ FP32 provides speedup: {fp64_time/fp32_time:.2f}x faster than FP64")
    logger.info(f"  ✓ FP32 reduces memory usage: {fp64_mem/fp32_mem:.2f}x less memory than FP64")
    logger.info(f"  ✓ FP32 causes accuracy degradation: {fp32_acc['max_rel_error']:.2e} max relative error")
    
    return results


def benchmark_linear_up(linear_up_fn, inputs, warmup=10, runs=50, precision_name=None, layer_number=None, device=None):

    traced_fn = torch.jit.trace(linear_up_fn, inputs)
    
    with torch.inference_mode():
        # if the inputs is a tuple, unpack it
        if isinstance(inputs, tuple):
            _ = traced_fn(*inputs)
        else:
            _ = traced_fn(inputs)
    if is_cuda_device(device):
        torch.cuda.synchronize(device)

    with torch.inference_mode():
        for _ in range(warmup):
            if isinstance(inputs, tuple):
                traced_fn(*inputs)
            else:
                traced_fn(inputs)
    if is_cuda_device(device):
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()
    gc.collect()
    if is_cuda_device(device):
        torch.cuda.reset_peak_memory_stats(device=device)

    if is_cuda_device(device):
        get_memory_allocation.start_record_memory_history()

    time_ms = []
    with torch.inference_mode():
        if is_cuda_device(device):
            for _ in range(runs):
                start = torch.cuda.Event(enable_timing=True)
                end   = torch.cuda.Event(enable_timing=True)
                start.record()
                if isinstance(inputs, tuple):
                    traced_fn(*inputs)
                else:
                    traced_fn(inputs)
                end.record()
                torch.cuda.synchronize(device)
                time_ms.append(start.elapsed_time(end))
        else:
            import time
            for _ in range(runs):
                t0 = time.perf_counter()
                if isinstance(inputs, tuple):
                    traced_fn(*inputs)
                else:
                    traced_fn(inputs)
                t1 = time.perf_counter()
                time_ms.append((t1 - t0) * 1000.0)

    mean_time_ms = (sum(time_ms)/len(time_ms))
    if is_cuda_device(device):
        peak_mem_bytes = torch.cuda.max_memory_allocated(device)
        peak_mem_reserved = torch.cuda.max_memory_reserved(device)
        get_memory_allocation.export_memory_snapshot(precision_name, layer_number)
        get_memory_allocation.stop_record_memory_history()
    else:
        peak_mem_bytes = 0
        peak_mem_reserved = 0
    
    # Test output precision
    test_output = traced_fn(*inputs) if isinstance(inputs, tuple) else traced_fn(inputs)
    output_dtype = test_output.dtype
    
    return mean_time_ms, peak_mem_bytes, peak_mem_reserved, output_dtype

def main():
    # Check available devices and select the best one
    if torch.cuda.is_available():
        device = ensure_torch_device(get_gpu_with_least_memory())
        logger.info(f"Using device: {device}")
    else:
        device = torch.device("cpu")
        logger.info("CUDA not available, using CPU")
    
    torch.manual_seed(0)
    
    _, _, _, z_table = data_prep()
    cfg   = get_default_model_config(z_table)
    model = modules.MACE(**cfg).to(device=device, dtype=torch.float64)

    # grab the two TensorProduct modules because we want to get the dimensions of the weights
    tp0 = model.interactions[0].conv_tp # first layer
    tp1 = model.interactions[1].conv_tp # second layer

    # turn their irreps into e3nn.Irreps to get dimensions
    in_ir0, attr_ir0, out_ir0 = (
        Irreps(tp0.irreps_in1),
        Irreps(tp0.irreps_in2),
        Irreps(tp0.irreps_out),
    )
    in_ir1, attr_ir1, out_ir1 = (
        Irreps(tp1.irreps_in1),
        Irreps(tp1.irreps_in2),
        Irreps(tp1.irreps_out),
    )

    N = 1000
    nf0 = torch.randn(N, in_ir0.dim, dtype=torch.float64).to(device)
    nf1 = torch.randn(N, in_ir1.dim, dtype=torch.float64).to(device)

    # Test precision effects on e3nn linear layers (without cuEquivariance)
    logger.info("\n" + "="*60)
    logger.info("PRECISION EFFECTS TESTING (e3nn Linear Layers - No cuEquivariance)")
    logger.info("="*60)

    # Benchmark precision effects for linear_up0_e3nn
    logger.info("\n" + "="*60)
    logger.info("BENCHMARK: Precision Effects on e3nn Linear Layer 0")
    logger.info("="*60)
    precision_results_l0 = benchmark_precision_effects(in_ir0, out_ir0, nf0, device, use_cueq=False, layer_number=0)
    analyze_precision_results(precision_results_l0)

    # Benchmark precision effects for linear_up1_e3nn
    logger.info("\n" + "="*60)
    logger.info("BENCHMARK: Precision Effects on e3nn Linear Layer 1")
    logger.info("="*60)
    precision_results_l1 = benchmark_precision_effects(in_ir1, out_ir1, nf1, device, use_cueq=False, layer_number=1)
    analyze_precision_results(precision_results_l1)

if __name__ == "__main__":
    main()