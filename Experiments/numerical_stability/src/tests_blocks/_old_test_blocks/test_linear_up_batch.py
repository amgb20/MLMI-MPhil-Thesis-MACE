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

def run_linear_backward_pass(
    layer: LinearLayer,
    input_tensor: torch.Tensor,
):
    """
    Compute loss = out.pow(2).sum() for out = layer(input_tensor) and run backward.

    Returns a tuple of (loss, input_grad, param_grads_dict), where:
    - loss is a scalar tensor (detached)
    - input_grad matches input shape
    - param_grads_dict maps parameter names in layer.impl to their gradient tensors
    """
    # Ensure params have no stale grads
    if hasattr(layer, "impl"):
        layer.impl.zero_grad(set_to_none=True)
    else:
        layer.zero_grad(set_to_none=True)

    # Prepare input to require gradients and to match parameter dtype/device
    x = input_tensor.detach().clone()
    try:
        first_param = next(layer.impl.parameters())
        target_dtype = first_param.dtype
        target_device = first_param.device
    except StopIteration:
        # No parameters; fall back to layer device/dtype via output
        target_dtype = x.dtype
        target_device = x.device

    x = x.to(device=target_device, dtype=target_dtype)
    x.requires_grad_(True)

    # Forward -> loss -> backward
    out = layer(x)
    loss = out.pow(2).sum()
    loss.backward()

    # Collect gradients
    input_grad = x.grad.detach().clone() if x.grad is not None else None
    param_grads = {}
    for name, param in layer.impl.named_parameters():
        param_grads[name] = None if param.grad is None else param.grad.detach().clone()

    return loss.detach(), input_grad, param_grads

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

def benchmark_precision_effects(irreps_in, irreps_out, input_tensor, device, warmup=10, runs=50, use_cueq=False, layer_number=None, batch_size: int | None = None):
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
    try:
        total_samples = int(torch.tensor(input_tensor.shape[:-1]).prod().item())
    except Exception:
        total_samples = None
    if batch_size is None:
        # Infer from input tensor if it has at least two dims before feature dim
        if input_tensor.ndim >= 2:
            inferred_b = input_tensor.shape[-2] if input_tensor.ndim >= 2 else 1
            batch_size = inferred_b if inferred_b > 1 else 1
        else:
            batch_size = 1
    logger.info(f"Input shape: {tuple(input_tensor.shape)}, batch_size={batch_size}, total_samples={total_samples}")
    
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
    # FP64 backward baseline (loss and gradients)
    fp64_loss_base, fp64_in_grad_base, fp64_param_grads_base = run_linear_backward_pass(fp64_layer, input_tensor)
    
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
        # Throughput: samples/sec, define samples as product of leading dims (excluding feature dim)
        try:
            leading_elems = int(torch.tensor(input_precision.shape[:-1]).prod().item())
            samples_per_s = (leading_elems) / (time_ms / 1000.0)
        except Exception:
            samples_per_s = None
        
        # Test accuracy degradation
        accuracy_results = test_precision_accuracy(layer, input_precision, reference_output, precision_name)
        
        # Backward pass with loss = out.pow(2).sum()
        loss, input_grad, param_grads = run_linear_backward_pass(layer, input_precision)
        input_grad_norm = None if input_grad is None else input_grad.norm().item()
        total_param_grad_sq = 0.0
        per_param_grad_norms = {}
        for p_name, p_grad in param_grads.items():
            if p_grad is not None:
                gn = p_grad.norm().item()
                per_param_grad_norms[p_name] = gn
                total_param_grad_sq += gn * gn
        param_grad_norm = total_param_grad_sq ** 0.5

        # Compare backward results to FP64 baseline
        eps = 1e-12
        loss_abs_diff = abs(loss.item() - fp64_loss_base.item())
        loss_rel_diff = loss_abs_diff / max(abs(fp64_loss_base.item()), eps)

        # Input gradient diffs
        if input_grad is not None and fp64_in_grad_base is not None:
            g64 = fp64_in_grad_base.to(device=input_grad.device, dtype=input_grad.dtype)
            in_diff = input_grad - g64
            in_rel_l2 = in_diff.norm().item() / max(g64.norm().item(), eps)
            in_max_abs = in_diff.abs().max().item()
        else:
            in_rel_l2 = None
            in_max_abs = None

        # Parameter gradient diffs (aggregate and per-parameter)
        total_num_sq = 0.0
        total_den_sq = 0.0
        per_param_grad_rel_l2 = {}
        for name, g64 in fp64_param_grads_base.items():
            g = param_grads.get(name, None)
            if g is None or g64 is None:
                per_param_grad_rel_l2[name] = None
                continue
            g64_cast = g64.to(device=g.device, dtype=g.dtype)
            diff = g - g64_cast
            num = diff.norm().item()
            den = g64_cast.norm().item()
            per_param_grad_rel_l2[name] = None if den == 0.0 else num / max(den, eps)
            total_num_sq += float(diff.pow(2).sum().item())
            total_den_sq += float(g64_cast.pow(2).sum().item())
        total_param_grad_rel_l2 = (total_num_sq ** 0.5) / max(total_den_sq ** 0.5, eps)

        # Store results
        results[precision_name] = {
            'time_ms': time_ms,
            'samples_per_s': samples_per_s,
            'peak_mem_mb': peak_mem_bytes / 1024**2,
            'peak_mem_reserved_mb': peak_mem_reserved / 1024**2,
            'output_dtype': output_dtype,
            'accuracy': accuracy_results,
            'backward': {
                'loss': float(loss.item()),
                'input_grad_norm': input_grad_norm,
                'param_grad_norm': param_grad_norm,
                'per_param_grad_norms': per_param_grad_norms,
            },
            'backward_vs_fp64': {
                'loss_abs_diff': float(loss_abs_diff),
                'loss_rel_diff': float(loss_rel_diff),
                'input_grad_rel_l2': in_rel_l2,
                'input_grad_max_abs': in_max_abs,
                'param_grad_rel_l2': float(total_param_grad_rel_l2),
                'per_param_grad_rel_l2': per_param_grad_rel_l2,
            }
        }
        
        logger.info(f"  Performance: {time_ms:.2f} ms, Peak Mem Allocated: {peak_mem_bytes/1024**2:.2f} MB, Peak Mem Reserved: {peak_mem_reserved/1024**2:.2f} MB, Throughput: {samples_per_s:.2f} samples/s" if samples_per_s is not None else f"  Performance: {time_ms:.2f} ms, Peak Mem Allocated: {peak_mem_bytes/1024**2:.2f} MB, Peak Mem Reserved: {peak_mem_reserved/1024**2:.2f} MB")
        logger.info(f"  Output dtype: {output_dtype}")
        logger.info(
            f"  Backward: loss={loss.item():.4e}, grad_x_norm={input_grad_norm if input_grad_norm is None else f'{input_grad_norm:.4e}'}, grad_param_norm={param_grad_norm:.4e}"
        )
        logger.info(
            f"  Backward vs FP64: loss_rel_diff={loss_rel_diff:.3e}, input_grad_rel_l2={in_rel_l2 if in_rel_l2 is None else f'{in_rel_l2:.3e}'}, param_grad_rel_l2={total_param_grad_rel_l2:.3e}"
        )
    
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
    if 'samples_per_s' in results['FP64'] and results['FP64']['samples_per_s'] is not None:
        logger.info(f"\nThroughput:")
        logger.info(f"  FP64: {results['FP64']['samples_per_s']:.2f} samples/s")
        logger.info(f"  FP32: {results['FP32']['samples_per_s']:.2f} samples/s ({results['FP32']['samples_per_s']/results['FP64']['samples_per_s']:.2f}x)")
    
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
    # Ensure deterministic, inference-like behavior during tracing
    if isinstance(linear_up_fn, torch.nn.Module):
        linear_up_fn = linear_up_fn.eval()

    # Trace with relaxed checking; fall back to eager on failure
    try:
        traced_fn = torch.jit.trace(linear_up_fn, inputs, check_trace=False)
    except Exception as exc:
        logger.warning(f"Tracing failed, falling back to eager execution for benchmarking: {exc}")
        traced_fn = linear_up_fn
    
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
    B = 32  # batch size
    nf0 = torch.randn(B, N, in_ir0.dim, dtype=torch.float64, device=device)
    nf1 = torch.randn(B, N, in_ir1.dim, dtype=torch.float64, device=device)

    # For a fair comparison, disable TF32 when running e3nn on CUDA
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        try:
            # Force exact FP32 matmuls (no TF32)
            torch.set_float32_matmul_precision("highest")
        except Exception:
            pass

    # Test precision effects on e3nn linear layers (without cuEquivariance)
    logger.info("\n" + "="*60)
    logger.info("PRECISION EFFECTS TESTING (e3nn Linear Layers - No cuEquivariance)")
    logger.info("="*60)

    # Benchmark precision effects for linear_up0_e3nn
    logger.info("\n" + "="*60)
    logger.info("BENCHMARK: Precision Effects on e3nn Linear Layer 0")
    logger.info("="*60)
    precision_results_l0 = benchmark_precision_effects(in_ir0, out_ir0, nf0, device, use_cueq=False, layer_number=0, batch_size=B)
    analyze_precision_results(precision_results_l0)

    # Benchmark precision effects for linear_up1_e3nn
    logger.info("\n" + "="*60)
    logger.info("BENCHMARK: Precision Effects on e3nn Linear Layer 1")
    logger.info("="*60)
    precision_results_l1 = benchmark_precision_effects(in_ir1, out_ir1, nf1, device, use_cueq=False, layer_number=1, batch_size=B)
    analyze_precision_results(precision_results_l1)

if __name__ == "__main__":
    main()