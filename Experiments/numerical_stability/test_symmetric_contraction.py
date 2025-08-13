# import sys
# sys.path.insert(0, '/homes/ab3149/Documents/MLMI-MPhil-Thesis-MACE')

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

import mace.modules.symmetric_contraction as SymmetricContraction

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
        # Prefer copying the wrapped impl/sc module if present
        state_src = (
            src_module.sc.state_dict() if hasattr(src_module, "sc") else src_module.state_dict()
        )
        cast_state = {}
        for key, value in state_src.items():
            if torch.is_tensor(value):
                cast_state[key] = value.detach().to(device=device, dtype=dtype)
            else:
                cast_state[key] = value
        if hasattr(dst_module, "sc"):
            dst_module.sc.load_state_dict(cast_state, strict=False)
        else:
            dst_module.load_state_dict(cast_state, strict=False)
    except Exception as exc:
        logger.warning(f"Could not copy state dict between modules: {exc}")


class SymmetricContractionWrapper(torch.nn.Module):
    """
    Thin wrapper over cuet.SymmetricContraction and SymmetricContraction 
    """

    def __init__(
        self,
        irreps_in: o3.Irreps,
        irreps_out: o3.Irreps,
        *,
        cueq_config=None,          # your CuEquivarianceConfig
        shared_weights: bool = True,
        internal_weights: bool = True,
        correlation: int,
        num_elements: int,
        use_cueq: bool = True,
        math_dtype: torch.dtype,
        device: torch.device
    ):
        super().__init__()
        self.is_cueq = bool(use_cueq)
        if self.is_cueq:
            self.sc = cuet.SymmetricContraction(
                cue.Irreps('O3', irreps_in),
                cue.Irreps('O3', irreps_out),
                layout_in=cue.ir_mul,
                layout_out=cue.mul_ir,
                contraction_degree=correlation,
                num_elements=num_elements,
                original_mace=True,
                dtype=math_dtype,
                math_dtype=math_dtype,
               ).to(device)
        else:
            prev_dtype = torch.get_default_dtype()
            try:
                torch.set_default_dtype(math_dtype)
                self.sc = SymmetricContraction.SymmetricContraction(
                    irreps_in=irreps_in,
                    irreps_out=irreps_out,
                    correlation=correlation,
                    num_elements=num_elements,
                ).to(device)
            finally:
                torch.set_default_dtype(prev_dtype)
            # Ensure parameters are in the correct dtype
            for param in self.sc.parameters():
                param.data = param.data.to(math_dtype)

    def forward(self, x: torch.Tensor, attrs_one_hot: torch.Tensor) -> torch.Tensor:
        """
        x : Tensor with shape [N, C, D] (mul_ir layout); attrs_one_hot: [N, num_elements]
        returns y : Tensor of shape [N, irreps_out.dim]
        """
        # Cast inputs to module dtype to avoid dtype mismatches
        param_example = next(self.sc.parameters(), None)
        target_dtype = param_example.dtype if param_example is not None else x.dtype
        if x.dtype != target_dtype:
            x = x.to(target_dtype)
        if attrs_one_hot.dtype != target_dtype:
            attrs_one_hot = attrs_one_hot.to(target_dtype)
        if self.is_cueq:
            # Match blocks.py: convert mul_ir [N, C, D] -> ir_mul [N, D, C], then flatten
            x = x.transpose(1, 2).contiguous()
            x_in = x.flatten(1).contiguous()
            # Convert one-hot attrs to integer indices
            indices = torch.nonzero(attrs_one_hot)[:, 1].to(torch.int32)
            output = self.sc(x_in, indices=indices)
        else:
            # e3nn expects (features [N, C, D], one-hot attrs [N, E])
            output = self.sc(x, attrs_one_hot)
        # Ensure output is in the same dtype as input
        if output.dtype != x.dtype:
            output = output.to(x.dtype)
        return output

def run_linear_backward_pass(
    layer: SymmetricContractionWrapper,
    inputs,
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

    # Unpack inputs (features, attrs)
    if isinstance(inputs, tuple):
        x, attrs = inputs
    else:
        x, attrs = inputs, None
    # Prepare input to require gradients and to match parameter dtype/device
    x = x.detach().clone()
    try:
        first_param = next((layer.sc.parameters() if hasattr(layer, "sc") else layer.parameters()))
        target_dtype = first_param.dtype
        target_device = first_param.device
    except StopIteration:
        target_dtype = x.dtype
        target_device = x.device

    x = x.to(device=target_device, dtype=target_dtype)
    if attrs is not None:
        attrs = attrs.to(device=target_device, dtype=target_dtype)
    x.requires_grad_(True)

    # Forward -> loss -> backward
    out = layer(x, attrs) if attrs is not None else layer(x)
    loss = out.pow(2).sum()
    loss.backward()

    # Collect gradients
    input_grad = x.grad.detach().clone() if x.grad is not None else None
    param_grads = {}
    for name, param in (layer.sc.named_parameters() if hasattr(layer, "sc") else layer.named_parameters()):
        param_grads[name] = None if param.grad is None else param.grad.detach().clone()

    return loss.detach(), input_grad, param_grads

def test_precision_accuracy(layer, inputs, reference_output, precision_name):
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
        
        # If tuple, report feature dtype
        feat_dtype = inputs[0].dtype if isinstance(inputs, tuple) else inputs.dtype
        logger.info(f"input_tensor.dtype: {feat_dtype}")
        logger.info(f"reference_output.dtype: {reference_output.dtype}")

        if isinstance(inputs, tuple):
            current_output = layer(*inputs)
        else:
            current_output = layer(inputs)
        
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

def benchmark_precision_effects(irreps_in, irreps_out, inputs, device, correlation, num_elements, warmup=10, runs=50, use_cueq=False, layer_number=None):
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
    logger.info(f"PRECISION EFFECTS TESTING (Symmetric Contraction)")
    logger.info(f"{'='*60}")
    
    # Precision plan:
    # - cuEq: FP32 only. Use FP32 as reference.
    # - e3nn: FP64 reference; compare FP16 and BF16 (as requested) against FP64.
    if use_cueq:
        ref_dtype = torch.float32
        precisions = [(torch.float32, "FP32")]
    else:
        ref_dtype = torch.float64
        precisions = [
            (torch.float64, "FP64"),
            (torch.float16, "FP16"),
            (torch.bfloat16, "BF16"),
        ]
    
    results = {}
    
    # First, get FP64 reference output with a single parameter set to be reused
    logger.info(f"\nGenerating {ref_dtype} reference output with fixed weights...")
    # Always build reference with e3nn (FP64 baseline independent of cuEq)
    ref_layer = SymmetricContractionWrapper(
        irreps_in,
        irreps_out,
        correlation=correlation,
        num_elements=num_elements,
        use_cueq=False,
        math_dtype=ref_dtype,
        device=device,
    )
    with torch.inference_mode():
        reference_output = ref_layer(*inputs)
    # Backward baseline (loss and gradients)
    base_loss, base_in_grad, base_param_grads = run_linear_backward_pass(ref_layer, inputs)
    
    # If cuEq, record FP64 timing once from the reference layer
    if use_cueq:
        t64_ms, t64_alloc, t64_resv, t64_dtype = benchmark_linear_up(
            ref_layer, inputs, warmup, runs, "FP64", layer_number, device
        )
        results["FP64"] = {
            "time_ms": t64_ms,
            "peak_mem_mb": t64_alloc / 1024**2,
            "peak_mem_reserved_mb": t64_resv / 1024**2,
            "output_dtype": t64_dtype,
            "accuracy": None,
        }

    # Test each precision
    for math_dtype, precision_name in precisions:
        logger.info(f"\n{'-'*40}")
        logger.info(f"Testing {precision_name} precision...")
        logger.info(f"{'-'*40}")
        
        # Create or reuse layer with specific precision, ensuring identical weights
        if precision_name == "FP64":
            layer = ref_layer
            feat, attrs = inputs
            input_precision = (feat, attrs)
        else:
            layer = SymmetricContractionWrapper(
                irreps_in,
                irreps_out,
                correlation=correlation,
                num_elements=num_elements,
                use_cueq=use_cueq,
                math_dtype=math_dtype,
                device=device,
            )
            copy_state_dict_cast(ref_layer, layer, dtype=math_dtype, device=device)
            # Convert input to same precision for fair comparison
            feat, attrs = inputs
            input_precision = (feat.to(math_dtype), attrs.to(math_dtype))
        
        # Benchmark performance
        time_ms, peak_mem_bytes, peak_mem_reserved, output_dtype = benchmark_linear_up(layer, input_precision, warmup, runs, precision_name, layer_number, device)
        
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

        # Compare backward results to reference baseline
        eps = 1e-6
        loss_abs_diff = abs(loss.item() - base_loss.item())
        loss_rel_diff = loss_abs_diff / max(abs(base_loss.item()), eps)

        # Input gradient diffs
        if input_grad is not None and base_in_grad is not None:
            gref = base_in_grad.to(device=input_grad.device, dtype=input_grad.dtype)
            in_diff = input_grad - gref
            in_rel_l2 = in_diff.norm().item() / max(gref.norm().item(), eps)
            in_max_abs = in_diff.abs().max().item()
        else:
            in_rel_l2 = None
            in_max_abs = None

        # Parameter gradient diffs (aggregate and per-parameter)
        total_num_sq = 0.0
        total_den_sq = 0.0
        per_param_grad_rel_l2 = {}
        for name, g64 in base_param_grads.items():
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
        
        logger.info(f"  Performance: {time_ms:.2f} ms, Peak Mem Allocated: {peak_mem_bytes/1024**2:.2f} MB, Peak Mem Reserved: {peak_mem_reserved/1024**2:.2f} MB")
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
    Works for both cuEq (FP64 baseline + FP32) and non-cuEq (FP64 baseline + FP16/BF16).
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"PRECISION EFFECTS ANALYSIS")
    logger.info(f"{'='*60}")

    if 'FP64' not in results:
        logger.warning("No FP64 baseline in results; cannot analyze.")
        return results

    fp64 = results['FP64']
    fp64_time = fp64['time_ms']
    fp64_mem = fp64['peak_mem_mb']

    # Consider any non-FP64 entries as comparison precisions
    compare_keys = [k for k in results.keys() if k != 'FP64']
    if not compare_keys:
        logger.info("Only FP64 baseline present. Nothing to compare.")
        return results

    logger.info("\nPerformance Comparison:")
    for k in compare_keys:
        kt = results[k]['time_ms']
        logger.info(f"  FP64: {fp64_time:.2f} ms (baseline) | {k}: {kt:.2f} ms ({fp64_time/kt:.2f}x speedup)")

    logger.info("\nMemory Usage:")
    for k in compare_keys:
        km = results[k]['peak_mem_mb']
        logger.info(f"  FP64: {fp64_mem:.2f} MB (baseline) | {k}: {km:.2f} MB ({fp64_mem/max(km,1e-9):.2f}x reduction)")

    logger.info("\nAccuracy Degradation Analysis:")
    for k in compare_keys:
        acc = results[k].get('accuracy')
        if not acc:
            logger.info(f"  {k} vs FP64: accuracy not recorded")
            continue
        logger.info(f"  {k} vs FP64:")
        logger.info(f"    Max Absolute Error: {acc['max_abs_error']:.2e}")
        logger.info(f"    Mean Absolute Error: {acc['mean_abs_error']:.2e}")
        logger.info(f"    Max Relative Error: {acc['max_rel_error']:.2e}")
        logger.info(f"    Mean Relative Error: {acc['mean_rel_error']:.2e}")

    logger.info("\nSummary:")
    for k in compare_keys:
        kt = results[k]['time_ms']
        logger.info(f"  ✓ {k} speedup vs FP64: {fp64_time/kt:.2f}x")

    return results


def benchmark_linear_up(linear_up_fn, inputs, warmup=10, runs=50, precision_name=None, layer_number=None, device=None):

    # Support tuple inputs (features, attrs)
    example_args = inputs if isinstance(inputs, tuple) else (inputs,)
    use_eager = getattr(linear_up_fn, "is_cueq", False) is True
    if not use_eager:
        traced_fn = torch.jit.trace(linear_up_fn, example_args)
    
    with torch.inference_mode():
        if use_eager:
            _ = linear_up_fn(*example_args)
        else:
            _ = traced_fn(*example_args)
    if is_cuda_device(device):
        torch.cuda.synchronize(device)

    with torch.inference_mode():
        for _ in range(warmup):
            if use_eager:
                linear_up_fn(*example_args)
            else:
                traced_fn(*example_args)
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
                if use_eager:
                    linear_up_fn(*example_args)
                else:
                    traced_fn(*example_args)
                end.record()
                torch.cuda.synchronize(device)
                time_ms.append(start.elapsed_time(end))
        else:
            import time
            for _ in range(runs):
                t0 = time.perf_counter()
                if use_eager:
                    linear_up_fn(*example_args)
                else:
                    traced_fn(*example_args)
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
    test_output = linear_up_fn(*example_args) if use_eager else traced_fn(*example_args)
    output_dtype = test_output.dtype
    
    return mean_time_ms, peak_mem_bytes, peak_mem_reserved, output_dtype

def main():
    # Check available devices and select the best one
    if torch.cuda.is_available():
        device = ensure_torch_device(get_gpu_with_least_memory())
        # device = torch.device("cuda:1")
        logger.info(f"Using device: {device}")
    else:
        device = torch.device("cpu")
        logger.info("CUDA not available, using CPU")
    
    torch.manual_seed(0)
    
    _, _, _, z_table = data_prep()
    cfg   = get_default_model_config(z_table)
    model = modules.MACE(**cfg).to(device=device, dtype=torch.float64)

    # Build SymmetricContraction inputs from product blocks' irreps
    # Layer 0 product block
    prod0 = model.products[0]
    prod1 = model.products[1]

    in_ir0 = Irreps(prod0.symmetric_contractions.irreps_in)
    out_ir0 = Irreps(prod0.symmetric_contractions.irreps_out)
    in_ir1 = Irreps(prod1.symmetric_contractions.irreps_in)
    out_ir1 = Irreps(prod1.symmetric_contractions.irreps_out)

    # This block defines a helper function to determine the input tensor shape for a given set of irreducible representations (irreps).
    # Specifically, for a given Irreps object `ir`, it computes:
    #   - `ch`: the number of channels, which is the count of scalar (l=0, even parity) irreps (i.e., 0e fields).
    #   - `num_ell`: the total number of "spherical harmonic" components needed, which is the sum over all unique angular momentum quantum numbers l of (2l+1).
    # This is used to construct input tensors of shape [N, ch, num_ell] for benchmarking symmetric contractions.
    def channels_and_num_ell(ir: Irreps):
        ch = ir.count(o3.Irrep(0, 1))  # Number of scalar (0e) channels
        unique_ls = sorted({irrep.l for _, irrep in ir})  # Unique l values present in the irreps
        num_ell = sum(2 * l + 1 for l in unique_ls)  # Total number of spherical harmonic components
        return ch, num_ell

    ch0, ell0 = channels_and_num_ell(in_ir0)
    ch1, ell1 = channels_and_num_ell(in_ir1)

    N = 100
    x0 = torch.randn(N, ch0, ell0, dtype=torch.float64, device=device)
    x1 = torch.randn(N, ch1, ell1, dtype=torch.float64, device=device)

    # Build element attrs one-hot of size num_elements
    num_elements = len(cfg["atomic_numbers"]) if "atomic_numbers" in cfg else int(cfg.get("num_elements", 2))
    attrs0 = torch.zeros(N, num_elements, dtype=torch.float64, device=device)
    attrs1 = torch.zeros(N, num_elements, dtype=torch.float64, device=device)
    # simple alternating assignment for benchmarking
    idx = torch.arange(N, device=device) % num_elements
    attrs0[torch.arange(N, device=device), idx] = 1.0
    attrs1[torch.arange(N, device=device), idx] = 1.0

    # Determine correlation per layer from config
    corr_cfg = cfg.get("correlation", 3)
    if isinstance(corr_cfg, (list, tuple)):
        corr0 = int(corr_cfg[0])
        corr1 = int(corr_cfg[1] if len(corr_cfg) > 1 else corr0)
    else:
        corr0 = int(corr_cfg)
        corr1 = int(corr_cfg)

    # For a fair comparison, disable TF32 when running e3nn on CUDA
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        try:
            # Force exact FP32 matmuls (no TF32)
            torch.set_float32_matmul_precision("highest") # this is the tf32x3 trick
        except Exception:
            pass

    # Test precision effects on Symmetric Contraction (without cuEquivariance)
    logger.info("\n" + "="*60)
    use_cueq = False
    if use_cueq:
        logger.info("PRECISION EFFECTS TESTING (Symmetric Contraction - cuEquivariance)")
    else:
        logger.info("PRECISION EFFECTS TESTING (Symmetric Contraction - No cuEquivariance)")
    logger.info("="*60)

    # Benchmark precision effects for linear_up0_e3nn
    logger.info("\n" + "="*60)
    logger.info("BENCHMARK: Precision Effects on Symmetric Contraction Layer 0")
    logger.info("="*60)
    precision_results_l0 = benchmark_precision_effects(
        in_ir0,
        out_ir0,
        (x0, attrs0),
        device,
        correlation=corr0,
        num_elements=num_elements,
        use_cueq=use_cueq,
        layer_number=0,
    )
    analyze_precision_results(precision_results_l0)

    # Benchmark precision effects for linear_up1_e3nn
    logger.info("\n" + "="*60)
    logger.info("BENCHMARK: Precision Effects on Symmetric Contraction Layer 1")
    logger.info("="*60)
    precision_results_l1 = benchmark_precision_effects(
        in_ir1,
        out_ir1,
        (x1, attrs1),
        device,
        correlation=corr1,
        num_elements=num_elements,
        use_cueq=use_cueq,
        layer_number=1,
    )
    analyze_precision_results(precision_results_l1)

if __name__ == "__main__":
    main()