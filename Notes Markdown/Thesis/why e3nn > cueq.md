Short answer: your explanation is **mostly right**, with one important caveat about TF32 defaults. Here’s a quick fact-check, point by point.

### Verdict on each claim

* **“cuEq is in a regime where kernel-launch/dispatch overhead dominates; e3nn reduces to very fast GEMMs.”**
  ✅ Plausible/likely for **small shapes and batches**. cuEq’s custom kernels pay a fixed launch cost; when the arithmetic intensity is low, overhead dominates. NVIDIA’s guidance is to do “enough work per launch” (rule-of-thumb \~1 ms per kernel), otherwise you’re overhead-bound. ([NVIDIA Docs][1], [NVIDIA Developer Forums][2])

* **“e3nn wins with small-ish shapes and FP64 math.”**
  ✅ Reasonable. e3nn’s `Linear`/TP paths funnel into PyTorch matmuls (cuBLAS/cuBLASLt), which are extremely optimized. FP64 has lower throughput than FP32/TF32/FP16 on most GPUs (even A100’s FP64, while decent, is far below TF32/FP16 tensor-core rates), so a GEMM path can look relatively better versus custom FP64 kernels. ([alcf.anl.gov][3], [NVIDIA Developer][4])

* **“PyTorch/e3nn matmuls likely run with TF32 enabled by default, giving big speedups; cuEq custom op doesn’t use TF32.”**
  ⚠️ **Nuance:** On modern PyTorch (1.12+), **TF32 is *not* enabled by default for matmuls**; it’s enabled for cuDNN convolutions, but matmuls use TF32 only if you allow it (e.g., `torch.set_float32_matmul_precision('high'|'medium')` or `torch.backends.cuda.matmul.allow_tf32=True`). Some NVIDIA containers flip an env var that *does* turn it on, which could explain “mysterious” speedups. cuEq kernels use explicit `math_dtype` (float32/float64), not TF32. ([PyTorch][5], [docs.monai.io][6], [NVIDIA Docs][7])

* **“cuEq kernels are optimized for FP32/BF16; FP64 is slow on most GPUs.”**
  ✅ Directionally right. cuEq’s Torch/JAX APIs expose **math dtypes = FP32/FP64** (I/O can be FP16/BF16), and GPU peak rates heavily favor TF32/FP16/BF16 over FP64. So avoiding FP64 for performance-critical paths is sensible unless you need it. ([NVIDIA Docs][7], [alcf.anl.gov][3])

* **“Benchmarking a plain `Linear` favors e3nn; cuEq shines more on fused TP / interaction ops and larger batches.”**
  ✅ Matches upstream guidance: cuEq’s biggest wins in MACE are `ChannelWiseTensorProduct` and `SymmetricContraction`; `Linear` may even fall back to FX because the kernel isn’t a big improvement at small sizes. ([NVIDIA Docs][7])

### How to make your A/B fairer (and reproducible)

* **Lock TF32 policy** so you aren’t comparing cuEq(FP32) vs e3nn(TF32):

  ```python
  torch.backends.cuda.matmul.allow_tf32 = False
  torch.backends.cudnn.allow_tf32 = False
  # or equivalently:
  torch.set_float32_matmul_precision("highest")
  ```

  (Enable TF32 deliberately for the “with-TF32” condition.) ([PyTorch][5])
* **Prefer FP32 or BF16 math** in cuEq unless you truly need FP64. On A100/H100 the TF32/FP16/BF16 paths are an order of magnitude faster than FP64. ([alcf.anl.gov][3])
* **Increase arithmetic intensity** (bigger N, measure the actual MACE `conv_tp`/`SymmetricContraction`, not just `Linear`). That’s where cuEq was designed to pay off. ([NVIDIA Docs][7])
* **Amortize launch costs** (warmup + enough iterations; avoid tiny loops with lots of Python work). If you still see overhead, consider CUDA Graphs to tame launch/dispatch overhead in steady-state. ([PyTorch][5])

### TL;DR

Your diagnosis—*“small problem sizes + FP64 + potential TF32 advantage for e3nn GEMMs ⇒ e3nn can beat cuEq”*—is **credible**. The only correction is that **TF32 for matmuls isn’t on by default in stock PyTorch**, so confirm whether your environment enabled it (and fix it explicitly for apples-to-apples). With TF32 policy controlled, FP32/BF16 math, and larger/fused tensor-product workloads, cuEq should pull ahead—as the MACE + cuEq guidance and A100 peak figures suggest. ([NVIDIA Docs][7], [PyTorch][5], [alcf.anl.gov][3])

If you want, paste your exact benchmark shapes/irreps/batch and I’ll sketch a tiny harness that runs both backends under the two TF32 regimes so we can see the crossover point.

[1]: https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/?utm_source=chatgpt.com "CUDA C++ Best Practices Guide"
[2]: https://forums.developer.nvidia.com/t/help-in-speeding-up-culaunchkernel-execution-time/232136?utm_source=chatgpt.com "Help in speeding up cuLaunchKernel execution time"
[3]: https://www.alcf.anl.gov/sites/default/files/2021-07/ALCF_A100_20210728%5B80%5D.pdf?utm_source=chatgpt.com "INSIDE THE NVIDIA AMPERE A100 GPU IN THETAGPU ..."
[4]: https://developer.nvidia.com/blog/getting-immediate-speedups-with-a100-tf32/?utm_source=chatgpt.com "Getting Immediate Speedups with NVIDIA A100 TF32"
[5]: https://pytorch.org/docs/stable/notes/cuda.html "CUDA semantics — PyTorch 2.8 documentation"
[6]: https://docs.monai.io/en/stable/precision_accelerating.html?utm_source=chatgpt.com "Precision and Accelerating — MONAI 1.5.0 Documentation"
[7]: https://docs.nvidia.com/cuda/cuequivariance/tutorials/mace.html "In depth example: MACE — cuEquivariance"
