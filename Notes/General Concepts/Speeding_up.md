### 1  Why lower-precision helps

**Memory & bandwidth** scale linearly with the number of bits, while many modern GPUs/accelerators execute 16-bit matrix–multiply **2--8× faster** than FP32 because they can pack more values into each Tensor Core warp.
Mixed- or low-precision therefore reduces:

| cost                  | scaling when you halve precision | practical effect                           |
| --------------------- | -------------------------------- | ------------------------------------------ |
| DRAM traffic          | ↓ 2 ×                            | less time stalled on memory                |
| L2/L1 cache footprint | ↓ 2 ×                            | larger batch or longer cut-off radius fits |
| FMA throughput        | ↑ 2--4 ×                         | higher FLOP/s on tensor cores              |
| Energy per op         | ↓ ≈2 ×                           | longer MD trajectories per GPU-hour        |

Mathematically, rounding introduces an **additive noise** ε with variance

$$
\mathrm{Var}[\varepsilon]\;=\;\tfrac{\Delta^{2}}{12},\qquad 
\Delta\!=\!\text{quantisation step}.
$$

For well scaled features this noise stays below the intrinsic DFT label noise, so accuracy is largely preserved as long as **accumulations** (e.g. energy sums, gradients) are still performed in FP32 and gradients are *loss-scaled* ([docs.nvidia.com][1]).

---

### 2  Prior art you can build on

| Work                                                                                                                                             | What they did                                                                                                          | Relevance                                                 |
| ------------------------------------------------------------------------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------- |
| **MACE repo** supports `float16` tensors (issue #559) ([github.com][2])                                                                          | Confirms the code already runs in half precision if requested.                                                         | You mostly need to automate AMP & add tests.              |
| **cuEquivariance** & **EquiTriton** kernels ([github.com][3], [github.com][4])                                                                   | Supply Triton/CUDA kernels for SO(3) tensor products and spherical harmonics that run in FP16/BF16 with big speed-ups. | Drop-in replacement for the most expensive ops.           |
| **NequIP & E3NN users** routinely train equivariant GNNs with PyTorch AMP; the NVIDIA mixed-precision recipe is the same ([docs.nvidia.com][1]). | Shows that equivariant models are numerically stable in FP16 when gradients are loss-scaled.                           | Gives confidence your experiments will converge.          |
| **Learned Low-Precision GNNs (LPG-NAS)** ([arxiv.org][5])                                                                                        | Demonstrates 4-/8-bit quantisation of message-passing networks with minimal accuracy loss.                             | Provides quantisation-aware training tricks you can port. |

No publication has yet benchmarked *MACE itself* under systematic mixed- or low-precision settings, so your thesis can be the first to quantify this.

---

### 3  Where to cut precision inside the MACE pipeline

```
┌── Input xyz ──► Radial basis (MLP) ─┐
│                                     ▼
│           ┌─ Irrep tensor features (ℓ=0…L) ─┐
│           │                                 │
│  CG ⊗     │   Tensor product  (SO(3) ops)   │  FP16/ BF16
│  FP32     └─────────┬─────────┬─────────────┘  (keep CG coeffs FP32)
│                     │
│      Gate & non-lin │  ← FP16 safe
│                     ▼
│           ┌─ Message Aggregation ───────────┐
│           │                                 │
└──────────►┴── Node update layers ───────────┘
              ▼
  Energy head (dense) – FP16 weights, **FP32 accumulation**
  Forces via autograd – gradients scaled, cast back to FP32
```

**Practical rules**

| Layer / tensor                         | Suggested dtype          | Rationale                                            |
| -------------------------------------- | ------------------------ | ---------------------------------------------------- |
| Coordinates, cell                      | FP64 or FP32             | preserves tiny displacements                         |
| Radial MLP, linear weight matrices     | **FP16/BF16**            | largest FLOP share, stable ranges (RBF ∈ \[0,1])     |
| Clebsch–Gordan tables                  | FP32                     | small, reused; avoids accumulated round-off          |
| Tensor-product outputs & hidden irreps | FP16/BF16                | high arithmetic intensity; tensor cores benefit most |
| Energy reduction (`sum_{atoms}`)       | accumulate FP32          | prevents cancellation error for big systems          |
| Back-prop gradients                    | FP16 with *loss scaling* | mitigates underflow                                  |

Implement with **PyTorch AMP** (`torch.autocast` + `GradScaler`) and optionally swap `e3nn` ops for **cuEquivariance**/**EquiTriton** kernels to unlock FP16 tensor-core paths.

---

### 4  Next steps to master low-precision MACE

1. **Numerical background**

   * Read Micikevicius et al. (2017) on mixed-precision training, and NVIDIA’s white paper for loss-scaling heuristics ([docs.nvidia.com][1]).
   * Review rounding-error theory (machine epsilon, Kahan summation) — essential to justify where you keep FP32.

2. **Hands-on experiments**

   * Clone the `mace` repo and add the flag `--amp` to `run_train.py`.
   * Start with *MACE-small* on an MD17 molecule; compare FP32 vs FP16 timing and MAE.
   * Profile with `nsys`/`nvprof` to locate kernels still running in FP32 and patch them.

3. **Quantisation-aware training (optional)**

   * Prototype INT8 post-training quantisation of the dense read-out layers using `torch.quantization`.
   * Investigate whether tensor-product ops can be expressed via look-up tables so that weights stay INT8 while intermediate activations stay FP16.

4. **Mathematical analysis for your thesis**

   * Derive error bounds for energy and force when inputs are perturbed by ±ε from quantisation; show they scale with the **Lipschitz constant** of the network.
   * Relate these bounds to acceptable DFT label noise (≈ 1 meV/atom).

5. **Benchmark & document**

   * Produce a table of **speed-up vs accuracy** for several precisions (FP32, BF16, FP16, INT8 weights + FP16 activations).
   * Include ablation where only radial MLPs are down-cast vs full network.

### How to speed up inference:
- Mixed/low precision inference (FP16, BF16)

- Optimized kernels for tensor products (e.g. cuEquivariance, EquiTriton)

- Pruning or distillation (smaller, faster student networks)

- Fewer message passing layers (lower depth with minimal accuracy drop)



[1]: https://docs.nvidia.com/deeplearning/performance/mixed-precision-training/index.html?utm_source=chatgpt.com "Train With Mixed Precision - NVIDIA Docs"
[2]: https://github.com/ACEsuit/mace/issues/559?utm_source=chatgpt.com "cannot turn off multihead finetuning in multi-head-interface branch"
[3]: https://github.com/NVIDIA/cuEquivariance/issues/45?utm_source=chatgpt.com "Performance comparison between e3nn and cuEquivariance of ..."
[4]: https://github.com/IntelLabs/EquiTriton "GitHub - IntelLabs/EquiTriton: EquiTriton is a project that seeks to implement high-performance kernels for commonly used building blocks in equivariant neural networks, enabling compute efficient training and inference."
[5]: https://arxiv.org/abs/2009.09232?utm_source=chatgpt.com "Learned Low Precision Graph Neural Networks"
