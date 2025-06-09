
## 🔬 **Low-Precision Training: What Can Be Tested**

Here's what you can test or experiment with **today** to address these computational bottlenecks using **low-precision techniques**:

| **Test/Strategy**                             | **Where to Apply**                         | **Expected Benefit**                            | **Tools to Use**                                                             |
| --------------------------------------------- | ------------------------------------------ | ----------------------------------------------- | ---------------------------------------------------------------------------- |
| **Mixed-Precision Training**                  | Everywhere except CG and SH ops            | 2–3× speedup, lower memory                      | `torch.cuda.amp` (PyTorch) or `jax.lax.precision`                            |
| **Low-Precision Tensor Contractions**         | Eq. (10) — node-level loop contractions    | Significant speedup at L ≥ 1 or ν ≥ 2           | Implement custom fused kernels or use `Triton`, `XLA`, `NVIDIA Tensor Cores` |
| **Keep Clebsch–Gordan / SH Terms in Float32** | Tensor ops involving rotation-equivariance | Avoids equivariance-breaking artifacts          | Store CG/Ylm tables as `float32` constants                                   |
| **Loss Scaling**                              | During backpropagation                     | Prevents underflow in float16 gradients         | Use `GradScaler` in PyTorch                                                  |
| **Quantize MLP Readout Layers**               | Final site energy prediction               | 8-bit quantization viable without accuracy loss | Use `torch.quantization` or ONNX INT8                                        |

---

## 📈 **Parallelism and Training Time Gains**

* **Observation**: MACE with L=0 is \~10× faster than NequIP/BOTNet while maintaining accuracy.
* **Training speedup**: MACE reaches BOTNet-level accuracy in **30 mins vs 24+ hrs**.
* **Reason**: Its tensor-based message formulation + small receptive field reduces GPU memory contention and communication overhead → **better scaling across multiple GPUs**.

This makes it especially attractive for **active learning workflows**, where fast retraining and re-evaluation are crucial.

---

## 🧪 **Future/Experimental Ideas**

You could also explore:

1. **Layer-wise Precision Scaling**:

   * Use float16 for readout and intermediate MLPs.
   * Use float32 for initial feature computation and CG terms.

2. **Progressive Correlation Order**:

   * Start training with ν = 1, then progressively raise it.
   * Benefit: lower early-phase compute + stable convergence.

3. **Neural Compression of Species Embeddings**:

   * Replace fixed species embeddings with **low-rank or quantized embeddings**.
   * Especially helpful for large-S datasets (e.g., periodic table-sized chemical space).

4. **Operator Fusion + Kernel Optimization**:

   * Fuse radial, angular, and contraction steps to reduce memory reads/writes.
   * Implement with Triton or TVM for custom GPUs.

---

## ✅ **Takeaway Summary**

* **Computation bottlenecks in MACE** are dominated by equivariant tensor algebra, not neighbor gathering.
* **MACE already addresses** some scaling issues with smart architectural design (species compression, small receptive field, loop contractions).
* **Low-precision training and mixed-precision inference** are highly promising: they target exactly the bottlenecks that remain.
* With current tools (PyTorch AMP, JAX, Triton), you can **immediately test** these strategies for real-world speedup.

Let me know if you'd like a benchmarking plan or starter script to begin testing these strategies.
