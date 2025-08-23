Great prompt. Here’s a tight, citable way to motivate **why cuEquivariance (cuEq) tends to be faster than e3nn for MACE**, plus where the speed actually comes from and how to verify it.

# E3NN vs cuEquivariance — what’s theoretically different?

**Same math, different execution model.**
Both libraries implement the same $O(3)$/$E(3)$ equivariant algebra (Clebsch–Gordan tensor products, irreps bookkeeping, etc.). The big differences are *how* those algebraic ops are laid out in memory and compiled into GPU kernels:

1. **Segmented polynomials & fused tensor products (cuEq).**
   cuEq expresses equivariant ops as *segmented polynomials* built from *Segmented Tensor Products (STP)* and a small set of “triangular” primitives. This lets it **fuse** CG tensor products, gathers/scatters, and downstream contractions into a few custom kernels, cutting global-memory traffic and kernel-launch overhead—exactly the bottlenecks in MACE. ([NVIDIA Docs][1])

2. **GPU-friendly data layouts.**
   cuEq promotes an *(ir, mul)* layout (irreps outermost) that coalesces memory and maps cleanly onto register/shared-memory tiling; e3nn historically uses *(mul, ir)*, which is flexible but less GPU-optimal in many cases. ([NVIDIA Docs][2])

3. **Tensor Core usage & autotuning.**
   cuEq ships kernels that explicitly target Tensor Cores and expose precision knobs (TF32/BF16/FP16 variants) and **AOT/On-Demand autotuning** for key ops. e3nn relies more on PyTorch primitives, so you often don’t get the same level of fusion or TC specialization by default. ([GitHub][3], [NVIDIA Docs][4])

4. **Specialized MACE primitives.**
   cuEq provides a drop-in **SymmetricContraction** with an `original_mace=True` mode to mimic MACE’s historical implementation, but executed by a fused CUDA kernel. This is one of the heaviest layers in larger MACE models. ([NVIDIA Docs][5])

# What speedups are reported (for MACE & related ops)?

* **MACE docs**: “up to \~5×” speedup in training/inference for large models on a single GPU when switching to cuEq. ([mace-docs.readthedocs.io][6])
* **NVIDIA cuEq page**: headline figures up to **10× end-to-end MACE**, **200×** on the symmetric contraction micro-op, and 3.5× on triangle ops (used in other ENNs). Treat as vendor-reported but directionally accurate. ([NVIDIA Developer][7])
* **Release notes**: new JAX 1D-uniform STP kernel gave **\~3×** MACE speedup (JAX cuEq v0.3 vs older torch backend), showing how kernel design dominates constants. ([NVIDIA Docs][8])
* **Community & research baselines**: independent GPU kernel papers/libraries (e.g., FlashTP; sparse CG generators) consistently show *large* gains vs e3nn and often also vs early cuEq releases by **kernel fusion + sparsity-aware tiling**—evidence the bottleneck is memory movement and fragmentation rather than pure FLOPs. (You can cite them for the *principle* even if you don’t adopt them.) ([openreview.net][9], [arXiv][10], [arXiv][11])

# So, why pick cuEq over e3nn for *speed*?

* **Fewer, fatter kernels** (less launch overhead).
* **Lower DRAM traffic** via fusion (tensor product + contraction + scatters combined).
* **Better occupancy/throughput** by aligning irreps/multiplicities with TC-friendly tiles.
* **Autotuned kernels** for your exact shapes/irreps mix.
  These are *engineering* advantages that preserve the same equivariant math. (e3nn’s advantages are flexibility, clarity, and rapid prototyping; it’s the “reference” math toolkit, and it’s evolving too. But its default execution is rarely as fused/TC-aware as cuEq out of the box.) ([docs.e3nn.org][12])

# Where does the MACE acceleration actually come from?

In MACE, the heaviest hotspots are typically:

* **CG tensor products** inside message blocks,
* the **symmetric (degree-$\nu$) contractions** that mix channels, and
* irreps reshapes/gathers around those ops.

cuEq specifically replaces these with STP + **SymmetricContraction** kernels and a GPU-optimal layout, which is where you see the multi-× gains. ([NVIDIA Docs][5])

# How to *show* it (what to measure)

1. **End-to-end A/B**: same model & batch/graph sizes, e3nn vs cuEq MACE with identical precision (e.g., TF32 on A100). Report **tokens/atoms per second** and wall-clock per epoch. (The MACE docs’ 5× figure uses this framing.) ([mace-docs.readthedocs.io][6])
2. **Microbenchmarks**: time standalone **SymmetricContraction** and a representative **tensor product block** with identical irreps specs, varying $l_{\max}$, multiplicities, and degree $\nu$. Compare kernel counts and **bytes moved / FLOP** (roofline). ([NVIDIA Docs][5])
3. **Nsight Systems/Compute**: show (i) fewer kernel launches, (ii) higher achieved occupancy/TC utilization, (iii) lower DRAM transactions per output element on cuEq. (cuEq’s triangle ops pages document tuning/precision toggles—use the same knobs philosophy for MACE primitives.) ([NVIDIA Docs][4])
4. **Layout sensitivity**: switch cuEq between *(ir, mul)* and e3nn-compatible *(mul, ir)* and show the delta—this cleanly isolates the layout effect. ([NVIDIA Docs][2])

# Phrasing you can drop into your Motivation section

> **Why cuEquivariance?**
> The algebra implemented by e3nn and cuEquivariance is identical; both realize $O(3)$-equivariant tensor products and contractions. The difference is execution: cuEquivariance compiles the same CG algebra into a small set of **fused, Tensor-Core-aware kernels** using **segmented tensor products** and a GPU-optimal data layout. In MACE, this directly targets the dominant hotspots—CG tensor products and symmetric contractions—reducing global memory traffic and kernel launches. Public benchmarks and MACE’s own documentation report **multi-× end-to-end speedups** for large models with no change in model predictions when precision is held constant. ([mace-docs.readthedocs.io][6], [NVIDIA Docs][1])

---

## Sources you can cite

* **MACE docs — cuEq integration & observed speedups** (“up to \~5×” on large models). ([mace-docs.readthedocs.io][6])
* **NVIDIA cuEq product page** (headline speedups incl. MACE and per-op gains). ([NVIDIA Developer][7])
* **cuEq tutorials & API** (STP, data layouts, SymmetricContraction, triangle ops & tuning). ([NVIDIA Docs][1])
* **cuEq release notes** (new kernels → measured MACE speedups). ([NVIDIA Docs][8])
* **e3nn docs** (reference tensor products & irreps; baseline design). ([docs.e3nn.org][13])
* **Comparative/adjacent research** showing why fusion+sparsity-aware tiling wins for ENNs (helpful background when arguing the *principle*): FlashTP and sparse CG kernel papers. ([openreview.net][9], [arXiv][10], [arXiv][11])
* **Community comparison thread** (cuEq vs e3nn test case). ([GitHub][14])

If you want, I can also rewrite your subsection cleanly and drop in citation macros around these points.

[1]: https://docs.nvidia.com/cuda/cuequivariance/tutorials/index.html?utm_source=chatgpt.com "Tutorials — cuEquivariance"
[2]: https://docs.nvidia.com/cuda/cuequivariance/tutorials/layout.html?utm_source=chatgpt.com "Data Layouts — cuEquivariance"
[3]: https://github.com/NVIDIA/cuEquivariance/releases?utm_source=chatgpt.com "Releases · NVIDIA/cuEquivariance"
[4]: https://docs.nvidia.com/cuda/cuequivariance/api/generated/cuequivariance_torch.triangle_multiplicative_update.html?utm_source=chatgpt.com "triangle_multiplicative_update — cuEquivariance"
[5]: https://docs.nvidia.com/cuda/cuequivariance/api/generated/cuequivariance_torch.SymmetricContraction.html?utm_source=chatgpt.com "SymmetricContraction — cuEquivariance"
[6]: https://mace-docs.readthedocs.io/en/latest/guide/cuda_acceleration.html?utm_source=chatgpt.com "CUDA Acceleration with cuEquivariance Library"
[7]: https://developer.nvidia.com/cuequivariance?utm_source=chatgpt.com "cuEquivariance CUDA-X Library"
[8]: https://docs.nvidia.com/cuda/cuequivariance/changelog.html?utm_source=chatgpt.com "Release Notes — cuEquivariance"
[9]: https://openreview.net/forum?id=wiQe95BPaB&referrer=%5Bthe+profile+of+Seungwu+Han%5D%28%2Fprofile%3Fid%3D~Seungwu_Han1%29&utm_source=chatgpt.com "FlashTP: Fused, Sparsity-Aware Tensor Product for Machine ..."
[10]: https://www.arxiv.org/pdf/2501.13986v4?utm_source=chatgpt.com "arXiv:2501.13986v4 [cs.LG] 8 May 2025"
[11]: https://arxiv.org/html/2501.13986v1?utm_source=chatgpt.com "An Efficient Sparse Kernel Generator for O(3)-Equivariant ..."
[12]: https://docs.e3nn.org/?utm_source=chatgpt.com "Euclidean neural networks — e3nn 0.5.1 documentation"
[13]: https://docs.e3nn.org/en/stable/api/o3/o3_tp.html?utm_source=chatgpt.com "Tensor Product — e3nn 0.5.1 documentation"
[14]: https://github.com/NVIDIA/cuEquivariance/issues/45?utm_source=chatgpt.com "Issue #45 · NVIDIA/cuEquivariance"
