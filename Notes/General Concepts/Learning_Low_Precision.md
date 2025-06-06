

## 1  Get comfortable with equivariant building blocks

| Why you need it                                                                 | Best resource                                                                                                                                                                                     |
| ------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Understand irreps, tensor products, gating – the algebra all SO(3) models share | **e3nn User Guide + Jupyter notebooks** – after installing, open `e3nn_tutorial` and step through the notebooks; they mirror the code that MACE uses under the hood. ([e3nn.org][1])              |
| Hear the intuition from the authors                                             | Hannes Stark’s LOGaG reading-group talk *“MACE: Higher-Order Equivariant …”* (YouTube, 40 min) ([youtube.com][2])                                                                                 |
| See the same math applied to generic GNNs (good warm-up)                        | PyTorch Geometric’s “Creating Message Passing Networks” guide; shows how to subclass `MessagePassing` and swap aggregation, update and message functions. ([pytorch-geometric.readthedocs.io][3]) |

---

## 2  Open the hood of MACE itself

| What to read / run                                                                                                                      | What you’ll learn                                                                                                                              |
| --------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------- |
| **MACE Colab Tutorial 1–3** (training, active-learning, and an *advanced* “theory & code” walk-through) ([mace-docs.readthedocs.io][4]) | Maps every mathematical symbol in the paper to the exact PyTorch lines; shows where to inject custom layers.                                   |
| **`mace/guide/cuda_acceleration_with_cuequivariance`** page in the docs ([mace-docs.readthedocs.io][5])                                 | Lists the two kernels (tensor product & spherical harmonics) that dominate runtime and how to replace them with faster low-precision versions. |
| **Repo README → “MACE layers” sub-module** ([github.com][6])                                                                            | Minimal, reusable layers (radial MLP, gated tensor product block) – perfect starting point for architecture surgery.                           |

---

## 3  Master low-precision & mixed-precision tooling

| Skill                                                                               | Go-to tutorial                                                                                                                                   |
| ----------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------ |
| Automatic Mixed Precision (AMP) in PyTorch                                          | Official recipe – shows `torch.autocast` + `GradScaler`, when to keep accumulations in FP32, and typical 2–3× speed-ups. ([docs.pytorch.org][7]) |
| Fine-grained control of precision inside CUDA kernels                               | NVIDIA **cuEquivariance tutorials** – walk through FP16/BF16 tensor-product kernels for SO(3). ([docs.nvidia.com][8])                            |
| Writing custom Triton kernels (if you need an op that cuEquivariance doesn’t cover) | Community Triton tutorial repo & video series – starts from matrix-multiply and ends with attention-like ops. ([github.com][9])                  |
| Background reading on mixed-precision pitfalls                                      | PyTorch blog *“What every user should know about mixed precision training”*. ([pytorch.org][10])                                                 |
| Why cuEquivariance matters for science models                                       | NVIDIA technical blog on the library’s speed-ups for equivariant NNs. ([developer.nvidia.com][11])                                               |

---

## 4  Where, concretely, to down-cast inside MACE

```
┌── Radial MLP (RBF) ──┐        # safe to FP16/BF16
│                      ▼
│  CG tensor-product ─────────┐ # swap to cuEquivariance FP16 kernel
│                              │
│  Gated non-linearity ────────┤ # FP16
│                              ▼
└─► Node-update MLP ───────────┘ # FP16 weights, FP32 accum.
Energy/force heads -------------  # keep reductions FP32
```

*Keep Clebsch–Gordan coefficient tables in FP32 (tiny, reused often).*

---

## 5  Suggested learning path & next experiments

1. **Clone the repo, enable AMP** (`torch.autocast(device_type='cuda', dtype=torch.float16)` around the forward pass + `GradScaler` in the trainer loop).
2. **Profile** with `nsys` or the PyTorch profiler, identify any kernels still running FP32.
3. **Drop-in the cuEquivariance FP16 tensor-product op** (docs above).
4. **Optional:** write a Triton kernel for the radial basis or gate if they dominate after step 3.
5. **Benchmark** accuracy vs speed on a tiny MD17 system, then on a larger benchmark set.
6. **Document error bounds**: derive how quantisation noise propagates through a CG product; compare to DFT label noise.

---

### TL;DR cheat-sheet

| Phase                   | Key tutorial                                     |
| ----------------------- | ------------------------------------------------ |
| Understand equivariance | e3nn notebooks ([e3nn.org][1])                   |
| Learn MACE internals    | Colab Tutorial 3 ([mace-docs.readthedocs.io][4]) |
| Add mixed precision     | PyTorch AMP recipe ([docs.pytorch.org][7])       |
| Replace heavy kernels   | cuEquivariance tutorials ([docs.nvidia.com][8])  |
| Custom ops (optional)   | Triton tutorial repo ([github.com][9])           |

Work through these in order and you’ll have all the ingredients to **redesign layers, change precision, and measure real-world speed-ups** in MACE. If you get stuck on a specific kernel or convergence issue, just let me know and we can debug it together.

[1]: https://e3nn.org/ "Welcome to e3nn! {#welcome} | e3nn"
[2]: https://www.youtube.com/watch?v=I9Y2le9e74A&utm_source=chatgpt.com "MACE: Higher Order Equivariant Message Passing Neural Networks ..."
[3]: https://pytorch-geometric.readthedocs.io/en/2.6.1/notes/create_gnn.html "Creating Message Passing Networks — pytorch_geometric  documentation"
[4]: https://mace-docs.readthedocs.io/en/latest/examples/tutorials.html "Tutorials on MACE training and architecture — mace 0.3.13 documentation"
[5]: https://mace-docs.readthedocs.io/en/latest/guide/intro.html "Introduction — mace 0.3.13 documentation"
[6]: https://github.com/ACEsuit/mace "GitHub - ACEsuit/mace: MACE - Fast and accurate machine learning interatomic potentials with higher order equivariant message passing."
[7]: https://docs.pytorch.org/tutorials/recipes/recipes/amp_recipe.html "Automatic Mixed Precision — PyTorch Tutorials 2.7.0+cu126 documentation"
[8]: https://docs.nvidia.com/cuda/cuequivariance/tutorials/index.html "Tutorials — cuEquivariance"
[9]: https://github.com/VikParuchuri/triton_tutorial/?utm_source=chatgpt.com "Tutorials for Triton, a language for writing gpu kernels - GitHub"
[10]: https://pytorch.org/blog/what-every-user-should-know-about-mixed-precision-training-in-pytorch/?utm_source=chatgpt.com "What Every User Should Know About Mixed Precision Training in ..."
[11]: https://developer.nvidia.com/blog/accelerate-drug-and-material-discovery-with-new-math-library-nvidia-cuequivariance/?utm_source=chatgpt.com "Accelerate Drug and Material Discovery with New Math Library ..."
