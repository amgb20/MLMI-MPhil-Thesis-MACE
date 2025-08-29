Perfect—let’s switch the profiler mindset to **low-latency inference** and turn it into concrete knobs you can flip, plus a decision flow that maps profiler symptoms → actions.

# A. How to profile **inference** (so results are actionable)

Use a tiny harness that reflects how you’ll serve the model:

```python
model.eval()
torch.set_grad_enabled(False)
from torch import inference_mode

with inference_mode():                    # kills autograd & tensor versioning overhead
    # warmup a few runs to build caches
    for _ in range(20): out = model(batch0)

from torch.profiler import profile, ProfilerActivity, schedule, tensorboard_trace_handler

sched = schedule(wait=5, warmup=5, active=50, repeat=1)  # small, steady-state window

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    schedule=sched, record_shapes=True, profile_memory=True,
    on_trace_ready=tensorboard_trace_handler("./log/mace_infer")
) as prof:
    for b in batches:               # batches here can be size=1 for latency or >1 for throughput
        _ = model(b)
        prof.step()
```

**KPIs to record per run:** p50/p95 latency (per structure and per batch), throughput (structures/s), GPU util proxy (CUDA total / wall-time), # kernel launches per inference, H2D time, top-10 ops (CPU/CUDA totals).

---

# B. Read the profiler like a playbook (inference edition)

| **If you see…**                                                   | **What it means**                        | **Action**                                                                                                                                                                                     |
| ----------------------------------------------------------------- | ---------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `cudaLaunchKernel` large CPU self time; thousands of tiny kernels | Launch overhead dominates                | **CUDA Graphs** + **static shapes** (bucket/pad); or `torch.compile(..., mode="reduce-overhead")`                                                                                              |
| Many `aten::clone`, `aten::copy_`, `aten::to`, `Memcpy DtoD`      | Redundant copies / dtype or layout churn | Create tensors directly on target device/dtype; remove `.clone()/.contiguous()` in hot path; pre-allocate outputs; use `non_blocking=True` with pinned host mem                                |
| `cuequivariance_ops::*jit` heavy **Self CPU** but tiny CUDA       | cuEq host JIT/packing dominates          | **Bucket shapes** (fixed L, channels, edge counts per bucket); warm up each bucket; keep process hot; prefer **larger batches** if latency budget allows                                       |
| GEMMs (`mm/bmm/einsum`) dominate CUDA                             | Compute-bound on matmul                  | On **A100**: enable **TF32** (`torch.set_float32_matmul_precision('high')`), enlarge batch/atoms to hit Tensor Cores’ sweet spots; for e3nn backends consider **AMP/BF16** (cuEq only FP32/64) |
| Large H2D bars                                                    | Input pipeline still the bottleneck      | Keep inputs on GPU across calls if possible; use **pinned** memory + `non_blocking`; prefetch next batch; precompute graphs & features offline                                                 |
| Allocator churn (`aten::empty`, `zeros`, big ± CUDA Mem deltas)   | Reallocating temps every call            | Pre-allocate persistent buffers; use larger batch to amortize; keep **static shapes** → lets CUDA Graphs reuse memory                                                                          |

---

# C. Switches & features to flip for **faster MACE inference**

### 1) Zero autograd cost

```python
model.eval()
torch.set_grad_enabled(False)
from torch import inference_mode
# wrap the whole serve loop in inference_mode()
```

This alone often removes the big “autograd::\*” blocks you saw.

### 2) Eliminate launch overhead

**Best**: **CUDA Graphs** (needs static shapes for tensors & workspace sizes).

```python
# 1) Warmup with representative shapes and allocate static buffers
static_in = make_static(batch_template).to(device)

# 2) Capture once
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g):
    static_out = model(static_in)  # captured

# 3) Serve: copy inputs into static buffers, replay graph
def run(in_batch):
    copy_into_static(static_in, in_batch)  # no allocs; same shapes
    g.replay()
    return static_out
```

To use this with variable graphs, **bucket** requests so each bucket has fixed (max) atoms/edges and you **pad** smaller ones.

**Second best**: `torch.compile` (PyTorch 2.1+)

```python
model = torch.compile(model, mode="reduce-overhead")  # or "max-autotune" when stable
```

This fuses ops and reduces the number of launches; works even with moderate dynamism. Combine with bucketing for best results.

### 3) Use Tensor Cores where legal

* **A100 only:** enable **TF32** for FP32 matmuls (good default for MACE unless you’ve proven accuracy loss):

```python
torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision('high')  # same intent at higher level
```

* **BF16/FP16:**

  * **e3nn** / GEMM paths can benefit from `autocast(dtype=torch.bfloat16)` + FP32 accum;
  * **cuEq** supports **FP32/FP64 only** → skip AMP for those kernels; TF32 won’t help cuEq’s custom ops.

### 4) Make the math chunkier

* For **throughput**: batch multiple structures (or micro-batch then fuse). Larger problem sizes reduce overhead and hit HMMA shapes better.
* For **latency** with CUDA Graphs: still pad to static capacity so the captured graph runs fast.

### 5) Kill copies & D2D moves

* Create inputs **directly** on device with the right dtype; avoid `.to(device)` in the hot path.
* Avoid `.clone()/.contiguous()` unless correctness demands it.
* Keep outputs’ storage persistent and write into them (no fresh alloc per call).

### 6) Precompute & cache CPU-heavy bits

If your inference builds neighbor lists, edge vectors, or spherical harmonics:

* Precompute once (for static systems) and store `{edge_index, distances, Y_lm, Z}` in LMDB/NPZ.
* If inference positions change, compute **on GPU** in the first layer (unit vectors, Y\_lm) to remove CPU stalls.

### 7) Don’t hobble the libraries

* Ensure determinism knobs aren’t forcing slow paths at inference:
  `torch.use_deterministic_algorithms(False)`; **unset** `CUBLAS_WORKSPACE_CONFIG`.
* cuDNN flags don’t matter here; MACE doesn’t use CNNs.

### 8) Memory allocator hygiene

* If you see fragmentation, try `PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128` (or larger) and keep shapes static (graphs/compile help more).

---

# D. Minimal experiment matrix (so decisions are justified)

Run each row on the **same** input set; report **p50/p95 latency**, **throughput**, **#kernel launches/inference**, and **Top-10 CPU/CUDA totals**:

1. **Baseline:** eval + inference\_mode.
2. Baseline + **TF32** (A100).
3. (If e3nn) + **BF16 autocast**.
4. Baseline + **torch.compile(mode="reduce-overhead")**.
5. Baseline + **buckets** (pad to static shapes).
6. Buckets + **CUDA Graphs**.
7. Buckets + CUDA Graphs + TF32 (and BF16 if applicable).

You’ll typically see:

* 1→2: **1.2–1.4×** if GEMMs are material (A100).
* 2→4: **1.1–1.3×** from fusion & fewer launches.
* 4→6: another **1.2–1.6×** if you were launch-bound.
* Copies cleanup can add **\~5–20%** depending on how many `clone/copy_` you remove.

---

# E. Quick decision tree

* Launch overhead high? → **CUDA Graphs** (needs **buckets/padding**) → else `torch.compile`.
* GEMMs heavy and A100? → **TF32 on**. If e3nn path → try **BF16**.
* Copies everywhere? → Create tensors on target device/dtype; remove `.clone()`; reuse buffers.
* cuEq host cost high? → **bucket shapes**, warm caches; increase batch size if latency budget allows.

Apply these in that order, profiling each change. That will give you fast, defensible inference wins on MACE with minimal code churn.


The MACE inference equation can be written as:

**Energy Computation:**
$$E = \sum_{i} E_i^{(0)} + \sum_{l=1}^{L} \sum_{i} \text{MLP}_l(\mathbf{h}_i^{(l)})$$

**Node Feature Updates (Interaction Blocks):**
$$\mathbf{h}_i^{(l+1)} = \mathbf{h}_i^{(l)} + \sum_{j \in \mathcal{N}(i)} \text{Conv}_l(\mathbf{h}_i^{(l)}, \mathbf{h}_j^{(l)}, \mathbf{r}_{ij}, \mathbf{Y}_{lm}(\hat{\mathbf{r}}_{ij}))$$

**Tensor Product (Product Blocks):**
$$\text{Conv}_l = \sum_{\lambda} \mathbf{W}_\lambda \otimes \mathbf{Y}_{lm}(\hat{\mathbf{r}}_{ij}) \otimes \mathbf{h}_j^{(l)}$$

**Forces (via Autograd):**
$$\mathbf{F}_i = -\nabla_{\mathbf{r}_i} E$$

$$\mathbf{r}_i, Z_i → \text{NN}_{\text{forward}} → E$$
    ↓
$$\mathbf{F}_i = -\nabla_{\mathbf{r}_i} E$$

Where:
- $E_i^{(0)}$ are atomic reference energies
- $\mathbf{h}_i^{(l)}$ are node features at layer $l$
- $\mathcal{N}(i)$ are neighbors of atom $i$
- $\mathbf{Y}_{lm}$ are spherical harmonics
- $\mathbf{r}_{ij}$ is the vector between atoms $i$ and $j$
- $\text{Conv}_l$ are equivariant convolutions
- $\text{MLP}_l$ are readout networks

The key insight is that forces are computed as the negative gradient of the energy with respect to atomic positions, which PyTorch handles automatically through the computational graph built during the forward pass.

## 
ScaleShiftMACE(
  (node_embedding): LinearNodeEmbeddingBlock(
    (linear): Linear(
      shared_weights=True, internal_weights=True, weight_numel=1280
      (transpose_in): TransposeIrrepsLayout((irrep,mul) -> (irrep,mul))
      (transpose_out): TransposeIrrepsLayout((irrep,mul) -> (irrep,mul))
      (f): ╭ a=[1280:1⨯(10,128)] b=[10:1⨯(1,10)] -> C=[128:1⨯(1,128)]
      ╰─ []·a[uv]·b[iu]➜C[iv] ─ num_paths=1 i=1 u=10 v=128
      SegmentedPolynomial(
        (m): SegmentedPolynomialNaive(
          (graphs): ModuleList(
            (0): uv,iu,iv operands=[(10, 128)],[(1, 10)],[(1, 128)] paths=[op0[0]*op1[0]*op2[0]*0.32]
          )
        )
        (fallback): SegmentedPolynomialNaive(
          (graphs): ModuleList(
            (0): uv,iu,iv operands=[(10, 128)],[(1, 10)],[(1, 128)] paths=[op0[0]*op1[0]*op2[0]*0.32]
          )
        )
      )
    )
  )
  (radial_embedding): RadialEmbeddingBlock(
    (bessel_fn): BesselBasis(r_max=6.0, num_basis=8, trainable=False)
    (cutoff_fn): PolynomialCutoff(p=5, r_max=6.0)
  )
  (spherical_harmonics): SphericalHarmonics()
  (atomic_energies_fn): AtomicEnergiesBlock(energies=[[-13.5720, -1030.5671, -1486.3750, -2043.9337, -2715.3186, -9287.4072, -10834.4844, -12522.6494, -70045.2812, -8102.5244]])
  (interactions): ModuleList(
    (0): RealAgnosticInteractionBlock(
      (linear_up): Linear(
        shared_weights=True, internal_weights=True, weight_numel=16384
        (transpose_in): TransposeIrrepsLayout((irrep,mul) -> (irrep,mul))
        (transpose_out): TransposeIrrepsLayout((irrep,mul) -> (irrep,mul))
        (f): ╭ a=[16384:1⨯(128,128)] b=[128:1⨯(1,128)] -> C=[128:1⨯(1,128)]
        ╰─ []·a[uv]·b[iu]➜C[iv] ─ num_paths=1 i=1 u=128 v=128
        SegmentedPolynomial(
          (m): SegmentedPolynomialNaive(
            (graphs): ModuleList(
              (0): uv,iu,iv operands=[(128, 128)],[(1, 128)],[(1, 128)] paths=[op0[0]*op1[0]*op2[0]*0.09]
            )
          )
          (fallback): SegmentedPolynomialNaive(
            (graphs): ModuleList(
              (0): uv,iu,iv operands=[(128, 128)],[(1, 128)],[(1, 128)] paths=[op0[0]*op1[0]*op2[0]*0.09]
            )
          )
        )
      )
      (conv_tp): ╭ a=[512:4⨯(128)] b=[128:1⨯(128)] c=[16:16⨯()] -> D=[2048:16⨯(128)]
      ╰─ []·a[u]·b[u]·c[]➜D[u] ─ num_paths=16 u=128
      SegmentedPolynomial(
        (m): SegmentedPolynomialFromUniform1dJit()
        (fallback): SegmentedPolynomialFromUniform1dJit()
      )
      (conv_tp_weights): FullyConnectedNet[8, 64, 64, 64, 512]
      (linear): Linear(
        shared_weights=True, internal_weights=True, weight_numel=65536
        (transpose_in): TransposeIrrepsLayout((irrep,mul) -> (irrep,mul))
        (transpose_out): TransposeIrrepsLayout((irrep,mul) -> (irrep,mul))
        (f): ╭ a=[65536:4⨯(128,128)] b=[2048:(1,128)+(3,128)+...] -> C=[2048:(1,128)+(3,128)+...]
        ╰─ []·a[uv]·b[iu]➜C[iv] ─ num_paths=4 i={1, 3, 5, 7} u=128 v=128
        SegmentedPolynomial(
          (m): SegmentedPolynomialNaive(
            (graphs): ModuleList(
              (0): uv,iu,iv sizes=65536,2048,2048 num_segments=4,4,4 num_paths=4 i={1, 3, 5, 7} u=128 v=128
            )
          )
          (fallback): SegmentedPolynomialNaive(
            (graphs): ModuleList(
              (0): uv,iu,iv sizes=65536,2048,2048 num_segments=4,4,4 num_paths=4 i={1, 3, 5, 7} u=128 v=128
            )
          )
        )
      )
      (skip_tp): FullyConnectedTensorProduct(
        shared_weights=True, internal_weights=True, weight_numel=655360
        (transpose_in1): TransposeIrrepsLayout((irrep,mul) -> (irrep,mul))
        (transpose_in2): TransposeIrrepsLayout((irrep,mul) -> (irrep,mul))
        (transpose_out): TransposeIrrepsLayout((irrep,mul) -> (irrep,mul))
        (f): ╭ a=[655360:4⨯(128,10,128)] b=[2048:(1,128)+(3,128)+...] c=[10:1⨯(1,10)] -> D=[2048:(1,128)+(3,128)+...]
        ╰─ [ijk]·a[uvw]·b[iu]·c[jv]➜D[kw] ─ num_paths=4 i={1, 3, 5, 7} j=1 k={1, 3, 5, 7} u=128 v=10 w=128
        SegmentedPolynomial(
          (m): SegmentedPolynomialNaive(
            (graphs): ModuleList(
              (0): uvw,iu,jv,kw+ijk sizes=655360,2048,10,2048 num_segments=4,4,1,4 num_paths=4 i={1, 3, 5, 7} j=1 k={1, 3, 5, 7} u=128 v=10 w=128
            )
          )
          (fallback): SegmentedPolynomialNaive(
            (graphs): ModuleList(
              (0): uvw,iu,jv,kw+ijk sizes=655360,2048,10,2048 num_segments=4,4,1,4 num_paths=4 i={1, 3, 5, 7} j=1 k={1, 3, 5, 7} u=128 v=10 w=128
            )
          )
        )
      )
      (reshape): reshape_irreps()
    )
    (1): RealAgnosticResidualInteractionBlock(
      (linear_up): Linear(
        shared_weights=True, internal_weights=True, weight_numel=32768
        (transpose_in): TransposeIrrepsLayout((irrep,mul) -> (irrep,mul))
        (transpose_out): TransposeIrrepsLayout((irrep,mul) -> (irrep,mul))
        (f): ╭ a=[32768:2⨯(128,128)] b=[512:(1,128)+(3,128)] -> C=[512:(1,128)+(3,128)]
        ╰─ []·a[uv]·b[iu]➜C[iv] ─ num_paths=2 i={1, 3} u=128 v=128
        SegmentedPolynomial(
          (m): SegmentedPolynomialNaive(
            (graphs): ModuleList(
              (0): uv,iu,iv sizes=32768,512,512 num_segments=2,2,2 num_paths=2 i={1, 3} u=128 v=128
            )
          )
          (fallback): SegmentedPolynomialNaive(
            (graphs): ModuleList(
              (0): uv,iu,iv sizes=32768,512,512 num_segments=2,2,2 num_paths=2 i={1, 3} u=128 v=128
            )
          )
        )
      )
      (conv_tp): ╭ a=[1280:10⨯(128)] b=[512:4⨯(128)] c=[16:16⨯()] -> D=[5120:40⨯(128)]
      ╰─ []·a[u]·b[u]·c[]➜D[u] ─ num_paths=86 u=128
      SegmentedPolynomial(
        (m): SegmentedPolynomialFromUniform1dJit()
        (fallback): SegmentedPolynomialFromUniform1dJit()
      )
      (conv_tp_weights): FullyConnectedNet[8, 64, 64, 64, 1280]
      (linear): Linear(
        shared_weights=True, internal_weights=True, weight_numel=163840
        (transpose_in): TransposeIrrepsLayout((irrep,mul) -> (irrep,mul))
        (transpose_out): TransposeIrrepsLayout((irrep,mul) -> (irrep,mul))
        (f): ╭ a=[163840:10⨯(128,128)] b=[5120:(1,128)+(1,128)+...] -> C=[2048:(1,128)+(3,128)+...]
        ╰─ []·a[uv]·b[iu]➜C[iv] ─ num_paths=10 i={1, 3, 5, 7} u=128 v=128
        SegmentedPolynomial(
          (m): SegmentedPolynomialNaive(
            (graphs): ModuleList(
              (0): uv,iu,iv sizes=163840,5120,2048 num_segments=10,10,4 num_paths=10 i={1, 3, 5, 7} u=128 v=128
            )
          )
          (fallback): SegmentedPolynomialNaive(
            (graphs): ModuleList(
              (0): uv,iu,iv sizes=163840,5120,2048 num_segments=10,10,4 num_paths=10 i={1, 3, 5, 7} u=128 v=128
            )
          )
        )
      )
      (skip_tp): FullyConnectedTensorProduct(
        shared_weights=True, internal_weights=True, weight_numel=163840
        (transpose_in1): TransposeIrrepsLayout((irrep,mul) -> (irrep,mul))
        (transpose_in2): TransposeIrrepsLayout((irrep,mul) -> (irrep,mul))
        (transpose_out): TransposeIrrepsLayout((irrep,mul) -> (irrep,mul))
        (f): ╭ a=[163840:1⨯(128,10,128)] b=[512:(1,128)+(3,128)] c=[10:1⨯(1,10)] -> D=[128:1⨯(1,128)]
        ╰─ [ijk]·a[uvw]·b[iu]·c[jv]➜D[kw] ─ num_paths=1 i={1, 3} j=1 k=1 u=128 v=10 w=128
        SegmentedPolynomial(
          (m): SegmentedPolynomialNaive(
            (graphs): ModuleList(
              (0): uvw,iu,jv,kw+ijk sizes=163840,512,10,128 num_segments=1,2,1,1 num_paths=1 i={1, 3} j=1 k=1 u=128 v=10 w=128
            )
          )
          (fallback): SegmentedPolynomialNaive(
            (graphs): ModuleList(
              (0): uvw,iu,jv,kw+ijk sizes=163840,512,10,128 num_segments=1,2,1,1 num_paths=1 i={1, 3} j=1 k=1 u=128 v=10 w=128
            )
          )
        )
      )
      (reshape): reshape_irreps()
    )
  )
  (products): ModuleList(
    (0): EquivariantProductBasisBlock(
      (symmetric_contractions): SymmetricContraction(
        contraction_degree=3, weight_shape=(86, 128)
        (transpose_in): TransposeIrrepsLayout((irrep,mul) -> (irrep,mul))
        (transpose_out): TransposeIrrepsLayout((irrep,mul) -> (irrep,mul))
        (f): ╭ a=[3712:29⨯(128)] b=[2048:16⨯(128)] -> C=[512:4⨯(128)]
        │  []·a[u]·b[u]➜C[u] ─────────── num_paths=4 u=128
        │  []·a[u]·b[u]·b[u]➜C[u] ────── num_paths=86 u=128
        ╰─ []·a[u]·b[u]·b[u]·b[u]➜C[u] ─ num_paths=1949 u=128
        SegmentedPolynomial(
          (m): SegmentedPolynomialFromUniform1dJit()
          (fallback): SegmentedPolynomialFromUniform1dJit()
        )
      )
      (linear): Linear(
        shared_weights=True, internal_weights=True, weight_numel=32768
        (transpose_in): TransposeIrrepsLayout((irrep,mul) -> (irrep,mul))
        (transpose_out): TransposeIrrepsLayout((irrep,mul) -> (irrep,mul))
        (f): ╭ a=[32768:2⨯(128,128)] b=[512:(1,128)+(3,128)] -> C=[512:(1,128)+(3,128)]
        ╰─ []·a[uv]·b[iu]➜C[iv] ─ num_paths=2 i={1, 3} u=128 v=128
        SegmentedPolynomial(
          (m): SegmentedPolynomialNaive(
            (graphs): ModuleList(
              (0): uv,iu,iv sizes=32768,512,512 num_segments=2,2,2 num_paths=2 i={1, 3} u=128 v=128
            )
          )
          (fallback): SegmentedPolynomialNaive(
            (graphs): ModuleList(
              (0): uv,iu,iv sizes=32768,512,512 num_segments=2,2,2 num_paths=2 i={1, 3} u=128 v=128
            )
          )
        )
      )
    )
    (1): EquivariantProductBasisBlock(
      (symmetric_contractions): SymmetricContraction(
        contraction_degree=3, weight_shape=(28, 128)
        (transpose_in): TransposeIrrepsLayout((irrep,mul) -> (irrep,mul))
        (transpose_out): TransposeIrrepsLayout((irrep,mul) -> (irrep,mul))
        (f): ╭ a=[1664:13⨯(128)] b=[2048:16⨯(128)] -> C=[128:1⨯(128)]
        │  []·a[u]·b[u]➜C[u] ─────────── num_paths=1 u=128
        │  []·a[u]·b[u]·b[u]➜C[u] ────── num_paths=16 u=128
        ╰─ []·a[u]·b[u]·b[u]·b[u]➜C[u] ─ num_paths=353 u=128
        SegmentedPolynomial(
          (m): SegmentedPolynomialFromUniform1dJit()
          (fallback): SegmentedPolynomialFromUniform1dJit()
        )
      )
      (linear): Linear(
        shared_weights=True, internal_weights=True, weight_numel=16384
        (transpose_in): TransposeIrrepsLayout((irrep,mul) -> (irrep,mul))
        (transpose_out): TransposeIrrepsLayout((irrep,mul) -> (irrep,mul))
        (f): ╭ a=[16384:1⨯(128,128)] b=[128:1⨯(1,128)] -> C=[128:1⨯(1,128)]
        ╰─ []·a[uv]·b[iu]➜C[iv] ─ num_paths=1 i=1 u=128 v=128
        SegmentedPolynomial(
          (m): SegmentedPolynomialNaive(
            (graphs): ModuleList(
              (0): uv,iu,iv operands=[(128, 128)],[(1, 128)],[(1, 128)] paths=[op0[0]*op1[0]*op2[0]*0.09]
            )
          )
          (fallback): SegmentedPolynomialNaive(
            (graphs): ModuleList(
              (0): uv,iu,iv operands=[(128, 128)],[(1, 128)],[(1, 128)] paths=[op0[0]*op1[0]*op2[0]*0.09]
            )
          )
        )
      )
    )
  )
  (readouts): ModuleList(
    (0): LinearReadoutBlock(
      (linear): Linear(
        shared_weights=True, internal_weights=True, weight_numel=128
        (transpose_in): TransposeIrrepsLayout((irrep,mul) -> (irrep,mul))
        (transpose_out): TransposeIrrepsLayout((irrep,mul) -> (irrep,mul))
        (f): ╭ a=[128:1⨯(128,1)] b=[512:(1,128)+(3,128)] -> C=[1:1⨯(1,1)]
        ╰─ []·a[uv]·b[iu]➜C[iv] ─ num_paths=1 i={1, 3} u=128 v=1
        SegmentedPolynomial(
          (m): SegmentedPolynomialNaive(
            (graphs): ModuleList(
              (0): uv,iu,iv sizes=128,512,1 num_segments=1,2,1 num_paths=1 i={1, 3} u=128 v=1
            )
          )
          (fallback): SegmentedPolynomialNaive(
            (graphs): ModuleList(
              (0): uv,iu,iv sizes=128,512,1 num_segments=1,2,1 num_paths=1 i={1, 3} u=128 v=1
            )
          )
        )
      )
    )
    (1): NonLinearReadoutBlock(
      (linear_1): Linear(
        shared_weights=True, internal_weights=True, weight_numel=2048
        (transpose_in): TransposeIrrepsLayout((irrep,mul) -> (irrep,mul))
        (transpose_out): TransposeIrrepsLayout((irrep,mul) -> (irrep,mul))
        (f): ╭ a=[2048:1⨯(128,16)] b=[128:1⨯(1,128)] -> C=[16:1⨯(1,16)]
        ╰─ []·a[uv]·b[iu]➜C[iv] ─ num_paths=1 i=1 u=128 v=16
        SegmentedPolynomial(
          (m): SegmentedPolynomialNaive(
            (graphs): ModuleList(
              (0): uv,iu,iv operands=[(128, 16)],[(1, 128)],[(1, 16)] paths=[op0[0]*op1[0]*op2[0]*0.09]
            )
          )
          (fallback): SegmentedPolynomialNaive(
            (graphs): ModuleList(
              (0): uv,iu,iv operands=[(128, 16)],[(1, 128)],[(1, 16)] paths=[op0[0]*op1[0]*op2[0]*0.09]
            )
          )
        )
      )
      (non_linearity): Activation [x] (16x0e -> 16x0e)
      (linear_2): Linear(
        shared_weights=True, internal_weights=True, weight_numel=16
        (transpose_in): TransposeIrrepsLayout((irrep,mul) -> (irrep,mul))
        (transpose_out): TransposeIrrepsLayout((irrep,mul) -> (irrep,mul))
        (f): ╭ a=[16:1⨯(16,1)] b=[16:1⨯(1,16)] -> C=[1:1⨯(1,1)]
        ╰─ []·a[uv]·b[iu]➜C[iv] ─ num_paths=1 i=1 u=16 v=1
        SegmentedPolynomial(
          (m): SegmentedPolynomialNaive(
            (graphs): ModuleList(
              (0): uv,iu,iv operands=[(16, 1)],[(1, 16)],[(1, 1)] paths=[op0[0]*op1[0]*op2[0]*0.25]
            )
          )
          (fallback): SegmentedPolynomialNaive(
            (graphs): ModuleList(
              (0): uv,iu,iv operands=[(16, 1)],[(1, 16)],[(1, 1)] paths=[op0[0]*op1[0]*op2[0]*0.25]
            )
          )
        )
      )
    )
  )
  (scale_shift): ScaleShiftBlock(scale=1.0818, shift=0.0000)
)