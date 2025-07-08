**Precision Comparison Script Documentation**

This document provides an overview and detailed explanation of the `precision_3.py` script, which measures numerical differences between FP64, FP32, and FP16 executions of the MACE interaction blocks with cuEquivariance acceleration.

---

## Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Project Structure](#project-structure)
4. [Script Workflow](#script-workflow)

   * [Data Preparation](#data-preparation)
   * [Model Configuration](#model-configuration)
   * [Master FP64 Model Creation](#master-fp64-model-creation)
   * [Precision Loop](#precision-loop)

     * [Re-initialize & Cast Model](#re-initialize--cast-model)
     * [Recompute Embeddings](#recompute-embeddings)
     * [Prepare Inputs](#prepare-inputs)
     * [Interaction Blocks Execution](#interaction-blocks-execution)
   * [Error Metrics Computation](#error-metrics-computation)
5. [How to Run](#how-to-run)
6. [Troubleshooting](#troubleshooting)

---

## Overview

The `precision_3.py` script benchmarks a MACE model's interaction blocks under different numerical precisions (FP64, FP32, optional FP16), using cuEquivariance (cuEq) acceleration. It ensures:

* **Weight consistency:** all precision tests start from identical FP64 parameters.
* **Fresh inputs:** embeddings recomputed per precision.
* **Complete dtype alignment:** `node_feats`, `node_attrs`, and cuEq weights share the same dtype.
* **Gradient retention:** intermediate tensors retain gradients for backward‐error measurement.

The output is saved as `precision_3.xlsx`, containing absolute and relative forward/backward errors per block and precision.

---

## Prerequisites

* Python 3.8+ with the following packages installed:

  ```bash
  pip install torch torchvision
  pip install e3nn ase pandas numpy scipy
  pip install cuequivariance_torch  # for cuEq support
  ```
* Access to a CUDA‑capable GPU for FP16 tests (optional).
* The `solvent_rotated.xyz` data file located at `Experiments/Official MACE notebook/data/`.

---

## Project Structure

```
MLMI-MPhil-Thesis-MACE/
├── Experiments/
│   └── precision_3/
│       ├── precision_3.py    # This script
│       └── data/
│           └── solvent_rotated.xyz
└── mace/                     # MACE library
```

---

## Script Workflow

### Data Preparation

1. **Read molecule**: using ASE to load the first frame of `solvent_rotated.xyz`.
2. **Build atomic graph**: create a `Configuration` and convert to `AtomicData` with cutoff radius.
3. **Compute edges**: obtain `edge_index`, `shifts`, then calculate edge vectors and lengths.

### Model Configuration

Defined in `get_default_model_config()`:

* Radial & spherical parameters (cutoff, bessel functions, max ℓ).
* Interaction classes (`RealAgnosticResidualInteractionBlock`).
* Equivariance settings:

  ```python
  cueq_config = CuEquivarianceConfig(enabled=True, layout="ir_mul", group="O3_e3nn", optimize_all=True)
  oeq_config = OEQConfig(enabled=True, optimize_all=True, conv_fusion="atomic")
  ```

### Master FP64 Model Creation

Within `run_precision_comparison()`:

```python
# Ensure default dtype is FP64
torch.set_default_dtype(torch.float64)
# Build and snapshot
master = modules.MACE(**config).to(device, dtype=torch.float64)
master_state = master.state_dict()
```

This FP64 "master" serves as the single source of truth for weights.

### Precision Loop

Iterate over `precisions = [torch.float64, torch.float32]` (and optionally `torch.float16`).

#### Re-initialize & Cast Model

```python
m = modules.MACE(**config)
m.load_state_dict(master_state)
m = m.to(device, dtype=dtype)
```

* **`load_state_dict`** loads identical FP64 weights.
* **`.to(dtype)`** recasts every parameter and buffer (including cuEq FX submodules).

#### Recompute Embeddings

```python
batch_d = batch.to(device, dtype=dtype)
x0 = m.node_embedding(batch_d.node_attrs)
e_feats, _ = m.radial_embedding(lengths_d, batch_d.node_attrs, batch_d.edge_index, z_table)
e_attrs = m.spherical_harmonics(vectors_d)
```

All embeddings are generated at the current precision.

#### Prepare Inputs

```python
node_attrs_d = batch_d.node_attrs.to(dtype)  # ensure float
inputs0 = {
  'node_feats': x0.requires_grad_(),
  'node_attrs': node_attrs_d,
  'edge_feats': e_feats,
  'edge_attrs': e_attrs,
  'edge_index': batch_d.edge_index.long(),
}
```

Casting `node_attrs` avoids `tensordot` dtype mismatches.

#### Interaction Blocks Execution

```python
block0 = m.interactions[0]
product0 = m.products[0]
block1 = m.interactions[1]

# Forward + backward block0
out0, _ = block0(**inputs0)
out0_prod = product0(node_feats=out0, sc=None, node_attrs=node_attrs_d)
loss0 = (out0**2).sum(); loss0.backward()

# Prepare inputs1 from out0_prod, repeat for block1
```

Intermediate tensors use `.retain_grad()` to capture non-leaf gradients.

### Error Metrics Computation

After both blocks run at each precision, FP64 outputs/gradients are compared against FP32 (and FP16) to compute:

* **Max absolute** and **max relative** errors for forward and backward passes.
* Results are tabulated in pandas DataFrames and saved to `precision_3.xlsx`.

---

## How to Run

```bash
cd Experiments/precision_3
python3 precision_3.py
```

Watch the console logs for:

* Device and dtype setup.
* Edge & node counts.
* Per-precision debug prints confirming matching dtypes:

  ```
  ```

skip\_tp.weight: torch.float32 node\_feats: torch.float32 node\_attrs: torch.float32

```

The Excel workbook `precision_3.xlsx` will contain sheets `block_0` and `block_1` with error metrics.

---

## Troubleshooting

- **`RuntimeError: both inputs should have same dtype`**: Ensure you cast *all* inputs and model buffers (including `node_attrs`) to the same dtype before each forward.
- **`Gradient is None`**: Call `.retain_grad()` on non-leaf tensors or use `.requires_grad_()` on leaf tensors.
- **cuEq unavailable**: If `cuequivariance` isn’t installed or fails import, set `enabled=False` in `CuEquivarianceConfig`.

---

*End of documentation.*

```
