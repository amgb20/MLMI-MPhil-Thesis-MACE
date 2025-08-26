# 3. Inference Benchmark

Great—this is the perfect moment to separate **speed** from **fidelity**. Do it on two tracks:

# 1) Numerical fidelity (vs FP64 reference)

Treat the fp64/CUEq run as your *numerical oracle*. For each structure $i$ with $N_i$ atoms:

**Energy errors**

* $\Delta E_i = \hat E_i - E^{64}_i$
* MAE: $\mathrm{MAE}_E = \frac{1}{M}\sum_i |\Delta E_i|$  (optionally per-atom: divide by $N_i$)
* RMSE: $\sqrt{\frac{1}{M}\sum_i \Delta E_i^2}$
* Rel-err: $\frac{|\Delta E_i|}{|E^{64}_i|+\varepsilon}$
* Report: mean, std, median, P95, P99, and max.

**Force errors** (per-atom vectors $\mathbf F_{ij}\in\mathbb R^3$)

* Vector diff: $\Delta \mathbf F_{ij} = \hat{\mathbf F}_{ij} - \mathbf F^{64}_{ij}$
* Per-structure L2 rel-err:
  $r^F_i = \frac{\|\Delta \mathbf F_i\|_F}{\|\mathbf F^{64}_i\|_F+\varepsilon}$ with $\|\cdot\|_F$ over all atoms/components
* MAE (componentwise): $\mathrm{MAE}_F = \frac{1}{3\sum_i N_i}\sum_{i,j} \|\Delta \mathbf F_{ij}\|_1$
* RMSE: $\sqrt{\frac{1}{3\sum_i N_i}\sum_{i,j}\|\Delta \mathbf F_{ij}\|_2^2}$
* Cosine similarity:
  $\cos\theta_{ij} = \frac{\hat{\mathbf F}_{ij}\cdot \mathbf F^{64}_{ij}}{\|\hat{\mathbf F}_{ij}\|\|\mathbf F^{64}_{ij}\|+\varepsilon}$ (aggregate mean/P95)
* Count “catastrophic” cases: $\|\Delta \mathbf F_{ij}\|_2 > \tau_F$ (choose $\tau_F$ in eV/Å).

**Equivariance & invariance checks (very important for MACE)**

* Rotate coordinates by $R$. Energy should be invariant:
  $\delta E_{\text{rot}} = |E(\mathbf x) - E(R\mathbf x)|$
* Forces should be equivariant:
  $\delta \mathbf F_{\text{rot}} = \|\mathbf F(R\mathbf x) - R\,\mathbf F(\mathbf x)\|_F$
  Track MAE/P95; compare across precisions—low precision often raises these residuals.

**Energy–force consistency**

* Spot-check $\mathbf F \approx -\nabla E$ via finite differences:
  $F_{k}^{\text{FD}} \approx -\frac{E(\mathbf x+\epsilon \mathbf e_k)-E(\mathbf x-\epsilon \mathbf e_k)}{2\epsilon}$.
  Report $\|\mathbf F - \mathbf F^{\text{FD}}\|$ on a small set; look for growth with lower precision.

**Batch-size sensitivity**

* Re-run a subset with batch=1 vs large batch. Report deltas vs fp64 to catch accumulation/overflow issues.

**What to publish (plots)**

* CDF of $|\Delta E|$ and $r^F_i$ per variant.
* Scatter: $\|\mathbf F^{64}\|$ vs $\|\Delta \mathbf F\|$ (exposes magnitude-dependent error).
* Violin/box for cosine similarity of forces.

# 2) Physical accuracy (vs ground-truth labels, if you have them)

If you have DFT labels $(E^\*, \mathbf F^\*)$, compute **degradation** relative to fp64:

* $\Delta\mathrm{MAE}_E = \mathrm{MAE}_E^{\text{variant}} - \mathrm{MAE}_E^{64}$
* $\Delta\mathrm{MAE}_F = \mathrm{MAE}_F^{\text{variant}} - \mathrm{MAE}_F^{64}$

This shows whether mixed precision changes real accuracy, not just agreement with fp64.

# 3) Downstream stability checks (fast but telling)

Pick 2–3 representative systems and run short ASE MD with each precision variant using the *same* weights:

* **NVE energy drift**: fit a line to total energy vs time; report drift (e.g., meV/atom/ps).
* **Thermostatted stability (NVT)**: temperature variance, no NaNs/Infs.
* **Geometry relaxations**: final energy difference and RMSD between minima (fp64 vs variant).
  Optional for materials: lattice constants/volume after relaxation; for liquids: RDF overlap.

# 4) Acceptance thresholds (set them *before* looking)

Define margins relative to fp64 (example template—tune to your system scale):

* Energy: P95 $|\Delta E|$ ≤ $\delta_E$ (e.g., 0.1–1.0 meV/atom)
* Forces: P95 $r^F_i$ ≤ $\delta_F$ (e.g., 0.5–2%) and mean cosine similarity ≥ 0.999
* Equivariance residuals: within 2–5× fp64 baseline but below an absolute cap
* MD drift: within 1.1× fp64 drift, no instabilities

# 5) Run matrix (what you listed)

* **CUEq**: fp64, fp32
* **e3nn**: fp32, fp16 (and bf16 if kernel supports), and “fp16-on-linear” with safe casts
* **AMP** (autocast+GradScaler off for pure inference)
  Keep seeds fixed; `model.eval(); torch.use_deterministic_algorithms(True)` where possible; log NaN/Inf rate.

# 6) Practical tips for targeted mixed precision

* **Safe boundaries**: cast inputs to the sub-block to `float16`/`bfloat16`, keep accumulators and outputs in `float32` (or `float64` for the reference), e.g.:

  * `with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True): linear(...)`
  * Immediately `.to(torch.float32)` at the boundary if downstream isn’t safe.
* **Dynamic range**: FP16 can overflow on large intermediate norms—watch for spikes in $\|\mathbf h\|$; clamp/scale if needed.
* **Layer norms / softmax / reductions**: keep in fp32 (or at least use fp32 accumulators).

# 7) Minimal metric snippets (PyTorch-ish)

**Forces (structure-wise rel error & cosine):**

```python
def force_metrics(F_hat, F_ref, eps=1e-12):
    # F_*: [n_atoms, 3]
    diff = F_hat - F_ref
    rel = diff.norm(p=2) / (F_ref.norm(p=2) + eps)
    cos = (F_hat * F_ref).sum(dim=1) / (F_hat.norm(dim=1) * F_ref.norm(dim=1) + eps)
    return rel.item(), cos.mean().item()
```

**Equivariance residual:**

```python
def rot_equiv_err(model, pos, Z, R):  # R: [3,3] rotation
    E1, F1 = model(Z, pos)             # forward
    pos_r = pos @ R.T
    E2, F2 = model(Z, pos_r)
    E_inv = abs(E1 - E2).item()
    F_eq = (F2 - F1 @ R.T).norm().item()
    return E_inv, F_eq
```

---

### What to report in your Results section

1. A table summarizing **latency / throughput / memory** for each variant.
2. A table of **numerical fidelity** vs fp64 (MAE/RMSE/P95 for E & F, cosine sim, equivariance residuals).
3. A small figure panel with:

   * CDF of force rel-error
   * Scatter of $\|\mathbf F^{64}\|$ vs $\|\Delta \mathbf F\|$
   * NVE drift comparison
4. If you have labels, a table with **ΔMAE vs DFT** to prove you didn’t sacrifice physical accuracy.

With this setup, you’ll be able to say, for example: “**fp16-on-linear** delivers **1.8×** speedup vs fp32 with **P95 force rel-error = 0.6%**, no increase in NVE drift, and equivariance residuals within **1.5×** of fp64.”

If you want, I can sketch a quick benchmark harness scaffold tailored to your MACE + CUEq/e3nn loaders so you can drop it straight into your repo.
Absolutely — I opened your notebook (`T01_MACE_Practice_I.ipynb`) and mapped the “targeted mixed precision” strategy onto what you already have. Here’s a concrete, drop-in plan tied to **your actual cells and files**.

---

# What’s already in your notebook (anchors)

* **Data & labels (XTB)**: Cells **7–9** produce `data/solvent_xtb_[train|test].xyz` with `energy_xtb` / `forces_xtb`.
* **Training**: Cells **10–12** train a MACE model (`MACE_models/mace01_run-123_stagetwo.model`).
* **Evaluation**: Cells **15–18** run `eval_mace` and plot **XTB vs MACE**.
* **Inference/MD**: Cell **20** creates `MACECalculator(..., default_dtype="float32")` and runs MD.

This is perfect scaffolding to add **precision variants** and **fidelity/latency** benchmarking.

---

# A. Run precision variants (FP64 / FP32 / TF32) from the notebook

👉 Add **one new cell after Cell 20**:

```python
# === Precision variants on the *same* trained model ===
import torch
import numpy as np
from ase.io import read
from mace.calculators import MACECalculator

MODEL = "MACE_models/mace01_run-123_stagetwo.model"
TEST = 'data/solvent_xtb_test.xyz'  # you can subset for speed, e.g. ':200'

def make_calc(dtype="float64", tf32=False):
    torch.backends.cuda.matmul.allow_tf32 = bool(tf32)
    torch.backends.cudnn.allow_tf32 = bool(tf32)
    # 'default_dtype' in MACECalculator reliably supports "float64" and "float32"
    return MACECalculator(model_paths=[MODEL], device='cuda', default_dtype=dtype)

def collect_preds(configs, calc):
    E, F = [], []
    for at in configs:
        at.calc = calc
        E.append(at.get_potential_energy())   # scalar
        F.append(at.get_forces().copy())      # [N,3]
    return np.array(E, dtype=np.float64), np.array(F, dtype=np.float64)

db = read(TEST, ':300')  # choose a stable slice for reproducibility

# Reference (numerical oracle): FP64
calc64 = make_calc("float64")
E64, F64 = collect_preds(db, calc64)

# FP32
calc32 = make_calc("float32", tf32=False)
E32, F32 = collect_preds(db, calc32)

# TF32 (on A100 this changes matmul paths; still uses float32 interfaces)
torch.set_float32_matmul_precision('high')
calc_tf32 = make_calc("float32", tf32=True)
Etf, Ftf = collect_preds(db, calc_tf32)
```

> Why TF32 here? It’s a **free speedup** on A100 for matmuls with minimal accuracy cost, and works seamlessly with your current `MACECalculator(default_dtype="float32")`.

---

# B. Numerical fidelity vs FP64 (energy/forces + equivariance)

👉 Add **one cell** for metrics:

```python
def l2(A): return np.sqrt((A*A).sum())

def force_metrics(F_hat, F_ref, eps=1e-12):
    # Structure-wise relative error + cosine similarity
    rel = []
    cos = []
    for fhat, fref in zip(F_hat, F_ref):
        rel.append(l2(fhat - fref) / (l2(fref) + eps))
        # per-atom cosine, then mean
        c = (fhat * fref).sum(-1) / (np.linalg.norm(fhat, axis=-1)*np.linalg.norm(fref, axis=-1) + eps)
        cos.append(np.nanmean(c))
    return {
        "rel_mean": float(np.mean(rel)),
        "rel_p95": float(np.percentile(rel, 95)),
        "cos_mean": float(np.mean(cos)),
        "cos_p05": float(np.percentile(cos, 5)),
    }

def energy_metrics(E_hat, E_ref, per_atom=False, atoms_list=None, eps=1e-12):
    if per_atom and atoms_list is not None:
        N = np.array([len(at) for at in atoms_list])
        diff = (E_hat - E_ref) / N
        denom = np.maximum(np.abs(E_ref)/N, eps)
    else:
        diff = (E_hat - E_ref)
        denom = np.maximum(np.abs(E_ref), eps)
    return {
        "MAE": float(np.mean(np.abs(diff))),
        "RMSE": float(np.sqrt(np.mean(diff**2))),
        "rel_mean": float(np.mean(np.abs(diff)/denom)),
        "rel_p95": float(np.percentile(np.abs(diff)/denom, 95)),
        "max_abs": float(np.max(np.abs(diff))),
    }

# Summaries vs FP64
atoms_list = db  # used for per-atom scaling
summ = {
    "FP32_vs_FP64": {
        "E": energy_metrics(E32, E64, per_atom=True, atoms_list=atoms_list),
        "F": force_metrics(F32, F64),
    },
    "TF32_vs_FP64": {
        "E": energy_metrics(Etf, E64, per_atom=True, atoms_list=atoms_list),
        "F": force_metrics(Ftf, F64),
    },
}
summ
```

👉 **Equivariance / invariance** (small, surgical check):

```python
# Rotate one representative structure and compare
import math
def rotz(theta):
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c,-s,0],[s,c,0],[0,0,1]], dtype=np.float64)

def rot_check(calc, atoms, theta=1.234):
    R = rotz(theta)
    at = atoms.copy()
    pos = at.get_positions()
    E1 = calc.get_potential_energy(at); F1 = calc.get_forces(at)

    at_r = atoms.copy()
    at_r.set_positions(pos @ R.T)
    E2 = calc.get_potential_energy(at_r); F2 = calc.get_forces(at_r)

    E_inv = abs(E1 - E2)
    F_eq = np.linalg.norm(F2 - F1 @ R.T)
    return E_inv, F_eq

E_inv32, F_eq32 = rot_check(calc32, db[0])
E_inv64, F_eq64 = rot_check(calc64, db[0])
(E_inv32, F_eq32, E_inv64, F_eq64)
```

Report: the FP32 residuals should be close to FP64; if FP32/TF32 inflate `F_eq` notably, that’s a red flag for too-aggressive precision in equivariant blocks.

---

# C. Latency, throughput, and memory (end-to-end)

👉 Add **one cell** with GPU timers + memory:

```python
import torch, time

def time_inference(calc, atoms_list, warmup=5, iters=30):
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    # Warmup
    for _ in range(warmup):
        _ = atoms_list[0].copy(); _.calc = calc; _.get_potential_energy(); _.get_forces()
    torch.cuda.synchronize()

    # Timed loop
    start = torch.cuda.Event(enable_timing=True); end = torch.cuda.Event(enable_timing=True)
    times_ms = []
    for _ in range(iters):
        at = atoms_list[_ % len(atoms_list)].copy(); at.calc = calc
        start.record()
        _ = at.get_potential_energy(); _ = at.get_forces()
        end.record()
        torch.cuda.synchronize()
        times_ms.append(start.elapsed_time(end))
    peak_mem = torch.cuda.max_memory_allocated()
    return {
        "latency_ms_mean": float(np.mean(times_ms)),
        "latency_ms_p95": float(np.percentile(times_ms, 95)),
        "peak_mem_MB": float(peak_mem / (1024**2)),
    }

bench = {
    "FP64": time_inference(calc64, db),
    "FP32": time_inference(calc32, db),
    "TF32": time_inference(calc_tf32, db),
}
bench
```

> This reflects the **“end-to-end”** view you want (calculator + full forward incl. E & F), consistent with your results section.

---

# D. (Optional) NVE drift check (fast stability sanity)

Your `simpleMD` uses **Langevin** (thermostatted). For drift, run a short **NVE**:

```python
from ase.md.verlet import VelocityVerlet
from ase import units

def nve_drift(atoms, calc, dt_fs=0.5, steps=2000):
    at = atoms.copy(); at.calc = calc
    dyn = VelocityVerlet(at, dt_fs * units.fs)
    Es = []
    def log(): Es.append((at.get_total_energy()).item())
    dyn.attach(log, interval=1)
    dyn.run(steps)
    # meV/atom/ps drift
    t_ps = np.arange(len(Es)) * dt_fs / 1000.0
    p = np.polyfit(t_ps, np.array(Es)/len(at), 1)
    return float(p[0]*1000)  # meV/atom/ps

drift64 = nve_drift(db[0], calc64)
drift32 = nve_drift(db[0], calc32)
drift_tf = nve_drift(db[0], calc_tf32)
(drift64, drift32, drift_tf)
```

Accept if FP32/TF32 drift ≤ \~1.1× FP64 and no instabilities.

---

# E. Where do FP16 / “fp16-on-linear” fit here?

* `MACECalculator(default_dtype=…)` supports `"float64"` and `"float32"` cleanly. FP16 typically requires **surgical casting** of specific sub-modules (e.g., e3nn linears) with **safe FP32 accumulators**. That’s outside what your current notebook exposes via the calculator API.
* If you still want to prototype it *in this notebook*, you can **access the loaded torch model** and monkey-patch specific linears to run in half **with FP32 boundaries**. (This is experimental—do it on a copy of the model.)

Minimal pattern (works on many MACE/e3nn versions; adapt class checks as needed):

```python
import types, torch

def grab_model(calc):
    return getattr(calc, "model", None) or (getattr(calc, "models", [None]) or [None])[0]

def fp16_on_linears(calc):
    m = grab_model(calc)
    if m is None:
        raise RuntimeError("Could not access internal model from MACECalculator.")

    # Fallback test by name to avoid tight coupling to a specific e3nn type
    def is_linear(mod):
        return mod.__class__.__name__.lower().endswith("linear")

    patched = 0
    for name, mod in m.named_modules():
        if is_linear(mod):
            old_forward = mod.forward
            def new_forward(self, x, *a, **k):
                x16 = x.to(torch.float16)
                out = old_forward(x16, *a, **k)
                return out.to(torch.float32)
            mod.forward = types.MethodType(new_forward, mod)
            patched += 1
    return patched

# Example usage:
calc32 = make_calc("float32")
num_patched = fp16_on_linears(calc32)
print("Patched linears:", num_patched)
# Then reuse the same metrics/timing functions on calc32 (now with fp16 linears)
```

* Use the same **fidelity** and **latency** cells to compare “fp16-on-linear” vs FP32/FP64/TF32.
* If you need **CUEq vs e3nn** specifically: your current notebook doesn’t expose the backend toggle. Keep doing that in your dedicated bench scripts, then feed their predictions into the **same metric functions** above for apples-to-apples accuracy and speed summaries.

---

# F. What to put in your results tables/plots (from these cells)

* **Latency/memory**: `bench` dict → table (mean/P95 latency, peak MB) for FP64 / FP32 / TF32 / (fp16-on-linear if used).
* **Numerical fidelity**: `summ` dict → table of Energy (MAE/RMSE/rel P95 per atom) and Forces (rel mean/P95, cosine mean/P05).
* **Equivariance residuals**: `(E_inv*, F_eq*)` values for FP32/TF32 vs FP64.
* **NVE drift**: `drift*` (meV/atom/ps).

That gives you exactly the “**inférence bout-à-bout vs fp64, fp32, AMP/TF32, and ‘fp16-sur-linear’**: latence, débit, mémoire, erreurs E/F, stabilité” you want — using **your existing data, model, and file layout**.


Absolutely — I opened your notebook (`T01_MACE_Practice_I.ipynb`) and mapped the “targeted mixed precision” strategy onto what you already have. Here’s a concrete, drop-in plan tied to **your actual cells and files**.

---

# What’s already in your notebook (anchors)

* **Data & labels (XTB)**: Cells **7–9** produce `data/solvent_xtb_[train|test].xyz` with `energy_xtb` / `forces_xtb`.
* **Training**: Cells **10–12** train a MACE model (`MACE_models/mace01_run-123_stagetwo.model`).
* **Evaluation**: Cells **15–18** run `eval_mace` and plot **XTB vs MACE**.
* **Inference/MD**: Cell **20** creates `MACECalculator(..., default_dtype="float32")` and runs MD.

This is perfect scaffolding to add **precision variants** and **fidelity/latency** benchmarking.

---

# A. Run precision variants (FP64 / FP32 / TF32) from the notebook

👉 Add **one new cell after Cell 20**:

```python
# === Precision variants on the *same* trained model ===
import torch
import numpy as np
from ase.io import read
from mace.calculators import MACECalculator

MODEL = "MACE_models/mace01_run-123_stagetwo.model"
TEST = 'data/solvent_xtb_test.xyz'  # you can subset for speed, e.g. ':200'

def make_calc(dtype="float64", tf32=False):
    torch.backends.cuda.matmul.allow_tf32 = bool(tf32)
    torch.backends.cudnn.allow_tf32 = bool(tf32)
    # 'default_dtype' in MACECalculator reliably supports "float64" and "float32"
    return MACECalculator(model_paths=[MODEL], device='cuda', default_dtype=dtype)

def collect_preds(configs, calc):
    E, F = [], []
    for at in configs:
        at.calc = calc
        E.append(at.get_potential_energy())   # scalar
        F.append(at.get_forces().copy())      # [N,3]
    return np.array(E, dtype=np.float64), np.array(F, dtype=np.float64)

db = read(TEST, ':300')  # choose a stable slice for reproducibility

# Reference (numerical oracle): FP64
calc64 = make_calc("float64")
E64, F64 = collect_preds(db, calc64)

# FP32
calc32 = make_calc("float32", tf32=False)
E32, F32 = collect_preds(db, calc32)

# TF32 (on A100 this changes matmul paths; still uses float32 interfaces)
torch.set_float32_matmul_precision('high')
calc_tf32 = make_calc("float32", tf32=True)
Etf, Ftf = collect_preds(db, calc_tf32)
```

> Why TF32 here? It’s a **free speedup** on A100 for matmuls with minimal accuracy cost, and works seamlessly with your current `MACECalculator(default_dtype="float32")`.

---

# B. Numerical fidelity vs FP64 (energy/forces + equivariance)

👉 Add **one cell** for metrics:

```python
def l2(A): return np.sqrt((A*A).sum())

def force_metrics(F_hat, F_ref, eps=1e-12):
    # Structure-wise relative error + cosine similarity
    rel = []
    cos = []
    for fhat, fref in zip(F_hat, F_ref):
        rel.append(l2(fhat - fref) / (l2(fref) + eps))
        # per-atom cosine, then mean
        c = (fhat * fref).sum(-1) / (np.linalg.norm(fhat, axis=-1)*np.linalg.norm(fref, axis=-1) + eps)
        cos.append(np.nanmean(c))
    return {
        "rel_mean": float(np.mean(rel)),
        "rel_p95": float(np.percentile(rel, 95)),
        "cos_mean": float(np.mean(cos)),
        "cos_p05": float(np.percentile(cos, 5)),
    }

def energy_metrics(E_hat, E_ref, per_atom=False, atoms_list=None, eps=1e-12):
    if per_atom and atoms_list is not None:
        N = np.array([len(at) for at in atoms_list])
        diff = (E_hat - E_ref) / N
        denom = np.maximum(np.abs(E_ref)/N, eps)
    else:
        diff = (E_hat - E_ref)
        denom = np.maximum(np.abs(E_ref), eps)
    return {
        "MAE": float(np.mean(np.abs(diff))),
        "RMSE": float(np.sqrt(np.mean(diff**2))),
        "rel_mean": float(np.mean(np.abs(diff)/denom)),
        "rel_p95": float(np.percentile(np.abs(diff)/denom, 95)),
        "max_abs": float(np.max(np.abs(diff))),
    }

# Summaries vs FP64
atoms_list = db  # used for per-atom scaling
summ = {
    "FP32_vs_FP64": {
        "E": energy_metrics(E32, E64, per_atom=True, atoms_list=atoms_list),
        "F": force_metrics(F32, F64),
    },
    "TF32_vs_FP64": {
        "E": energy_metrics(Etf, E64, per_atom=True, atoms_list=atoms_list),
        "F": force_metrics(Ftf, F64),
    },
}
summ
```

👉 **Equivariance / invariance** (small, surgical check):

```python
# Rotate one representative structure and compare
import math
def rotz(theta):
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c,-s,0],[s,c,0],[0,0,1]], dtype=np.float64)

def rot_check(calc, atoms, theta=1.234):
    R = rotz(theta)
    at = atoms.copy()
    pos = at.get_positions()
    E1 = calc.get_potential_energy(at); F1 = calc.get_forces(at)

    at_r = atoms.copy()
    at_r.set_positions(pos @ R.T)
    E2 = calc.get_potential_energy(at_r); F2 = calc.get_forces(at_r)

    E_inv = abs(E1 - E2)
    F_eq = np.linalg.norm(F2 - F1 @ R.T)
    return E_inv, F_eq

E_inv32, F_eq32 = rot_check(calc32, db[0])
E_inv64, F_eq64 = rot_check(calc64, db[0])
(E_inv32, F_eq32, E_inv64, F_eq64)
```

Report: the FP32 residuals should be close to FP64; if FP32/TF32 inflate `F_eq` notably, that’s a red flag for too-aggressive precision in equivariant blocks.

---

# C. Latency, throughput, and memory (end-to-end)

👉 Add **one cell** with GPU timers + memory:

```python
import torch, time

def time_inference(calc, atoms_list, warmup=5, iters=30):
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    # Warmup
    for _ in range(warmup):
        _ = atoms_list[0].copy(); _.calc = calc; _.get_potential_energy(); _.get_forces()
    torch.cuda.synchronize()

    # Timed loop
    start = torch.cuda.Event(enable_timing=True); end = torch.cuda.Event(enable_timing=True)
    times_ms = []
    for _ in range(iters):
        at = atoms_list[_ % len(atoms_list)].copy(); at.calc = calc
        start.record()
        _ = at.get_potential_energy(); _ = at.get_forces()
        end.record()
        torch.cuda.synchronize()
        times_ms.append(start.elapsed_time(end))
    peak_mem = torch.cuda.max_memory_allocated()
    return {
        "latency_ms_mean": float(np.mean(times_ms)),
        "latency_ms_p95": float(np.percentile(times_ms, 95)),
        "peak_mem_MB": float(peak_mem / (1024**2)),
    }

bench = {
    "FP64": time_inference(calc64, db),
    "FP32": time_inference(calc32, db),
    "TF32": time_inference(calc_tf32, db),
}
bench
```

> This reflects the **“end-to-end”** view you want (calculator + full forward incl. E & F), consistent with your results section.

---

# D. (Optional) NVE drift check (fast stability sanity)

Your `simpleMD` uses **Langevin** (thermostatted). For drift, run a short **NVE**:

```python
from ase.md.verlet import VelocityVerlet
from ase import units

def nve_drift(atoms, calc, dt_fs=0.5, steps=2000):
    at = atoms.copy(); at.calc = calc
    dyn = VelocityVerlet(at, dt_fs * units.fs)
    Es = []
    def log(): Es.append((at.get_total_energy()).item())
    dyn.attach(log, interval=1)
    dyn.run(steps)
    # meV/atom/ps drift
    t_ps = np.arange(len(Es)) * dt_fs / 1000.0
    p = np.polyfit(t_ps, np.array(Es)/len(at), 1)
    return float(p[0]*1000)  # meV/atom/ps

drift64 = nve_drift(db[0], calc64)
drift32 = nve_drift(db[0], calc32)
drift_tf = nve_drift(db[0], calc_tf32)
(drift64, drift32, drift_tf)
```

Accept if FP32/TF32 drift ≤ \~1.1× FP64 and no instabilities.

---

# E. Where do FP16 / “fp16-on-linear” fit here?

* `MACECalculator(default_dtype=…)` supports `"float64"` and `"float32"` cleanly. FP16 typically requires **surgical casting** of specific sub-modules (e.g., e3nn linears) with **safe FP32 accumulators**. That’s outside what your current notebook exposes via the calculator API.
* If you still want to prototype it *in this notebook*, you can **access the loaded torch model** and monkey-patch specific linears to run in half **with FP32 boundaries**. (This is experimental—do it on a copy of the model.)

Minimal pattern (works on many MACE/e3nn versions; adapt class checks as needed):

```python
import types, torch

def grab_model(calc):
    return getattr(calc, "model", None) or (getattr(calc, "models", [None]) or [None])[0]

def fp16_on_linears(calc):
    m = grab_model(calc)
    if m is None:
        raise RuntimeError("Could not access internal model from MACECalculator.")

    # Fallback test by name to avoid tight coupling to a specific e3nn type
    def is_linear(mod):
        return mod.__class__.__name__.lower().endswith("linear")

    patched = 0
    for name, mod in m.named_modules():
        if is_linear(mod):
            old_forward = mod.forward
            def new_forward(self, x, *a, **k):
                x16 = x.to(torch.float16)
                out = old_forward(x16, *a, **k)
                return out.to(torch.float32)
            mod.forward = types.MethodType(new_forward, mod)
            patched += 1
    return patched

# Example usage:
calc32 = make_calc("float32")
num_patched = fp16_on_linears(calc32)
print("Patched linears:", num_patched)
# Then reuse the same metrics/timing functions on calc32 (now with fp16 linears)
```

* Use the same **fidelity** and **latency** cells to compare “fp16-on-linear” vs FP32/FP64/TF32.
* If you need **CUEq vs e3nn** specifically: your current notebook doesn’t expose the backend toggle. Keep doing that in your dedicated bench scripts, then feed their predictions into the **same metric functions** above for apples-to-apples accuracy and speed summaries.

---

# F. What to put in your results tables/plots (from these cells)

* **Latency/memory**: `bench` dict → table (mean/P95 latency, peak MB) for FP64 / FP32 / TF32 / (fp16-on-linear if used).
* **Numerical fidelity**: `summ` dict → table of Energy (MAE/RMSE/rel P95 per atom) and Forces (rel mean/P95, cosine mean/P05).
* **Equivariance residuals**: `(E_inv*, F_eq*)` values for FP32/TF32 vs FP64.
* **NVE drift**: `drift*` (meV/atom/ps).

That gives you exactly the “**inférence bout-à-bout vs fp64, fp32, AMP/TF32, and ‘fp16-sur-linear’**: latence, débit, mémoire, erreurs E/F, stabilité” you want — using **your existing data, model, and file layout**.
