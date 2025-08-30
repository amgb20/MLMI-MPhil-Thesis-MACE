# === Phase 4 MD Impact: Points 1–5 in one file ===
# 1) Upgraded simpleMD (repro velocities, correct friction units, returns arrays)
# 2) Run all precisions with identical initial state
# 3) Compare numerically vs FP64 (RMSE/bias for E/atom, T; max|F| ratio)
# 4) Overlay plots (E/atom & T plus deltas)
# 5) NVE micro-runs and drift (meV/atom/ps)

import os, time, numpy as np, matplotlib.pyplot as plt
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # determinism for CUDA GEMMs (set before torch import)

import torch
from ase import units
from ase.io import read
from ase.md.langevin import Langevin
from ase.md.verlet import VelocityVerlet
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary, ZeroRotation
from mace.calculators import MACECalculator

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))


from src.utils.get_model_audition import audit_model_dtypes, normalize_fp64_only, summarize_model_dtypes

# ---------------------- CONFIG ----------------------
MODEL   = "Experiments/numerical_stability/src/inference/model/MACE-OFF24_medium.model"
DATA_XYZ= "Experiments/Official MACE notebook/data/solvent_molecs.xyz"  # first structure used
OUTDIR  = "Experiments/numerical_stability/src/molecular-dynamic/results_phase4"
TEMP_K  = 1200.0
DT_FS   = 0.5
FRIC_PER_FS = 0.02        # Langevin gamma in 1/fs (converted to s^-1 internally)
STEPS_NVT   = 2000
STEPS_NVE   = 4000
SEED_INIT   = 123         # ensures identical velocities across variants
DEVICE  = "cuda" if torch.cuda.is_available() else "cpu"

VARIANTS = {
    "FP64": dict(default_dtype="float64", layer_dtype="float64", tf32=False),
    "FP32": dict(default_dtype="float32", layer_dtype="float32", tf32=False),
    "FP32_BF16": dict(default_dtype="float32", layer_dtype="bfloat16", tf32=False),  # optional
    "FP32_FP16": dict(default_dtype="float32", layer_dtype="float16", tf32=False),  # optional
    # "fp16-on-linear": {...}  # add if you have a patched model
}

# Acceptance guideposts (tune to your system)
THRESH = dict(E_RMSE_meV_atom=2.0, T_RMSE_K=10.0, Fmax_ratio_tol=0.05, NVE_drift_ratio=1.1)

# ---------------------- UTILS ----------------------
def set_repro(seed=SEED_INIT):
    np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark   = False
    torch.backends.cudnn.deterministic = True

def make_calc(default_dtype="float64", layer_dtype=None, tf32=False, cueq=True):
    torch.backends.cuda.matmul.allow_tf32 = bool(tf32)
    torch.backends.cudnn.allow_tf32       = bool(tf32)
    if layer_dtype is None: layer_dtype = default_dtype
    mace_calc = MACECalculator(
        model_paths=[MODEL],
        device=DEVICE,
        default_dtype=default_dtype,
        layer_default_dtype=layer_dtype,
        enable_cueq=cueq
    )

    audit_model_dtypes(mace_calc.models[0])
    if default_dtype == "float32":
        normalize_fp64_only(mace_calc.models[0], default_target=torch.float32)  # keep FP16 layers intact
        audit_model_dtypes(mace_calc.models[0])               # should now be pure float32             # should now be pure float64

    summarize_model_dtypes(mace_calc.models[0])

    return mace_calc

def clone_init_structure(data_xyz):
    # Use the very first structure; if your file has multiple, pick the right selector here
    return read(data_xyz, index=0).copy()

def simpleMD(init_conf, temp_K, calc, dt_fs=DT_FS, steps=STEPS_NVT, friction_per_fs=FRIC_PER_FS,
             seed=SEED_INIT, traj_path=None, sample_every=10):
    # Reproducible velocities/identical state across variants
    rng = np.random.default_rng(seed)
    atoms = init_conf.copy()
    MaxwellBoltzmannDistribution(atoms, temperature_K=temp_K, rng=rng)
    Stationary(atoms); ZeroRotation(atoms)

    atoms.calc = calc
    gamma = friction_per_fs / units.fs  # ASE Langevin expects s^-1
    dyn = Langevin(atoms, dt_fs*units.fs, temperature_K=temp_K, friction=gamma, fixcm=True)

    t_fs, T_K, E_atom, Fmax = [], [], [], []
    if traj_path: 
        try: os.remove(traj_path)
        except FileNotFoundError: pass

    def log():
        with torch.no_grad():
            t = dyn.get_time()/units.fs
            e = atoms.get_potential_energy()/len(atoms)
            Tk= atoms.get_temperature()
            F = atoms.get_forces()
        t_fs.append(t); E_atom.append(e); T_K.append(Tk); Fmax.append(np.linalg.norm(F,axis=1).max())
        if traj_path and (len(t_fs) % sample_every == 0):
            atoms.write(traj_path, append=True)
    dyn.attach(log, interval=1)
    dyn.run(steps)
    return dict(t_fs=np.array(t_fs), E_atom=np.array(E_atom), T_K=np.array(T_K),
                Fmax=np.array(Fmax), final_atoms=atoms.copy())

def nve_drift(init_conf, calc, dt_fs=DT_FS, steps=STEPS_NVE, seed=SEED_INIT):
    rng = np.random.default_rng(seed)
    atoms = init_conf.copy()
    MaxwellBoltzmannDistribution(atoms, temperature_K=TEMP_K, rng=rng)
    Stationary(atoms); ZeroRotation(atoms)
    atoms.calc = calc

    dyn = VelocityVerlet(atoms, dt_fs*units.fs)
    E_tot = []
    def log():
        with torch.no_grad():
            E_tot.append(atoms.get_total_energy()/len(atoms))  # eV/atom
    dyn.attach(log, interval=1)
    dyn.run(steps)
    E_tot = np.array(E_tot)
    t_ps  = (np.arange(len(E_tot))*dt_fs)/1000.0
    slope_eV_per_atom_ps = np.polyfit(t_ps, E_tot, 1)[0]
    drift_meV_per_atom_ps = 1000.0 * slope_eV_per_atom_ps
    return float(drift_meV_per_atom_ps), t_ps, E_tot

def save_npz(path, **arrays):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, **arrays)

def overlay_energy(results, out_png):
    ref = results["FP64"]
    plt.figure(figsize=(7,7))
    # E/atom
    plt.subplot(2,1,1)
    for name,res in results.items():
        plt.plot(res["t_fs"], res["E_atom"], label=name)
    plt.ylabel("E (eV/atom)"); plt.legend()
    # ΔE/atom vs FP64
    plt.subplot(2,1,2)
    for name,res in results.items():
        if name=="FP64": continue
        n = min(len(ref["t_fs"]), len(res["t_fs"]))
        plt.plot(ref["t_fs"][:n], res["E_atom"][:n]-ref["E_atom"][:n], label=f"{name}-FP64")
    plt.axhline(0, ls="--", lw=1); plt.ylabel("ΔE (eV/atom)"); plt.xlabel("Time (fs)")
    plt.tight_layout(); os.makedirs(os.path.dirname(out_png), exist_ok=True); plt.savefig(out_png, dpi=300)

def overlay_temperature(results, out_png):
    ref = results["FP64"]
    plt.figure(figsize=(7,7))
    # T
    plt.subplot(2,1,1)
    for name,res in results.items():
        plt.plot(res["t_fs"], res["T_K"], label=name)
    plt.ylabel("T (K)"); plt.legend()
    # ΔT vs FP64
    plt.subplot(2,1,2)
    for name,res in results.items():
        if name=="FP64": continue
        n = min(len(ref["t_fs"]), len(res["t_fs"]))
        plt.plot(ref["t_fs"][:n], res["T_K"][:n]-ref["T_K"][:n], label=f"{name}-FP64")
    plt.axhline(0, ls="--", lw=1); plt.ylabel("ΔT (K)"); plt.xlabel("Time (fs)")
    plt.tight_layout(); os.makedirs(os.path.dirname(out_png), exist_ok=True); plt.savefig(out_png, dpi=300)

def compare_numeric(results, out_csv):
    ref = results["FP64"]
    rows = [["Variant","E_RMSE (meV/atom)","E_bias (meV/atom)","T_RMSE (K)","T_bias (K)","|ΔT|max (K)","max|F| ratio"]]
    for name,res in results.items():
        n = min(len(ref["t_fs"]), len(res["t_fs"]))
        dE = res["E_atom"][:n] - ref["E_atom"][:n]
        dT = res["T_K"][:n]    - ref["T_K"][:n]
        E_RMSE_meV = 1000.0*np.sqrt(np.mean(dE**2))
        E_bias_meV = 1000.0*np.mean(dE)
        T_RMSE     = float(np.sqrt(np.mean(dT**2)))
        T_bias     = float(np.mean(dT))
        Tmax_delta = float(np.max(np.abs(dT)))
        Fmax_ratio = float(np.max(res["Fmax"]) / max(1e-12, np.max(ref["Fmax"])))
        rows.append([name,f"{E_RMSE_meV:.3f}",f"{E_bias_meV:.3f}",f"{T_RMSE:.2f}",f"{T_bias:.2f}",f"{Tmax_delta:.1f}",f"{Fmax_ratio:.3f}"])
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv,"w") as f:
        for r in rows: f.write(",".join(r)+"\n")
    return rows

def add_nve_drift_to_table(rows, drift_map, out_csv):
    # add a drift column to the CSV (keeps previous columns)
    with open(out_csv,"r") as f:
        lines = [ln.strip() for ln in f.readlines()]
    # insert header with drift
    header = lines[0] + ",NVE drift (meV/atom/ps),Drift Ratio vs FP64"
    out = [header]
    ref_drift = drift_map["FP64"]
    for ln in lines[1:]:
        name = ln.split(",")[0]
        d    = drift_map[name]
        ratio= d / (ref_drift if ref_drift != 0 else np.nan)
        out.append(ln + f",{d:.3f},{ratio:.3f}")
    with open(out_csv,"w") as f:
        f.write("\n".join(out))

# ---------------------- MAIN ----------------------
if __name__ == "__main__":
    os.makedirs(OUTDIR, exist_ok=True)
    set_repro(SEED_INIT)
    print("Device:", DEVICE)

    init_conf = clone_init_structure(DATA_XYZ)

    # 2) Run all precisions with identical initial state
    results = {}
    for name, cfg in VARIANTS.items():
        calc = make_calc(**cfg)
        res  = simpleMD(init_conf, TEMP_K, calc, dt_fs=DT_FS, steps=STEPS_NVT,
                        friction_per_fs=FRIC_PER_FS, seed=SEED_INIT,
                        traj_path=os.path.join(OUTDIR, f"{name}.xyz"), sample_every=10)
        results[name] = res
        save_npz(os.path.join(OUTDIR, f"md_{name}.npz"), **res)

    # 3) Numeric comparison vs FP64
    table_rows = compare_numeric(results, os.path.join(OUTDIR,"md_numeric_summary.csv"))
    print("\nNumeric summary:")
    for r in table_rows: print(", ".join(r))

    # 4) Overlay plots
    overlay_energy(results, os.path.join(OUTDIR, "overlay_energy.png"))
    overlay_temperature(results, os.path.join(OUTDIR, "overlay_temperature.png"))

    # 5) NVE micro-runs (drift)
    drift_map = {}
    for name, cfg in VARIANTS.items():
        calc = make_calc(**cfg)  # fresh calc per variant
        drift, t_ps, E_tot = nve_drift(init_conf, calc, dt_fs=DT_FS, steps=STEPS_NVE, seed=SEED_INIT)
        drift_map[name] = drift
        save_npz(os.path.join(OUTDIR, f"nve_{name}.npz"), t_ps=t_ps, E_tot=E_tot, drift_meV_per_atom_ps=drift)
    add_nve_drift_to_table(table_rows, drift_map, os.path.join(OUTDIR,"md_numeric_summary.csv"))

    # Optional: quick acceptance report
    ref_drift = drift_map["FP64"]
    print("\nAcceptance check (tunable thresholds):")
    for r in table_rows[1:]:
        name = r[0]
        E_ok  = float(r[1]) <= THRESH["E_RMSE_meV_atom"]
        T_ok  = float(r[3]) <= THRESH["T_RMSE_K"]
        F_ok  = abs(float(r[6]) - 1.0) <= THRESH["Fmax_ratio_tol"]
        drift_ratio = drift_map[name] / (ref_drift if ref_drift != 0 else np.nan)
        D_ok  = drift_ratio <= THRESH["NVE_drift_ratio"]
        verdict = "ACCEPT" if (E_ok and T_ok and F_ok and D_ok) else "REVIEW"
        print(f"{name}: E_RMSE={r[1]} meV/atom ({'OK' if E_ok else 'NO'}), "
              f"T_RMSE={r[3]} K ({'OK' if T_ok else 'NO'}), "
              f"max|F| ratio={r[6]} ({'OK' if F_ok else 'NO'}), "
              f"NVE drift ratio={drift_ratio:.3f} ({'OK' if D_ok else 'NO'}) -> {verdict}")
