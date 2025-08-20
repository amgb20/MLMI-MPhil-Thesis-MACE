from mace.calculators import MACECalculator
from ase import build
import numpy as np
import torch
from torch.profiler import ProfilerActivity, profile, tensorboard_trace_handler
from torch.profiler import record_function


# Simple profiling of MACE calculator load and energy/forces evaluation
activities = [ProfilerActivity.CPU]
if torch.cuda.is_available():
    activities.append(ProfilerActivity.CUDA)

trace_dir = 'Experiments/numerical_stability/src/inference'

with profile(
    activities=activities,
    profile_memory=True,
    with_stack=False,
    with_modules=False,
    on_trace_ready=tensorboard_trace_handler(trace_dir),
) as prof:
    with record_function("Setup/LoadMACECalculator"):
        mace_calc = MACECalculator(
            model_paths="Experiments/numerical_stability/src/inference/model/MACE-OFF24_medium.model",
            default_dtype="float32",
            enable_cueq=True,
            device='cuda:1'
        )

    atoms = build.molecule("C60")
    atoms.calc = mace_calc

    for i in range(10):
        with record_function("Infer/get_potential_energy"):
            energy = atoms.get_potential_energy()
        with record_function("Infer/get_forces"):
            forces = atoms.get_forces()
        if i % 5 == 0:
            fmax = float(np.max(np.linalg.norm(forces, axis=1)))
            print(f"Iter {i+1}/10: E={float(energy):.6f}, F_max={fmax:.6f}")
        prof.step()

sort_key = "cuda_time_total" if torch.cuda.is_available() else "cpu_time_total"
print(prof.key_averages().table(sort_by=sort_key, row_limit=50))