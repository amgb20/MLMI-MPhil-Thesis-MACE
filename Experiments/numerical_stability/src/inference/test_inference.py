from ase.io import read
from aseMolec import extAtoms as ea
import numpy as np
from mace.calculators.mace import MACECalculator
import torch
import logging


#let us start with a single molecule
init_conf = ea.sel_by_info_val(read('Experiments/Official MACE notebook/data/solvent_molecs.xyz',':'), 'Nmols', 1)[0].copy()

# check the device
if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

logging.info(f"Using device: {device}")

time_ms = []

start = torch.cuda.Event(enable_timing=True)
end   = torch.cuda.Event(enable_timing=True)
start.record()
#we can use MACE as a calculator in ASE!
mace_calc = MACECalculator(model_paths=['Experiments/Official MACE notebook/MACE_models/mace01_run-123_stagetwo.model'], device=device, default_dtype="float64")
# print(mace_calc.models[0])

end.record()
torch.cuda.synchronize(device)
time_ms.append(start.elapsed_time(end))

init_conf.calc = mace_calc

energy = init_conf.get_potential_energy()
forces = init_conf.get_forces()


# Output like eval_mace
print(f"\n=== MACE Evaluation Results ===")
print(f"Structure: {init_conf.get_chemical_formula()}")
print(f"Number of atoms: {len(init_conf)}")
print(f"Total energy: {energy:.8f} eV")
print(f"Energy per atom: {energy/len(init_conf):.8f} eV/atom")
print(f"Forces shape: {forces.shape}")
print(f"Maximum force: {np.max(np.linalg.norm(forces, axis=1)):.8f} eV/Å")
print(f"Time taken: {time_ms[0]:.2f} ms")

# simpleMD(init_conf, temp=1200, calc=mace_calc, fname='moldyn/mace01_md.xyz', s=10, T=2000)
