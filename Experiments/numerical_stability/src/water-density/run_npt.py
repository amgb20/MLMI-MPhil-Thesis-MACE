import sys
import time
from datetime import date
import numpy as np
from ase import units
from ase.io import read, write, Trajectory
from ase.io.lammpsdata import read_lammps_data, write_lammps_data
from ase.md.npt import NPT
from ase.md.velocitydistribution import Stationary, ZeroRotation
from mace.calculators import MACECalculator
from macetools.calculators.localsources import MACELocalSymmetricCharges

import time
import argparse
import torch



parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, required=True)
parser.add_argument("--structure", type=str, required=True)
parser.add_argument("--temp", type=float, required=True)
parser.add_argument("--runtime", type=int, required=True)
parser.add_argument("--label", type=str, required=True)
args = parser.parse_args()

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'



densfact = (units.m/1.0e2)**3/units.mol

out_name  = f"{args.label}_{args.temp}"

MD_T  = args.runtime
interval_save_config = 10000
interval_log = 50
temp  = args.temp
pres  = 1.013

ttime   = 100*units.fs
B_water = 2.2*units.GPa #vs 100*units.GPa recommended default
ptime   = 500*units.fs


calculator = MACECalculator(
        model_paths="Experiments/numerical_stability/src/inference/model/MACE-OFF24_medium.model",
        default_dtype=args.default_dtype,
        enable_cueq=args.enable_cueq,
        device=device,
        layer_default_dtype=args.layer_default_dtype,
    )

# special for just this one...
calculator.model.coulomb_energy.ewald_energy.kspace_cutoff *= (1/1.25)

start_config = read(args.structure, '-1')
start_config.set_cell(np.triu(start_config.get_cell()))
start_config.calc = calculator

md = NPT(atoms=start_config, timestep=1*units.fs, temperature_K=temp, externalstress=pres*units.bar, ttime=ttime, pfactor=ptime**2*B_water)
md.set_fraction_traceless(0) #I think this is equivalent to an isotropic stress tensor


start_config_positions = start_config.positions.copy()
start_config_com       = start_config.get_center_of_mass().copy()

thermo_traj = open(out_name + '.thermo', 'a')
coord_traj_name = out_name + '.xyz'

def print_traj(a=start_config):
    wall_time = time.time() - start_time
    calc_time = md.get_time()/units.fs
    calc_temp = a.get_temperature()
    calc_dens = np.sum(a.get_masses())/a.get_volume()*densfact
    calc_pres = -np.trace(a.get_stress(include_ideal_gas=True, voigt=False))/3/units.bar
    calc_epot = a.get_potential_energy()
    calc_msd  = (((a.positions-a.get_center_of_mass())-(start_config_positions-start_config_com))**2).mean(0).sum(0)
    calc_drft = ((a.get_center_of_mass()-start_config_com)**2).sum(0)
    calc_tens = -a.get_stress(include_ideal_gas=True, voigt=True)/units.bar
    if md.nsteps % interval_log == 0:
        thermo_traj.write(('%12d'+' %17.6f'*13+'\n') % (calc_time, calc_temp, calc_dens, wall_time, calc_pres, calc_epot, calc_msd, calc_drft, *tuple(calc_tens)))
        thermo_traj.flush()
    if md.nsteps % interval_save_config == 0:
        write(coord_traj_name, a, append=True)


#thermo_traj.write('# ASE Dynamics. Date: '+date.today().strftime("%d %b %Y")+'\n')
#thermo_traj.write('#   Time(fs)      Temperature(K)   Density(g/cm$^3$)    Wall_time(s)   Pressure(bar)      Energy(eV)         MSD(A$^2$)        COMSD(A$^2$)    P$_{xx}$(bar)     P$_{yy}$(bar)     P$_{zz}$(bar)     P$_{yz}$(bar)     P$_{xz}$(bar)     P$_{xy}$(bar)\n')
#open(coord_traj_name, 'w').close()

start_time = time.time()
print_traj(start_config)
md.attach(print_traj)
md.run(MD_T)
thermo_traj.close()


