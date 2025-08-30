from ase.io import read, write
from ase import units
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import Stationary, ZeroRotation, MaxwellBoltzmannDistribution
from mace.calculators import MACECalculator
from aseMolec import extAtoms as ea
import torch
from ase import build
import argparse


import random
import os
import time
import numpy as np
import pylab as pl
import sys
import logging
logging.basicConfig(level=logging.INFO)

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

def parse_args():
    parser = argparse.ArgumentParser(description="MACE profiling with different calculation modes")
    parser.add_argument("--mode", 
                       choices=["energy_only", "energy_forces", "inference_only"],
                       default="inference_only",
                       help="Calculation mode to profile")
    parser.add_argument("--supercell_size", type=int, default=3, help="Size of water box")
    parser.add_argument("--warmup", type=int, default=100, help="Number of warmup steps")
    parser.add_argument("--num_iters", type=int, default=100, help="Number of iterations")
    parser.add_argument("--tf32", action="store_true", help="Enable TF32")
    parser.add_argument("--default_dtype", type=str, default="float64", help="Default dtype for MACECalculator")
    parser.add_argument("--layer_default_dtype", type=str, default="float64", help="Default dtype for layers")
    parser.add_argument("--enable_cueq", action="store_true", help="Enable CUEQ")
    parser.add_argument("--autocast", action="store_true", help="Enable autocast")
    args = parser.parse_args()
    return args

def simpleMD(args, init_conf, temp, calc, fname, s, T):
    init_conf.set_calculator(calc)

    #initialize the temperature
    random.seed(701) #just making sure the MD failure is reproducible
    MaxwellBoltzmannDistribution(init_conf, temperature_K=300) #initialize temperature at 300
    Stationary(init_conf)
    ZeroRotation(init_conf)

    dyn = Langevin(init_conf, 1.0*units.fs, temperature_K=temp, friction=0.1) #drive system to desired temperature

    time_fs = []
    temperature = []
    energies = []

    #remove previously stored trajectory with the same name
    os.system('rm -rfv '+fname)

    # Create output directory for plots if it doesn't exist
    plot_dir = 'Experiments/numerical_stability/src/molecular-dynamic/results'
    os.makedirs(plot_dir, exist_ok=True)

    fig, ax = pl.subplots(2, 1, figsize=(6,6), sharex='all', gridspec_kw={'hspace': 0, 'wspace': 0})

    def write_frame():
            dyn.atoms.write(fname, append=True)
            time_fs.append(dyn.get_time()/units.fs)
            temperature.append(dyn.atoms.get_temperature())
            energies.append(dyn.atoms.get_potential_energy()/len(dyn.atoms))

            # Clear previous plots
            ax[0].clear()
            ax[1].clear()
            
            ax[0].plot(np.array(time_fs), np.array(energies), color="b")
            ax[0].set_ylabel('E (eV/atom)')

            # plot the temperature of the system as subplots
            ax[1].plot(np.array(time_fs), temperature, color="r")
            ax[1].set_ylabel('T (K)')
            ax[1].set_xlabel('Time (fs)')

    dyn.attach(write_frame, interval=s)
    t0 = time.time()
    dyn.run(T)
    t1 = time.time()
    print("MD finished in {0:.2f} minutes!".format((t1-t0)/60))
    
    # Save final comprehensive plot
    final_plot = f"{plot_dir}/md_final_ddtype_{args.default_dtype}_dll_{args.layer_default_dtype}.png"
    pl.savefig(final_plot, dpi=300, bbox_inches='tight')
    print(f"Final plot saved to: {final_plot}")

def main():
    args = parse_args()
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    # torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    np.random.seed(701)

    if torch.cuda.is_available():
        logging.info("CUDA is available")
        device = 'cuda'
        torch.cuda.init()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.reset_accumulated_memory_stats()
        torch.cuda.synchronize()
    else:
        logging.info("CUDA is not available")
        device = 'cpu'

     #let us start with a single molecule
    init_conf = ea.sel_by_info_val(read('Experiments/Official MACE notebook/data/solvent_molecs.xyz',':'), 'Nmols', 1)[0].copy()


    #we can use MACE as a calculator in ASE!
    mace_calc = MACECalculator(
        model_paths="Experiments/numerical_stability/src/inference/model/MACE-OFF24_medium.model",
        default_dtype=args.default_dtype,
        enable_cueq=args.enable_cueq,
        device=device,
        layer_default_dtype=args.layer_default_dtype,
    )

    simpleMD(args, init_conf, temp=1200, calc=mace_calc, fname='Experiments/Official MACE notebook/moldyn/mace01_md.xyz', s=10, T=2000)


if __name__ == "__main__":
    main()