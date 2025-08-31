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

def calculate_g_r(trajectory_file, plot_dir, args):
    """
    Calculate and plot the radial distribution function g(r) from MD trajectory
    """
    from ase.io import read
    import numpy as np
    import pylab as pl
    
    print("Calculating radial distribution function...")
    
    # Read trajectory
    try:
        trajectory = read(trajectory_file, ':')
        print(f"Read {len(trajectory)} frames from trajectory")
    except Exception as e:
        print(f"Error reading trajectory: {e}")
        return
    
    if len(trajectory) < 10:
        print("Warning: Very few frames in trajectory, g(r) may be unreliable")
    
    # Parameters for g(r) calculation
    r_max = 10.0  # Maximum distance in Å
    dr = 0.1      # Bin width in Å
    nbins = int(r_max / dr)
    
    # Initialize histogram
    hist = np.zeros(nbins)
    r_values = np.arange(nbins) * dr + dr/2
    
    # Calculate g(r) from all frames
    total_pairs = 0
    
    # Handle volume calculation - use bounding box if no cell defined
    first_frame = trajectory[0]
    if first_frame.cell.any():
        # Periodic system
        volume = first_frame.get_volume()
        print(f"Using periodic cell volume: {volume:.2f} Å³")
        use_pbc = True
    else:
        # Non-periodic system - use bounding box
        positions = first_frame.get_positions()
        min_coords = np.min(positions, axis=0)
        max_coords = np.max(positions, axis=0)
        box_size = max_coords - min_coords
        # Add some padding to avoid edge effects
        padding = 2.0  # Å
        box_size += padding
        volume = np.prod(box_size)
        print(f"Using bounding box volume: {volume:.2f} Å³")
        use_pbc = False
    
    density = len(first_frame) / volume
    
    for frame_idx, atoms in enumerate(trajectory):
        if frame_idx % max(1, len(trajectory)//10) == 0:
            print(f"Processing frame {frame_idx}/{len(trajectory)}")
        
        positions = atoms.get_positions()
        
        # Count pairs in each distance bin
        for i in range(len(positions)):
            for j in range(i+1, len(positions)):
                # Calculate distance
                if use_pbc and atoms.cell.any():
                    # Periodic boundary conditions
                    cell = atoms.get_cell()
                    diff = positions[i] - positions[j]
                    diff = diff - np.round(diff / cell) * cell
                    distance = np.linalg.norm(diff)
                else:
                    # No periodic boundary conditions
                    diff = positions[i] - positions[j]
                    distance = np.linalg.norm(diff)
                
                if distance < r_max:
                    bin_idx = int(distance / dr)
                    if bin_idx < nbins:
                        hist[bin_idx] += 2  # Count both i-j and j-i
                        total_pairs += 1
    
    # Normalize to get g(r)
    # g(r) = (N(r) / (4πr²dr)) / ρ
    # where N(r) is the number of pairs in bin, ρ is the density
    g_r = np.zeros(nbins)
    for i in range(nbins):
        r = r_values[i]
        if r > 0:
            # Volume of spherical shell: 4πr²dr
            shell_volume = 4 * np.pi * r**2 * dr
            # Normalize by density and number of frames
            g_r[i] = hist[i] / (shell_volume * density * len(trajectory))
    
    # Plot g(r)
    fig, ax = pl.subplots(1, 1, figsize=(8, 6))
    ax.plot(r_values, g_r, 'b-', linewidth=2, label='g(r)')
    ax.axhline(y=1, color='k', linestyle='--', alpha=0.5, label='g(r) = 1')
    
    ax.set_xlabel('r [Å]')
    ax.set_ylabel('g(r)')
    ax.set_title('Radial Distribution Function')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, r_max)
    
    # Save the plot
    gr_plot = f"{plot_dir}/g_r_ddtype_{args.default_dtype}_dll_{args.layer_default_dtype}.png"
    pl.savefig(gr_plot, dpi=300, bbox_inches='tight')
    print(f"g(r) plot saved to: {gr_plot}")
    
    # Save the data
    gr_data = f"{plot_dir}/g_r_data_ddtype_{args.default_dtype}_dll_{args.layer_default_dtype}.npz"
    np.savez(gr_data, r_values=r_values, g_r=g_r, hist=hist)
    print(f"g(r) data saved to: {gr_data}")
    
    return r_values, g_r

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

    # Calculate and plot g(r) after MD simulation
    plot_dir = 'Experiments/numerical_stability/src/molecular-dynamic/results'
    calculate_g_r('Experiments/Official MACE notebook/moldyn/mace01_md.xyz', plot_dir, args)


if __name__ == "__main__":
    main()