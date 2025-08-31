import os, time, numpy as np, matplotlib.pyplot as plt
from scipy.interpolate import interp1d

def clean_and_interpolate_data(data_dict, temp_threshold=500, energy_threshold=-1156.4):
    """
    Clean data by filtering extreme values and interpolating through them
    """
    cleaned_data = {}
    
    for name, data in data_dict.items():
        cleaned_data[name] = {}
        
        # Clean temperature data
        t_fs = data["t_fs"]
        T_K = data["T_K"].copy()
        
        # Find indices where temperature exceeds threshold
        high_temp_mask = T_K > temp_threshold
        if np.any(high_temp_mask):
            print(f"Found {np.sum(high_temp_mask)} temperature values above {temp_threshold}K in {name}")
            # Interpolate through high temperature regions
            valid_indices = ~high_temp_mask
            if np.sum(valid_indices) > 1:  # Need at least 2 points for interpolation
                valid_times = t_fs[valid_indices]
                valid_temps = T_K[valid_indices]
                if len(valid_times) > 1:
                    temp_interp = interp1d(valid_times, valid_temps, kind='linear', 
                                          bounds_error=False, fill_value='extrapolate')
                    T_K = temp_interp(t_fs)
                    # Ensure interpolated values don't exceed threshold
                    T_K[high_temp_mask] = temp_threshold
        
        # Clean energy data
        E_atom = data["E_atom"].copy()
        
        # Find indices where energy goes below threshold
        low_energy_mask = E_atom < energy_threshold
        if np.any(low_energy_mask):
            print(f"Found {np.sum(low_energy_mask)} energy values below {energy_threshold} eV/atom in {name}")
            # Interpolate through low energy regions
            valid_indices = ~low_energy_mask
            if np.sum(valid_indices) > 1:  # Need at least 2 points for interpolation
                valid_times = t_fs[valid_indices]
                valid_energies = E_atom[valid_indices]
                if len(valid_times) > 1:
                    energy_interp = interp1d(valid_times, valid_energies, kind='linear', 
                                            bounds_error=False, fill_value='extrapolate')
                    E_atom = energy_interp(t_fs)
                    # Ensure interpolated values don't go below threshold
                    E_atom[low_energy_mask] = energy_threshold
        
        # Store cleaned data
        cleaned_data[name]["t_fs"] = t_fs
        cleaned_data[name]["T_K"] = T_K
        cleaned_data[name]["E_atom"] = E_atom
        
        # Copy other data if it exists and is numeric
        for key in data.keys():
            if key not in ["t_fs", "T_K", "E_atom"]:
                try:
                    # Only copy numeric arrays
                    if isinstance(data[key], np.ndarray) and np.issubdtype(data[key].dtype, np.number):
                        cleaned_data[name][key] = data[key]
                except:
                    # Skip non-numeric or problematic data
                    continue
    
    return cleaned_data

def overlay_energy(results, out_png):
    ref = results["FP64"]
    plt.figure(figsize=(7,7))
    # E/atom
    plt.subplot(2,1,1)
    for name,res in results.items():
        plt.plot(res["t_fs"], res["E_atom"], label=name, marker='o', markersize=2, markevery=10)
    plt.ylabel("E (eV/atom)"); plt.legend()
    # ΔE/atom vs FP64
    plt.subplot(2,1,2)
    for name,res in results.items():
        if name=="FP64": continue
        n = min(len(ref["t_fs"]), len(res["t_fs"]))
        plt.plot(ref["t_fs"][:n], res["E_atom"][:n]-ref["E_atom"][:n], label=f"{name}-FP64", marker='s', markersize=2, markevery=10)
    plt.axhline(0, ls="--", lw=1); plt.ylabel("ΔE (eV/atom)"); plt.xlabel("Time (fs)")
    plt.tight_layout(); os.makedirs(os.path.dirname(out_png), exist_ok=True); plt.savefig(out_png, dpi=300)

def overlay_temperature(results, out_png):
    ref = results["FP64"]
    plt.figure(figsize=(7,7))
    # T
    plt.subplot(2,1,1)
    for name,res in results.items():
        plt.plot(res["t_fs"], res["T_K"], label=name, marker='o', markersize=2, markevery=10)
    plt.ylabel("T (K)"); plt.legend()
    # ΔT vs FP64
    plt.subplot(2,1,2)
    for name,res in results.items():
        if name=="FP64": continue
        n = min(len(ref["t_fs"]), len(res["t_fs"]))
        plt.plot(ref["t_fs"][:n], res["T_K"][:n]-ref["T_K"][:n], label=f"{name}-FP64", marker='s', markersize=2, markevery=10)
    plt.axhline(0, ls="--", lw=1); plt.ylabel("ΔT (K)"); plt.xlabel("Time (fs)")
    plt.tight_layout(); os.makedirs(os.path.dirname(out_png), exist_ok=True); plt.savefig(out_png, dpi=300)

def plot_radial_distribution(results_dir, out_png):
    """
    Plot radial distribution function g(r) vs r[Å] for all precision variants
    """
    # Look for g(r) data files
    gr_files = {}
    precision_names = ["FP32_BF16", "FP32_FP16", "FP32", "FP64"]
    
    for name in precision_names:
        # Try to find g(r) data file
        gr_data_file = f"{results_dir}/g_r_data_ddtype_{name.lower()}_dll_{name.lower()}.npz"
        if os.path.exists(gr_data_file):
            gr_files[name] = gr_data_file
            print(f"Found g(r) data for {name}: {gr_data_file}")
        else:
            print(f"No g(r) data found for {name}")
    
    if not gr_files:
        print("No g(r) data files found. Creating empty plot.")
        plt.figure(figsize=(8, 6))
        plt.text(0.5, 0.5, 'No g(r) data available', ha='center', va='center', transform=plt.gca().transAxes)
        plt.xlabel('r [Å]')
        plt.ylabel('g(r)')
        plt.title('Radial Distribution Function')
        plt.tight_layout()
        os.makedirs(os.path.dirname(out_png), exist_ok=True)
        plt.savefig(out_png, dpi=300)
        return
    
    # Create the plot
    plt.figure(figsize=(10, 8))
    
    # Plot g(r) for each precision variant
    for name, file_path in gr_files.items():
        try:
            data = np.load(file_path, allow_pickle=True)
            r_values = data['r_values']
            g_r = data['g_r']
            
            plt.plot(r_values, g_r, label=name, linewidth=2, marker='o', markersize=3, markevery=20)
            
        except Exception as e:
            print(f"Error loading g(r) data for {name}: {e}")
            continue
    
    # Add reference line at g(r) = 1
    plt.axhline(y=1, color='k', linestyle='--', alpha=0.5, label='g(r) = 1')
    
    plt.xlabel('r [Å]')
    plt.ylabel('g(r)')
    plt.title('Radial Distribution Function - Comparison Across Precisions')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 10)  # Assuming r_max = 10 Å
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_png, dpi=300)
    print(f"g(r) plot saved to: {out_png}")


# read 4 .npz files and plot the results
file_fp32_bf16 = 'Experiments/numerical_stability/src/molecular-dynamic/results_20250831_105428/md_FP32_BF16.npz'
file_fp32_fp16 = 'Experiments/numerical_stability/src/molecular-dynamic/results_20250831_105428/md_FP32_FP16.npz'
file_fp32_fp32 = 'Experiments/numerical_stability/src/molecular-dynamic/results_20250831_105428/md_FP32.npz'
file_fp64 = 'Experiments/numerical_stability/src/molecular-dynamic/results_20250831_105428/md_FP64.npz'

file_list = [file_fp32_bf16, file_fp32_fp16, file_fp32_fp32, file_fp64]
precision_names = ["FP32_BF16", "FP32_FP16", "FP32", "FP64"]

results = {}
for i, (file_path, name) in enumerate(zip(file_list, precision_names)):
    if os.path.exists(file_path):
        try:
            # Load with allow_pickle=True to handle object arrays
            results[name] = np.load(file_path, allow_pickle=True)
            print(f"Loaded {name} from {file_path}")
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            continue
    else:
        print(f"Warning: File {file_path} not found")

if len(results) > 0:
    print("Cleaning data and interpolating through extreme values...")
    # Clean and interpolate the data
    cleaned_results = clean_and_interpolate_data(results)
    
    # Create plots with cleaned data
    overlay_energy(cleaned_results, "Experiments/numerical_stability/src/molecular-dynamic/results_20250831_105428/energy_cleaned.png")
    overlay_temperature(cleaned_results, "Experiments/numerical_stability/src/molecular-dynamic/results_20250831_105428/temperature_cleaned.png")
    print("Cleaned plots saved successfully!")
    
    # Also create original plots for comparison
    overlay_energy(results, "Experiments/numerical_stability/src/molecular-dynamic/results_20250831_105428/energy_original.png")
    overlay_temperature(results, "Experiments/numerical_stability/src/molecular-dynamic/results_20250831_105428/temperature_original.png")
    print("Original plots saved for comparison!")
    
    # Create g(r) plot
    trajectory_path_fp32_bf16 = 'Experiments/numerical_stability/src/molecular-dynamic/results_20250831_105428/FP32_BF16.xyz'
    trajectory_path_fp32_fp16 = 'Experiments/numerical_stability/src/molecular-dynamic/results_20250831_105428/FP32_FP16.xyz'
    trajectory_path_fp32_fp32 = 'Experiments/numerical_stability/src/molecular-dynamic/results_20250831_105428/FP32.xyz'
    trajectory_path_fp64 = 'Experiments/numerical_stability/src/molecular-dynamic/results_20250831_105428/FP64.xyz'
    trajectory_path_list = [trajectory_path_fp32_bf16, trajectory_path_fp32_fp16, trajectory_path_fp32_fp32, trajectory_path_fp64]
    trajectory_list = ['FP32_BF16', 'FP32_FP16', 'FP32', 'FP64']
    for trajectory_path, trajectory_name in zip(trajectory_path_list, trajectory_list):
        plot_radial_distribution(trajectory_path, f"{trajectory_name}.png")
        print("g(r) plot created!")
else:
    print("No data files found to plot")