import torch
from torch.serialization import add_safe_globals  # NEW

add_safe_globals([slice])  # allowlist 'slice' for the current process

from e3nn import o3  # now this import should succeed

import cuequivariance as cue
import cuequivariance_torch as cuet


irreps_in = o3.Irreps("10x0e")
irreps_out = o3.Irreps("128x0e")
shared_weights = True
internal_weights = True
layout = "mul_ir"

torch.manual_seed(0); 
torch.set_default_dtype(torch.float32)

# Create first layer with seed
inner_cueq = cuet.Linear(
                cue.Irreps("O3", irreps_in),
                cue.Irreps("O3", irreps_out),
                layout=layout,
                shared_weights=shared_weights,
                use_fallback=True,
            )

# Reset seed to same value for identical initialization
torch.manual_seed(0)
inner_o3 = o3.Linear(
                irreps_in, irreps_out,
                shared_weights=shared_weights,
                internal_weights=internal_weights,
            )

# Alternative approach: Copy weights from CUEQ to O3 to ensure identical weights
# This ensures both layers have exactly the same weights
if hasattr(inner_cueq, 'weight') and hasattr(inner_o3, 'weight'):
    with torch.no_grad():
        # Handle different weight shapes between backends
        cueq_weight = inner_cueq.weight
        o3_weight = inner_o3.weight
        
        print(f"CUEQ weight shape: {cueq_weight.shape}")
        print(f"O3 weight shape: {o3_weight.shape}")
        
        # Reshape weights to match if needed
        if cueq_weight.shape != o3_weight.shape:
            if cueq_weight.numel() == o3_weight.numel():
                # Same number of elements, just different shape
                reshaped_weight = cueq_weight.view(o3_weight.shape)
                inner_o3.weight.copy_(reshaped_weight)
                print(f"Reshaped CUEQ weight from {cueq_weight.shape} to {o3_weight.shape}")
            else:
                print("Warning: Weight tensors have different number of elements - cannot copy")
        else:
            inner_o3.weight.copy_(cueq_weight)
            
        # Handle bias if present
        if hasattr(inner_cueq, 'bias') and hasattr(inner_o3, 'bias'):
            if inner_cueq.bias is not None and inner_o3.bias is not None:
                inner_o3.bias.copy_(inner_cueq.bias)
            elif inner_cueq.bias is not None:
                print("CUEQ has bias but O3 doesn't")
            elif inner_o3.bias is not None:
                print("O3 has bias but CUEQ doesn't")

# Verify weights are identical
if hasattr(inner_cueq, 'weight') and hasattr(inner_o3, 'weight'):
    # Compare weights accounting for different shapes
    cueq_weight = inner_cueq.weight.flatten()
    o3_weight = inner_o3.weight.flatten()
    
    if cueq_weight.shape == o3_weight.shape:
        weight_diff = (cueq_weight - o3_weight).abs().max()
        print(f'Weight difference between CUEQ and O3: {weight_diff.item():.2e}')
        if weight_diff < 1e-6:
            print('✓ Weights are identical (within numerical precision)')
        else:
            print('⚠ Weights are different - comparison may not be fair')
    else:
        print('⚠ Weight tensors have different shapes - cannot compare directly')

#create a tensor of scaler 10x0e
torch.manual_seed(42)  # Fixed seed for input generation
x = torch.randn(1, irreps_in.dim)
y_cueq = inner_cueq(x)
y_o3 = inner_o3(x)
print('y_cueq.shape: ', y_cueq.shape)
print('y_o3.shape: ', y_o3.shape)
print('abs error: ', abs(y_cueq - y_o3).max())
print('relative error: ', abs(y_cueq - y_o3).max() / y_o3.max())
print('===============================================================')
print('compare_outputs_stats')
print('===============================================================')


import torch
import numpy as np

@torch.no_grad()
def compare_outputs_stats(inner_cueq, inner_o3, in_dim, n=256, dtype=torch.float32, device="cpu"):
    torch.manual_seed(0)  # seed inputs only
    X = torch.randn(n, in_dim, dtype=dtype, device=device)

    Yc = inner_cueq(X)
    Yo = inner_o3(X)

    diff = Yc - Yo
    rel_l2 = diff.norm(dim=1) / (Yo.norm(dim=1).clamp_min(1e-12))
    cos_sim = torch.sum(Yc*Yo, dim=1) / (Yc.norm(dim=1).clamp_min(1e-12) * Yo.norm(dim=1).clamp_min(1e-12))

    stats = {
        "Linf_diff_max": float(diff.abs().max().cpu()),
        "RelL2_mean": float(rel_l2.mean().cpu()),
        "RelL2_p95": float(rel_l2.quantile(0.95).cpu()),
        "CosSim_mean": float(cos_sim.mean().cpu()),
        "CosSim_p05": float(cos_sim.quantile(0.05).cpu()),
    }
    return stats

# usage:
stats = compare_outputs_stats(inner_cueq, inner_o3, irreps_in.dim)
print(stats)

import torch
print('===============================================================')
print('operator_gap_by_basis')
print('===============================================================')

@torch.no_grad()
def operator_gap_by_basis(inner_cueq, inner_o3, in_dim, dtype=torch.float32, device="cpu"):
    # Build input basis (I_in)
    I = torch.eye(in_dim, dtype=dtype, device=device)  # shape [in_dim, in_dim]
    Wc = inner_cueq(I).T  # columns are responses to basis -> operator columns; shape [out_dim, in_dim]
    Wo = inner_o3(I).T

    # Frobenius norms and relative gap
    diff = Wc - Wo
    fro_diff = diff.norm().item()
    fro_ref  = Wo.norm().item()
    rel_fro  = fro_diff / max(fro_ref, 1e-12)

    # Spectral norm gap (largest singular value)
    spec_diff = torch.linalg.svdvals(diff).max().item()

    return {
        "Fro_norm_diff": fro_diff,
        "Fro_norm_ref": fro_ref,
        "Rel_Fro_gap": rel_fro,
        "Spectral_norm_diff": spec_diff,
        "Wc_shape": tuple(Wc.shape),
        "Wo_shape": tuple(Wo.shape),
    }

# usage:
op_stats = operator_gap_by_basis(inner_cueq, inner_o3, irreps_in.dim)
print(op_stats)

print('===============================================================')
print('Compare node_feats from pickle files')
print('===============================================================')

import pickle
import numpy as np
import matplotlib.pyplot as plt
import glob
import os

def load_and_compare_node_feats(cueq_file, o3_file):
    """Load and compare node_feats from pickle files"""
    try:
        # Load CUEQ node features
        with open(cueq_file, 'rb') as f:
            cueq_data = pickle.load(f)
        
        # Load O3 node features  
        with open(o3_file, 'rb') as f:
            o3_data = pickle.load(f)
        
        # Extract node features (purely tensor format)
        cueq_feats = cueq_data
        o3_feats = o3_data
        
        # Convert tensors to numpy arrays
        if hasattr(cueq_feats, 'detach'):
            cueq_feats = cueq_feats.detach().cpu().numpy()
        elif hasattr(cueq_feats, 'numpy'):
            cueq_feats = cueq_feats.numpy()
            
        if hasattr(o3_feats, 'detach'):
            o3_feats = o3_feats.detach().cpu().numpy()
        elif hasattr(o3_feats, 'numpy'):
            o3_feats = o3_feats.numpy()
            
        # Convert to float32 for consistent comparison
        cueq_feats = cueq_feats.astype(np.float32)
        o3_feats = o3_feats.astype(np.float32)
        
        print(f"CUEQ node_feats shape: {cueq_feats.shape}")
        print(f"O3 node_feats shape: {o3_feats.shape}")
        
        # Compare shapes and reconcile if possible
        if cueq_feats.shape != o3_feats.shape:
            print(f"⚠️ Shape mismatch: CUEQ {cueq_feats.shape} vs O3 {o3_feats.shape}")
            # If total elements are the same, compare on flattened arrays
            if cueq_feats.size == o3_feats.size:
                cueq_feats = cueq_feats.reshape(-1)
                o3_feats = o3_feats.reshape(-1)
                print(f"→ Reshaped to: CUEQ {cueq_feats.shape} vs O3 {o3_feats.shape}")
            else:
                return None
        
        # Calculate differences
        diff = np.abs(cueq_feats - o3_feats)
        max_diff = np.max(diff)
        mean_diff = np.mean(diff)
        rel_diff = max_diff / (np.max(np.abs(o3_feats)) + 1e-12)
        
        # Calculate correlation
        cueq_flat = cueq_feats.flatten()
        o3_flat = o3_feats.flatten()
        correlation = np.corrcoef(cueq_flat, o3_flat)[0, 1]
        
        # Calculate relative L2 error
        rel_l2 = np.linalg.norm(diff) / (np.linalg.norm(o3_feats) + 1e-12)
        
        stats = {
            "Max_abs_diff": float(max_diff),
            "Mean_abs_diff": float(mean_diff),
            "Relative_diff": float(rel_diff),
            "Relative_L2": float(rel_l2),
            "Correlation": float(correlation),
            "CUEQ_range": [float(np.min(cueq_feats)), float(np.max(cueq_feats))],
            "O3_range": [float(np.min(o3_feats)), float(np.max(o3_feats))],
        }
        
        print("Node features comparison results:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
        
        # Check if they're identical within numerical precision
        if max_diff < 1e-6:
            print("✓ Node features are identical (within numerical precision)")
        elif max_diff < 1e-3:
            print("⚠ Node features are very close but not identical")
        else:
            print("⚠ Node features differ significantly")
            
        return stats
        
    except FileNotFoundError as e:
        print(f"Error: Could not find pickle file - {e}")
        return None
    except Exception as e:
        print(f"Error loading pickle files: {e}")
        return None

# Load and compare the pickle files
# Adjust these filenames to match your actual pickle files
cueq_pickle_file = "Experiments/numerical_stability/src/water-density/node_feats_fp32_cueq.pkl"  # or "node_feats_cueq_YYYYMMDD_HHMMSS.pkl"
o3_pickle_file = "Experiments/numerical_stability/src/water-density/node_feats_fp32_e3nn.pkl"      # or "node_feats_o3_YYYYMMDD_HHMMSS.pkl"

print(f"Loading CUEQ node features from: {cueq_pickle_file}")
print(f"Loading O3 node features from: {o3_pickle_file}")

node_feats_stats = load_and_compare_node_feats(cueq_pickle_file, o3_pickle_file)

if node_feats_stats is not None:
    print("\n" + "="*60)
    print("SUMMARY: Node features comparison")
    print("="*60)
    print(f"Maximum absolute difference: {node_feats_stats['Max_abs_diff']:.2e}")
    print(f"Relative difference: {node_feats_stats['Relative_diff']:.2e}")
    print(f"Correlation coefficient: {node_feats_stats['Correlation']:.6f}")
    print(f"Relative L2 error: {node_feats_stats['Relative_L2']:.2e}")
else:
    print("Could not compare node features - check file paths and formats")

print('===============================================================')
print('Plot comparison across multiple pickle files')
print('===============================================================')

def debug_file_pairing(cueq_pattern, o3_pattern):
    """Debug function to show how files are being paired"""
    cueq_files = sorted(glob.glob(cueq_pattern))
    o3_files = sorted(glob.glob(o3_pattern))
    
    print("="*60)
    print("FILE PAIRING DEBUG")
    print("="*60)
    print(f"CUEQ files found ({len(cueq_files)}):")
    for i, f in enumerate(cueq_files):
        print(f"  {i}: {os.path.basename(f)}")
    
    print(f"\nE3NN files found ({len(o3_files)}):")
    for i, f in enumerate(o3_files):
        print(f"  {i}: {os.path.basename(f)}")
    
    print(f"\nProposed pairing:")
    min_files = min(len(cueq_files), len(o3_files))
    for i in range(min_files):
        print(f"  {i}: {os.path.basename(cueq_files[i])} ↔ {os.path.basename(o3_files[i])}")
    
    if len(cueq_files) != len(o3_files):
        print(f"\n⚠️  MISMATCH: {len(cueq_files)} CUEQ files vs {len(o3_files)} E3NN files")
    else:
        print(f"\n✓ Perfect match: {len(cueq_files)} files each")
    print("="*60)

def plot_comparison_across_files(cueq_pattern, o3_pattern, save_path=None):
    """
    Plot comparison metrics across multiple pickle files
    
    Args:
        cueq_pattern: Glob pattern for CUEQ pickle files (e.g., "node_feats_fp32_cueq*.pkl")
        o3_pattern: Glob pattern for O3 pickle files (e.g., "node_feats_fp32_e3nn*.pkl")
        save_path: Path to save the plot (optional)
    """
    # Find all matching files
    cueq_files = sorted(glob.glob(cueq_pattern))
    o3_files = sorted(glob.glob(o3_pattern))
    
    print(f"Found {len(cueq_files)} CUEQ files: {cueq_files}")
    print(f"Found {len(o3_files)} O3 files: {o3_files}")
    
    if len(cueq_files) == 0 or len(o3_files) == 0:
        print("No matching files found!")
        return
    
    # Initialize lists to store metrics
    file_names = []
    max_diffs = []
    mean_diffs = []
    rel_diffs = []
    rel_l2s = []
    correlations = []
    
    # Ensure we have the same number of files for both backends
    min_files = min(len(cueq_files), len(o3_files))
    if len(cueq_files) != len(o3_files):
        print(f"Warning: Different number of files - CUEQ: {len(cueq_files)}, E3NN: {len(o3_files)}")
        print(f"Using first {min_files} files from each")
    
    # Compare each pair of files
    for i in range(min_files):
        cueq_file = cueq_files[i]
        o3_file = o3_files[i]
        print(f"\nComparing {i+1}/{min_files}: {os.path.basename(cueq_file)} vs {os.path.basename(o3_file)}")
        
        stats = load_and_compare_node_feats(cueq_file, o3_file)
        
        if stats is not None:
            # Use iteration number as the label
            file_name = f"iter_{i}"
            file_names.append(file_name)
            max_diffs.append(stats['Max_abs_diff'])
            mean_diffs.append(stats['Mean_abs_diff'])
            rel_diffs.append(stats['Relative_diff'])
            rel_l2s.append(stats['Relative_L2'])
            correlations.append(stats['Correlation'])
            print(f"  → Max abs diff: {stats['Max_abs_diff']:.2e}")
            print(f"  → Relative diff: {stats['Relative_diff']:.2e}")
        else:
            print(f"Skipping {cueq_file} vs {o3_file} due to comparison error")
    
    if not file_names:
        print("No successful comparisons found!")
        return
    
    # Create subplots with line plots for iteration tracking
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('CUEQ vs E3NN Node Features Comparison Across Iterations', fontsize=16)
    
    # Create proper iteration mapping based on file order
    # Since we're comparing pairs of files, use sequential numbering
    iterations = np.arange(len(file_names))
    
    print(f"File pairs and their iteration numbers:")
    for i, (cueq_file, o3_file) in enumerate(zip(cueq_files, o3_files)):
        print(f"  Iteration {i}: {os.path.basename(cueq_file)} vs {os.path.basename(o3_file)}")
    
    # Plot 1: Maximum Absolute Difference over iterations
    axes[0, 0].plot(iterations, max_diffs, 'o-', color='red', linewidth=2, markersize=6)
    axes[0, 0].set_title('Maximum Absolute Difference vs Iteration')
    axes[0, 0].set_xlabel('Iteration')
    axes[0, 0].set_ylabel('Max |CUEQ - E3NN|')
    axes[0, 0].set_yscale('log')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Relative Difference over iterations (main focus)
    axes[0, 1].plot(iterations, rel_diffs, 'o-', color='orange', linewidth=2, markersize=6)
    axes[0, 1].set_title('Relative Difference vs Iteration')
    axes[0, 1].set_xlabel('Iteration')
    axes[0, 1].set_ylabel('Max |CUEQ - E3NN| / Max |E3NN|')
    axes[0, 1].set_yscale('log')
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[0, 1].legend()
    
    # Plot 3: Relative L2 Error over iterations
    axes[1, 0].plot(iterations, rel_l2s, 'o-', color='blue', linewidth=2, markersize=6)
    axes[1, 0].set_title('Relative L2 Error vs Iteration')
    axes[1, 0].set_xlabel('Iteration')
    axes[1, 0].set_ylabel('||CUEQ - E3NN||₂ / ||E3NN||₂')
    axes[1, 0].set_yscale('log')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Correlation Coefficient over iterations
    axes[1, 1].plot(iterations, correlations, 'o-', color='green', linewidth=2, markersize=6)
    axes[1, 1].set_title('Correlation Coefficient vs Iteration')
    axes[1, 1].set_xlabel('Iteration')
    axes[1, 1].set_ylabel('Correlation(CUEQ, E3NN)')
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].grid(True, alpha=0.3)
    
    axes[1, 1].legend()
    
    plt.tight_layout()
    
    # Save plot if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    plt.show()
    
    # Create a focused plot for relative difference over iterations
    plt.figure(figsize=(10, 6))
    plt.plot(iterations, rel_diffs, 'o-', color='orange', linewidth=3, markersize=8, label='Relative Difference')
    plt.axhline(y=1e-3, color='red', linestyle='--', alpha=0.7, linewidth=2, label='Significant Difference (1e-3)')
    
    plt.title('CUEQ vs E3NN Relative Difference Over Iterations', fontsize=14, fontweight='bold')
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Relative Difference (Max |CUEQ - E3NN| / Max |E3NN|)', fontsize=12)
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11)
    
    # Add text annotations for key points
    for i, (iter_val, rel_diff) in enumerate(zip(iterations, rel_diffs)):
        if i % max(1, len(iterations)//10) == 0:  # Annotate every 10th point or fewer
            plt.annotate(f'{rel_diff:.2e}', (iter_val, rel_diff), 
                        textcoords="offset points", xytext=(0,10), ha='center', fontsize=9)
    
    plt.tight_layout()
    
    # Save focused plot
    if save_path:
        focused_save_path = save_path.replace('.png', '_focused_relative_diff_linear_up.png')
        plt.savefig(focused_save_path, dpi=300, bbox_inches='tight')
        print(f"Focused relative difference plot saved to: {focused_save_path}")
    
    plt.show()
    
    # Print summary statistics
    print("\n" + "="*60)
    print("SUMMARY STATISTICS ACROSS ALL FILES")
    print("="*60)
    print(f"Number of files compared: {len(file_names)}")
    print(f"Max absolute difference - Mean: {np.mean(max_diffs):.2e}, Std: {np.std(max_diffs):.2e}")
    print(f"Relative difference - Mean: {np.mean(rel_diffs):.2e}, Std: {np.std(rel_diffs):.2e}")
    print(f"Relative L2 error - Mean: {np.mean(rel_l2s):.2e}, Std: {np.std(rel_l2s):.2e}")
    print(f"Correlation - Mean: {np.mean(correlations):.6f}, Std: {np.std(correlations):.6f}")
    
    # Check for perfect matches
    perfect_matches = sum(1 for diff in max_diffs if diff < 1e-6)
    print(f"Perfect matches (diff < 1e-6): {perfect_matches}/{len(file_names)}")
    
    return {
        'file_names': file_names,
        'max_diffs': max_diffs,
        'mean_diffs': mean_diffs,
        'rel_diffs': rel_diffs,
        'rel_l2s': rel_l2s,
        'correlations': correlations
    }

# Example usage - adjust patterns to match your file naming convention
cueq_pattern = "Experiments/numerical_stability/src/water-density/linear_up/fp32_cueq/linear_up/*.pkl"
o3_pattern = "Experiments/numerical_stability/src/water-density/linear_up/fp32_e3nn/linear_up/*.pkl"

print(f"Searching for CUEQ files with pattern: {cueq_pattern}")
print(f"Searching for O3 files with pattern: {o3_pattern}")

# Debug file pairing first
debug_file_pairing(cueq_pattern, o3_pattern)

# Run the comparison and plotting
comparison_results = plot_comparison_across_files(
    cueq_pattern, 
    o3_pattern, 
    save_path="Experiments/numerical_stability/src/water-density/cueq_e3nn_comparison_linear_up.png"
)





