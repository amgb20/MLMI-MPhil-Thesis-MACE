# Comprehensive plotting code for MACE benchmark analysis
# Add this as a new cell in your notebook after the DataFrame display

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def create_benchmark_plots(df_results):
    """
    Create comprehensive visualizations for MACE benchmark results.
    
    Args:
        df_results: DataFrame with columns ['dtype', 'L0_time', 'L0_mem', 'L1_time', 'L1_mem', 'L1/L0_time', 'L1/L0_mem']
    """
    
    # Set style for better plots
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create a comprehensive visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes[1,2].axis('off')
    fig.suptitle('MACE Layer Benchmark Analysis', fontsize=16, fontweight='bold')
    
    # 1. Time comparison (Layer 0 vs Layer 1)
    ax1 = axes[0, 0]
    x = np.arange(len(df_results))
    width = 0.35
    ax1.bar(x - width/2, df_results['L0_time'], width, label='Layer 0', alpha=0.8)
    ax1.bar(x + width/2, df_results['L1_time'], width, label='Layer 1', alpha=0.8)
    ax1.set_xlabel('Data Type')
    ax1.set_ylabel('Time (ms)')
    ax1.set_title('Forward Pass Time Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels(df_results['dtype'], rotation=45)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Memory comparison (Layer 0 vs Layer 1)
    ax2 = axes[0, 1]
    ax2.bar(x - width/2, df_results['L0_mem'], width, label='Layer 0', alpha=0.8)
    ax2.bar(x + width/2, df_results['L1_mem'], width, label='Layer 1', alpha=0.8)
    ax2.set_xlabel('Data Type')
    ax2.set_ylabel('Memory (MB)')
    ax2.set_title('Memory Usage Comparison')
    ax2.set_xticks(x)
    ax2.set_xticklabels(df_results['dtype'], rotation=45)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Layer 1/Layer 0 ratios
    ax3 = axes[0, 2]
    ax3_twin = ax3.twinx()
    line1 = ax3.plot(df_results['dtype'], df_results['L1/L0_time'], 'o-', 
                      label='Time Ratio', linewidth=2, markersize=8, color='blue')
    line2 = ax3_twin.plot(df_results['dtype'], df_results['L1/L0_mem'], 's-', 
                           label='Memory Ratio', linewidth=2, markersize=8, color='red')
    ax3.set_xlabel('Data Type')
    ax3.set_ylabel('Time Ratio (L1/L0)', color='blue')
    ax3_twin.set_ylabel('Memory Ratio (L1/L0)', color='red')
    ax3.set_title('Layer 1 vs Layer 0 Ratios')
    ax3.tick_params(axis='y', labelcolor='blue')
    ax3_twin.tick_params(axis='y', labelcolor='red')
    ax3.grid(True, alpha=0.3)
    ax3.set_xticklabels(df_results['dtype'], rotation=45)
    
    # Combine legends
    lines1, labels1 = ax3.get_legend_handles_labels()
    lines2, labels2 = ax3_twin.get_legend_handles_labels()
    ax3.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    # 4. Speedup relative to FP64
    ax4 = axes[1, 0]
    fp64_time_l0 = df_results[df_results['dtype'] == 'FP64']['L0_time'].iloc[0]
    fp64_time_l1 = df_results[df_results['dtype'] == 'FP64']['L1_time'].iloc[0]
    speedup_l0 = fp64_time_l0 / df_results['L0_time']
    speedup_l1 = fp64_time_l1 / df_results['L1_time']
    
    ax4.bar(x - width/2, speedup_l0, width, label='Layer 0', alpha=0.8)
    ax4.bar(x + width/2, speedup_l1, width, label='Layer 1', alpha=0.8)
    ax4.set_xlabel('Data Type')
    ax4.set_ylabel('Speedup vs FP64')
    ax4.set_title('Speedup Relative to FP64')
    ax4.set_xticks(x)
    ax4.set_xticklabels(df_results['dtype'], rotation=45)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.axhline(y=1, color='black', linestyle='--', alpha=0.5)
    
    # 5. Memory efficiency (MB per ms)
    ax5 = axes[1, 1]
    memory_efficiency_l0 = df_results['L0_mem'] / df_results['L0_time']
    memory_efficiency_l1 = df_results['L1_mem'] / df_results['L1_time']
    
    ax5.bar(x - width/2, memory_efficiency_l0, width, label='Layer 0', alpha=0.8)
    ax5.bar(x + width/2, memory_efficiency_l1, width, label='Layer 1', alpha=0.8)
    ax5.set_xlabel('Data Type')
    ax5.set_ylabel('Memory Efficiency (MB/ms)')
    ax5.set_title('Memory Efficiency')
    ax5.set_xticks(x)
    ax5.set_xticklabels(df_results['dtype'], rotation=45)
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return fig

def print_summary_statistics(df_results):
    """
    Print comprehensive summary statistics for the benchmark results.
    
    Args:
        df_results: DataFrame with benchmark results
    """
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    
    print(f"\nFastest Layer 0: {df_results.loc[df_results['L0_time'].idxmin(), 'dtype']} "
          f"({df_results['L0_time'].min():.3f} ms)")
    print(f"Fastest Layer 1: {df_results.loc[df_results['L1_time'].idxmin(), 'dtype']} "
          f"({df_results['L1_time'].min():.3f} ms)")
    
    print(f"\nMost Memory Efficient Layer 0: {df_results.loc[df_results['L0_mem'].idxmin(), 'dtype']} "
          f"({df_results['L0_mem'].min():.3f} MB)")
    print(f"Most Memory Efficient Layer 1: {df_results.loc[df_results['L1_mem'].idxmin(), 'dtype']} "
          f"({df_results['L1_mem'].min():.3f} MB)")
    
    print(f"\nLayer 1 is on average {df_results['L1/L0_time'].mean():.2f}x slower than Layer 0")
    print(f"Layer 1 uses on average {df_results['L1/L0_mem'].mean():.2f}x more memory than Layer 0")
    
    # Calculate speedups vs FP64
    fp64_idx = df_results[df_results['dtype'] == 'FP64'].index[0]
    speedup_l0_vs_fp64 = df_results.loc[fp64_idx, 'L0_time'] / df_results['L0_time']
    speedup_l1_vs_fp64 = df_results.loc[fp64_idx, 'L1_time'] / df_results['L1_time']
    
    print(f"\nSpeedup vs FP64 (Layer 0):")
    for dtype, speedup in zip(df_results['dtype'], speedup_l0_vs_fp64):
        print(f"  {dtype}: {speedup:.2f}x")
    
    print(f"\nSpeedup vs FP64 (Layer 1):")
    for dtype, speedup in zip(df_results['dtype'], speedup_l1_vs_fp64):
        print(f"  {dtype}: {speedup:.2f}x")

# Comprehensive plotting code for numerical stability benchmark results
# Add this as a new cell in your notebook after the DataFrame displays

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

def create_numerical_stability_plots(df_layer0_comparison, df_layer1_comparison):
    """
    Create comprehensive visualizations for numerical stability benchmark results.
    
    Args:
        df_layer0_comparison: DataFrame with Layer 0 error comparisons
        df_layer1_comparison: DataFrame with Layer 1 error comparisons
    """
    
    # Set style for better plots
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create a comprehensive visualization
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle('MACE Numerical Stability Analysis', fontsize=16, fontweight='bold')
    
    # Prepare data for plotting
    precisions = ['FP32', 'TF32', 'FP16', 'BF16']
    
    # Extract forward and backward errors for each layer
    layer0_forward = df_layer0_comparison[df_layer0_comparison['metric'] == 'forward']
    layer0_backward = df_layer0_comparison[df_layer0_comparison['metric'] == 'backward']
    layer1_forward = df_layer1_comparison[df_layer1_comparison['metric'] == 'forward']
    layer1_backward = df_layer1_comparison[df_layer1_comparison['metric'] == 'backward']
    
    # 1. Absolute Error Comparison (Forward Pass)
    ax1 = axes[0, 0]
    x = np.arange(len(precisions))
    width = 0.35
    
    abs_errors_l0_fwd = [layer0_forward[layer0_forward['precision'] == p]['max_abs_error'].iloc[0] for p in precisions]
    abs_errors_l1_fwd = [layer1_forward[layer1_forward['precision'] == p]['max_abs_error'].iloc[0] for p in precisions]
    
    ax1.bar(x - width/2, abs_errors_l0_fwd, width, label='Layer 0', alpha=0.8)
    ax1.bar(x + width/2, abs_errors_l1_fwd, width, label='Layer 1', alpha=0.8)
    ax1.set_xlabel('Precision')
    ax1.set_ylabel('Max Absolute Error')
    ax1.set_title('Forward Pass - Absolute Error')
    ax1.set_xticks(x)
    ax1.set_xticklabels(precisions, rotation=45)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')  # Log scale for better visualization
    
    # 2. Relative Error Comparison (Forward Pass)
    ax2 = axes[0, 1]
    rel_errors_l0_fwd = [layer0_forward[layer0_forward['precision'] == p]['max_rel_error'].iloc[0] for p in precisions]
    rel_errors_l1_fwd = [layer1_forward[layer1_forward['precision'] == p]['max_rel_error'].iloc[0] for p in precisions]
    
    ax2.bar(x - width/2, rel_errors_l0_fwd, width, label='Layer 0', alpha=0.8)
    ax2.bar(x + width/2, rel_errors_l1_fwd, width, label='Layer 1', alpha=0.8)
    ax2.set_xlabel('Precision')
    ax2.set_ylabel('Max Relative Error')
    ax2.set_title('Forward Pass - Relative Error')
    ax2.set_xticks(x)
    ax2.set_xticklabels(precisions, rotation=45)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')
    
    # 3. Absolute Error Comparison (Backward Pass)
    ax3 = axes[0, 2]
    abs_errors_l0_bwd = [layer0_backward[layer0_backward['precision'] == p]['max_abs_error'].iloc[0] for p in precisions]
    abs_errors_l1_bwd = [layer1_backward[layer1_backward['precision'] == p]['max_abs_error'].iloc[0] for p in precisions]
    
    ax3.bar(x - width/2, abs_errors_l0_bwd, width, label='Layer 0', alpha=0.8)
    ax3.bar(x + width/2, abs_errors_l1_bwd, width, label='Layer 1', alpha=0.8)
    ax3.set_xlabel('Precision')
    ax3.set_ylabel('Max Absolute Error')
    ax3.set_title('Backward Pass - Absolute Error')
    ax3.set_xticks(x)
    ax3.set_xticklabels(precisions, rotation=45)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_yscale('log')
    
    # 4. Relative Error Comparison (Backward Pass)
    ax4 = axes[1, 0]
    rel_errors_l0_bwd = [layer0_backward[layer0_backward['precision'] == p]['max_rel_error'].iloc[0] for p in precisions]
    rel_errors_l1_bwd = [layer1_backward[layer1_backward['precision'] == p]['max_rel_error'].iloc[0] for p in precisions]
    
    ax4.bar(x - width/2, rel_errors_l0_bwd, width, label='Layer 0', alpha=0.8)
    ax4.bar(x + width/2, rel_errors_l1_bwd, width, label='Layer 1', alpha=0.8)
    ax4.set_xlabel('Precision')
    ax4.set_ylabel('Max Relative Error')
    ax4.set_title('Backward Pass - Relative Error')
    ax4.set_xticks(x)
    ax4.set_xticklabels(precisions, rotation=45)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_yscale('log')
    
    # 5. Layer 0 vs Layer 1 Error Ratio
    ax5 = axes[1, 1]
    # Calculate ratio of Layer 1 to Layer 0 errors
    fwd_abs_ratio = [l1/l0 if l0 > 0 else 0 for l0, l1 in zip(abs_errors_l0_fwd, abs_errors_l1_fwd)]
    bwd_abs_ratio = [l1/l0 if l0 > 0 else 0 for l0, l1 in zip(abs_errors_l0_bwd, abs_errors_l1_bwd)]
    
    ax5.plot(precisions, fwd_abs_ratio, 'o-', label='Forward Pass', linewidth=2, markersize=8)
    ax5.plot(precisions, bwd_abs_ratio, 's-', label='Backward Pass', linewidth=2, markersize=8)
    ax5.set_xlabel('Precision')
    ax5.set_ylabel('Layer 1 / Layer 0 Error Ratio')
    ax5.set_title('Error Ratio (Layer 1 / Layer 0)')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    ax5.axhline(y=1, color='black', linestyle='--', alpha=0.5, label='Equal Error')
    ax5.set_xticklabels(precisions, rotation=45)
    
    # 6. Heatmap of all errors
    ax6 = axes[1, 2]
    
    # Prepare data for heatmap
    heatmap_data = []
    metrics = ['Forward Abs', 'Forward Rel', 'Backward Abs', 'Backward Rel']
    
    for metric in metrics:
        if 'Forward Abs' in metric:
            values = abs_errors_l0_fwd + abs_errors_l1_fwd
        elif 'Forward Rel' in metric:
            values = rel_errors_l0_fwd + rel_errors_l1_fwd
        elif 'Backward Abs' in metric:
            values = abs_errors_l0_bwd + abs_errors_l1_bwd
        else:  # Backward Rel
            values = rel_errors_l0_bwd + rel_errors_l1_bwd
        heatmap_data.append(values)
    
    # Create labels for heatmap
    heatmap_labels = [f"{p}_L0" for p in precisions] + [f"{p}_L1" for p in precisions]
    
    # Normalize data for better visualization
    heatmap_array = np.array(heatmap_data)
    heatmap_norm = (heatmap_array - heatmap_array.min()) / (heatmap_array.max() - heatmap_array.min())
    
    im = ax6.imshow(heatmap_norm, cmap='viridis', aspect='auto')
    ax6.set_xticks(range(len(heatmap_labels)))
    ax6.set_yticks(range(len(metrics)))
    ax6.set_xticklabels(heatmap_labels, rotation=45, ha='right')
    ax6.set_yticklabels(metrics)
    ax6.set_title('Normalized Error Heatmap')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax6)
    cbar.set_label('Normalized Error Value')
    
    plt.tight_layout()
    plt.show()
    
    return fig

def create_detailed_error_analysis(df_layer0_comparison, df_layer1_comparison):
    """
    Create detailed error analysis plots with additional insights.
    """
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Detailed Numerical Stability Analysis', fontsize=16, fontweight='bold')
    
    precisions = ['FP32', 'TF32', 'FP16', 'BF16']
    
    # Extract data
    layer0_forward = df_layer0_comparison[df_layer0_comparison['metric'] == 'forward']
    layer0_backward = df_layer0_comparison[df_layer0_comparison['metric'] == 'backward']
    layer1_forward = df_layer1_comparison[df_layer1_comparison['metric'] == 'forward']
    layer1_backward = df_layer1_comparison[df_layer1_comparison['metric'] == 'backward']
    
    # 1. Forward vs Backward Error Comparison (Layer 0)
    ax1 = axes[0, 0]
    x = np.arange(len(precisions))
    width = 0.35
    
    fwd_abs_l0 = [layer0_forward[layer0_forward['precision'] == p]['max_abs_error'].iloc[0] for p in precisions]
    bwd_abs_l0 = [layer0_backward[layer0_backward['precision'] == p]['max_abs_error'].iloc[0] for p in precisions]
    
    ax1.bar(x - width/2, fwd_abs_l0, width, label='Forward Pass', alpha=0.8)
    ax1.bar(x + width/2, bwd_abs_l0, width, label='Backward Pass', alpha=0.8)
    ax1.set_xlabel('Precision')
    ax1.set_ylabel('Max Absolute Error')
    ax1.set_title('Layer 0 - Forward vs Backward')
    ax1.set_xticks(x)
    ax1.set_xticklabels(precisions, rotation=45)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    
    # 2. Forward vs Backward Error Comparison (Layer 1)
    ax2 = axes[0, 1]
    fwd_abs_l1 = [layer1_forward[layer1_forward['precision'] == p]['max_abs_error'].iloc[0] for p in precisions]
    bwd_abs_l1 = [layer1_backward[layer1_backward['precision'] == p]['max_abs_error'].iloc[0] for p in precisions]
    
    ax2.bar(x - width/2, fwd_abs_l1, width, label='Forward Pass', alpha=0.8)
    ax2.bar(x + width/2, bwd_abs_l1, width, label='Backward Pass', alpha=0.8)
    ax2.set_xlabel('Precision')
    ax2.set_ylabel('Max Absolute Error')
    ax2.set_title('Layer 1 - Forward vs Backward')
    ax2.set_xticks(x)
    ax2.set_xticklabels(precisions, rotation=45)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')
    
    # 3. Relative Error Comparison
    ax3 = axes[1, 0]
    fwd_rel_l0 = [layer0_forward[layer0_forward['precision'] == p]['max_rel_error'].iloc[0] for p in precisions]
    fwd_rel_l1 = [layer1_forward[layer1_forward['precision'] == p]['max_rel_error'].iloc[0] for p in precisions]
    
    ax3.plot(precisions, fwd_rel_l0, 'o-', label='Layer 0', linewidth=2, markersize=8)
    ax3.plot(precisions, fwd_rel_l1, 's-', label='Layer 1', linewidth=2, markersize=8)
    ax3.set_xlabel('Precision')
    ax3.set_ylabel('Max Relative Error')
    ax3.set_title('Forward Pass - Relative Error')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_yscale('log')
    
    # 4. Error Stability Analysis
    ax4 = axes[1, 1]
    # Calculate coefficient of variation (stability metric)
    def calc_stability(forward_errors, backward_errors):
        return [abs(f-b)/(f+b) if f+b > 0 else 0 for f, b in zip(forward_errors, backward_errors)]
    
    stability_l0 = calc_stability(fwd_abs_l0, bwd_abs_l0)
    stability_l1 = calc_stability(fwd_abs_l1, bwd_abs_l1)
    
    ax4.plot(precisions, stability_l0, 'o-', label='Layer 0', linewidth=2, markersize=8)
    ax4.plot(precisions, stability_l1, 's-', label='Layer 1', linewidth=2, markersize=8)
    ax4.set_xlabel('Precision')
    ax4.set_ylabel('Error Stability (|Fwd-Bwd|/(Fwd+Bwd))')
    ax4.set_title('Numerical Stability Analysis')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return fig

def print_numerical_stability_summary(df_layer0_comparison, df_layer1_comparison):
    """
    Print comprehensive summary of numerical stability results.
    """
    
    print("\n" + "="*80)
    print("NUMERICAL STABILITY SUMMARY")
    print("="*80)
    
    precisions = ['FP32', 'TF32', 'FP16', 'BF16']
    
    # Extract data
    layer0_forward = df_layer0_comparison[df_layer0_comparison['metric'] == 'forward']
    layer0_backward = df_layer0_comparison[df_layer0_comparison['metric'] == 'backward']
    layer1_forward = df_layer1_comparison[df_layer1_comparison['metric'] == 'forward']
    layer1_backward = df_layer1_comparison[df_layer1_comparison['metric'] == 'backward']
    
    print(f"\nMost Stable Precision (Layer 0):")
    min_abs_l0 = layer0_forward['max_abs_error'].min()
    best_prec_l0 = layer0_forward[layer0_forward['max_abs_error'] == min_abs_l0]['precision'].iloc[0]
    print(f"  Forward Pass: {best_prec_l0} (Error: {min_abs_l0:.2e})")
    
    min_abs_l0_bwd = layer0_backward['max_abs_error'].min()
    best_prec_l0_bwd = layer0_backward[layer0_backward['max_abs_error'] == min_abs_l0_bwd]['precision'].iloc[0]
    print(f"  Backward Pass: {best_prec_l0_bwd} (Error: {min_abs_l0_bwd:.2e})")
    
    print(f"\nMost Stable Precision (Layer 1):")
    min_abs_l1 = layer1_forward['max_abs_error'].min()
    best_prec_l1 = layer1_forward[layer1_forward['max_abs_error'] == min_abs_l1]['precision'].iloc[0]
    print(f"  Forward Pass: {best_prec_l1} (Error: {min_abs_l1:.2e})")
    
    min_abs_l1_bwd = layer1_backward['max_abs_error'].min()
    best_prec_l1_bwd = layer1_backward[layer1_backward['max_abs_error'] == min_abs_l1_bwd]['precision'].iloc[0]
    print(f"  Backward Pass: {best_prec_l1_bwd} (Error: {min_abs_l1_bwd:.2e})")
    
    # Calculate average errors
    print(f"\nAverage Absolute Errors:")
    for precision in precisions:
        l0_fwd = layer0_forward[layer0_forward['precision'] == precision]['max_abs_error'].iloc[0]
        l0_bwd = layer0_backward[layer0_backward['precision'] == precision]['max_abs_error'].iloc[0]
        l1_fwd = layer1_forward[layer1_forward['precision'] == precision]['max_abs_error'].iloc[0]
        l1_bwd = layer1_backward[layer1_backward['precision'] == precision]['max_abs_error'].iloc[0]
        
        avg_error = (l0_fwd + l0_bwd + l1_fwd + l1_bwd) / 4
        print(f"  {precision}: {avg_error:.2e}")
    
    # Find worst precision
    worst_precision = None
    worst_error = 0
    for precision in precisions:
        max_error = max(
            layer0_forward[layer0_forward['precision'] == precision]['max_abs_error'].iloc[0],
            layer0_backward[layer0_backward['precision'] == precision]['max_abs_error'].iloc[0],
            layer1_forward[layer1_forward['precision'] == precision]['max_abs_error'].iloc[0],
            layer1_backward[layer1_backward['precision'] == precision]['max_abs_error'].iloc[0]
        )
        if max_error > worst_error:
            worst_error = max_error
            worst_precision = precision
    
    print(f"\nLeast Stable Precision: {worst_precision} (Max Error: {worst_error:.2e})")
    
    # Layer comparison
    l0_avg = (layer0_forward['max_abs_error'].mean() + layer0_backward['max_abs_error'].mean()) / 2
    l1_avg = (layer1_forward['max_abs_error'].mean() + layer1_backward['max_abs_error'].mean()) / 2
    
    print(f"\nLayer Comparison:")
    print(f"  Layer 0 Average Error: {l0_avg:.2e}")
    print(f"  Layer 1 Average Error: {l1_avg:.2e}")
    print(f"  Layer 1 is {l1_avg/l0_avg:.2f}x less stable than Layer 0")