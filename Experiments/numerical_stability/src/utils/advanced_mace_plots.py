import argparse
import os
import re
import sys
from typing import List, Tuple, Dict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle


def to_ms(s: str) -> float:
    """Convert a string like '1.234ms', '567.8us', '0.12s' to milliseconds."""
    if s is None:
        return 0.0
    s = str(s).strip()
    if not s or s == "0" or s.lower() in ("nan", "none"):
        return 0.0
    try:
        if s.endswith("ms"):
            return float(s[:-2])
        if s.endswith("us"):
            return float(s[:-2]) / 1000.0
        if s.endswith("s"):
            return float(s[:-1]) * 1000.0
        # fall back: just parse float
        return float(s)
    except Exception:
        return 0.0


EXPECTED_COLS = {
    "Name": ["Name", "name"],
    "CPU time avg": ["CPU time avg", "CPU avg", "Avg CPU time", "cpu_avg"],
    "CUDA time avg": ["CUDA time avg", "CUDA avg", "Avg CUDA time", "cuda_avg"],
}


def find_col(df: pd.DataFrame, candidates: List[str]) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    # try case-insensitive
    lower_map = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in lower_map:
            return lower_map[c.lower()]
    raise KeyError(f"Could not find any of the columns: {candidates} in: {list(df.columns)}")


def process_sheet_data(df: pd.DataFrame, sheet_name: str) -> pd.DataFrame:
    """Process data from a specific sheet to separate CUDA time and CPU overhead."""
    if df.empty:
        return pd.DataFrame(columns=["Name", "CUDA_ms", "CPU_overhead_ms"])
    
    # Normalize columns
    name_col = find_col(df, EXPECTED_COLS["Name"])
    cpu_col = find_col(df, EXPECTED_COLS["CPU time avg"])
    cuda_col = find_col(df, EXPECTED_COLS["CUDA time avg"])
    
    # Select and rename columns
    df = df[[name_col, cpu_col, cuda_col]].rename(
        columns={name_col: "Name", cpu_col: "CPU time avg", cuda_col: "CUDA time avg"}
    )
    
    # Convert to milliseconds
    df["CPU_avg_ms"] = df["CPU time avg"].apply(to_ms)
    df["CUDA_avg_ms"] = df["CUDA time avg"].apply(to_ms)
    
    # Separate CUDA time and CPU overhead
    # CUDA time: rows where CPU time is 0.000us (or very close to 0)
    # CPU overhead: rows where CPU time is non-zero
    cuda_rows = df[df["CPU_avg_ms"] < 0.001]  # Less than 0.001ms (1us)
    cpu_overhead_rows = df[df["CPU_avg_ms"] >= 0.001]  # 1us or more
    
    # Create result dataframe
    result_data = []
    
    # Process CUDA time rows
    for _, row in cuda_rows.iterrows():
        result_data.append({
            "Name": row["Name"],
            "CUDA_ms": row["CUDA_avg_ms"],
            "CPU_overhead_ms": 0.0,  # No CPU overhead for pure CUDA operations
            "Type": "CUDA"
        })
    
    # Process CPU overhead rows
    for _, row in cpu_overhead_rows.iterrows():
        result_data.append({
            "Name": row["Name"],
            "CUDA_ms": 0.0,  # No CUDA time for CPU overhead operations
            "CPU_overhead_ms": row["CPU_avg_ms"],
            "Type": "CPU_Overhead"
        })
    
    return pd.DataFrame(result_data)


def clean_name_for_display(name: str, sheet_name: str) -> str:
    """Clean and format names for better display on plots."""
    if sheet_name == "Embeddings":
        # Handle embedding names
        if "NodeEmbedding" in name:
            return "Node Embeddings"
        elif "RadialEmbedding" in name:
            return "Radial Embeddings"
        elif "SphericalHarmonics" in name:
            return "Spherical Harmonics"
        elif "AtomicEnergies" in name:
            return "Atomic Energies"
        elif "ScaleShift" in name:
            return "Scale Shift"
        else:
            # Extract the last part after the last slash
            return name.split("/")[-1] if "/" in name else name
    
    elif sheet_name in ["interaction", "product", "readout"]:
        # Handle interaction/product/readout names
        if "/Main" in name:
            return ""  # Return empty string to filter out Main entries
        
        # Extract the meaningful part and clean it up
        parts = name.split("/")
        if len(parts) >= 3:
            # Format: MACE/Interaction[0]/Something -> Something
            meaningful_part = parts[2]
            # Clean up common patterns
            meaningful_part = meaningful_part.replace("_", " ").title()
            return meaningful_part
        else:
            return name.split("/")[-1] if "/" in name else name
    
    else:
        # Default: just take the last part
        return name.split("/")[-1] if "/" in name else name


def plot_performance_distribution(all_data: pd.DataFrame, out_path: str) -> None:
    """Create histogram plots showing distribution of CUDA times and CPU overheads."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12), dpi=160)
    
    # Filter out zero values for better distribution visualization
    cuda_data = all_data[all_data["CUDA_ms"] > 0]["CUDA_ms"]
    cpu_data = all_data[all_data["CPU_overhead_ms"] > 0]["CPU_overhead_ms"]
    
    # CUDA time distribution
    ax1.hist(cuda_data, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    ax1.set_xlabel('CUDA Time (ms)')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Distribution of CUDA Execution Times')
    ax1.grid(True, alpha=0.3)
    
    # CPU overhead distribution
    ax2.hist(cpu_data, bins=30, alpha=0.7, color='lightcoral', edgecolor='black')
    ax2.set_xlabel('CPU Overhead (ms)')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Distribution of CPU Overhead Times')
    ax2.grid(True, alpha=0.3)
    
    # Log scale for better visualization of wide ranges
    ax3.hist(cuda_data, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    ax3.set_xlabel('CUDA Time (ms)')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Distribution of CUDA Times (Log Scale)')
    ax3.set_xscale('log')
    ax3.grid(True, alpha=0.3)
    
    # Box plot comparison
    data_to_plot = [all_data[all_data["Sheet"] == sheet]["CUDA_ms"].values for sheet in all_data["Sheet"].unique()]
    labels = [sheet.title() for sheet in all_data["Sheet"].unique()]
    ax4.boxplot(data_to_plot, labels=labels)
    ax4.set_ylabel('CUDA Time (ms)')
    ax4.set_title('CUDA Time Distribution by Sheet')
    ax4.tick_params(axis='x', rotation=45)
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_efficiency_analysis(all_data: pd.DataFrame, out_path: str) -> None:
    """Create scatter plots and efficiency analysis plots."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12), dpi=160)
    
    # Scatter plot: CUDA time vs CPU overhead
    ax1.scatter(all_data["CUDA_ms"], all_data["CPU_overhead_ms"], alpha=0.6, s=50)
    ax1.set_xlabel('CUDA Time (ms)')
    ax1.set_ylabel('CPU Overhead (ms)')
    ax1.set_title('CUDA Time vs CPU Overhead')
    ax1.grid(True, alpha=0.3)
    
    # Add trend line
    if len(all_data) > 1:
        z = np.polyfit(all_data["CUDA_ms"], all_data["CPU_overhead_ms"], 1)
        p = np.poly1d(z)
        ax1.plot(all_data["CUDA_ms"], p(all_data["CUDA_ms"]), "r--", alpha=0.8)
    
    # Efficiency ratio plot (CUDA/CPU ratio)
    efficiency_data = all_data.copy()
    efficiency_data["efficiency_ratio"] = efficiency_data["CUDA_ms"] / (efficiency_data["CPU_overhead_ms"] + 1e-6)  # Avoid division by zero
    
    ax2.hist(efficiency_data["efficiency_ratio"], bins=30, alpha=0.7, color='green', edgecolor='black')
    ax2.set_xlabel('Efficiency Ratio (CUDA/CPU)')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Distribution of Efficiency Ratios')
    ax2.grid(True, alpha=0.3)
    
    # Top 10 most efficient operations
    top_efficient = efficiency_data.nlargest(10, "efficiency_ratio")
    ax3.barh(range(len(top_efficient)), top_efficient["efficiency_ratio"], color='green', alpha=0.7)
    ax3.set_yticks(range(len(top_efficient)))
    ax3.set_yticklabels([clean_name_for_display(name, "general") for name in top_efficient["Name"]], fontsize=8)
    ax3.set_xlabel('Efficiency Ratio (CUDA/CPU)')
    ax3.set_title('Top 10 Most Efficient Operations')
    ax3.grid(True, alpha=0.3)
    
    # Bottom 10 least efficient operations
    bottom_efficient = efficiency_data.nsmallest(10, "efficiency_ratio")
    ax4.barh(range(len(bottom_efficient)), bottom_efficient["efficiency_ratio"], color='red', alpha=0.7)
    ax4.set_yticks(range(len(bottom_efficient)))
    ax4.set_yticklabels([clean_name_for_display(name, "general") for name in bottom_efficient["Name"]], fontsize=8)
    ax4.set_xlabel('Efficiency Ratio (CUDA/CPU)')
    ax4.set_title('Bottom 10 Least Efficient Operations')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_time_breakdown(all_data: pd.DataFrame, out_path: str) -> None:
    """Create pie charts and time breakdown plots."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12), dpi=160)
    
    # Total time breakdown by sheet
    sheet_totals = all_data.groupby("Sheet").agg({
        "CUDA_ms": "sum",
        "CPU_overhead_ms": "sum"
    }).reset_index()
    
    # Pie chart of total CUDA time by sheet
    ax1.pie(sheet_totals["CUDA_ms"], labels=[s.title() for s in sheet_totals["Sheet"]], 
             autopct='%1.1f%%', startangle=90)
    ax1.set_title('Total CUDA Time Distribution by Sheet')
    
    # Pie chart of total CPU overhead by sheet
    ax2.pie(sheet_totals["CPU_overhead_ms"], labels=[s.title() for s in sheet_totals["Sheet"]], 
             autopct='%1.1f%%', startangle=90)
    ax2.set_title('Total CPU Overhead Distribution by Sheet')
    
    # Stacked bar chart of time breakdown
    x_pos = range(len(sheet_totals))
    ax3.bar(x_pos, sheet_totals["CUDA_ms"], label='CUDA Time', alpha=0.8, color='skyblue')
    ax3.bar(x_pos, sheet_totals["CPU_overhead_ms"], bottom=sheet_totals["CUDA_ms"], 
             label='CPU Overhead', alpha=0.8, color='lightcoral')
    ax3.set_xlabel('Sheet')
    ax3.set_ylabel('Time (ms)')
    ax3.set_title('Total Time Breakdown by Sheet')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels([s.title() for s in sheet_totals["Sheet"]], rotation=45)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Cumulative time plot
    sorted_data = all_data.sort_values("CUDA_ms", ascending=False)
    cumulative_cuda = np.cumsum(sorted_data["CUDA_ms"])
    cumulative_cpu = np.cumsum(sorted_data["CPU_overhead_ms"])
    
    ax4.plot(range(len(sorted_data)), cumulative_cuda, label='CUDA Time', linewidth=2, color='skyblue')
    ax4.plot(range(len(sorted_data)), cumulative_cpu, label='CPU Overhead', linewidth=2, color='lightcoral')
    ax4.set_xlabel('Operation Rank')
    ax4.set_ylabel('Cumulative Time (ms)')
    ax4.set_title('Cumulative Time by Operation Rank')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_bottleneck_analysis(all_data: pd.DataFrame, out_path: str) -> None:
    """Create plots to identify performance bottlenecks."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12), dpi=160)
    
    # Top 15 time-consuming operations
    top_operations = all_data.nlargest(15, "CUDA_ms")
    y_pos = range(len(top_operations))
    
    ax1.barh(y_pos, top_operations["CUDA_ms"], color='red', alpha=0.7)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels([clean_name_for_display(name, "general") for name in top_operations["Name"]], fontsize=8)
    ax1.set_xlabel('CUDA Time (ms)')
    ax1.set_title('Top 15 Most Time-Consuming Operations')
    ax1.grid(True, alpha=0.3)
    
    # Top 15 CPU overhead operations
    top_cpu_ops = all_data.nlargest(15, "CPU_overhead_ms")
    y_pos = range(len(top_cpu_ops))
    
    ax2.barh(y_pos, top_cpu_ops["CPU_overhead_ms"], color='orange', alpha=0.7)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels([clean_name_for_display(name, "general") for name in top_cpu_ops["Name"]], fontsize=8)
    ax2.set_xlabel('CPU Overhead (ms)')
    ax2.set_title('Top 15 Highest CPU Overhead Operations')
    ax2.grid(True, alpha=0.3)
    
    # Performance vs overhead scatter with size indicating total time
    total_time = all_data["CUDA_ms"] + all_data["CPU_overhead_ms"]
    scatter = ax3.scatter(all_data["CUDA_ms"], all_data["CPU_overhead_ms"], 
                          s=total_time*100, alpha=0.6, c=total_time, cmap='viridis')
    ax3.set_xlabel('CUDA Time (ms)')
    ax3.set_ylabel('CPU Overhead (ms)')
    ax3.set_title('Performance vs Overhead (Size = Total Time)')
    ax3.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax3, label='Total Time (ms)')
    
    # Heatmap of performance by sheet and operation type
    pivot_data = all_data.pivot_table(
        values="CUDA_ms", 
        index="Sheet", 
        columns="Type", 
        aggfunc="mean", 
        fill_value=0
    )
    
    if not pivot_data.empty:
        sns.heatmap(pivot_data, annot=True, fmt='.2f', cmap='YlOrRd', ax=ax4)
        ax4.set_title('Average CUDA Time by Sheet and Operation Type')
        ax4.set_xlabel('Operation Type')
        ax4.set_ylabel('Sheet')
    
    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def create_summary_table(all_data: pd.DataFrame, out_path: str) -> None:
    """Create a summary statistics table."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    
    # Calculate summary statistics
    summary_stats = []
    
    for sheet in all_data["Sheet"].unique():
        sheet_data = all_data[all_data["Sheet"] == sheet]
        
        cuda_times = sheet_data[sheet_data["CUDA_ms"] > 0]["CUDA_ms"]
        cpu_overheads = sheet_data[sheet_data["CPU_overhead_ms"] > 0]["CPU_overhead_ms"]
        
        if len(cuda_times) > 0:
            summary_stats.append({
                "Sheet": sheet.title(),
                "Total Operations": len(sheet_data),
                "CUDA Operations": len(cuda_times),
                "CPU Overhead Operations": len(cpu_overheads),
                "Total CUDA Time (ms)": cuda_times.sum(),
                "Total CPU Overhead (ms)": cpu_overheads.sum(),
                "Avg CUDA Time (ms)": cuda_times.mean(),
                "Avg CPU Overhead (ms)": cpu_overheads.mean(),
                "Max CUDA Time (ms)": cuda_times.max(),
                "Max CPU Overhead (ms)": cpu_overheads.max()
            })
    
    # Create summary DataFrame
    summary_df = pd.DataFrame(summary_stats)
    
    # Save as CSV
    csv_path = out_path.replace('.png', '_summary.csv')
    summary_df.to_csv(csv_path, index=False)
    
    # Create visual table
    fig, ax = plt.subplots(figsize=(16, 8), dpi=160)
    ax.axis('tight')
    ax.axis('off')
    
    # Create table
    table = ax.table(cellText=summary_df.values, 
                     colLabels=summary_df.columns,
                     cellLoc='center',
                     loc='center',
                     bbox=[0, 0, 1, 1])
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)
    
    # Color header row
    for i in range(len(summary_df.columns)):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Color alternating rows
    for i in range(1, len(summary_df) + 1):
        for j in range(len(summary_df.columns)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#f0f0f0')
    
    plt.title('MACE Performance Summary Statistics', fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    fig.savefig(out_path, bbox_inches='tight', dpi=160)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Generate advanced MACE performance analysis plots")
    ap.add_argument("--xlsx", default="Experiments/numerical_stability/src/inference/results/block_level_cost/xlsx/filtered/cuda_time_total_comprehensive.xlsx", help="Path to the XLSX file")
    ap.add_argument("--out", default="Experiments/numerical_stability/src/inference/results/block_level_cost/images", help="Output directory")
    ap.add_argument("--prefix", default="mace_advanced", help="Output filename prefix")
    args = ap.parse_args()

    # Define sheets to process
    sheets_to_process = ["Embeddings", "interaction", "product", "readout"]
    
    # Read Excel file
    try:
        # Read all sheets at once
        all_sheets = pd.read_excel(args.xlsx, sheet_name=sheets_to_process)
    except Exception as e:
        print(f"Failed to read XLSX: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Process all sheets and combine data
    all_export_data = []
    
    for sheet_name in sheets_to_process:
        if sheet_name not in all_sheets:
            print(f"Warning: Sheet '{sheet_name}' not found in the Excel file", file=sys.stderr)
            continue
            
        print(f"Processing sheet: {sheet_name}")
        df = all_sheets[sheet_name]
        
        # Process the sheet data
        processed_df = process_sheet_data(df, sheet_name)
        
        if not processed_df.empty:
            # Add sheet information
            processed_df["Sheet"] = sheet_name
            
            # Add to combined data
            all_export_data.append(processed_df)
            print(f"  Processed {len(processed_df)} rows")
        else:
            print(f"  No data to process for sheet: {sheet_name}")
    
    if not all_export_data:
        print("No data to analyze!")
        return
    
    # Combine all data
    all_data = pd.concat(all_export_data, ignore_index=True)
    print(f"\nTotal data points: {len(all_data)}")
    
    # Create output directory
    os.makedirs(args.out, exist_ok=True)
    
    # Generate all plots
    print("\nGenerating advanced analysis plots...")
    
    # 1. Performance Distribution Plots
    print("  1. Performance Distribution Plots...")
    plot_performance_distribution(all_data, os.path.join(args.out, f"{args.prefix}_performance_distribution.png"))
    
    # 2. Efficiency Analysis Plots
    print("  2. Efficiency Analysis Plots...")
    plot_efficiency_analysis(all_data, os.path.join(args.out, f"{args.prefix}_efficiency_analysis.png"))
    
    # 3. Time Breakdown Plots
    print("  3. Time Breakdown Plots...")
    plot_time_breakdown(all_data, os.path.join(args.out, f"{args.prefix}_time_breakdown.png"))
    
    # 4. Bottleneck Analysis Plots
    print("  4. Bottleneck Analysis Plots...")
    plot_bottleneck_analysis(all_data, os.path.join(args.out, f"{args.prefix}_bottleneck_analysis.png"))
    
    # 5. Summary Table
    print("  5. Summary Statistics Table...")
    create_summary_table(all_data, os.path.join(args.out, f"{args.prefix}_summary_table.png"))
    
    # Export all data as CSV
    csv_path = os.path.join(args.out, f"{args.prefix}_complete_data.csv")
    all_data.to_csv(csv_path, index=False)
    print(f"Complete data exported to: {csv_path}")
    
    print(f"\nAll advanced analysis plots generated successfully!")
    print(f"Output directory: {os.path.abspath(args.out)}")
    print(f"Files created:")
    print(f"  - {args.prefix}_performance_distribution.png")
    print(f"  - {args.prefix}_efficiency_analysis.png")
    print(f"  - {args.prefix}_time_breakdown.png")
    print(f"  - {args.prefix}_bottleneck_analysis.png")
    print(f"  - {args.prefix}_summary_table.png")
    print(f"  - {args.prefix}_complete_data.csv")


if __name__ == "__main__":
    main()