import argparse
import os
import re
import sys
from typing import List, Tuple

import pandas as pd
import matplotlib.pyplot as plt


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


def make_labels(parent: str, child: str) -> str:
    alias = {
        "Interaction[0]": "Int0",
        "Interaction[1]": "Int1",
        "Product[0]": "Prod0",
        "Product[1]": "Prod1",
        "Readout[0]": "Read0",
        "Readout[1]": "Read1",
    }
    p = alias.get(parent, parent)
    return f"{p}: {child}"


def sanitize_child(child: str) -> str:
    # Keep the full subpath under the parent for uniqueness/readability
    return str(child)


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


def plot_sheet_data(df: pd.DataFrame, sheet_name: str, out_path: str) -> None:
    """Generate plot for a specific sheet's data."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    
    if df.empty:
        fig, ax = plt.subplots(figsize=(8, 4), dpi=160)
        ax.set_title(f"{sheet_name} (no entries)")
        ax.axis("off")
        plt.tight_layout()
        fig.savefig(out_path)
        plt.close(fig)
        return
    
    # Filter out Main entries for interaction/product/readout sheets
    if sheet_name in ["interaction", "product", "readout"]:
        df = df[~df["Name"].str.contains("/Main", na=False)]
    
    if df.empty:
        fig, ax = plt.subplots(figsize=(8, 4), dpi=160)
        ax.set_title(f"{sheet_name} (no entries after filtering)")
        ax.axis("off")
        plt.tight_layout()
        fig.savefig(out_path)
        plt.close(fig)
        return
    
    # Separate CUDA and CPU overhead data
    cuda_data = df[df["Type"] == "CUDA"]
    cpu_data = df[df["Type"] == "CPU_Overhead"]
    
    # Prepare data for plotting
    all_names = list(set(df["Name"].tolist()))
    all_names.sort()  # Sort for consistent ordering
    
    # For interaction/product/readout sheets, organize by [0] vs [1]
    if sheet_name in ["interaction", "product", "readout"]:
        # Split names into [0] and [1] groups
        group_0_names = [name for name in all_names if "[0]" in name]
        group_1_names = [name for name in all_names if "[1]" in name]
        
        # Sort each group
        group_0_names.sort()
        group_1_names.sort()
        
        # Combine groups
        all_names = group_0_names + group_1_names
    
    cuda_values = []
    cpu_values = []
    display_names = []
    
    for name in all_names:
        cuda_row = cuda_data[cuda_data["Name"] == name]
        cpu_row = cpu_data[cpu_data["Name"] == name]
        
        cuda_val = cuda_row["CUDA_ms"].iloc[0] if not cuda_row.empty else 0.0
        cpu_val = cpu_row["CPU_overhead_ms"].iloc[0] if not cpu_row.empty else 0.0
        
        cuda_values.append(cuda_val)
        cpu_values.append(cpu_val)
        
        # Clean name for display
        clean_name = clean_name_for_display(name, sheet_name)
        display_names.append(clean_name)
    
    # Create plot
    n = len(all_names)
    fig_w = max(12, n * 0.5)
    
    fig, ax = plt.subplots(figsize=(fig_w, 6), dpi=160)
    x = range(n)
    
    # Create stacked bar chart
    bars1 = ax.bar(x, cuda_values, label="CUDA time (ms)", alpha=0.8, color='skyblue')
    bars2 = ax.bar(x, cpu_values, bottom=cuda_values, label="CPU overhead (ms)", alpha=0.8, color='lightcoral')
    
    ax.set_ylabel("Time (ms)")
    
    # Set title based on sheet name
    if sheet_name == "Embeddings":
        title = "Embeddings - CUDA vs CPU Overhead"
    elif sheet_name == "interaction":
        title = "Interactions - CUDA vs CPU Overhead"
    elif sheet_name == "product":
        title = "Products - CUDA vs CPU Overhead"
    elif sheet_name == "readout":
        title = "Readouts - CUDA vs CPU Overhead"
    else:
        title = f"{sheet_name} - CUDA vs CPU Overhead"
    
    ax.set_title(title)
    ax.set_xticks(list(x))
    ax.set_xticklabels(display_names, rotation=45, ha='right')
    ax.legend(loc="upper right")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    
    # Add value labels on bars
    for i, (cuda, cpu) in enumerate(zip(cuda_values, cpu_values)):
        if cuda > 0:
            ax.text(i, cuda/2, f"{cuda:.2f}", ha='center', va='center', fontsize=8, fontweight='bold')
        if cpu > 0:
            ax.text(i, cuda + cpu/2, f"{cpu:.2f}", ha='center', va='center', fontsize=8, fontweight='bold')
    
    # For interaction/product/readout sheets, add visual separator between [0] and [1] groups
    if sheet_name in ["interaction", "product", "readout"] and len(group_0_names) > 0 and len(group_1_names) > 0:
        separator_pos = len(group_0_names) - 0.5
        
        # Add the vertical separator line
        ax.axvline(x=separator_pos, color='gray', linestyle='--', alpha=0.7)
        
        # Calculate positions for the group labels
        y_max = ax.get_ylim()[1]
        
        # Position [0] label on the left side of the separator
        ax.text(separator_pos - 0.5, y_max * 0.95, 
                f"[0]", ha='center', va='top', 
                fontsize=10, fontweight='bold', color='gray', 
                bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
        
        # Position [1] label on the right side of the separator
        ax.text(separator_pos + 0.5, y_max * 0.95, 
                f"[1]", ha='center', va='top', 
                fontsize=10, fontweight='bold', color='gray',
                bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Generate MACE profiler charts from XLSX with specific sheets")
    ap.add_argument("--xlsx", default="Experiments/numerical_stability/src/inference/results/block_level_cost/xlsx/filtered/cuda_time_total_comprehensive.xlsx", help="Path to the XLSX file")
    ap.add_argument("--out", default="Experiments/numerical_stability/src/inference/results/block_level_cost/images", help="Output directory")
    ap.add_argument("--prefix", default="mace", help="Output filename prefix")
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
    
    # Process each sheet
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
            # Generate plot
            plot_filename = f"{args.prefix}_{sheet_name.lower()}_cuda_cpu.png"
            plot_path = os.path.join(args.out, plot_filename)
            plot_sheet_data(processed_df, sheet_name, plot_path)
            print(f"  Plot saved: {plot_path}")
            
            # Add to export data
            for _, row in processed_df.iterrows():
                all_export_data.append({
                    "Sheet": sheet_name,
                    "Name": row["Name"],
                    "CUDA_ms": row["CUDA_ms"],
                    "CPU_overhead_ms": row["CPU_overhead_ms"],
                    "Type": row["Type"]
                })
        else:
            print(f"  No data to plot for sheet: {sheet_name}")
    
    # Export aggregated data as CSV
    if all_export_data:
        export_df = pd.DataFrame(all_export_data)
        csv_path = os.path.join(args.out, f"{args.prefix}_all_sheets_metrics.csv")
        export_df.to_csv(csv_path, index=False)
        print(f"Metrics exported to: {csv_path}")
    
    print(f"All processing complete. Output directory: {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()