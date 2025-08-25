"""
Comprehensive plotting script for conv_tp benchmark results

This script creates various visualizations to analyze the performance and accuracy
of different backends (e3nn vs cuEq) across different precisions and batch sizes.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# Set style for better-looking plots
plt.style.use('default')
sns.set_palette("husl")

# Color scheme for backends
BACKEND_COLORS = {
    'e3nn': '#1f77b4',  # Blue
    'cueq': '#ff7f0e'   # Orange
}

# Precision order for consistent plotting
PRECISION_ORDER = ['fp64', 'fp32', 'bf16', 'fp16']

class ConvTPPlotter:
    """Class to handle all plotting operations for conv_tp benchmark results"""
    
    def __init__(self, csv_path, output_dir):
        """
        Initialize the plotter
        
        Args:
            csv_path (str): Path to the CSV results file
            output_dir (str): Directory to save output images
        """
        self.csv_path = csv_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load and preprocess data
        self.df = self.load_data()
        self.setup_plotting()
    
    def load_data(self):
        """Load and preprocess the CSV data"""
        try:
            df = pd.read_csv(self.csv_path)
            print(f"✓ Loaded data: {len(df)} rows, {len(df.columns)} columns")
            print(f"  Columns: {list(df.columns)}")
            return df
        except Exception as e:
            print(f"✗ Error loading data: {e}")
            return None
    
    def setup_plotting(self):
        """Setup plotting parameters"""
        # Set figure size and DPI for high-quality output
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['figure.dpi'] = 300
        plt.rcParams['savefig.dpi'] = 300
        plt.rcParams['font.size'] = 10
        plt.rcParams['axes.titlesize'] = 14
        plt.rcParams['axes.labelsize'] = 12
        plt.rcParams['xtick.labelsize'] = 10
        plt.rcParams['ytick.labelsize'] = 10
        plt.rcParams['legend.fontsize'] = 10
        
        # Set seaborn style
        sns.set_style("whitegrid")
        sns.set_context("paper", font_scale=1.2)
    
    def save_plot(self, filename, dpi=300, bbox_inches='tight'):
        """Save plot with consistent naming and quality"""
        filepath = self.output_dir / filename
        plt.savefig(filepath, dpi=dpi, bbox_inches=bbox_inches, facecolor='white')
        print(f"  ✓ Saved: {filepath}")
        plt.close()  # Close to free memory
    
    def plot_latency_comparison(self):
        """Plot 1: Forward and Backward Latency Comparison"""
        print("\n--- Creating Latency Comparison Plots ---")
        
        # Forward Latency
        plt.figure(figsize=(14, 6))
        
        plt.subplot(1, 2, 1)
        sns.barplot(
            data=self.df,
            x="Precision",
            y="Forward Latency (ms)",
            hue="Backend",
            palette=BACKEND_COLORS,
            order=PRECISION_ORDER
        )
        plt.title("Forward Latency by Precision and Backend", fontweight='bold')
        plt.ylabel("Latency (ms)")
        plt.xlabel("Precision")
        plt.legend(title="Backend", loc='upper right')
        
        # Backward Latency
        plt.subplot(1, 2, 2)
        sns.barplot(
            data=self.df,
            x="Precision",
            y="Backward Latency (ms)",
            hue="Backend",
            palette=BACKEND_COLORS,
            order=PRECISION_ORDER
        )
        plt.title("Backward Latency by Precision and Backend", fontweight='bold')
        plt.ylabel("Latency (ms)")
        plt.xlabel("Precision")
        plt.legend(title="Backend", loc='upper right')
        
        plt.tight_layout()
        self.save_plot("01_latency_comparison.png")
    
    def plot_speedup_analysis(self):
        """Plot 2: Speedup Analysis relative to FP64"""
        print("\n--- Creating Speedup Analysis Plots ---")
        
        # Filter out FP64 (baseline) and rows without speedup data
        speedup_df = self.df[
            (self.df["Precision"] != "fp64") & 
            (self.df["Forward Speedup vs FP64"].notna())
        ].copy()
        
        if len(speedup_df) == 0:
            print("  ⚠ No speedup data available")
            return
        
        plt.figure(figsize=(14, 6))
        
        # Forward Speedup
        plt.subplot(1, 2, 1)
        sns.barplot(
            data=speedup_df,
            x="Precision",
            y="Forward Speedup vs FP64",
            hue="Backend",
            palette=BACKEND_COLORS,
            order=[p for p in PRECISION_ORDER if p != "fp64"]
        )
        plt.title("Forward Speedup vs FP64 Baseline", fontweight='bold')
        plt.ylabel("Speedup (x)")
        plt.xlabel("Precision")
        plt.legend(title="Backend", loc='upper right')
        plt.axhline(y=1, color='red', linestyle='--', alpha=0.7, label='Baseline (1x)')
        
        # Backward Speedup
        plt.subplot(1, 2, 2)
        sns.barplot(
            data=speedup_df,
            x="Precision",
            y="Backward Speedup vs FP64",
            hue="Backend",
            palette=BACKEND_COLORS,
            order=[p for p in PRECISION_ORDER if p != "fp64"]
        )
        plt.title("Backward Speedup vs FP64 Baseline", fontweight='bold')
        plt.ylabel("Speedup (x)")
        plt.xlabel("Precision")
        plt.legend(title="Backend", loc='upper right')
        plt.axhline(y=1, color='red', linestyle='--', alpha=0.7, label='Baseline (1x)')
        
        plt.tight_layout()
        self.save_plot("02_speedup_analysis.png")
    
    def plot_memory_usage(self):
        """Plot 3: Memory Usage Analysis"""
        print("\n--- Creating Memory Usage Plots ---")
        
        plt.figure(figsize=(14, 6))
        
        # Peak GPU Memory
        plt.subplot(1, 2, 1)
        sns.barplot(
            data=self.df,
            x="Precision",
            y="Peak GPU Memory (MB)",
            hue="Backend",
            palette=BACKEND_COLORS,
            order=PRECISION_ORDER
        )
        plt.title("Peak GPU Memory Usage by Precision and Backend", fontweight='bold')
        plt.ylabel("Memory (MB)")
        plt.xlabel("Precision")
        plt.legend(title="Backend", loc='upper right')
        
        # Memory ratio (cuEq vs e3nn)
        plt.subplot(1, 2, 2)
        memory_ratio_data = []
        
        for precision in PRECISION_ORDER:
            for interaction in self.df["Interaction Block"].unique():
                e3nn_mem = self.df[
                    (self.df["Backend"] == "e3nn") & 
                    (self.df["Precision"] == precision) & 
                    (self.df["Interaction Block"] == interaction)
                ]["Peak GPU Memory (MB)"]
                
                cueq_mem = self.df[
                    (self.df["Backend"] == "cueq") & 
                    (self.df["Precision"] == precision) & 
                    (self.df["Interaction Block"] == interaction)
                ]["Peak GPU Memory (MB)"]
                
                if len(e3nn_mem) > 0 and len(cueq_mem) > 0:
                    ratio = cueq_mem.iloc[0] / e3nn_mem.iloc[0]
                    memory_ratio_data.append({
                        "Precision": precision,
                        "Interaction Block": interaction,
                        "Memory Ratio (cuEq/e3nn)": ratio
                    })
        
        if memory_ratio_data:
            ratio_df = pd.DataFrame(memory_ratio_data)
            sns.barplot(
                data=ratio_df,
                x="Precision",
                y="Memory Ratio (cuEq/e3nn)",
                hue="Interaction Block",
                order=PRECISION_ORDER
            )
            plt.title("Memory Usage Ratio: cuEq vs e3nn", fontweight='bold')
            plt.ylabel("Ratio (cuEq/e3nn)")
            plt.xlabel("Precision")
            plt.legend(title="Interaction Block", loc='upper right')
            plt.axhline(y=1, color='red', linestyle='--', alpha=0.7, label='Equal Memory (1.0)')
        
        plt.tight_layout()
        self.save_plot("03_memory_usage.png")
    
    def plot_accuracy_analysis(self):
        """Plot 4: Accuracy Analysis (Error Metrics)"""
        print("\n--- Creating Accuracy Analysis Plots ---")
        
        # Filter data with error metrics
        error_df = self.df[
            (self.df["Max Abs Error Forward"].notna()) & 
            (self.df["Max Abs Error Forward"] != "")
        ].copy()
        
        if len(error_df) == 0:
            print("  ⚠ No accuracy data available")
            return
        
        # Convert error columns to numeric
        error_columns = [
            "Max Abs Error Forward", "Max Rel Error Forward",
            "Max Abs Error Backward", "Max Rel Error Backward"
        ]
        
        for col in error_columns:
            if col in error_df.columns:
                error_df[col] = pd.to_numeric(error_df[col], errors='coerce')
        
        plt.figure(figsize=(16, 12))
        
        # Forward Absolute Error
        plt.subplot(2, 2, 1)
        sns.barplot(
            data=error_df,
            x="Precision",
            y="Max Abs Error Forward",
            hue="Backend",
            palette=BACKEND_COLORS,
            order=[p for p in PRECISION_ORDER if p in error_df["Precision"].unique()]
        )
        plt.title("Forward Pass: Maximum Absolute Error", fontweight='bold')
        plt.ylabel("Max Absolute Error")
        plt.xlabel("Precision")
        plt.legend(title="Backend", loc='upper right')
        plt.yscale('log')
        
        # Forward Relative Error
        plt.subplot(2, 2, 2)
        sns.barplot(
            data=error_df,
            x="Precision",
            y="Max Rel Error Forward",
            hue="Backend",
            palette=BACKEND_COLORS,
            order=[p for p in PRECISION_ORDER if p in error_df["Precision"].unique()]
        )
        plt.title("Forward Pass: Maximum Relative Error", fontweight='bold')
        plt.ylabel("Max Relative Error")
        plt.xlabel("Precision")
        plt.legend(title="Backend", loc='upper right')
        plt.yscale('log')
        
        # Backward Absolute Error
        plt.subplot(2, 2, 3)
        sns.barplot(
            data=error_df,
            x="Precision",
            y="Max Abs Error Backward",
            hue="Backend",
            palette=BACKEND_COLORS,
            order=[p for p in PRECISION_ORDER if p in error_df["Precision"].unique()]
        )
        plt.title("Backward Pass: Maximum Absolute Error", fontweight='bold')
        plt.ylabel("Max Absolute Error")
        plt.xlabel("Precision")
        plt.legend(title="Backend", loc='upper right')
        plt.yscale('log')
        
        # Backward Relative Error
        plt.subplot(2, 2, 4)
        sns.barplot(
            data=error_df,
            x="Precision",
            y="Max Rel Error Backward",
            hue="Backend",
            palette=BACKEND_COLORS,
            order=[p for p in PRECISION_ORDER if p in error_df["Precision"].unique()]
        )
        plt.title("Backward Pass: Maximum Relative Error", fontweight='bold')
        plt.ylabel("Max Relative Error")
        plt.xlabel("Precision")
        plt.legend(title="Backend", loc='upper right')
        plt.yscale('log')
        
        plt.tight_layout()
        self.save_plot("04_accuracy_analysis.png")
    
    def plot_interaction_block_comparison(self):
        """Plot 5: Comparison between Interaction Blocks"""
        print("\n--- Creating Interaction Block Comparison Plots ---")
        
        plt.figure(figsize=(16, 12))
        
        # Forward Latency by Interaction Block
        plt.subplot(2, 2, 1)
        sns.barplot(
            data=self.df,
            x="Precision",
            y="Forward Latency (ms)",
            hue="Interaction Block",
            palette="Set2",
            order=PRECISION_ORDER
        )
        plt.title("Forward Latency by Precision and Interaction Block", fontweight='bold')
        plt.ylabel("Latency (ms)")
        plt.xlabel("Precision")
        plt.legend(title="Interaction Block", loc='upper right')
        
        # Backward Latency by Interaction Block
        plt.subplot(2, 2, 2)
        sns.barplot(
            data=self.df,
            x="Precision",
            y="Backward Latency (ms)",
            hue="Interaction Block",
            palette="Set2",
            order=PRECISION_ORDER
        )
        plt.title("Backward Latency by Precision and Interaction Block", fontweight='bold')
        plt.ylabel("Latency (ms)")
        plt.xlabel("Precision")
        plt.legend(title="Interaction Block", loc='upper right')
        
        # Memory Usage by Interaction Block
        plt.subplot(2, 2, 3)
        sns.barplot(
            data=self.df,
            x="Precision",
            y="Peak GPU Memory (MB)",
            hue="Interaction Block",
            palette="Set2",
            order=PRECISION_ORDER
        )
        plt.title("Memory Usage by Precision and Interaction Block", fontweight='bold')
        plt.ylabel("Memory (MB)")
        plt.xlabel("Precision")
        plt.legend(title="Interaction Block", loc='upper right')
        
        # Backend comparison within each interaction block
        plt.subplot(2, 2, 4)
        interaction_data = []
        for interaction in self.df["Interaction Block"].unique():
            for backend in self.df["Backend"].unique():
                subset = self.df[
                    (self.df["Interaction Block"] == interaction) & 
                    (self.df["Backend"] == backend)
                ]
                if len(subset) > 0:
                    avg_fwd = subset["Forward Latency (ms)"].mean()
                    interaction_data.append({
                        "Interaction Block": f"Block {interaction}",
                        "Backend": backend,
                        "Avg Forward Latency (ms)": avg_fwd
                    })
        
        if interaction_data:
            interaction_df = pd.DataFrame(interaction_data)
            sns.barplot(
                data=interaction_df,
                x="Interaction Block",
                y="Avg Forward Latency (ms)",
                hue="Backend",
                palette=BACKEND_COLORS
            )
            plt.title("Average Forward Latency by Interaction Block and Backend", fontweight='bold')
            plt.ylabel("Average Latency (ms)")
            plt.xlabel("Interaction Block")
            plt.legend(title="Backend", loc='upper right')
        
        plt.tight_layout()
        self.save_plot("05_interaction_block_comparison.png")
    
    def plot_performance_heatmap(self):
        """Plot 6: Performance Heatmaps"""
        print("\n--- Creating Performance Heatmaps ---")
        
        plt.figure(figsize=(16, 8))
        
        # Forward Latency Heatmap
        plt.subplot(1, 2, 1)
        heatmap_data = self.df.pivot_table(
            values="Forward Latency (ms)",
            index="Precision",
            columns="Backend",
            aggfunc="mean"
        )
        
        # Reorder precision rows
        heatmap_data = heatmap_data.reindex(PRECISION_ORDER)
        
        sns.heatmap(
            heatmap_data,
            annot=True,
            fmt=".2f",
            cmap="YlOrRd",
            cbar_kws={"label": "Forward Latency (ms)"}
        )
        plt.title("Forward Latency Heatmap (Backend × Precision)", fontweight='bold')
        
        # Backward Latency Heatmap
        plt.subplot(1, 2, 2)
        heatmap_data_bwd = self.df.pivot_table(
            values="Backward Latency (ms)",
            index="Precision",
            columns="Backend",
            aggfunc="mean"
        )
        
        # Reorder precision rows
        heatmap_data_bwd = heatmap_data_bwd.reindex(PRECISION_ORDER)
        
        sns.heatmap(
            heatmap_data_bwd,
            annot=True,
            fmt=".2f",
            cmap="YlOrRd",
            cbar_kws={"label": "Backward Latency (ms)"}
        )
        plt.title("Backward Latency Heatmap (Backend × Precision)", fontweight='bold')
        
        plt.tight_layout()
        self.save_plot("06_performance_heatmap.png")
    
    def plot_statistical_summary(self):
        """Plot 7: Statistical Summary and Distribution Analysis"""
        print("\n--- Creating Statistical Summary Plots ---")
        
        plt.figure(figsize=(16, 12))
        
        # Latency distribution by backend
        plt.subplot(2, 2, 1)
        for backend in self.df["Backend"].unique():
            subset = self.df[self.df["Backend"] == backend]
            plt.hist(subset["Forward Latency (ms)"], alpha=0.7, label=backend, bins=15)
        plt.title("Forward Latency Distribution by Backend", fontweight='bold')
        plt.xlabel("Forward Latency (ms)")
        plt.ylabel("Frequency")
        plt.legend()
        
        # Latency distribution by precision
        plt.subplot(2, 2, 2)
        for precision in PRECISION_ORDER:
            if precision in self.df["Precision"].unique():
                subset = self.df[self.df["Precision"] == precision]
                plt.hist(subset["Forward Latency (ms)"], alpha=0.7, label=precision, bins=15)
        plt.title("Forward Latency Distribution by Precision", fontweight='bold')
        plt.xlabel("Forward Latency (ms)")
        plt.ylabel("Frequency")
        plt.legend()
        
        # Box plot: Latency by backend and precision
        plt.subplot(2, 2, 3)
        sns.boxplot(
            data=self.df,
            x="Precision",
            y="Forward Latency (ms)",
            hue="Backend",
            palette=BACKEND_COLORS,
            order=PRECISION_ORDER
        )
        plt.title("Forward Latency Distribution by Precision and Backend", fontweight='bold')
        plt.ylabel("Forward Latency (ms)")
        plt.xlabel("Precision")
        plt.legend(title="Backend", loc='upper right')
        
        # Memory vs Latency scatter plot
        plt.subplot(2, 2, 4)
        for backend in self.df["Backend"].unique():
            subset = self.df[self.df["Backend"] == backend]
            plt.scatter(
                subset["Peak GPU Memory (MB)"],
                subset["Forward Latency (ms)"],
                alpha=0.7,
                label=backend,
                s=50
            )
        plt.title("Memory Usage vs Forward Latency", fontweight='bold')
        plt.xlabel("Peak GPU Memory (MB)")
        plt.ylabel("Forward Latency (ms)")
        plt.legend()
        
        plt.tight_layout()
        self.save_plot("07_statistical_summary.png")
    
    def plot_backend_specific_analysis(self):
        """Plot 8: Backend-specific Performance Analysis"""
        print("\n--- Creating Backend-specific Analysis Plots ---")
        
        plt.figure(figsize=(16, 12))
        
        # cuEq vs e3nn performance ratio
        plt.subplot(2, 2, 1)
        performance_ratio_data = []
        
        for precision in PRECISION_ORDER:
            for interaction in self.df["Interaction Block"].unique():
                e3nn_fwd = self.df[
                    (self.df["Backend"] == "e3nn") & 
                    (self.df["Precision"] == precision) & 
                    (self.df["Interaction Block"] == interaction)
                ]["Forward Latency (ms)"]
                
                cueq_fwd = self.df[
                    (self.df["Backend"] == "cueq") & 
                    (self.df["Precision"] == precision) & 
                    (self.df["Interaction Block"] == interaction)
                ]["Forward Latency (ms)"]
                
                if len(e3nn_fwd) > 0 and len(cueq_fwd) > 0:
                    ratio = e3nn_fwd.iloc[0] / cueq_fwd.iloc[0]
                    performance_ratio_data.append({
                        "Precision": precision,
                        "Interaction Block": interaction,
                        "Performance Ratio (e3nn/cuEq)": ratio
                    })
        
        if performance_ratio_data:
            ratio_df = pd.DataFrame(performance_ratio_data)
            sns.barplot(
                data=ratio_df,
                x="Precision",
                y="Performance Ratio (e3nn/cuEq)",
                hue="Interaction Block",
                order=[p for p in PRECISION_ORDER if p in ratio_df["Precision"].unique()]
            )
            plt.title("Performance Ratio: e3nn/cuEq (Higher = e3nn slower)", fontweight='bold')
            plt.ylabel("Performance Ratio (e3nn/cuEq)")
            plt.xlabel("Precision")
            plt.legend(title="Interaction Block", loc='upper right')
            plt.axhline(y=1, color='red', linestyle='--', alpha=0.7, label='Equal Performance (1.0)')
        
        # Memory efficiency comparison
        plt.subplot(2, 2, 2)
        memory_efficiency_data = []
        
        for precision in PRECISION_ORDER:
            for interaction in self.df["Interaction Block"].unique():
                e3nn_mem = self.df[
                    (self.df["Backend"] == "e3nn") & 
                    (self.df["Precision"] == precision) & 
                    (self.df["Interaction Block"] == interaction)
                ]["Peak GPU Memory (MB)"]
                
                cueq_mem = self.df[
                    (self.df["Backend"] == "cueq") & 
                    (self.df["Precision"] == precision) & 
                    (self.df["Interaction Block"] == interaction)
                ]["Peak GPU Memory (MB)"]
                
                if len(e3nn_mem) > 0 and len(cueq_mem) > 0:
                    ratio = cueq_mem.iloc[0] / e3nn_mem.iloc[0]
                    memory_efficiency_data.append({
                        "Precision": precision,
                        "Interaction Block": interaction,
                        "Memory Ratio (cuEq/e3nn)": ratio
                    })
        
        if memory_efficiency_data:
            mem_df = pd.DataFrame(memory_efficiency_data)
            sns.barplot(
                data=mem_df,
                x="Precision",
                y="Memory Ratio (cuEq/e3nn)",
                hue="Interaction Block",
                order=[p for p in PRECISION_ORDER if p in mem_df["Precision"].unique()]
            )
            plt.title("Memory Efficiency: cuEq vs e3nn", fontweight='bold')
            plt.ylabel("Memory Ratio (cuEq/e3nn)")
            plt.xlabel("Precision")
            plt.legend(title="Interaction Block", loc='upper right')
            plt.axhline(y=1, color='red', linestyle='--', alpha=0.7, label='Equal Memory (1.0)')
        
        # Precision-specific performance analysis
        plt.subplot(2, 2, 3)
        precision_performance = self.df.groupby(["Precision", "Backend"])["Forward Latency (ms)"].mean().unstack()
        precision_performance = precision_performance.reindex(PRECISION_ORDER)
        
        precision_performance.plot(kind='bar', ax=plt.gca(), color=[BACKEND_COLORS['e3nn'], BACKEND_COLORS['cueq']])
        plt.title("Average Forward Latency by Precision and Backend", fontweight='bold')
        plt.ylabel("Average Forward Latency (ms)")
        plt.xlabel("Precision")
        plt.legend(title="Backend")
        plt.xticks(rotation=45)
        
        # Interaction block performance by backend
        plt.subplot(2, 2, 4)
        interaction_performance = self.df.groupby(["Interaction Block", "Backend"])["Forward Latency (ms)"].mean().unstack()
        
        interaction_performance.plot(kind='bar', ax=plt.gca(), color=[BACKEND_COLORS['e3nn'], BACKEND_COLORS['cueq']])
        plt.title("Average Forward Latency by Interaction Block and Backend", fontweight='bold')
        plt.ylabel("Average Forward Latency (ms)")
        plt.xlabel("Interaction Block")
        plt.legend(title="Backend")
        plt.xticks(rotation=0)
        
        plt.tight_layout()
        self.save_plot("08_backend_specific_analysis.png")
    
    def generate_summary_report(self):
        """Generate a comprehensive summary report"""
        print("\n--- Generating Summary Report ---")
        
        report_path = self.output_dir / "conv_tp_benchmark_summary.txt"
        
        with open(report_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("CONV_TP BENCHMARK RESULTS SUMMARY\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"Data Summary:\n")
            f.write(f"  Total records: {len(self.df)}\n")
            f.write(f"  Backends: {', '.join(self.df['Backend'].unique())}\n")
            f.write(f"  Precisions: {', '.join(self.df['Precision'].unique())}\n")
            f.write(f"  Interaction Blocks: {', '.join(map(str, self.df['Interaction Block'].unique()))}\n")
            f.write(f"  Batch Sizes: {', '.join(map(str, self.df['Batch Size'].unique()))}\n\n")
            
            # Performance summary
            f.write("Performance Summary:\n")
            f.write("-" * 40 + "\n")
            
            for backend in self.df['Backend'].unique():
                f.write(f"\n{backend.upper()} Backend:\n")
                backend_data = self.df[self.df['Backend'] == backend]
                
                for precision in PRECISION_ORDER:
                    if precision in backend_data['Precision'].unique():
                        prec_data = backend_data[backend_data['Precision'] == precision]
                        avg_fwd = prec_data['Forward Latency (ms)'].mean()
                        avg_bwd = prec_data['Backward Latency (ms)'].mean()
                        avg_mem = prec_data['Peak GPU Memory (MB)'].mean()
                        
                        f.write(f"  {precision.upper()}: Fwd={avg_fwd:.2f}ms, Bwd={avg_bwd:.2f}ms, Mem={avg_mem:.1f}MB\n")
            
            # Speedup analysis
            f.write("\nSpeedup Analysis (vs FP64):\n")
            f.write("-" * 40 + "\n")
            
            speedup_data = self.df[
                (self.df["Precision"] != "fp64") & 
                (self.df["Forward Speedup vs FP64"].notna())
            ]
            
            if len(speedup_data) > 0:
                for backend in speedup_data['Backend'].unique():
                    f.write(f"\n{backend.upper()}:\n")
                    backend_speedup = speedup_data[speedup_data['Backend'] == backend]
                    
                    for precision in backend_speedup['Precision'].unique():
                        prec_speedup = backend_speedup[backend_speedup['Precision'] == precision]
                        avg_fwd_speedup = prec_speedup['Forward Speedup vs FP64'].mean()
                        avg_bwd_speedup = prec_speedup['Backward Speedup vs FP64'].mean()
                        
                        f.write(f"  {precision.upper()}: Fwd={avg_fwd_speedup:.2f}x, Bwd={avg_bwd_speedup:.2f}x\n")
            
            f.write("\n" + "=" * 80 + "\n")
            f.write("Report generated automatically by conv_tp_plots.py\n")
            f.write("=" * 80 + "\n")
        
        print(f"  ✓ Summary report saved: {report_path}")
    
    def run_all_plots(self):
        """Run all plotting functions"""
        print("🚀 Starting comprehensive conv_tp benchmark visualization...")
        print(f"📊 Input data: {self.csv_path}")
        print(f"📁 Output directory: {self.output_dir}")
        
        if self.df is None:
            print("❌ Cannot proceed without data")
            return
        
        # Run all plotting functions
        self.plot_latency_comparison()
        self.plot_speedup_analysis()
        self.plot_memory_usage()
        self.plot_accuracy_analysis()
        self.plot_interaction_block_comparison()
        self.plot_performance_heatmap()
        self.plot_statistical_summary()
        self.plot_backend_specific_analysis()
        
        # Generate summary report
        self.generate_summary_report()
        
        print(f"\n🎉 All plots generated successfully!")
        print(f"📁 Check the output directory: {self.output_dir}")


def main():
    """Main function to run the plotting script"""
    # Define paths
    csv_path = "Experiments/numerical_stability/src/tests_blocks/conv_tp/results/conv_tp_benchmark_final_results.csv"
    output_dir = "Experiments/numerical_stability/src/tests_blocks/conv_tp/results/images"
    
    # Check if CSV file exists
    if not os.path.exists(csv_path):
        print(f"❌ CSV file not found: {csv_path}")
        print("Please run the benchmark first to generate results.")
        return
    
    # Create plotter and run all plots
    plotter = ConvTPPlotter(csv_path, output_dir)
    plotter.run_all_plots()


if __name__ == "__main__":
    main()
