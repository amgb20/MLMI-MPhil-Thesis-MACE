import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from datetime import datetime
from pathlib import Path

def load_and_aggregate(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    # Normalize
    df["_backend_norm"]   = df["Backend"].astype(str).str.lower().str.strip()
    df["_precision_norm"] = df["Precision"].astype(str).str.lower().str.strip()
    df["_block_idx"]      = df["Interaction Block"].astype(str).str.strip()

    # Map block indices
    block_label_map = {"0": "conv_tp[0]", "1": "conv_tp[1]"}
    df["_block_label"] = df["_block_idx"].map(block_label_map).fillna("conv_tp[" + df["_block_idx"] + "]")

    # Build agg dict based on available cols
    agg_spec = {"Forward Latency (ms)": "mean"}
    if "Backward Latency (ms)" in df.columns:
        agg_spec["Backward Latency (ms)"] = "mean"
    if "Forward Std Dev (ms)" in df.columns:
        agg_spec["Forward Std Dev (ms)"] = "mean"
    if "Backward Std Dev (ms)" in df.columns:
        agg_spec["Backward Std Dev (ms)"] = "mean"

    agg = (df.groupby(["_block_label", "_backend_norm", "_precision_norm"], as_index=False)
             .agg(agg_spec)
             .rename(columns={
                 "Forward Latency (ms)": "forward_latency_ms",
                 "Backward Latency (ms)": "backward_latency_ms",
                 "Forward Std Dev (ms)": "forward_std_ms",
                 "Backward Std Dev (ms)": "backward_std_ms",
             }))
    return agg

def make_yerr(values, stds, log_scale: bool):
    """Return error bars suitable for linear/log axes."""
    y = np.array(values, dtype=float)
    s = np.array(stds, dtype=float) if stds is not None else np.zeros_like(y)

    # Replace NaNs/infs in std with 0
    s[~np.isfinite(s)] = 0.0

    if not log_scale:
        return s

    # On log scale, ensure lower bound stays > 0: clip std to < y
    y_safe = np.where(np.isfinite(y), y, 1.0)
    # clip to at most 0.99*y (and >= tiny epsilon)
    s = np.minimum(s, np.maximum(1e-12, 0.99 * y_safe))
    return s

def plot_backend(sub: pd.DataFrame, backend: str, blocks, precisions, color_map, out_dir: Path, log_scale: bool):
    fig, ax = plt.subplots(figsize=(9, 4.5))
    bar_width = 0.20
    x = np.arange(len(blocks), dtype=float)
    offsets = np.linspace(-bar_width*(len(precisions)-1)/2.0,
                          bar_width*(len(precisions)-1)/2.0,
                          len(precisions))

    # Determine y-range base
    vals = []
    if "forward_latency_ms" in sub:
        vals += [v for v in sub["forward_latency_ms"].values if np.isfinite(v)]
    if "backward_latency_ms" in sub:
        vals += [v for v in sub["backward_latency_ms"].values if np.isfinite(v)]
    max_val = max(vals) if vals else 1.0

    # Build fp64 baselines per block for speedup annotations
    base_fwd = {}
    base_bwd = {}
    for b in blocks:
        r64 = sub[(sub["_block_label"] == b) & (sub["_precision_norm"] == "fp64")]
        base_fwd[b] = float(r64["forward_latency_ms"].iloc[0]) if len(r64) else np.nan
        if "backward_latency_ms" in r64.columns and len(r64) and np.isfinite(r64["backward_latency_ms"].iloc[0]):
            base_bwd[b] = float(r64["backward_latency_ms"].iloc[0])
        else:
            base_bwd[b] = np.nan

    for i, prec in enumerate(precisions):
        rows_f, rows_b, std_f, std_b = [], [], [], []
        for b in blocks:
            row = sub[(sub["_block_label"] == b) & (sub["_precision_norm"] == prec)]
            if len(row):
                rows_f.append(float(row["forward_latency_ms"].iloc[0]))
                std_f.append(float(row.get("forward_std_ms", np.nan)))
                if "backward_latency_ms" in row.columns:
                    rows_b.append(float(row["backward_latency_ms"].iloc[0]) if np.isfinite(row["backward_latency_ms"].iloc[0]) else np.nan)
                    std_b.append(float(row.get("backward_std_ms", np.nan)))
                else:
                    rows_b.append(np.nan); std_b.append(np.nan)
            else:
                rows_f.append(np.nan); std_f.append(np.nan)
                rows_b.append(np.nan); std_b.append(np.nan)

        # Prepare error bars
        yerr_f = make_yerr(rows_f, std_f, log_scale)
        yerr_b = make_yerr(rows_b, std_b, log_scale)

        # Forward bars
        ax.bar(x + offsets[i], rows_f, width=bar_width,
               color=color_map.get(prec, "gray"),
               edgecolor="black", linewidth=0.6, label=prec.upper(),
               yerr=yerr_f, capsize=3, error_kw=dict(linewidth=1))

        # Backward overlays (hatched)
        if np.isfinite(np.array(rows_b, dtype=float)).any():
            ax.bar(x + offsets[i], rows_b, width=bar_width,
                   facecolor='none', edgecolor=color_map.get(prec, "gray"),
                   hatch='///', linewidth=1.2,
                   yerr=yerr_b, capsize=3, error_kw=dict(linewidth=1))

        # Annotate speedups vs fp64 baseline above each bar
        for j, b in enumerate(blocks):
            xf = x[j] + offsets[i]
            yf = rows_f[j]
            yb = rows_b[j]
            bf = base_fwd.get(b, np.nan)
            bb = base_bwd.get(b, np.nan)
            # Forward speedup
            if np.isfinite(bf) and np.isfinite(yf) and yf > 0:
                sf = bf / yf
                ytext = yf * (1.10 if log_scale else 1.0) + (0.0 if log_scale else max_val * 0.02)
                ax.text(xf, ytext, f"F×{sf:.2f}", ha='center', va='bottom', fontsize=8)
            # Backward speedup
            if np.isfinite(bb) and np.isfinite(yb) and yb > 0:
                sb = bb / yb
                ytext_b = yb * (1.20 if log_scale else 1.0) + (0.0 if log_scale else max_val * 0.035)
                ax.text(xf, ytext_b, f"B×{sb:.2f}", ha='center', va='bottom', fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(blocks)
    ax.set_xlabel("conv_tp")
    ax.set_ylabel("Latency (ms)" + (" (log)" if log_scale else ""))

    if log_scale:
        # Pick a positive lower bound for log scale
        positives = [v for v in vals if v > 0]
        min_pos = min(positives) if positives else 1e-3
        ax.set_yscale("log")
        ax.set_ylim(min_pos * 0.8, max_val * 1.2)
    else:
        ax.set_ylim(0, max_val * 1.2)

    # Legends
    color_handles = [Patch(facecolor=color_map.get(p, "gray"), edgecolor="black", label=p.upper()) for p in precisions]
    leg1 = ax.legend(handles=color_handles, title="Precision", loc="upper left")
    ax.add_artist(leg1)
    style_handles = [
        Patch(facecolor='black', edgecolor='black', label='Forward'),
        Patch(facecolor='white', edgecolor='black', hatch='///', label='Backward')
    ]
    ax.legend(handles=style_handles, title="Style", loc="upper right")

    ax.set_title(f"{backend.upper()} - Interaction Block Convolution Latency")
    plt.tight_layout()

    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"conv_tp_latency_{backend}_{'log' if log_scale else 'linear'}_{stamp}.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return str(out_path)

def plot_backend_errors(df: pd.DataFrame,
                        backend: str,
                        blocks,
                        precisions,
                        color_map,
                        out_dir: str,
                        log_scale: bool = True,
                        eps: float = 1e-15):
    """
    Create two plots for a given backend:
      1) Max Absolute Error (Forward solid, Backward hatched)
      2) Max Relative Error (Forward solid, Backward hatched)

    Parameters mirror your latency plotter:
      - df: raw dataframe (not pre-aggregated)
      - backend: "e3nn" or "cueq"
      - blocks: e.g., ["conv_tp[0]", "conv_tp[1]"]
      - precisions: list like ["fp64","fp32","fp16","bf16"] filtered to what's available
      - color_map: dict like {"fp64":"C0","fp32":"C1","fp16":"C2","bf16":"C3"}
      - out_dir: directory to save PNGs
      - log_scale: bool to toggle y-axis scaling
      - eps: small positive to visualize zeros on log scale
    Returns: list of saved file paths
    """

    # Ensure helper columns exist (OK if already added upstream)
    tmp = df.copy()
    if "_backend_norm" not in tmp:
        tmp["_backend_norm"] = tmp["Backend"].astype(str).str.lower().str.strip()
    if "_precision_norm" not in tmp:
        tmp["_precision_norm"] = tmp["Precision"].astype(str).str.lower().str.strip()
    if "_block_idx" not in tmp:
        tmp["_block_idx"] = tmp["Interaction Block"].astype(str).str.strip()
    if "_block_label" not in tmp:
        blk_map = {"0": "conv_tp[0]", "1": "conv_tp[1]"}
        tmp["_block_label"] = tmp["_block_idx"].map(blk_map).fillna(
            "conv_tp[" + tmp["_block_idx"] + "]"
        )

    # Coerce error columns to numeric, missing -> NaN
    for c in ["Max Abs Error Forward", "Max Abs Error Backward",
              "Max Rel Error Forward", "Max Rel Error Backward"]:
        if c not in tmp.columns:
            tmp[c] = np.nan
        tmp[c] = pd.to_numeric(tmp[c], errors="coerce")

    sub = tmp[tmp["_backend_norm"] == backend.lower()].copy()
    if sub.empty:
        return []

    # Aggregate (mean across repeated runs / batch sizes)
    agg = (sub.groupby(["_block_label", "_precision_norm"], as_index=False)
              .agg({
                  "Max Abs Error Forward": "mean",
                  "Max Abs Error Backward": "mean",
                  "Max Rel Error Forward": "mean",
                  "Max Rel Error Backward": "mean",
              })
              .rename(columns={
                  "Max Abs Error Forward": "max_abs_fwd",
                  "Max Abs Error Backward": "max_abs_bwd",
                  "Max Rel Error Forward": "max_rel_fwd",
                  "Max Rel Error Backward": "max_rel_bwd",
              }))

    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    saved = []

    def _plot_metric(fwd_col, bwd_col, title_suffix, y_label, fname_stub):
        fig, ax = plt.subplots(figsize=(9, 4.5))
        bar_width = 0.20
        x = np.arange(len(blocks), dtype=float)
        offsets = np.linspace(-bar_width*(len(precisions)-1)/2.0,
                              bar_width*(len(precisions)-1)/2.0,
                              len(precisions))

        # Collect all values to determine sensible y-limits
        pool_vals = []

        for i, prec in enumerate(precisions):
            rows_f, rows_b = [], []
            for b in blocks:
                r = agg[(agg["_block_label"] == b) & (agg["_precision_norm"] == prec)]
                fv = float(r[fwd_col].iloc[0]) if len(r) and pd.notna(r[fwd_col].iloc[0]) else np.nan
                bv = float(r[bwd_col].iloc[0]) if len(r) and pd.notna(r[bwd_col].iloc[0]) else np.nan
                rows_f.append(fv); rows_b.append(bv)

            pool_vals.extend([v for v in rows_f if np.isfinite(v)])
            pool_vals.extend([v for v in rows_b if np.isfinite(v)])

            # For log scale, replace zeros/negatives with eps for visibility
            plot_rows_f = np.array(rows_f, dtype=float)
            plot_rows_b = np.array(rows_b, dtype=float)
            if log_scale:
                plot_rows_f = np.where(np.isfinite(plot_rows_f) & (plot_rows_f > 0), plot_rows_f, np.nan)
                plot_rows_b = np.where(np.isfinite(plot_rows_b) & (plot_rows_b > 0), plot_rows_b, np.nan)
                plot_rows_f = np.where(np.isnan(plot_rows_f), np.nan, np.maximum(plot_rows_f, eps))
                plot_rows_b = np.where(np.isnan(plot_rows_b), np.nan, np.maximum(plot_rows_b, eps))

            # Forward (solid)
            ax.bar(x + offsets[i], plot_rows_f, width=bar_width,
                   color=color_map.get(prec, "gray"), edgecolor="black", linewidth=0.6,
                   label=prec.upper() if i == 0 else None)
            # Backward (hatched overlay)
            if np.isfinite(plot_rows_b).any():
                ax.bar(x + offsets[i], plot_rows_b, width=bar_width,
                       facecolor='none', edgecolor=color_map.get(prec, "gray"),
                       hatch='///', linewidth=1.2)

        ax.set_xticks(x); ax.set_xticklabels(blocks)
        ax.set_xlabel("conv_tp")
        ax.set_ylabel(y_label + (" (log)" if log_scale else ""))

        if log_scale:
            pos_vals = [v for v in pool_vals if np.isfinite(v) and v > 0]
            lo = (min(pos_vals) if pos_vals else eps) * 0.8
            hi = (max(pos_vals) if pos_vals else 1.0) * 1.2
            ax.set_yscale("log"); ax.set_ylim(lo, hi)
        else:
            hi = (max(pool_vals) if pool_vals else 1.0) * 1.2
            ax.set_ylim(0, hi)

        # Legends (precision + style)
        color_handles = [Patch(facecolor=color_map.get(p, "gray"), edgecolor="black", label=p.upper()) for p in precisions]
        leg1 = ax.legend(handles=color_handles, title="Precision", loc="upper left")
        ax.add_artist(leg1)
        style_handles = [Patch(facecolor='black', edgecolor='black', label='Forward'),
                         Patch(facecolor='white', edgecolor='black', hatch='///', label='Backward')]
        ax.legend(handles=style_handles, title="Style", loc="upper right")

        ax.set_title(f"{backend.upper()} - {title_suffix}")
        plt.tight_layout()
        out_path = out_dir / f"{fname_stub}_{backend}_{'log' if log_scale else 'linear'}_{stamp}.png"
        plt.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        saved.append(str(out_path))

    # ---- Make both figures ----
    _plot_metric("max_abs_fwd", "max_abs_bwd",
                 title_suffix="Max Absolute Error",
                 y_label="Max Abs Error",
                 fname_stub="conv_tp_max_abs_error")

    _plot_metric("max_rel_fwd", "max_rel_bwd",
                 title_suffix="Max Relative Error",
                 y_label="Max Rel Error",
                 fname_stub="conv_tp_max_rel_error")

    return saved

def main():
    p = argparse.ArgumentParser(description="Plot conv_tp latency per backend/precision with optional log scale.")
    p.add_argument("--csv", type=str, required=True,
                   help="Path to the benchmark CSV (e.g., conv_tp_benchmark_final_results_REAL.csv)")
    p.add_argument("--log-scale", action="store_true",
                   help="Use logarithmic y-axis. Omit for linear scale.")
    p.add_argument("--out-dir", type=str, default="Experiments/numerical_stability/src/tests_blocks/conv_tp/results/images",
                   help="Directory to save images.")
    args = p.parse_args()

    # Load both aggregated view and raw dataframe (for error plots)
    agg = load_and_aggregate(args.csv)
    raw_df = pd.read_csv(args.csv)

    # Colors
    color_map = {"fp64": "C0", "fp32": "C1", "fp16": "C2", "bf16": "C3"}

    # Blocks present (keep order conv_tp[0], conv_tp[1] if they exist)
    blocks_all = ["conv_tp[0]", "conv_tp[1]"]
    blocks = [b for b in blocks_all if b in agg["_block_label"].unique()]
    if not blocks:
        raise RuntimeError("No conv_tp[0] or conv_tp[1] entries found in the data.")

    out_dir = Path(args.out_dir)

    # Plot Cueq
    cueq_sub = agg[agg["_backend_norm"] == "cueq"].copy()
    if len(cueq_sub):
        precs = [p for p in ["fp64", "fp32"] if p in set(cueq_sub["_precision_norm"])]
        path = plot_backend(cueq_sub, "cueq", blocks, precs, color_map, out_dir, log_scale=args.log_scale)
        print(f"Saved: {path}")
        # Error plots (if error columns exist, function will handle gracefully otherwise)
        error_paths = plot_backend_errors(raw_df, backend="cueq", blocks=blocks, precisions=precs,
                                          color_map=color_map, out_dir=out_dir, log_scale=args.log_scale)
        for pth in error_paths:
            print(f"Saved: {pth}")
    else:
        print("No cueq rows found; skipping cueq plot.")

    # Plot E3NN
    e3nn_sub = agg[agg["_backend_norm"] == "e3nn"].copy()
    if len(e3nn_sub):
        precs = [p for p in ["fp64", "fp32", "fp16", "bf16"] if p in set(e3nn_sub["_precision_norm"])]
        path = plot_backend(e3nn_sub, "e3nn", blocks, precs, color_map, out_dir, log_scale=args.log_scale)
        print(f"Saved: {path}")
        error_paths = plot_backend_errors(raw_df, backend="e3nn", blocks=blocks, precisions=precs,
                                          color_map=color_map, out_dir=out_dir, log_scale=args.log_scale)
        for pth in error_paths:
            print(f"Saved: {pth}")
    else:
        print("No e3nn rows found; skipping e3nn plot.")

if __name__ == "__main__":
    main()
