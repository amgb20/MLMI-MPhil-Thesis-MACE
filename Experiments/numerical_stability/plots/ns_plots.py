import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from collections import defaultdict

def plot_precision_comparison(df_perf, save_path):
    # plot the time and memory usage for each precision using bar charts
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Time bar chart
    ax1.bar(df_perf['precision'], df_perf['time_ms'])
    ax1.set_xlabel('Precision')
    ax1.set_ylabel('Time (ms)')
    ax1.set_title('Time vs Precision')
    ax1.tick_params(axis='x', rotation=45)
    
    # Memory bar chart
    ax2.bar(df_perf['precision'], df_perf['memory_bytes'])
    ax2.set_xlabel('Precision')
    ax2.set_ylabel('Memory (bytes)')
    ax2.set_title('Memory vs Precision')
    ax2.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()

def plot_error_distributions(results, batch, z_table, ref_precision="fp64"):
    """
    For each interaction block:
      1) Compute per-atom mean absolute fwd/bwd error vs FP64: we take the abs diff between each precision's outputs (or gradients)
      and the fp64 ref, then avg over all feature dimensions. That gives one scalar "mean error" per atom.
      2) Group atoms by element type: We recover the atomic number of each atom via the one-hot batch.node_attrs, then split the error scalars into lists for H, C, O (or whatever elements you have).
      3) Boxplot the distributions for each precision and each element

    What do we get from this plot: 
    - Uniformity: If all element boxes are similarly narrow and centered near zero, errors are small and consistent.
    - Outliers: A taller whisker or point far from the box indicates an outlier atom (maybe at the cutoff boundary or unusual geometry).
    - Element-specific bias: If one element's box is systematically higher, it suggests your lower-precision scheme struggles more for that atom type.
    """
    # 1) recover element index for each atom
    node_attrs = batch.node_attrs.cpu().numpy()             # one-hot [n_nodes, n_elements]
    elem_idx = node_attrs.argmax(axis=1)                    # which column is “1”
    elements = z_table.zs

    for block_idx in [0,1]:
        ref_out  = results[block_idx][ref_precision]['output']
        ref_grad = results[block_idx][ref_precision]['grad']
        for kind in ("fwd","bwd"):
            plt.figure(figsize=(8,5))
            plt.title(f"Block {block_idx} {kind.upper()} error by element")
            all_box_data = []
            all_labels   = []
            for prec, vals in results[block_idx].items():
                if prec == ref_precision: 
                    continue
                arr = vals['output'] if kind=="fwd" else vals['grad']
                # mean abs error per atom (collapse all feature dims)
                err_per_atom = np.mean(np.abs(arr - (ref_out if kind=="fwd" else ref_grad)),
                                       axis=tuple(range(1, arr.ndim)))
                # collect one list per element
                for i, Z in enumerate(elements):
                    mask = (elem_idx == i)
                    all_box_data.append(err_per_atom[mask])
                    all_labels.append(str(Z))
            plt.boxplot(all_box_data, labels=all_labels, showfliers=False)
            plt.ylabel("Mean |error|")
            plt.xlabel("Element (atomic number)")
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()
            plt.savefig(f'Experiments/precision_3/figs/error_distributions_{kind}.png')


# -----------------------------------------------
def plot_error_distributions_2(results, batch, z_table, ref_precision="fp64", save_path: str = None):
    """
    Four box-plots in one figure:
        Block-0 forward  |  Block-0 backward
        Block-1 forward  |  Block-1 backward

    X-axis = element symbol, grouped; each coloured box inside a group = one lower-precision mode.
    Y-axis = mean absolute error per atom (log scale).
    """
    # Helper: element symbol from atomic number
    periodic = {1: "H", 6: "C", 7: "N", 8: "O", 9: "F",
                14: "Si", 16: "S", 17: "Cl", 35: "Br", 53: "I"}
    elements_Z = z_table.zs
    elements_sym = [periodic.get(Z, str(Z)) for Z in elements_Z]
    print(elements_sym)

    # Decode which element each atom belongs to
    elem_idx = batch.node_attrs.cpu().numpy().argmax(1)  # [n_atoms] -> column id 0..n_elem-1

    # ---- iterate over blocks ------------------------------------------------
    for block_idx in [0, 1]:
        ref_out  = results[block_idx][ref_precision]["output"]
        ref_grad = results[block_idx][ref_precision]["grad"]

        fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
        for j, kind in enumerate(("fwd", "bwd")):        # 0=fwd,1=bwd
            ax = axes[j]
            ax.set_title(f"Block {block_idx} – {kind.upper()} error")

            # Collect data: dict[(elem_sym, precision)] -> list[err]
            box_data = defaultdict(list)

            for prec, vals in results[block_idx].items():
                if prec == ref_precision:
                    continue

                arr = vals["output"] if kind == "fwd" else vals["grad"]
                ref = ref_out         if kind == "fwd" else ref_grad
                err_per_atom = np.mean(np.abs(arr - ref),
                                       axis=tuple(range(1, arr.ndim)))  # --> [n_atoms]

                for col, Z in enumerate(elements_Z):
                    mask = (elem_idx == col)
                    sym = elements_sym[col]
                    box_data[(sym, prec)].extend(err_per_atom[mask])

            # ----- turn into ordered lists (cluster-by-element) -------------
            # unique order of precisions (to get consistent colours)
            precs = [p for p in results[block_idx] if p != ref_precision]
            colours = plt.cm.tab10(np.linspace(0, 1, len(precs)))
            prec2color = dict(zip(precs, colours))

            positions = []
            data      = []
            labels    = []
            tick_pos  = []

            offset = 0
            width  = 0.8 / len(precs)   # cluster width ~= 0.8
            for e_idx, sym in enumerate(elements_sym):
                tick_pos.append(offset + 0.4)  # centre of the cluster
                for i, prec in enumerate(precs):
                    positions.append(offset + i * width)
                    data.append(box_data[(sym, prec)])
                    labels.append(f"{sym}\n{prec}")
                offset += 1  # start next cluster

            bp = ax.boxplot(data, positions=positions, widths=width,
                            patch_artist=True, showfliers=False)

            # colour the boxes
            for patch, lab in zip(bp['boxes'], labels):
                *sym, prec = lab.split('\n')  # last line is precision
                patch.set_facecolor(prec2color[prec])

            ax.set_xticks(tick_pos)
            ax.set_xticklabels(elements_sym)
            ax.set_ylabel("Mean |error|")
            ax.set_yscale("log")

            # build legend only once
            if j == 1:
                handles = [mpatches.Patch(color=prec2color[p], label=p)
                           for p in precs]
                ax.legend(handles=handles, title="precision", loc="upper left")

        plt.tight_layout()
        fig.savefig(save_path)
        plt.show()

