"""
plot_network_graph_2d.py

2D-grid counterpart of plot_network_graph.py, for test_2d_grid.cpp. Compares the network's
connectivity before STDP (homogeneous all-to-all, random weights) against after STDP (once
long-running, multi-source Gaussian-pulse stimulation has reshaped the weights).

The key difference from the 1D version: nodes are laid out at their actual (x, y) grid
positions (from positions.csv), so edges are plotted as straight lines directly in physical
space rather than needing arcs to avoid overlapping along a single axis. There's also no
natural "forward vs. backward" direction on a 2D grid (unlike the 1D chain's j>i / j<i), so
edges are colored/shaded by weight instead.

  Panel 1 - Initial network graph: every directed connection drawn with opacity/width
            proportional to its (essentially random) initial weight.
  Panel 2 - Final network graph: only connections above `--edge_threshold` are drawn.
  Panel 3 - Weight distribution: initial vs. final histograms overlaid.
  Panel 4 - Weight change vs. Euclidean distance: mean (final - initial) vs. |i - j|.

Expects these files (produced by test_2d_grid.cpp) in the given output directory:
    positions.csv          neuron,row,col,x,y
    weights_initial.csv    iNeuron,jNeuron,weight
    weights_summary.csv    iNeuron,jNeuron,weight
    stimulation.csv        time,neuron,pulseCenter,offset   (for marking stimulated neurons)
"""

import argparse
import os

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt


def load_matrix(path, n_neurons):
    df = pd.read_csv(path)
    mat = df.pivot(index="iNeuron", columns="jNeuron", values="weight")
    mat = mat.reindex(index=range(n_neurons), columns=range(n_neurons))
    return mat.values


def build_graph(w, n_neurons, threshold=0.0):
    G = nx.DiGraph()
    G.add_nodes_from(range(n_neurons))
    for i in range(n_neurons):
        for j in range(n_neurons):
            if i == j:
                continue
            wij = w[i, j]
            if np.isnan(wij) or wij <= threshold:
                continue
            G.add_edge(i, j, weight=wij)
    return G


def draw_network(ax, G, pos, target_neurons, title, max_alpha=0.85, min_alpha=0.05,
                  edge_color="#3B75AF", width_scale=2.5):
    edges = list(G.edges())
    if edges:
        weights = np.array([G[u][v]["weight"] for u, v in edges])
        alphas = min_alpha + (max_alpha - min_alpha) * weights
        widths = 0.4 + width_scale * weights
        for (u, v), a, w_ in zip(edges, alphas, widths):
            nx.draw_networkx_edges(
                G, pos, edgelist=[(u, v)], ax=ax,
                edge_color=edge_color, alpha=float(a), width=float(w_),
                arrowsize=6, node_size=220, connectionstyle="arc3,rad=0.08",
            )

    node_colors = ["#C44E52" if i in target_neurons else "#4C4C4C" for i in G.nodes()]
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=220, node_color=node_colors, edgecolors="black", linewidths=0.5)
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=6, font_color="white")

    ax.set_title(title)
    ax.set_aspect("equal")
    ax.set_xlabel("Grid x position")
    ax.set_ylabel("Grid y position")
    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    pad = 0.8
    ax.set_xlim(min(xs) - pad, max(xs) + pad)
    ax.set_ylim(min(ys) - pad, max(ys) + pad)


def plot_weight_distribution(ax, w_init, w_final):
    init_vals = w_init[~np.eye(w_init.shape[0], dtype=bool)]
    final_vals = w_final[~np.eye(w_final.shape[0], dtype=bool)]

    bins = np.linspace(0, 1, 31)
    ax.hist(init_vals, bins=bins, alpha=0.55, color="#8172B2", label="Initial")
    ax.hist(final_vals, bins=bins, alpha=0.55, color="#C44E52", label="Final (post-STDP)")
    ax.set_xlabel("Weight")
    ax.set_ylabel("Count (directed edges)")
    ax.set_title("Weight distribution: initial vs. final")
    ax.legend(fontsize=8)


def plot_weight_change_vs_distance(ax, w_init, w_final, dist_matrix, n_neurons):
    rows = []
    for i in range(n_neurons):
        for j in range(n_neurons):
            if i == j:
                continue
            rows.append((round(dist_matrix[i, j], 2), w_final[i, j] - w_init[i, j]))
    df = pd.DataFrame(rows, columns=["distance", "delta"])
    stats = df.groupby("distance")["delta"].agg(["mean", "std"])

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.errorbar(stats.index, stats["mean"], yerr=stats["std"], fmt="o-", ms=4, capsize=3, color="#55A868")
    ax.set_xlabel("Euclidean distance |i - j|")
    ax.set_ylabel("Mean weight change\n(final - initial)")
    ax.set_title("STDP-induced weight change vs. distance")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", default="Codigo/code_grid/output_2D", help="Directory with simulation CSV outputs")
    parser.add_argument("--save", default="Codigo/code_grid/output_2D/network_graph_comparison_2d.png", help="Path to save the figure")
    parser.add_argument("--n_neurons", type=int, default=20, help="Number of neurons in the grid")
    parser.add_argument("--edge_threshold", type=float, default=0.5,
                         help="Only draw final-network edges with weight above this value")
    args = parser.parse_args()

    positions = pd.read_csv(os.path.join(args.outdir, "positions.csv")).sort_values("neuron")
    pos = {int(row.neuron): (row.x, row.y) for row in positions.itertuples()}

    xy = positions[["x", "y"]].values
    diff = xy[:, None, :] - xy[None, :, :]
    dist_matrix = np.sqrt((diff ** 2).sum(axis=-1))

    w_init = load_matrix(os.path.join(args.outdir, "weights_initial.csv"), args.n_neurons)
    w_final = load_matrix(os.path.join(args.outdir, "weights_summary.csv"), args.n_neurons)

    stim_path = os.path.join(args.outdir, "stimulation.csv")
    if os.path.exists(stim_path):
        target_neurons = set(pd.read_csv(stim_path)["neuron"].unique().tolist())
    else:
        target_neurons = set()

    G_init = build_graph(w_init, args.n_neurons, threshold=0.0)
    G_final = build_graph(w_final, args.n_neurons, threshold=args.edge_threshold)

    fig = plt.figure(figsize=(12, 14), layout="constrained")
    gs = fig.add_gridspec(3, 2, height_ratios=[1.3, 1.3, 1])

    ax_init = fig.add_subplot(gs[0, 0])
    ax_final = fig.add_subplot(gs[0, 1])
    ax_dist = fig.add_subplot(gs[1, 0])
    ax_delta = fig.add_subplot(gs[1, 1])

    draw_network(ax_init, G_init, pos, target_neurons,
                 title=f"Initial connectivity\n(all-to-all, N edges={G_init.number_of_edges()})")
    draw_network(ax_final, G_final, pos, target_neurons,
                 title=f"Final connectivity after STDP\n(weight > {args.edge_threshold}, "
                       f"N edges={G_final.number_of_edges()} / {args.n_neurons*(args.n_neurons-1)})")

    plot_weight_distribution(ax_dist, w_init, w_final)
    plot_weight_change_vs_distance(ax_delta, w_init, w_final, dist_matrix, args.n_neurons)

    handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#C44E52", markeredgecolor="black", markersize=8, label="Stimulated neuron"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#4C4C4C", markeredgecolor="black", markersize=8, label="Non-stimulated neuron"),
        plt.Line2D([0], [0], color="#3B75AF", lw=2, label="Directed edge (opacity/width = weight)"),
    ]
    fig.legend(handles=handles, loc="upper right", fontsize=8, ncol=1)

    fig.suptitle("2D Grid Neuron Array: connectivity before vs. after STDP", fontsize=13)
    fig.savefig(args.save, dpi=150)
    print(f"Saved figure to {args.save}")
    print(f"Initial edges: {G_init.number_of_edges()}, Final edges above threshold {args.edge_threshold}: {G_final.number_of_edges()}")


if __name__ == "__main__":
    main()