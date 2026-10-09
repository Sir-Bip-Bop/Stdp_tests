"""
plot_network_graph.py

Compares the network's connectivity BEFORE STDP (the homogeneous, randomly-weighted
all-to-all graph) against AFTER STDP (the structure that emerges once long-running,
multi-source Gaussian-pulse stimulation has had time to reshape the weights), for the
1D random-walk neuron array (test_1d_network.cpp).

  Panel 1 - Initial network graph: every directed connection (i -> j) drawn with opacity
            proportional to its (essentially random, ~uniform) initial weight.
  Panel 2 - Final network graph: only connections above `--edge_threshold` are drawn, so
            you can see which links actually survived/strengthened after STDP, versus the
            majority that decayed towards 0 and are omitted here.
  Panel 3 - Weight distribution: initial vs. final histograms overlaid, showing the shift
            away from the initial near-uniform spread towards a bimodal
            decayed-vs-potentiated split.
  Panel 4 - Weight change vs. distance: mean (final - initial) as a function of |i - j|,
            showing whether STDP is systematically strengthening short-range / feedforward
            connections and pruning long-range ones (or some other spatial pattern).

Nodes are laid out along a single horizontal axis matching their physical position in the
1D array (matching test_1d_network.cpp's `spacing`), so the plots read left-to-right the
same way the array is physically arranged.

Expects these files (produced by test_1d_network.cpp) in the given output directory:
    weights_initial.csv   iNeuron,jNeuron,weight   (snapshot essentially at t=0)
    weights_summary.csv   iNeuron,jNeuron,weight   (final matrix)
    stimulation.csv       time,neuron,pulseCenter,offset   (used only to mark which
                           neurons were externally stimulated, for context)
"""

import argparse
import os

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def load_matrix(path, n_neurons):
    df = pd.read_csv(path)
    mat = df.pivot(index="iNeuron", columns="jNeuron", values="weight")
    mat = mat.reindex(index=range(n_neurons), columns=range(n_neurons))
    return mat.values


def build_graph(w, n_neurons, threshold=0.0):
    """Directed graph from a weight matrix, keeping only edges with weight > threshold."""
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


def draw_network(ax, G, n_neurons, spacing, target_neurons, title, max_alpha=0.9, min_alpha=0.05,
                  color_forward="#3B75AF", color_backward="#999999", width_scale=2.5):
    pos = {i: (i * spacing, 0.0) for i in range(n_neurons)}

    forward_edges = [(u, v) for u, v in G.edges() if v > u]
    backward_edges = [(u, v) for u, v in G.edges() if v < u]

    for edges, color, rad in [(forward_edges, color_forward, 0.15), (backward_edges, color_backward, -0.15)]:
        if not edges:
            continue
        weights = np.array([G[u][v]["weight"] for u, v in edges])
        alphas = min_alpha + (max_alpha - min_alpha) * weights
        widths = 0.4 + width_scale * weights
        for (u, v), a, w_ in zip(edges, alphas, widths):
            nx.draw_networkx_edges(
                G, pos, edgelist=[(u, v)], ax=ax,
                edge_color=color, alpha=float(a), width=float(w_),
                connectionstyle=f"arc3,rad={rad}", arrowsize=6,
                node_size=180,
            )

    node_colors = ["#C44E52" if i in target_neurons else "#4C4C4C" for i in range(n_neurons)]
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=180, node_color=node_colors, edgecolors="black", linewidths=0.5)
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=6, font_color="white")

    ax.set_title(title)
    ax.set_xlabel("Neuron position along array")
    ax.set_yticks([])
    ax.set_xlim(-spacing, n_neurons * spacing)
    ax.set_ylim(-1, 1)
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)


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


def plot_weight_change_vs_distance(ax, w_init, w_final, n_neurons):
    rows = []
    for i in range(n_neurons):
        for j in range(n_neurons):
            if i == j:
                continue
            rows.append((abs(i - j), w_final[i, j] - w_init[i, j]))
    df = pd.DataFrame(rows, columns=["distance", "delta"])
    stats = df.groupby("distance")["delta"].agg(["mean", "std"])

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.errorbar(stats.index, stats["mean"], yerr=stats["std"], fmt="o-", ms=4, capsize=3, color="#55A868")
    ax.set_xlabel("Distance |i - j|")
    ax.set_ylabel("Mean weight change\n(final - initial)")
    ax.set_title("STDP-induced weight change vs. distance")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", default="Codigo/code_1D_array/output_graph", help="Directory with simulation CSV outputs")
    parser.add_argument("--save", default="Codigo/code_1D_array/output_graph/network_graph_comparison.png", help="Path to save the figure")
    parser.add_argument("--n_neurons", type=int, default=20, help="Number of neurons in the array")
    parser.add_argument("--spacing", type=float, default=1.0, help="Physical spacing between neurons (must match the C++ `spacing` parameter, only affects layout)")
    parser.add_argument("--edge_threshold", type=float, default=0.5,
                         help="Only draw final-network edges with weight above this value")
    args = parser.parse_args()

    w_init = load_matrix(os.path.join(args.outdir, "weights_initial.csv"), args.n_neurons)
    w_final = load_matrix(os.path.join(args.outdir, "weights_summary.csv"), args.n_neurons)

    stim_path = os.path.join(args.outdir, "stimulation.csv")
    if os.path.exists(stim_path):
        target_neurons = set(pd.read_csv(stim_path)["neuron"].unique().tolist())
    else:
        target_neurons = set()

    G_init = build_graph(w_init, args.n_neurons, threshold=0.0)
    G_final = build_graph(w_final, args.n_neurons, threshold=args.edge_threshold)

    fig = plt.figure(figsize=(12, 12), layout="constrained")
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1.1])

    ax_init = fig.add_subplot(gs[0, :])
    ax_final = fig.add_subplot(gs[1, :])
    ax_dist = fig.add_subplot(gs[2, 0])
    ax_delta = fig.add_subplot(gs[2, 1])

    draw_network(ax_init, G_init, args.n_neurons, args.spacing, target_neurons,
                 title=f"Initial connectivity (all-to-all, random weights, N edges={G_init.number_of_edges()})")
    draw_network(ax_final, G_final, args.n_neurons, args.spacing, target_neurons,
                 title=f"Final connectivity after STDP (edges with weight > {args.edge_threshold}, "
                       f"N edges={G_final.number_of_edges()} / {args.n_neurons*(args.n_neurons-1)})")

    plot_weight_distribution(ax_dist, w_init, w_final)
    plot_weight_change_vs_distance(ax_delta, w_init, w_final, args.n_neurons)

    # Legend explaining node/edge encoding, attached to the figure once
    handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#C44E52", markeredgecolor="black", markersize=8, label="Stimulated neuron"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#4C4C4C", markeredgecolor="black", markersize=8, label="Non-stimulated neuron"),
        plt.Line2D([0], [0], color="#3B75AF", lw=2, label="Feedforward edge (j > i)"),
        plt.Line2D([0], [0], color="#999999", lw=2, label="Feedback edge (j < i)"),
    ]
    fig.legend(handles=handles, loc="upper right", fontsize=8, ncol=1)

    fig.suptitle("1D Random-Walk Neuron Array: connectivity before vs. after STDP", fontsize=13)
    fig.savefig(args.save, dpi=150)
    print(f"Saved figure to {args.save}")
    print(f"Initial edges: {G_init.number_of_edges()}, Final edges above threshold {args.edge_threshold}: {G_final.number_of_edges()}")


if __name__ == "__main__":
    main()