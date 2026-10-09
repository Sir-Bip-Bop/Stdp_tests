"""
plot_1d_network.py

Visualizes the output of test_1d_network.cpp (rwN::RWNetwork::init1DArray simulation):

  Panel 1 - Raster plot: spike times for every neuron in the array, with the external
            stimulation events marked (stimulation is delivered only to neuron 0, every
            `externalEvery` timesteps, per the driver code).
  Panel 2 - Weight time evolution: how the nearest-neighbor forward weight w[i -> i+1]
            changes over the course of the simulation, for every position i in the array.
            Shown as a heatmap (time x position) so you can see potentiation propagate
            down the chain as the stimulus-triggered wave repeatedly sweeps through it.
  Panel 3 - Final spatial structure: the fully-trained weight matrix, plus weight vs.
            distance |i-j| to show how connection strength decays (or doesn't) with
            distance along the array once training has converged.

Expects these files (produced by test_1d_network.cpp) in the given output directory:
    spikes.csv              time,neuron
    stimulation.csv         time,neuron
    weights_evolution.csv   time,iNeuron,jNeuron,weight   (periodic snapshots)
    weights_summary.csv     iNeuron,jNeuron,weight        (final matrix only)
"""

import argparse
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_data(outdir):
    spikes = pd.read_csv(os.path.join(outdir, "spikes.csv"))
    stim = pd.read_csv(os.path.join(outdir, "stimulation.csv"))
    w_evo = pd.read_csv(os.path.join(outdir, "weights_evolution.csv"))
    w_final = pd.read_csv(os.path.join(outdir, "weights_summary.csv"))
    return spikes, stim, w_evo, w_final


def plot_raster(ax, spikes, stim, n_neurons):
    ax.scatter(spikes["time"], spikes["neuron"], s=8, c="black", label="Spike")
    # Mark stimulation events on the neuron(s) that actually receive them
    ax.scatter(
        stim["time"], stim["neuron"],
        marker="o", s=40, facecolors="none", edgecolors="crimson",
        linewidths=1.3, label="External stimulation", zorder=5,
    )
    ax.set_ylabel("neuron index")
    ax.set_title("spike raster plot")
    ax.set_ylim(-1, n_neurons)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)


def plot_weight_evolution(ax, w_evo, n_neurons):
    # Forward nearest-neighbor weight w[i -> i+1] as a function of time and position i
    nn = w_evo[w_evo["jNeuron"] == w_evo["iNeuron"] + 1].copy()
    pivot = nn.pivot(index="iNeuron", columns="time", values="weight")
    pivot = pivot.reindex(index=range(n_neurons - 1))  # ensure full/ordered index

    im = ax.imshow(
        pivot.values,
        aspect="auto",
        origin="lower",
        extent=[pivot.columns.min(), pivot.columns.max(), -0.5, n_neurons - 1.5],
        cmap="viridis",
        vmin=0, vmax=1,
    )
    ax.set_ylabel("neuron pair (i -> i+1)")
    ax.set_title("nearest-neighbor weight over time")
    cbar = plt.colorbar(im, ax=ax, pad=0.01)
    cbar.set_label("weight")


def plot_final_structure(ax_matrix, ax_distance, w_final, n_neurons):
    mat = w_final.pivot(index="iNeuron", columns="jNeuron", values="weight")
    mat = mat.reindex(index=range(n_neurons), columns=range(n_neurons))

    im = ax_matrix.imshow(mat.values, cmap="viridis", vmin=0, vmax=1, origin="lower")
    ax_matrix.set_xlabel("j (post-synaptic)")
    ax_matrix.set_ylabel("i (pre-synaptic)")
    ax_matrix.set_title("final weight matrix")
    plt.colorbar(im, ax=ax_matrix, pad=0.01, fraction=0.046)

    # Weight vs. distance along the array (final, spatial structure)
    df = w_final.copy()
    df["distance"] = (df["iNeuron"] - df["jNeuron"]).abs()
    df = df[df["distance"] > 0]
    stats = df.groupby("distance")["weight"].agg(["mean", "std"])

    ax_distance.errorbar(
        stats.index, stats["mean"], yerr=stats["std"],
        fmt="o-", ms=4, capsize=3, color="darkorange",
    )
    ax_distance.set_xlabel("distance |i - j|")
    ax_distance.set_ylabel("weight")
    ax_distance.set_title("final weight vs. distance")
    ax_distance.set_ylim(-0.05, 1.05)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", default="Codigo/code_1D_array/output_1D", help="Directory with simulation CSV outputs")
    parser.add_argument("--save", default="Codigo/code_1D_array/output_1D/1d_network_summary.png", help="Path to save the figure")
    parser.add_argument("--n_neurons", type=int, default=20, help="Number of neurons in the array")
    args = parser.parse_args()

    spikes, stim, w_evo, w_final = load_data(args.outdir)

    # constrained layout keeps axes widths aligned/centered even when some panels
    # have a colorbar (which eats into their width) and others don't.
    fig = plt.figure(figsize=(11, 11), layout="constrained")
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1.1])

    ax_raster = fig.add_subplot(gs[0, :])
    ax_evo = fig.add_subplot(gs[1, :])
    ax_matrix = fig.add_subplot(gs[2, 0])
    ax_distance = fig.add_subplot(gs[2, 1])

    plot_raster(ax_raster, spikes, stim, args.n_neurons)
    plot_weight_evolution(ax_evo, w_evo, args.n_neurons)
    plot_final_structure(ax_matrix, ax_distance, w_final, args.n_neurons)
    ax_distance.set_xlabel("distance |i - j|")
    ax_evo.set_xlabel("time [timesteps]")
    ax_raster.set_xlabel("time [timesteps]")

    # Reserve a colorbar-width spacer on the raster panel (invisible) so its
    # plotted width lines up with panels below/beside it that do have colorbars.
    spacer_cbar = fig.colorbar(plt.cm.ScalarMappable(cmap="viridis"), ax=ax_raster, pad=0.01)
    spacer_cbar.ax.set_visible(False)

    fig.suptitle("1D Random-Walk Neuron Array: initial exploration", fontsize=13)
    fig.savefig(args.save, dpi=150)
    print(f"Saved figure to {args.save}")


if __name__ == "__main__":
    main()