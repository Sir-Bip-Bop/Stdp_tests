"""
plot_2d_network.py

2D-grid counterpart of plot_1d_network.py, for test_2d_grid.cpp. The simulation and STDP
mechanics are identical to the 1D case; what changes is that "distance" between neurons is
now Euclidean distance on the grid (read from positions.csv) rather than |i - j| along a line.

  Panel 1 - Raster plot: spike times for every neuron, with external stimulation micro-spikes
            overlaid and colored by their offset from the pulse center (blue = early, red = late).
  Panel 2 - Weight time evolution: mean weight as a function of time and Euclidean distance
            (binned), shown as a heatmap. This is the 2D-grid generalization of the 1D script's
            "nearest-neighbor forward weight over time" panel -- since there's no single natural
            "next neuron" on a grid, we instead track how weight-vs-distance structure develops
            over the course of the run.
  Panel 3 - Pulse shape: micro-spike count vs. offset from pulse center, summed over all pulses.
  Panel 4 - Zoom on a single representative pulse (same as the 1D script; note that because
            neuron index only varies smoothly within a grid row, this zoom mostly captures
            same-row neighbors of the target, not full 2D neighborhoods).
  Panel 5 - Final spatial structure: final weight matrix, plus weight vs. Euclidean distance.

Expects these files (produced by test_2d_grid.cpp) in the given output directory:
    positions.csv            neuron,row,col,x,y
    spikes.csv               time,neuron
    stimulation.csv          time,neuron,pulseCenter,offset
    weights_evolution.csv    time,iNeuron,jNeuron,weight   (periodic snapshots)
    weights_summary.csv      iNeuron,jNeuron,weight        (final matrix only)
"""

import argparse
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_data(outdir):
    positions = pd.read_csv(os.path.join(outdir, "positions.csv"))
    spikes = pd.read_csv(os.path.join(outdir, "spikes.csv"))
    stim = pd.read_csv(os.path.join(outdir, "stimulation.csv"))
    w_evo = pd.read_csv(os.path.join(outdir, "weights_evolution.csv"))
    w_final = pd.read_csv(os.path.join(outdir, "weights_summary.csv"))
    return positions, spikes, stim, w_evo, w_final


def euclidean_distance_matrix(positions):
    """n_neurons x n_neurons matrix of Euclidean distances from a positions dataframe
    (columns: neuron, x, y)."""
    positions = positions.sort_values("neuron")
    xy = positions[["x", "y"]].values
    diff = xy[:, None, :] - xy[None, :, :]
    return np.sqrt((diff ** 2).sum(axis=-1))


def plot_raster(fig, ax, spikes, stim, n_neurons):
    ax.scatter(spikes["time"], spikes["neuron"], s=8, c="black", label="Spike")

    offset_extent = max(1, stim["offset"].abs().max())
    sc = ax.scatter(
        stim["time"], stim["neuron"],
        marker="v", s=35, c=stim["offset"], cmap="coolwarm",
        vmin=-offset_extent, vmax=offset_extent,
        edgecolors="none", alpha=0.85, zorder=5, label="External stimulation",
    )
    cbar = fig.colorbar(sc, ax=ax, pad=0.01)
    cbar.set_label("Stim. offset from pulse center\n(early < 0 < late)")

    ax.set_ylabel("Neuron index")
    ax.set_title("Spike raster with external stimulation")
    ax.set_ylim(-1, n_neurons)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)


def plot_weight_evolution_vs_distance(ax, w_evo, dist_matrix, n_bins=8):
    """Heatmap of mean weight vs. (time, Euclidean-distance bin)."""
    df = w_evo.copy()
    df["distance"] = dist_matrix[df["iNeuron"].values, df["jNeuron"].values]

    max_d = dist_matrix.max()
    bins = np.linspace(0, max_d, n_bins + 1)
    df["distance_bin"] = pd.cut(df["distance"], bins=bins, include_lowest=True)

    pivot = df.groupby(["distance_bin", "time"], observed=True)["weight"].mean().unstack("time")
    bin_mids = [interval.mid for interval in pivot.index]

    im = ax.imshow(
        pivot.values, aspect="auto", origin="lower",
        extent=[pivot.columns.min(), pivot.columns.max(), bins[0], bins[-1]],
        cmap="viridis", vmin=0, vmax=1,
    )
    ax.set_ylabel("Euclidean distance |i - j|")
    ax.set_title("Mean weight vs. time and distance")
    cbar = plt.colorbar(im, ax=ax, pad=0.01)
    cbar.set_label("weight")


def plot_pulse_shape(ax, stim):
    counts = stim.groupby("offset").size().sort_index()
    colors = ["#4C72B0" if o < 0 else ("#C44E52" if o > 0 else "#55A868") for o in counts.index]

    ax.bar(counts.index, counts.values, color=colors, width=0.9)
    ax.axvline(0, color="black", linewidth=1, linestyle="--")
    ax.set_xlabel("Offset from pulse center (timesteps)")
    ax.set_ylabel("Micro-spike count\n(summed over all pulses)")
    ax.set_title("Stimulation pulse shape")
    ax.text(0.03, 0.95, "early", transform=ax.transAxes, color="#4C72B0",
            fontsize=9, va="top", ha="left")
    ax.text(0.97, 0.95, "late", transform=ax.transAxes, color="#C44E52",
            fontsize=9, va="top", ha="right")


def plot_pulse_zoom(fig, ax, spikes, stim, n_neurons, pulse_index=3):
    centers = np.sort(stim["pulseCenter"].unique())
    if len(centers) == 0:
        ax.set_visible(False)
        return
    center = centers[min(pulse_index, len(centers) - 1)]
    period = int(np.min(np.diff(centers))) if len(centers) > 1 else 50

    sub_stim = stim[stim["pulseCenter"] == center]
    offset_extent = max(1, stim["offset"].abs().max())

    t_lo = center - offset_extent - 5
    t_hi = center + period
    sub_spikes = spikes[(spikes["time"] >= t_lo) & (spikes["time"] <= t_hi)]

    target_neuron = stim["neuron"].iloc[0]

    sc = ax.scatter(
        sub_stim["time"], sub_stim["neuron"],
        marker="v", s=110, c=sub_stim["offset"], cmap="coolwarm",
        vmin=-offset_extent, vmax=offset_extent, edgecolors="k", linewidths=0.4,
        zorder=5, label="External micro-spike",
    )
    ax.scatter(
        sub_spikes["time"], sub_spikes["neuron"],
        s=40, c="black", zorder=4, label="Network spike",
    )
    ax.axvline(center, color="gray", linestyle=":", linewidth=1)
    ax.set_xlim(t_lo, t_hi)
    ax.set_ylim(target_neuron - 3, target_neuron + 3)
    ax.set_xlabel("Time (timesteps)")
    ax.set_ylabel("Neuron index")
    ax.set_title(f"Zoom: single pulse (center t={center})")
    ax.legend(loc="upper right", fontsize=7, framealpha=0.9)
    fig.colorbar(sc, ax=ax, pad=0.01).set_label("Offset")


def plot_final_structure(ax_matrix, ax_distance, w_final, dist_matrix, n_neurons):
    mat = w_final.pivot(index="iNeuron", columns="jNeuron", values="weight")
    mat = mat.reindex(index=range(n_neurons), columns=range(n_neurons))

    im = ax_matrix.imshow(mat.values, cmap="viridis", vmin=0, vmax=1, origin="lower")
    ax_matrix.set_xlabel("j (post-synaptic)")
    ax_matrix.set_ylabel("i (pre-synaptic)")
    ax_matrix.set_title("Final weight matrix")
    plt.colorbar(im, ax=ax_matrix, pad=0.01, fraction=0.046)

    df = w_final.copy()
    df["distance"] = dist_matrix[df["iNeuron"].values, df["jNeuron"].values]
    df = df[df["distance"] > 0]
    # Round distance to group essentially-identical grid distances together
    df["distance_rounded"] = df["distance"].round(2)
    stats = df.groupby("distance_rounded")["weight"].agg(["mean", "std"])

    ax_distance.errorbar(
        stats.index, stats["mean"], yerr=stats["std"],
        fmt="o-", ms=4, capsize=3, color="darkorange",
    )
    ax_distance.set_xlabel("Euclidean distance |i - j|")
    ax_distance.set_ylabel("Weight (mean +/- std)")
    ax_distance.set_title("Final weight vs. distance")
    ax_distance.set_ylim(-0.05, 1.05)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", default="Codigo/code_grid/output_2D", help="Directory with simulation CSV outputs")
    parser.add_argument("--save", default="Codigo/code_grid/output_2D/2d_network_summary.png", help="Path to save the figure")
    parser.add_argument("--n_neurons", type=int, default=20, help="Number of neurons in the grid")
    parser.add_argument("--pulse_index", type=int, default=3,
                         help="Which pulse (0-indexed) to show in the zoom panel")
    args = parser.parse_args()

    positions, spikes, stim, w_evo, w_final = load_data(args.outdir)
    dist_matrix = euclidean_distance_matrix(positions)

    fig = plt.figure(figsize=(11, 16), layout="constrained")
    gs = fig.add_gridspec(4, 2, height_ratios=[1, 1, 1, 1.1])

    ax_raster = fig.add_subplot(gs[0, :])
    ax_evo = fig.add_subplot(gs[1, :])
    ax_pulse_shape = fig.add_subplot(gs[2, 0])
    ax_pulse_zoom = fig.add_subplot(gs[2, 1])
    ax_matrix = fig.add_subplot(gs[3, 0])
    ax_distance = fig.add_subplot(gs[3, 1])

    plot_raster(fig, ax_raster, spikes, stim, args.n_neurons)
    plot_weight_evolution_vs_distance(ax_evo, w_evo, dist_matrix)
    plot_pulse_shape(ax_pulse_shape, stim)
    plot_pulse_zoom(fig, ax_pulse_zoom, spikes, stim, args.n_neurons, pulse_index=args.pulse_index)
    plot_final_structure(ax_matrix, ax_distance, w_final, dist_matrix, args.n_neurons)
    ax_evo.set_xlabel("Time (timesteps)")
    ax_raster.set_xlabel("Time (timesteps)")

    fig.suptitle("2D Grid Neuron Array: activity and STDP weight development", fontsize=13)
    fig.savefig(args.save, dpi=150)
    print(f"Saved figure to {args.save}")


if __name__ == "__main__":
    main()