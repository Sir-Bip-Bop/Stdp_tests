"""
plot_1d_network.py

Visualizes the output of test_1d_network.cpp (rwN::RWNetwork::init1DArray simulation),
including the Gaussian-shaped external stimulation pulses (see deliverGaussianPulse() in
the C++ driver): each "stimulation event" is now a small volley of micro-spikes spread out
in time around a pulse center, so part of a pulse can land early (large negative offset)
and part can land late (small negative / positive offset) relative to any given postsynaptic
spike -- this is what we want to inspect for its effect on STDP.

  Panel 1 - Raster plot: spike times for every neuron, with external stimulation micro-spikes
            overlaid and colored by their offset from the pulse center (blue = early, red = late).
  Panel 2 - Weight time evolution: how the nearest-neighbor forward weight w[i -> i+1]
            changes over the course of the simulation, for every position i in the array.
            Shown as a heatmap (time x position) so you can see potentiation propagate
            down the chain as the stimulus-triggered wave repeatedly sweeps through it.
  Panel 3 - Pulse shape: micro-spike count vs. offset from pulse center, summed across all
            pulses -- a direct look at the Gaussian envelope (early / peak / late).
  Panel 4 - Zoom on a single representative pulse: the stimulated neuron's micro-spikes
            (colored by offset) alongside nearby neurons' actual spikes in that time window,
            so you can see the early/late spikes relative to real network activity.
  Panel 5 - Final spatial structure: the fully-trained weight matrix, plus weight vs.
            distance |i-j| to show how connection strength decays (or doesn't) with
            distance along the array once training has converged.

Expects these files (produced by test_1d_network.cpp) in the given output directory:
    spikes.csv              time,neuron
    stimulation.csv         time,neuron,pulseCenter,offset
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


def plot_raster(fig, ax, spikes, stim, n_neurons):
    ax.scatter(spikes["time"], spikes["neuron"], s=8, c="black", label="Spike")

    # Mark external stimulation micro-spikes, colored by offset from their pulse's center:
    # negative (blue) = arrived early, zero (white) = at the pulse peak, positive (red) = late.
    has_offset = "offset" in stim.columns
    if has_offset:
        offset_extent = max(1, stim["offset"].abs().max())
        sc = ax.scatter(
            stim["time"], stim["neuron"],
            marker="v", s=35, c=stim["offset"], cmap="coolwarm",
            vmin=-offset_extent, vmax=offset_extent,
            edgecolors="none", alpha=0.85, zorder=5, label="External stimulation",
        )
        cbar = fig.colorbar(sc, ax=ax, pad=0.01)
        cbar.set_label("Stim. offset from pulse center\n(early < 0 < late)")
    else:
        ax.scatter(
            stim["time"], stim["neuron"],
            marker="v", s=60, facecolors="none", edgecolors="crimson",
            linewidths=1.3, label="External stimulation", zorder=5,
        )

    ax.set_ylabel("Neuron index")
    ax.set_title("Spike raster with external stimulation")
    ax.set_ylim(-1, n_neurons)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)


def plot_pulse_shape(ax, stim):
    """Bar chart of micro-spike count vs. offset from pulse center, summed over all pulses.
    Directly shows the Gaussian envelope: sparse early tail, peak at 0, sparse late tail."""
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
    """Zoom in on one representative, non-edge-truncated pulse: shows the stimulated
    neuron's micro-spikes (colored by offset) alongside real spikes from nearby neurons
    that occur in the same window, so early vs. late timing is visible directly."""
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
    ax.set_ylabel("Neuron pair (i -> i+1)")
    ax.set_title("Nearest-neighbor forward weight over time")
    cbar = plt.colorbar(im, ax=ax, pad=0.01)
    cbar.set_label("weight")


def plot_final_structure(ax_matrix, ax_distance, w_final, n_neurons):
    mat = w_final.pivot(index="iNeuron", columns="jNeuron", values="weight")
    mat = mat.reindex(index=range(n_neurons), columns=range(n_neurons))

    im = ax_matrix.imshow(mat.values, cmap="viridis", vmin=0, vmax=1, origin="lower")
    ax_matrix.set_xlabel("j (post-synaptic)")
    ax_matrix.set_ylabel("i (pre-synaptic)")
    ax_matrix.set_title("Final weight matrix")
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
    ax_distance.set_xlabel("Distance |i - j|")
    ax_distance.set_ylabel("Weight (mean +/- std)")
    ax_distance.set_title("Final weight vs. distance")
    ax_distance.set_ylim(-0.05, 1.05)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", default="Codigo/code_1D_array/output_graph", help="Directory with simulation CSV outputs")
    parser.add_argument("--save", default="Codigo/code_1D_array/output_graph/1d_network_summary.png", help="Path to save the figure")
    parser.add_argument("--n_neurons", type=int, default=20, help="Number of neurons in the array")
    parser.add_argument("--pulse_index", type=int, default=3,
                         help="Which pulse (0-indexed, in chronological order) to show in the zoom panel. "
                              "Use a small positive value > 0 to avoid the first pulse, which is truncated "
                              "on its early side by the t=0 boundary.")
    args = parser.parse_args()

    spikes, stim, w_evo, w_final = load_data(args.outdir)

    # constrained layout keeps axes widths aligned/centered even when some panels
    # have a colorbar (which eats into their width) and others don't.
    fig = plt.figure(figsize=(11, 16), layout="constrained")
    gs = fig.add_gridspec(4, 2, height_ratios=[1, 1, 1, 1.1])

    ax_raster = fig.add_subplot(gs[0, :])
    ax_evo = fig.add_subplot(gs[1, :])
    ax_pulse_shape = fig.add_subplot(gs[2, 0])
    ax_pulse_zoom = fig.add_subplot(gs[2, 1])
    ax_matrix = fig.add_subplot(gs[3, 0])
    ax_distance = fig.add_subplot(gs[3, 1])

    plot_raster(fig, ax_raster, spikes, stim, args.n_neurons)
    plot_weight_evolution(ax_evo, w_evo, args.n_neurons)
    plot_pulse_shape(ax_pulse_shape, stim)
    plot_pulse_zoom(fig, ax_pulse_zoom, spikes, stim, args.n_neurons, pulse_index=args.pulse_index)
    plot_final_structure(ax_matrix, ax_distance, w_final, args.n_neurons)
    ax_distance.set_xlabel("Distance |i - j|")
    ax_evo.set_xlabel("Time (timesteps)")
    ax_raster.set_xlabel("Time (timesteps)")

    fig.suptitle("1D Random-Walk Neuron Array: activity and STDP weight development", fontsize=13)
    fig.savefig(args.save, dpi=150)
    print(f"Saved figure to {args.save}")


if __name__ == "__main__":
    main()