"""
run_sweep.py

Runs test_2d_grid multiple times across a set of parameter sweeps -- varying which neurons
receive external stimulation, and varying the characteristics of the Gaussian stimulation
pulse -- and compares the resulting final (post-STDP) connectivity graphs.

Design: each sweep varies ONE parameter at a time away from a shared baseline, rather than
a full cartesian product across every parameter. This keeps the number of runs small enough
to visualize as readable small-multiples figures, while still directly answering "what
happens to the final network if I change X". Each individual run only takes ~0.2s, so a much
larger/custom sweep is easy to add if you want finer resolution on any one axis later.

Sweeps included by default:
  1. target_pattern   - which neurons are externally stimulated (single neuron, opposite
                         corners, all 4 corners, edge midpoints, center)
  2. pulse_sigma       - temporal width of each Gaussian pulse (narrow -> wide)
  3. pulse_peak_spikes - how many micro-spikes land at the pulse's peak (weak -> strong pulse)
  4. pulse_stagger     - phase offset between different targets' pulse trains (synchronous -> staggered)

For each sweep, this script produces:
  - <sweep_name>_graphs.png   : small-multiples figure of the final network graph for every
                                 run in that sweep (same edge_threshold across the row, so
                                 they're visually comparable).
  - <sweep_name>_summary.png  : how simple scalar summaries (edges surviving threshold, mean
                                 final weight, fraction decayed/potentiated) change across
                                 the swept parameter.
  - sweep_summary.csv         : one row per run across ALL sweeps, with every parameter and
                                 every summary statistic, for your own further analysis.

Usage:
    python3 run_sweep.py --binary ./test_2d_grid --outdir sweep_output
"""

import argparse
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_2D_graph import build_graph, draw_network, load_matrix  # reuse existing graph logic


# -------------------------
# Baseline parameters (same defaults as test_2d_grid.cpp)
# -------------------------
BASELINE = dict(
    targets="0,4,15,19",
    tMax=8000,
    stagger=15,
    sigma=5.0,
    peakSpikes=6,
    bias=0.12,
    period=50,
)

TARGET_PATTERNS = {
    "single_corner":    "0",
    "opposite_corners": "0,19",
    "four_corners":     "0,4,15,19",
    "edge_midpoints":   "2,10,14,17",
    "center":           "12",
}

SIGMA_VALUES = [1.0, 5.0, 10.0, 20.0]
PEAK_SPIKES_VALUES = [2, 6, 12, 20]
STAGGER_VALUES = [0, 15, 30, 50]


def build_sweep_configs():
    """Returns a dict: sweep_name -> list of (run_name, param_overrides) tuples."""
    sweeps = {}

    sweeps["target_pattern"] = [
        (name, dict(targets=targets)) for name, targets in TARGET_PATTERNS.items()
    ]
    sweeps["pulse_sigma"] = [
        (f"sigma_{s:g}", dict(sigma=s)) for s in SIGMA_VALUES
    ]
    sweeps["pulse_peak_spikes"] = [
        (f"peak_{p}", dict(peakSpikes=p)) for p in PEAK_SPIKES_VALUES
    ]
    sweeps["pulse_stagger"] = [
        (f"stagger_{s}", dict(stagger=s)) for s in STAGGER_VALUES
    ]

    return sweeps


def run_one(binary, run_outdir, params):
    os.makedirs(run_outdir, exist_ok=True)
    cmd = [
        binary,
        str(params["targets"]),
        str(params["tMax"]),
        str(params["stagger"]),
        str(params["sigma"]),
        str(params["peakSpikes"]),
        str(params["bias"]),
        str(params["period"]),
        run_outdir,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  FAILED: {' '.join(cmd)}")
        print(result.stdout[-500:])
        print(result.stderr[-500:])
    return result.returncode == 0


def summarize_run(run_outdir, n_neurons=20, edge_threshold=0.5):
    w_init = load_matrix(os.path.join(run_outdir, "weights_initial.csv"), n_neurons)
    w_final = load_matrix(os.path.join(run_outdir, "weights_summary.csv"), n_neurons)

    mask = ~np.eye(n_neurons, dtype=bool)
    init_vals = w_init[mask]
    final_vals = w_final[mask]
    delta_vals = final_vals - init_vals

    G_final = build_graph(w_final, n_neurons, threshold=edge_threshold)

    return dict(
        n_edges_total=int(mask.sum()),
        n_edges_above_threshold=G_final.number_of_edges(),
        frac_edges_above_threshold=G_final.number_of_edges() / mask.sum(),
        mean_weight_initial=float(init_vals.mean()),
        mean_weight_final=float(final_vals.mean()),
        mean_weight_delta=float(delta_vals.mean()),
        frac_decayed_below_0p05=float((final_vals < 0.05).mean()),
        frac_potentiated_above_0p95=float((final_vals > 0.95).mean()),
    )


def plot_sweep_graphs(sweep_name, runs, positions_path, target_neurons_by_run, save_path, edge_threshold=0.5, n_neurons=20):
    positions = pd.read_csv(positions_path).sort_values("neuron")
    pos = {int(row.neuron): (row.x, row.y) for row in positions.itertuples()}

    n = len(runs)
    ncols = min(4, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 4.2 * nrows), layout="constrained")
    axes = np.atleast_1d(axes).flatten()

    for ax, (run_name, run_outdir) in zip(axes, runs):
        w_final = load_matrix(os.path.join(run_outdir, "weights_summary.csv"), n_neurons)
        G_final = build_graph(w_final, n_neurons, threshold=edge_threshold)
        targets = target_neurons_by_run[run_name]
        draw_network(ax, G_final, pos, targets,
                      title=f"{run_name}\n(edges: {G_final.number_of_edges()})")

    for ax in axes[len(runs):]:
        ax.set_visible(False)

    fig.suptitle(f"Sweep: {sweep_name} -- final connectivity (weight > {edge_threshold})", fontsize=13)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved {save_path}")


def plot_sweep_summary(sweep_name, df_sweep, x_col, save_path):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), layout="constrained")

    axes[0].plot(df_sweep[x_col], df_sweep["n_edges_above_threshold"], "o-", color="#3B75AF")
    axes[0].set_ylabel("N edges above threshold")
    axes[0].set_title("Surviving connections")

    axes[1].plot(df_sweep[x_col], df_sweep["mean_weight_final"], "o-", color="#C44E52", label="final")
    axes[1].plot(df_sweep[x_col], df_sweep["mean_weight_initial"], "o--", color="#8172B2", alpha=0.6, label="initial")
    axes[1].set_ylabel("Mean weight")
    axes[1].set_title("Mean weight")
    axes[1].legend(fontsize=8)

    axes[2].plot(df_sweep[x_col], df_sweep["frac_decayed_below_0p05"], "o-", color="#4C72B0", label="decayed (<0.05)")
    axes[2].plot(df_sweep[x_col], df_sweep["frac_potentiated_above_0p95"], "o-", color="#55A868", label="potentiated (>0.95)")
    axes[2].set_ylabel("Fraction of edges")
    axes[2].set_title("Decay vs. potentiation")
    axes[2].legend(fontsize=8)

    for ax in axes:
        ax.set_xlabel(x_col)
        if df_sweep[x_col].dtype == object:
            ax.tick_params(axis="x", rotation=30)

    fig.suptitle(f"Sweep: {sweep_name}", fontsize=13)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved {save_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--binary", default="./Codigo/code_grid/test_2D_grid", help="Path to the compiled test_2d_grid binary")
    parser.add_argument("--outdir", default="Codigo/code_grid/sweep_output", help="Root directory for all sweep runs and figures")
    parser.add_argument("--n_neurons", type=int, default=20)
    parser.add_argument("--edge_threshold", type=float, default=0.5)
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    sweeps = build_sweep_configs()

    all_rows = []

    for sweep_name, run_specs in sweeps.items():
        print(f"\n=== Sweep: {sweep_name} ({len(run_specs)} runs) ===")
        sweep_dir = os.path.join(args.outdir, sweep_name)
        os.makedirs(sweep_dir, exist_ok=True)

        runs = []                 # (run_name, run_outdir)
        target_neurons_by_run = {}

        for run_name, overrides in run_specs:
            params = dict(BASELINE)
            params.update(overrides)
            run_outdir = os.path.join(sweep_dir, run_name)

            print(f"  Running {run_name}: {overrides}")
            ok = run_one(args.binary, run_outdir, params)
            if not ok:
                continue

            runs.append((run_name, run_outdir))
            target_neurons_by_run[run_name] = set(int(t) for t in str(params["targets"]).split(","))

            stats = summarize_run(run_outdir, n_neurons=args.n_neurons, edge_threshold=args.edge_threshold)
            row = dict(sweep=sweep_name, run=run_name, **params, **stats)
            all_rows.append(row)

        if not runs:
            continue

        # x-axis for the summary plot: whichever parameter actually varies in this sweep
        df_all = pd.DataFrame(all_rows)
        df_sweep = df_all[df_all["sweep"] == sweep_name].reset_index(drop=True)
        varying_cols = [c for c in ["targets", "sigma", "peakSpikes", "stagger"]
                         if df_sweep[c].astype(str).nunique() > 1]
        x_col = varying_cols[0] if varying_cols else "run"

        positions_path = os.path.join(runs[0][1], "positions.csv")
        plot_sweep_graphs(
            sweep_name, runs, positions_path, target_neurons_by_run,
            save_path=os.path.join(args.outdir, f"{sweep_name}_graphs.png"),
            edge_threshold=args.edge_threshold, n_neurons=args.n_neurons,
        )
        plot_sweep_summary(
            sweep_name, df_sweep, x_col,
            save_path=os.path.join(args.outdir, f"{sweep_name}_summary.png"),
        )

    df_all = pd.DataFrame(all_rows)
    csv_path = os.path.join(args.outdir, "sweep_summary.csv")
    df_all.to_csv(csv_path, index=False)
    print(f"\nSaved combined summary table: {csv_path} ({len(df_all)} runs)")


if __name__ == "__main__":
    main()