"""
plot_fig1.py
============
Reproduces the Figure 1 layout for any two neurons, comparing two weight
conditions (A and B) on the same figure.

  Top panel    – neurons A & B under weight matrix A
  Bottom panel – neurons A & B under weight matrix B

Usage
-----
  python plot_fig1.py                        # neurons 0 vs 1, default output
  python plot_fig1.py --na 0 --nb 5          # choose neuron pair
  python plot_fig1.py --na 0 --nb 5 --out fig.png

Reads:
  output_figure1/voltage_A.csv   output_test/spikes_A.csv
  output_figure1/voltage_B.csv   output_test/spikes_B.csv
"""

import argparse, os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ── CLI ────────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--na",  type=int, default=0,  help="First neuron index  (default 0)")
parser.add_argument("--nb",  type=int, default=1,  help="Second neuron index (default 1)")
parser.add_argument("--out", type=str, default="Codigo/code_1D_array/output_figure1/fig1.png")
args = parser.parse_args()
nA, nB = args.na, args.nb

# ── Load ───────────────────────────────────────────────────────────────────────
def load(tag):
    vp = f"Codigo/code_1D_array/output_figure1/voltage_{tag}.csv"
    sp = f"Codigo/code_1D_array/output_figure1//spikes_{tag}.csv"
    if not os.path.exists(vp) or not os.path.exists(sp):
        sys.exit(f"Missing files for run '{tag}'. Run ./Fig1 first.")
    return pd.read_csv(vp), pd.read_csv(sp)

voltA, spikesA = load("A")
voltB, spikesB = load("B")
t = voltA["time"].values

def vspike(volt, spikes, n):
    return volt[f"v{n}"].values, spikes[spikes["neuron"] == n]["time"].values

vA_nA, spA_nA = vspike(voltA, spikesA, nA)
vA_nB, spA_nB = vspike(voltA, spikesA, nB)
vB_nA, spB_nA = vspike(voltB, spikesB, nA)
vB_nB, spB_nB = vspike(voltB, spikesB, nB)

# ── Colours ────────────────────────────────────────────────────────────────────
COL = {nA: "#d62728", nB: "#1f77b4"}   # red / blue

# ── Figure ─────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True,
                         gridspec_kw={"hspace": 0.12})
THRESHOLD, RESET = 1.0, 0.0

def draw_panel(ax, traces, run_label):
    ax.axhline(THRESHOLD, color="k", lw=0.7, ls="--", alpha=0.55, zorder=1)
    ax.axhline(RESET,     color="k", lw=0.7, ls="--", alpha=0.35, zorder=1)

    for (v, spk, n) in traces:
        col = COL[n]
        ax.plot(t, v, color=col, lw=0.9, alpha=0.85, label=f"Neuron {n}", zorder=2)
        for ts in spk:
            ax.axvline(ts, color=col, lw=0.8, alpha=0.55, ymin=0.88, ymax=1.0, zorder=3)

    ax.set_ylabel("Membrane\npotential  $v$", fontsize=10)
    ax.set_ylim(-0.07, 1.18)
    ax.set_yticks([RESET, THRESHOLD])
    ax.set_yticklabels(["$v^0=0$", r"$v^\theta=1$"], fontsize=9)
    ax.yaxis.set_minor_locator(ticker.NullLocator())
    ax.text(0.005, 0.94, run_label, transform=ax.transAxes,
            fontsize=11, fontweight="bold", va="top", ha="left", color="0.25")
    handles = [plt.Line2D([0],[0], color=COL[n], lw=1.8, label=f"Neuron {n}")
               for n in (nA, nB)]
    ax.legend(handles=handles, loc="upper right", fontsize=8,
              framealpha=0.7, handlelength=1.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

draw_panel(axes[0],
           [(vA_nA, spA_nA, nA), (vA_nB, spA_nB, nB)],
           "Weight matrix A  (random init)")

draw_panel(axes[1],
           [(vB_nA, spB_nA, nA), (vB_nB, spB_nB, nB)],
           "Weight matrix B  (all-zero)")

axes[1].set_xlabel("Time  (steps)", fontsize=10)
fig.suptitle(
    f"Reproducing Fig 1. – Membrane-potential traces for neurons {nA} & {nB}",
    fontsize=11, y=1.00
)

os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
fig.savefig(args.out, dpi=150, bbox_inches="tight")
print(f"Saved -> {args.out}")