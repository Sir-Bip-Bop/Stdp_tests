"""
plot_fig3.py
============
Reproduces Figure 3 of the paper: the relationship between the spike-timing
difference Delta-t^s and the resulting STDP weight change, for a chosen
presynaptic/postsynaptic neuron pair in the 1D array, at a FIXED synaptic
weight w.

Top panel    : histogram of Delta-t^s = t_post - t_pre - delay
Bottom panel : STDP kernel K(Delta-t) overlaid on the same x-axis, showing
               how each Delta-t bin contributes to potentiation/depression.

Usage
-----
  python plot_fig3.py --tag w025 --out output_test/fig3.png

Reads:
  output_test/deltaT_<tag>.csv
  output_test/meta_<tag>.txt
(produced by fig3_sim)
"""

import argparse, os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── CLI ────────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--tag", type=str, required=True,
                    help="Run tag used with fig3_sim (e.g. 'w025')")
parser.add_argument("--out", type=str, default=None,
                    help="Output image path (default: Codigo/code_1D_array/output_figure3/fig3_<tag>.png)")
parser.add_argument("--bins", type=int, default=61,
                    help="Number of histogram bins (default 61)")
args = parser.parse_args()

tag = args.tag
out = args.out or f"Codigo/code_1D_array/output_figure3/fig3_{tag}.png"

dt_path   = f"Codigo/code_1D_array/output_figure3/deltaT_{tag}.csv"
meta_path = f"Codigo/code_1D_array/output_figure3/meta_{tag}.txt"

if not os.path.exists(dt_path) or not os.path.exists(meta_path):
    sys.exit(f"Missing files for tag '{tag}'. Run ./fig3_sim first.")

# ── Load metadata ──────────────────────────────────────────────────────────────
meta = {}
with open(meta_path) as f:
    for line in f:
        k, v = line.strip().split("=")
        meta[k] = v

neuronA   = int(meta["neuronA"])
neuronB   = int(meta["neuronB"])
weight    = float(meta["weight"])
delayAB   = float(meta["delayAB"])
sigmaSTDP = float(meta["sigmaSTDP"])
ampSTDP   = float(meta["amplitudeSTDP"])
Tend      = meta["Tend"]
nSpikesA  = meta["nSpikesA"]
nSpikesB  = meta["nSpikesB"]

# ── Load Delta-t data ────────────────────────────────────────────────────────
df = pd.read_csv(dt_path)
deltaT = df["deltaT"].values

if len(deltaT) == 0:
    sys.exit("No Delta-t values recorded -- neurons never spiked close enough "
             "in time. Try a longer Tend or different neuron pair / weight.")

# ── STDP kernel as a function of OUR Delta-t convention ────────────────────────
# fig3_sim computes:        myDeltaT = t_post - t_pre - delay
# rwNeuron.h's deltaW_STDP uses:  codeDeltaT = t_pre + delay - t_post = -myDeltaT
# and applies deltaW = deltaW_STDP(codeDeltaT) to the synapse.
# Since deltaW_STDP is an odd function, the *actual* weight change as a
# function of myDeltaT is:  deltaW(myDeltaT) = deltaW_STDP(-myDeltaT) = -deltaW_STDP(myDeltaT)
def deltaW_STDP_raw(dt, A=ampSTDP, sigma=sigmaSTDP):
    return -A * dt * np.exp(-0.5 * (dt / sigma) ** 2) / (sigma**3 * np.sqrt(2 * np.pi))

def K(dt):
    """Actual synaptic weight change as a function of myDeltaT = t_post - t_pre - delay."""
    return -deltaW_STDP_raw(dt)

window = 5 * sigmaSTDP
dt_curve = np.linspace(-window, window, 1000)
K_curve  = K(dt_curve)

# ── Figure: 2 stacked panels (paper Fig. 3 style) ──────────────────────────────
fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True,
                         gridspec_kw={"hspace": 0.08, "height_ratios": [1.3, 1]})

# Top: histogram of Delta-t
ax_hist = axes[0]
bin_edges = np.linspace(-window, window, args.bins)
ax_hist.hist(deltaT, bins=bin_edges, color="0.35", edgecolor="0.2", linewidth=0.4)
ax_hist.axvline(0, color="k", lw=0.8, ls="-", alpha=0.5)
ax_hist.set_ylabel("Frequency", fontsize=10)
ax_hist.set_title(
    f"Neuron {neuronA} (pre) $\\to$ Neuron {neuronB} (post)   "
    f"$\\omega={weight:g}$,  delay$={delayAB:g}$\n"
    f"$T_{{end}}={Tend}$,  spikes: pre={nSpikesA}, post={nSpikesB}",
    fontsize=10
)
ax_hist.spines["top"].set_visible(False)
ax_hist.spines["right"].set_visible(False)

# Bottom: STDP kernel curve K(Delta-t)
ax_k = axes[1]
ax_k.plot(dt_curve, K_curve, color="#1f77b4", lw=1.6)
ax_k.axhline(0, color="k", lw=0.7, alpha=0.5)
ax_k.axvline(0, color="k", lw=0.8, ls="-", alpha=0.5)
ax_k.fill_between(dt_curve, K_curve, 0, where=(K_curve >= 0),
                  color="#1f77b4", alpha=0.15)
ax_k.fill_between(dt_curve, K_curve, 0, where=(K_curve < 0),
                  color="#d62728", alpha=0.15)
ax_k.set_xlabel(r"$\Delta t^s = t_{post} - t_{pre} - \tau$", fontsize=11)
ax_k.set_ylabel(r"$K(\Delta t^s)$", fontsize=10)
ax_k.spines["top"].set_visible(False)
ax_k.spines["right"].set_visible(False)

# Annotate potentiation / depression regions like the paper
ax_k.text(0.97, 0.92, "potentiation\n($\\Delta t>0$)", transform=ax_k.transAxes,
          fontsize=8, ha="right", va="top", color="#1f77b4")
ax_k.text(0.03, 0.92, "depression\n($\\Delta t<0$)", transform=ax_k.transAxes,
          fontsize=8, ha="left", va="top", color="#d62728")

os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved -> {out}")