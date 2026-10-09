"""
plot_fig8.py
============
Reproduces Figure 8 of the paper: dynamic behavior of synaptic weights under
a periodic external synchronizing signal.

Left panel  (schematic, like paper's Fig. 8 left): spike-timing diagram for
            neuron A and two example postsynaptic neurons B at distances
            d < D/2 and d > D/2, illustrating the causal delay geometry.
Right panel (numerical, like paper's Fig. 8 right): omega* vs distance for
            many neurons B_i, produced by fig8_sim -- shows the periodic
            d < D/2 -> 0,  d > D/2 -> 1  pattern with period D = v*T.

Usage
-----
  python plot_fig8.py --csv output_test/fig8_weights.csv --T 100 --v 1.0 \
                      --d-examples 25 75 --out output_test/fig8.png
"""

import argparse, os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── CLI ────────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--csv", type=str, default="Codigo/code_1D_array/output_figure8/fig8_weights.csv")
parser.add_argument("--T", type=float, required=True, help="Synchronizing period T")
parser.add_argument("--v", type=float, required=True, help="Propagation velocity")
parser.add_argument("--d-examples", type=float, nargs=2, default=None,
                    help="Two example distances for the schematic panel "
                         "(default: D/4 and 3D/4)")
parser.add_argument("--out", type=str, default="Codigo/code_1D_array/output_figure8/fig8.png")
args = parser.parse_args()

if not os.path.exists(args.csv):
    sys.exit(f"Missing sweep file: {args.csv}. Run fig8_sim first.")

df = pd.read_csv(args.csv)
T, v = args.T, args.v
D = v * T

d_examples = args.d_examples or [D/4, 3*D/4]
d1, d2 = d_examples   # d1 < D/2 (-> 0), d2 > D/2 (-> 1)

# ── Figure layout: left = schematic, right = omega* vs distance ────────────────
fig = plt.figure(figsize=(14, 5.5))
gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1.3], wspace=0.28)

# ── LEFT: schematic spike-timing diagram (paper's Fig. 8 left panel) ───────────
ax_s = fig.add_subplot(gs[0, 0])

n_periods = 2
t_max = n_periods * T * 1.15

# Neuron A's spike train (forced every T)
A_spikes = [n*T for n in range(n_periods+1)]
for ts in A_spikes:
    ax_s.plot([ts, ts], [2.6, 3.0], color="#d62728", lw=2.2)
ax_s.text(-0.06*t_max, 2.8, r"$\nu_A$", fontsize=12, color="#d62728", va="center", ha="right")

# Diagonal lines showing spike propagation at speed v (distance D per period)
for ts in A_spikes:
    ax_s.plot([ts, ts + D/v], [2.6, 1.0], color="#d62728", lw=0.8, alpha=0.45, ls="-")

# B at d1 < D/2: weakens -> 0
y_d1 = 1.7
for ts in A_spikes:
    ax_s.plot([ts, ts], [y_d1-0.18, y_d1+0.18], color="0.3", lw=1.0, alpha=0.3)  # forced spike marker faint
B1_spikes = [n*T for n in range(n_periods+1)]
for ts in B1_spikes:
    ax_s.plot([ts, ts], [y_d1-0.2, y_d1+0.2], color="#1f77b4", lw=2.2)
ax_s.text(-0.06*t_max, y_d1, f"$\\nu_B$\n$d={d1:g}$", fontsize=10, color="#1f77b4", va="center", ha="right")
ax_s.annotate("", xy=(D/v*0 + d1/v, y_d1+0.35), xytext=(0, 2.6),
              arrowprops=dict(arrowstyle="->", color="0.4", lw=1.0, alpha=0.7))
ax_s.text(d1/v, y_d1+0.45, r"$\Delta t^s<0$"+"\n(depression)", fontsize=8,
          color="0.3", ha="center")

# B at d2 > D/2: strengthens -> 1
y_d2 = 0.6
B2_spikes = [n*T for n in range(n_periods+1)]
for ts in B2_spikes:
    ax_s.plot([ts, ts], [y_d2-0.2, y_d2+0.2], color="#1f77b4", lw=2.2)
ax_s.text(-0.06*t_max, y_d2, f"$\\nu_B$\n$d={d2:g}$", fontsize=10, color="#1f77b4", va="center", ha="right")
ax_s.annotate("", xy=(T + (d2/v - T), y_d2+0.35), xytext=(0, 2.6),
              arrowprops=dict(arrowstyle="->", color="0.4", lw=1.0, alpha=0.7))
ax_s.text(T*0.5, y_d2+0.5, r"$\Delta t^s>0$"+"\n(potentiation, next cycle)", fontsize=8,
          color="0.3", ha="center")

for n in range(n_periods+1):
    ax_s.axvline(n*T, color="0.85", lw=0.7, zorder=0)
    if n < n_periods:
        ax_s.text(n*T + T/2, 3.3, f"$t={n}T$" if n>0 else "$t=0$", fontsize=8, ha="center", color="0.5")

ax_s.set_xlim(-0.12*t_max, t_max)
ax_s.set_ylim(0, 3.6)
ax_s.set_yticks([])
ax_s.set_xlabel("Time", fontsize=10)
ax_s.set_title(f"Schematic: spike timing vs. distance\n(T={T:g}, v={v:g}, D=vT={D:g})", fontsize=10)
for spine in ["top", "right", "left"]:
    ax_s.spines[spine].set_visible(False)

# ── RIGHT: omega* vs distance (numerical, paper's Fig 8 right panel) ───────────
ax_n = fig.add_subplot(gs[0, 1])

ax_n.scatter(df["distance"], df["finalWeight"], s=14, color="0.15", zorder=3)

# Shade the periodic d < D/2 (blue/disconnect) vs d > D/2 (red/connect) bands
dmax = df["distance"].max()
n_full_periods = int(np.ceil(dmax / D)) + 1
for k in range(n_full_periods):
    lo, hi = k*D, k*D + D/2
    ax_n.axvspan(lo, hi, color="#1f77b4", alpha=0.07, zorder=0)
    lo2, hi2 = k*D + D/2, (k+1)*D
    ax_n.axvspan(lo2, hi2, color="#d62728", alpha=0.07, zorder=0)

# Mark D/2 boundaries
for k in range(n_full_periods+1):
    boundary = k*D + D/2
    if boundary <= dmax:
        ax_n.axvline(boundary, color="0.6", lw=0.7, ls="--", zorder=1)

ax_n.set_xlabel("Distance  $d$", fontsize=11)
ax_n.set_ylabel(r"$\omega^{*}$", fontsize=12)
ax_n.set_ylim(-0.05, 1.05)
ax_n.set_xlim(0, dmax)
ax_n.set_title(f"Numerical verification: final synaptic weight vs. distance\n"
              f"(T={T:g}, v={v:g}, repeats with period D=vT={D:g})", fontsize=10)
ax_n.spines["top"].set_visible(False)
ax_n.spines["right"].set_visible(False)

handles = [
    plt.Rectangle((0,0),1,1, color="#1f77b4", alpha=0.2, label=r"$d<D/2$ (predicted $\omega^*=0$)"),
    plt.Rectangle((0,0),1,1, color="#d62728", alpha=0.2, label=r"$d>D/2$ (predicted $\omega^*=1$)"),
]
ax_n.legend(handles=handles, loc="center right", fontsize=8, framealpha=0.85)

fig.suptitle("Figure 8 reproduction — Periodic synaptic weight patterning vs. distance",
            fontsize=13, y=1.02)

os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
fig.savefig(args.out, dpi=150, bbox_inches="tight")
print(f"Saved -> {args.out}")