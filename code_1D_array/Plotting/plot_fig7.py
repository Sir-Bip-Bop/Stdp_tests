"""
plot_fig7.py
============
Reproduces Figure 7 of the paper: fixed points omega* of the STDP dynamics
as a function of distance d between two neurons (under a periodic external
synchronizing signal of period T).

Workflow (mirrors the paper's methodology, Sec. III A-B):
  1. Load the (distance, omega, deltaOmega) sweep produced by fig7_sim.
  2. For each distance, fit deltaOmega(omega) to a degree-2 polynomial.
  3. Find roots of that polynomial within [0, 1] -- these are interior fixed
     points. Boundary fixed points at omega=0 / omega=1 exist if the curve
     stays negative near 0, or positive near 1, respectively (paper's
     "boundary effects" discussion).
  4. Classify each fixed point: stable if the slope of the curve there is
     negative, unstable if positive.
  5. Plot:
       - Main panel: omega* vs distance, color-coded (blue = stable at 0,
         red = stable at >0, open circles = unstable, half circles =
         bifurcation points).
       - Four example sub-panels showing deltaOmega(omega) curves at chosen
         distances, with arrows showing the flow direction (paper Fig. 7 A-D).

Usage
-----
  python plot_fig7.py --csv output_test/fig7_sweep.csv --out output_test/fig7.png
  python plot_fig7.py --examples 50 350 425 700   # pick your own 4 distances
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
parser.add_argument("--csv", type=str, default="Codigo/code_1D_array/output_figure7/fig7_sweep.csv")
parser.add_argument("--out", type=str, default="Codigo/code_1D_array/output_figure7/fig7.png")
parser.add_argument("--examples", type=float, nargs=4, default=None,
                    help="4 distances to use for the example sub-panels "
                         "(default: auto-pick from data spread)")
args = parser.parse_args()

if not os.path.exists(args.csv):
    sys.exit(f"Missing sweep file: {args.csv}. Run fig7_sim first.")

df = pd.read_csv(args.csv)
distances = sorted(df["distance"].unique())

# ── Fit each distance's curve to a degree-2 polynomial and find fixed points ───
def analyze_distance(d):
    sub = df[df["distance"] == d].sort_values("omega")
    omega = sub["omega"].values
    dOmega = sub["deltaOmega"].values

    if len(omega) < 3:
        return None

    coeffs = np.polyfit(omega, dOmega, 2)   # degree-2 fit, as in the paper
    poly = np.poly1d(coeffs)
    dpoly = poly.deriv()

    # Roots of the polynomial (could be 0, 1, or 2 real roots in general)
    roots = np.roots(coeffs)
    real_roots = [r.real for r in roots if abs(r.imag) < 1e-6 and -0.02 <= r.real <= 1.02]
    real_roots = sorted(set(round(r, 4) for r in real_roots))
    real_roots = [min(max(r, 0.0), 1.0) for r in real_roots]

    fixed_points = []   # list of (omega*, stability) stability in {'stable','unstable'}
    for r in real_roots:
        slope = dpoly(r)
        stability = "stable" if slope < 0 else "unstable"
        fixed_points.append((r, stability))

    # Boundary fixed points (paper's discussion after Eq. and before Sec III B):
    # stable at omega=0 if poly(0) < 0 (depression pushes toward 0, can't go below)
    # stable at omega=1 if poly(1) > 0 (potentiation pushes toward 1, can't go above)
    if poly(0) < 0 and not any(abs(r) < 1e-3 for r, _ in fixed_points):
        fixed_points.append((0.0, "stable"))
    if poly(1) > 0 and not any(abs(r - 1) < 1e-3 for r, _ in fixed_points):
        fixed_points.append((1.0, "stable"))

    return {
        "omega": omega, "dOmega": dOmega, "poly": poly,
        "fixed_points": sorted(fixed_points, key=lambda x: x[0])
    }

results = {d: analyze_distance(d) for d in distances}
results = {d: r for d, r in results.items() if r is not None}

# ── Build main panel data: scatter of all fixed points vs distance ─────────────
main_d, main_w, main_stable, main_color = [], [], [], []
for d, r in results.items():
    for (w, stab) in r["fixed_points"]:
        main_d.append(d)
        main_w.append(w)
        main_stable.append(stab == "stable")
        # Blue = stable disconnection (omega*=0), Red = stable connection (omega*>0)
        if stab == "stable":
            main_color.append("#1f77b4" if w < 0.05 else "#d62728")
        else:
            main_color.append("white")  # open circle face

# ── Pick example distances for sub-panels ───────────────────────────────────────
if args.examples:
    example_ds = list(args.examples)
else:
    dmin, dmax = min(distances), max(distances)
    example_ds = [dmin + 0.1*(dmax-dmin), dmin + 0.35*(dmax-dmin),
                  dmin + 0.45*(dmax-dmin), dmin + 0.7*(dmax-dmin)]

def nearest_distance(target):
    return min(distances, key=lambda d: abs(d - target))

example_ds = [nearest_distance(d) for d in example_ds]

# ── Figure layout: 1 main panel (bottom, wide) + 4 small panels (top row) ──────
fig = plt.figure(figsize=(13, 9))
gs = gridspec.GridSpec(2, 4, height_ratios=[1, 1.3], hspace=0.35, wspace=0.35)

panel_letters = ["A", "B", "C", "D"]
for i, d in enumerate(example_ds):
    ax = fig.add_subplot(gs[0, i])
    r = results[d]
    omega_fine = np.linspace(0, 1, 200)
    ax.plot(omega_fine, r["poly"](omega_fine), color="0.15", lw=1.3)
    ax.scatter(r["omega"], r["dOmega"], s=10, color="0.5", alpha=0.6, zorder=2)
    ax.axhline(0, color="k", lw=0.6, alpha=0.5)

    # Mark and color fixed points
    for (w, stab) in r["fixed_points"]:
        col = "#d62728" if (stab == "stable" and w > 0.05) else \
              ("#1f77b4" if stab == "stable" else "white")
        edge = "k" if stab != "stable" else col
        ax.scatter([w], [0], s=70, facecolor=col, edgecolor="k",
                  zorder=5, linewidth=1.1)

    # Flow direction arrows (green, like the paper)
    omega_arrows = np.linspace(0.08, 0.92, 6)
    for wa in omega_arrows:
        slope_sign = np.sign(r["poly"](wa))
        dx = 0.04 * slope_sign
        ax.annotate("", xy=(wa+dx, -ax.get_ylim()[1]*0 + r["dOmega"].min()*0 - (abs(r["dOmega"]).max()*0.18)),
                    xytext=(wa, -(abs(r["dOmega"]).max()*0.18)),
                    arrowprops=dict(arrowstyle="->", color="green", lw=1.2, alpha=0.85))

    ax.set_title(f"{panel_letters[i]}:  d={d:g}", fontsize=10)
    ax.set_xlabel(r"$\omega$", fontsize=9)
    if i == 0:
        ax.set_ylabel(r"$\Delta\omega$", fontsize=9)
    ax.tick_params(labelsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

# ── Main panel: omega* vs distance ──────────────────────────────────────────────
ax_main = fig.add_subplot(gs[1, :])

for d, r in results.items():
    for (w, stab) in r["fixed_points"]:
        if stab == "stable":
            col = "#1f77b4" if w < 0.05 else "#d62728"
            ax_main.scatter([d], [w], s=28, color=col, zorder=3)
        else:
            ax_main.scatter([d], [w], s=28, facecolor="white",
                           edgecolor="k", linewidth=1.0, zorder=3)

# Mark example distances with vertical dashed guides
for d, letter in zip(example_ds, panel_letters):
    ax_main.axvline(d, color="0.7", lw=0.8, ls=":", zorder=1)
    ax_main.text(d, 1.04, letter, ha="center", fontsize=9, color="0.4")

ax_main.set_xlabel("Distance", fontsize=11)
ax_main.set_ylabel(r"$\omega^{*}$", fontsize=12)
ax_main.set_ylim(-0.05, 1.1)
ax_main.spines["top"].set_visible(False)
ax_main.spines["right"].set_visible(False)

# Legend
handles = [
    plt.Line2D([0],[0], marker='o', color='w', markerfacecolor='#1f77b4',
              markersize=8, label='Stable ($\\omega^*=0$)'),
    plt.Line2D([0],[0], marker='o', color='w', markerfacecolor='#d62728',
              markersize=8, label='Stable ($\\omega^*>0$)'),
    plt.Line2D([0],[0], marker='o', color='w', markerfacecolor='white',
              markeredgecolor='k', markersize=8, label='Unstable'),
]
ax_main.legend(handles=handles, loc="upper left", fontsize=9, framealpha=0.85)

fig.suptitle(
    "Figure 7 reproduction — Stability points of the STDP dynamics vs. distance",
    fontsize=13, y=0.995
)

os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
fig.savefig(args.out, dpi=150, bbox_inches="tight")
print(f"Saved -> {args.out}")