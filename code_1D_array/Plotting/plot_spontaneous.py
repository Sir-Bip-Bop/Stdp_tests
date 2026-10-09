"""
plot_spontaneous.py

Motivation
----------
Second half of Luis's spontaneous-activity experiment (runs produced by 1D_spontaneous.cpp, usually
through run_spontaneous_sweep.py). Every weight starts at omega, there is no external input, and
the questions are: (1) do the weights stabilise, either to fixed values or at least to a
stationary distance profile, or do they keep changing after a long time; (2) does a dependence of
the weight on distance (delay) emerge spontaneously; (3) how do (1) and (2) depend on omega and on
the delay, at fixed N. The weight-versus-distance plot from the report is the main readout, here
followed over time instead of only at the end.

What it measures
----------------
Per run (figure <figdir>/per_run/<name>.png, or <run>/summary.png with --run):
  * population rate (spikes per neuron per 1e4 steps) over the whole run and a raster of the last
    logged window, to tell sparse, cascading and self-sustained activity apart;
  * kymograph of the distance profile: mean weight at each distance |i-j| (both directions
    pooled) against time;
  * mean weight and fraction of weights at the bounds (< 0.01, > 0.99) against time;
  * weight versus distance at several times, the last one with a +-1 std band across pairs, plus
    the final mean pair asymmetry |w_ij - w_ji| (direction selection that pooling would hide);
  * final weight matrix W[pre][post];
  * convergence: RMS change of the full matrix and of the distance profile over a lag of 5 % of
    the run. A fixed point sends both to 0; a stationary but fluctuating state keeps the matrix
    change finite while the profile change drops to its noise floor.

Across the sweep (<figdir>/):
  * grid_profiles.png   omega (rows) x d0 (columns): weight versus distance at 10, 50, 75, 100 %
                        of the run, with regime and convergence labels;
  * delay_collapse.png  one panel per omega: last-quarter mean profiles of every d0 against the
                        physical delay |i-j|*d0 / sigma_STDP, testing whether delay alone sets the
                        profile;
  * grid_summary.png    heatmaps over (omega, d0): final mean weight, profile contrast, late
                        population rate (with regime), edge correlation between 75 % and 100 %;
  * sweep_summary.csv   one row per run with all scalar statistics below.

Definitions (late = last quarter of the snapshots, "third" = the quarter before it):
  regime          from per-neuron spike counts in 1000-step bins over the last quarter:
                  "sustained" if >= 99 % of bins have spikes and >= 90 % of bins have at least
                  half of the neurons active; "cascades" if >= 1 % of bins have half of the
                  neurons active; otherwise "sparse".
  profile_drift   RMS over distance of (mean late profile - mean third-quarter profile).
  edge_corr       Pearson correlation across all edges between W at 75 % and W at 100 % of the
                  run (1 means the individual weights no longer move).
  late_turnover   RMS over edges of W(100 %) - W(75 %): how far individual weights move in the
                  last quarter of the run.
  convergence     "frozen"     : late_turnover < --tol-frozen
                  "stationary" : profile_drift < --tol-profile and |mean-weight drift| < --tol-profile
                                 (individual weights may still turn over)
                  "drifting"   : neither.
  profile_contrast max - min of the late mean profile over distances with at least 20 pairs.

Usage
-----
    python Plotting/plot_spontaneous.py --sweep sweep_spont
    python Plotting/plot_spontaneous.py --sweep sweep_spont --no-per-run --figdir figs_spont
    python Plotting/plot_spontaneous.py --run sweep_spont/runs/w0.2500_d5.000_s1
    python Plotting/plot_spontaneous.py --run runA runB --tol-profile 0.01
"""

import argparse
import csv
import glob
import json
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors

INK = "#222222"
INK_MUTED = "#6b6b6b"
GRID = "#e3e3e3"

plt.rcParams.update({
    "axes.edgecolor": INK_MUTED, "axes.labelcolor": INK, "xtick.color": INK_MUTED,
    "ytick.color": INK_MUTED, "text.color": INK, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 9, "axes.titlesize": 10, "legend.frameon": False, "legend.fontsize": 8,
    "lines.linewidth": 1.6, "image.cmap": "viridis",
})


def seq_colors(n, cmap="Blues", lo=0.35, hi=1.0):
    """n colours from one sequential hue, light (early / small) to dark (late / large)."""
    cm = plt.get_cmap(cmap)
    return [cm(x) for x in np.linspace(lo, hi, max(n, 1))]


# -------------------------
# Loading
# -------------------------
def load_run(run_dir):
    with open(os.path.join(run_dir, "meta.json")) as f:
        meta = json.load(f)
    N = meta["N"]
    times = []
    with open(os.path.join(run_dir, "progress.csv")) as f:
        for row in csv.DictReader(f):
            times.append(int(row["t"]))
    times = np.array(times, dtype=np.int64)
    W = np.fromfile(os.path.join(run_dir, "weights.bin"), dtype=np.float32)
    n_snap = min(len(times), W.size // (N * N))
    W = W[: n_snap * N * N].reshape(n_snap, N, N)
    times = times[:n_snap]
    counts = np.fromfile(os.path.join(run_dir, "spike_counts.bin"), dtype=np.uint16)
    counts = counts[: (counts.size // N) * N].reshape(-1, N)
    sp_path = os.path.join(run_dir, "spikes.bin")
    spikes = np.fromfile(sp_path, dtype=np.int32).reshape(-1, 2) if os.path.exists(sp_path) else np.zeros((0, 2), np.int32)
    return dict(meta=meta, N=N, times=times, W=W, counts=counts, spikes=spikes,
                name=os.path.basename(os.path.normpath(run_dir)), dir=run_dir)


# -------------------------
# Analysis
# -------------------------
def distance_profiles(W):
    """Mean, std and pair asymmetry |w_ij - w_ji| per distance d = 1..N-1, for every snapshot."""
    n_snap, N, _ = W.shape
    mean = np.zeros((n_snap, N - 1))
    std = np.zeros((n_snap, N - 1))
    asym = np.zeros((n_snap, N - 1))
    for d in range(1, N):
        i = np.arange(N - d)
        j = i + d
        both = np.concatenate([W[:, i, j], W[:, j, i]], axis=1)   # forward and backward pooled
        mean[:, d - 1] = both.mean(axis=1)
        std[:, d - 1] = both.std(axis=1)
        asym[:, d - 1] = np.abs(W[:, i, j] - W[:, j, i]).mean(axis=1)
    return mean, std, asym


def activity_stats(counts, N, count_bin, frac_from=0.75):
    late = counts[int(frac_from * len(counts)):]
    if len(late) == 0:
        return dict(late_rate_per_1e4=np.nan, active_bin_frac=np.nan, big_bin_frac=np.nan, regime="n/a")
    active = (late > 0).sum(axis=1)
    active_frac = float((active > 0).mean())
    big_frac = float((active >= N / 2).mean())
    rate = float(late.sum() / N / (len(late) * count_bin / 1e4))
    if active_frac >= 0.99 and big_frac >= 0.9:
        regime = "sustained"
    elif big_frac >= 0.01:
        regime = "cascades"
    else:
        regime = "sparse"
    return dict(late_rate_per_1e4=rate, active_bin_frac=active_frac, big_bin_frac=big_frac, regime=regime)


def analyse(run, tol_frozen, tol_profile):
    W, N, times = run["W"], run["N"], run["times"]
    n = len(times)
    mean, std, asym = distance_profiles(W)
    off = ~np.eye(N, dtype=bool)
    mean_w = W[:, off].mean(axis=1)
    frac0 = (W[:, off] < 0.01).mean(axis=1)
    frac1 = (W[:, off] > 0.99).mean(axis=1)
    lag = max(1, int(round(0.05 * (n - 1))))
    if n > lag:
        dW = np.sqrt(((W[lag:] - W[:-lag]) ** 2)[:, off].mean(axis=1))
        dP = np.sqrt(((mean[lag:] - mean[:-lag]) ** 2).mean(axis=1))
    else:
        dW = dP = np.zeros(0)

    q3, q4 = int(0.5 * n), int(0.75 * n)
    q4 = min(q4, n - 1)
    late_prof = mean[q4:].mean(axis=0)
    third_prof = mean[q3:q4].mean(axis=0) if q4 > q3 else late_prof
    profile_drift = float(np.sqrt(((late_prof - third_prof) ** 2).mean()))
    mean_drift = float(mean_w[q4:].mean() - mean_w[q3:q4].mean()) if q4 > q3 else np.nan
    a, b = W[q4][off], W[-1][off]
    late_turnover = float(np.sqrt(((b - a) ** 2).mean()))
    edge_corr = float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else np.nan

    if late_turnover < tol_frozen:
        conv = "frozen"
    elif profile_drift < tol_profile and abs(mean_drift) < tol_profile:
        conv = "stationary"
    else:
        conv = "drifting"

    dmax_ok = N - 10 if N > 20 else N - 1          # distances with >= 20 pairs (both directions)
    contrast = float(late_prof[:dmax_ok].max() - late_prof[:dmax_ok].min())

    m = run["meta"]
    stats = dict(name=run["name"], omega=m["omega"], d0=m["d0"], seed=m["seed"], N=N, tmax=m["tmax"],
                 max_delay=m["max_delay"], status=m["status"], spikes_per_neuron=m.get("spikes_per_neuron"),
                 t_last=int(times[-1]), mean_w_final=float(mean_w[-1]), mean_w_late=float(mean_w[q4:].mean()),
                 mean_w_drift=mean_drift, frac0_final=float(frac0[-1]), frac1_final=float(frac1[-1]),
                 profile_contrast=contrast, profile_drift_rms=profile_drift, late_turnover_rms=late_turnover,
                 edge_corr_75_100=edge_corr, convergence=conv)
    stats.update(activity_stats(run["counts"], N, m["count_bin"]))
    series = dict(mean=mean, std=std, asym=asym, mean_w=mean_w, frac0=frac0, frac1=frac1,
                  dW=dW, dP=dP, lag=lag, late_prof=late_prof)
    return stats, series


# -------------------------
# Per-run figure
# -------------------------
def snapshot_indices(n, fracs):
    return sorted(set(min(n - 1, int(round(f * (n - 1)))) for f in fracs))


def per_run_figure(run, stats, s, path):
    N, times, meta = run["N"], run["times"], run["meta"]
    d0, sig = meta["d0"], meta["sigma_stdp"]
    tu = 1e6
    dist = np.arange(1, N)

    fig = plt.figure(figsize=(14, 11), layout="constrained")
    gs = fig.add_gridspec(3, 3, height_ratios=[0.8, 1, 1.1])

    # Population rate over the whole run
    ax = fig.add_subplot(gs[0, :2])
    c = run["counts"]
    if len(c):
        group = max(1, len(c) // 600)
        nb = (len(c) // group) * group
        pop = c[:nb].reshape(-1, group, N).sum(axis=(1, 2)) / N / (group * meta["count_bin"] / 1e4)
        tb = (np.arange(len(pop)) + 0.5) * group * meta["count_bin"]
        col = seq_colors(1)[0]
        ax.fill_between(tb / tu, 0, pop, color=col, alpha=0.35, lw=0)
        ax.plot(tb / tu, pop, color=col, lw=0.9)
        ax.set_ylim(0, None)
    ax.set_xlim(0, times[-1] / tu)
    ax.set_xlabel("time [10$^6$ steps]")
    ax.set_ylabel("spikes / neuron / 10$^4$ steps")
    ax.set_title(f"population rate   (late regime: {stats['regime']}; refractory limit "
                 f"{1e4 / (meta['refractory'] + 1):.0f})")

    # Raster of the last logged window
    ax = fig.add_subplot(gs[0, 2])
    sp = run["spikes"]
    wins = meta.get("raster_windows", [])
    if len(sp) and wins:
        a, b = wins[-1]
        sel = (sp[:, 0] >= a) & (sp[:, 0] < b)
        ax.scatter(sp[sel, 0] - a, sp[sel, 1], s=1.5, color=INK, linewidths=0)
        ax.set_xlim(0, b - a)
        ax.set_title(f"raster, last window (t0 = {a/tu:.2f}e6)")
    else:
        ax.text(0.5, 0.5, "no raw spikes logged", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("raster")
    ax.set_ylim(-1, N)
    ax.set_xlabel("time from t0 [steps]")
    ax.set_ylabel("neuron")
    ax.grid(False)

    # Kymograph of the distance profile
    ax = fig.add_subplot(gs[1, :2])
    im = ax.imshow(s["mean"].T, aspect="auto", origin="lower", vmin=0, vmax=1,
                   extent=[times[0] / tu, times[-1] / tu, 0.5, N - 0.5], interpolation="nearest")
    ax.set_xlabel("time [10$^6$ steps]")
    ax.set_ylabel("distance |i - j|")
    ax.set_title("mean weight at each distance over time")
    ax.grid(False)
    fig.colorbar(im, ax=ax, pad=0.01, label="mean weight")

    # Mean weight and bound fractions
    ax = fig.add_subplot(gs[1, 2])
    cols = seq_colors(3, "Greys", 0.45, 0.95)
    ax.plot(times / tu, s["mean_w"], color=seq_colors(1)[0], label="mean weight")
    ax.plot(times / tu, s["frac0"], color=cols[1], ls="--", label="fraction < 0.01")
    ax.plot(times / tu, s["frac1"], color=cols[2], ls=":", label="fraction > 0.99")
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("time [10$^6$ steps]")
    ax.set_title(f"mean weight (final {stats['mean_w_final']:.3f})")
    ax.legend(loc="best")

    # Weight vs distance at several times
    ax = fig.add_subplot(gs[2, 0])
    fr = [0, 0.1, 0.25, 0.5, 0.75, 1.0]
    idx = snapshot_indices(len(times), fr)
    colors = seq_colors(len(idx))
    for k, col in zip(idx, colors):
        ax.plot(dist, s["mean"][k], color=col, lw=1.4, label=f"t = {times[k]/tu:.2f}e6")
    k = idx[-1]
    ax.fill_between(dist, s["mean"][k] - s["std"][k], s["mean"][k] + s["std"][k], color=colors[-1], alpha=0.15, lw=0)
    ax.plot(dist, s["asym"][k], color=INK_MUTED, ls="--", lw=1.1, label="|w_ij - w_ji|, final")
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlim(0.5, N - 0.5)
    ax.set_xlabel("distance |i - j|")
    ax.set_ylabel("weight")
    ax.set_title("weight vs distance (band: +-1 std, final)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=3, fontsize=7)
    if d0 > 0:
        sec = ax.secondary_xaxis("top", functions=(lambda x: x * d0 / sig, lambda y: y * sig / d0))
        sec.set_xlabel("delay / $\\sigma_{STDP}$", fontsize=8)

    # Final weight matrix
    ax = fig.add_subplot(gs[2, 1])
    im = ax.imshow(run["W"][-1], origin="lower", vmin=0, vmax=1, interpolation="nearest")
    ax.set_xlabel("j (post-synaptic)")
    ax.set_ylabel("i (pre-synaptic)")
    ax.set_title("final weight matrix")
    ax.grid(False)
    fig.colorbar(im, ax=ax, pad=0.01, fraction=0.046)

    # Convergence
    ax = fig.add_subplot(gs[2, 2])
    if len(s["dW"]):
        tt = times[s["lag"]:] / tu
        c2 = seq_colors(2)
        ax.plot(tt, s["dW"], color=c2[1], lw=1.3, label="full matrix")
        ax.plot(tt, s["dP"], color=c2[0], lw=1.3, label="distance profile")
        ax.set_ylim(0, None)
        ax.legend(loc="best")
    ax.set_xlabel("time [10$^6$ steps]")
    ax.set_ylabel(f"RMS change over the last {s['lag'] * meta['snap_every']:.3g} steps")
    ax.set_title(f"convergence: {stats['convergence']}  (edge corr 75-100 %: {stats['edge_corr_75_100']:.2f})")

    status = "" if meta["status"] == "done" else f"   [status: {meta['status']}]"
    fig.suptitle(f"Spontaneous 1D array: N = {N}, omega = {meta['omega']:g}, d0 = {d0:g} "
                 f"(max delay {meta['max_delay']:g}), seed {meta['seed']}, tmax = {meta['tmax']:.3g}{status}",
                 fontsize=12)
    fig.savefig(path, dpi=130)
    plt.close(fig)


# -------------------------
# Sweep figures
# -------------------------
ABBR = {"sustained": "sust.", "cascades": "casc.", "sparse": "sparse", "frozen": "frozen",
        "stationary": "stat.", "drifting": "drift", "n/a": "n/a", "mixed": "mixed"}


def grid_axes_values(stats):
    omegas = sorted(set(r["omega"] for r in stats))
    d0s = sorted(set(r["d0"] for r in stats))
    return omegas, d0s


def by_cell(stats, series):
    cells = {}
    for st, se in zip(stats, series):
        cells.setdefault((st["omega"], st["d0"]), []).append((st, se))
    return cells


def grid_profiles_figure(stats, series, path):
    omegas, d0s = grid_axes_values(stats)
    cells = by_cell(stats, series)
    nr, nc = len(omegas), len(d0s)
    fig, axes = plt.subplots(nr, nc, figsize=(2.1 * nc + 1, 1.7 * nr + 1), sharex=True, sharey=True,
                             squeeze=False, layout="constrained")
    fr = [0.1, 0.5, 0.75, 1.0]
    colors = seq_colors(len(fr))
    for r, w in enumerate(omegas):
        for c, d in enumerate(d0s):
            ax = axes[r, c]
            ax.set_ylim(-0.03, 1.03)
            if (w, d) not in cells:
                ax.set_axis_off()
                continue
            group = cells[(w, d)]
            N = group[0][0]["N"]
            dist = np.arange(1, N)
            n = min(len(g[1]["mean"]) for g in group)
            for f, col in zip(fr, colors):
                k = min(n - 1, int(round(f * (n - 1))))
                prof = np.mean([g[1]["mean"][k] for g in group], axis=0)
                ax.plot(dist, prof, color=col, lw=1.1, label=f"{int(f*100)} %")
            st = group[0][0]
            ax.text(0.97, 0.95, f"{ABBR[st['regime']]} / {ABBR[st['convergence']]}", ha="right", va="top",
                    fontsize=6.5, color=INK_MUTED, transform=ax.transAxes)
            if r == 0:
                ax.set_title(f"d0 = {d:g}", fontsize=9)
            if c == 0:
                ax.set_ylabel(f"omega = {w:g}\nweight", fontsize=8)
            if r == nr - 1:
                ax.set_xlabel("|i - j|")
    handles, labels = next((a.get_legend_handles_labels() for a in axes.flat if a.lines), ([], []))
    try:
        fig.legend(handles, labels, title="snapshot, % of run", loc="outside lower center", ncol=len(fr))
    except (ValueError, TypeError):   # matplotlib < 3.7
        fig.legend(handles, labels, title="snapshot, % of run", loc="lower center", ncol=len(fr))
    fig.suptitle("Weight vs distance at 10, 50, 75, 100 % of the run\n"
                 "rows: initial omega, columns: nearest-neighbour delay d0, corner: activity regime / convergence",
                 fontsize=10)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def delay_collapse_figure(stats, series, path):
    omegas, d0s = grid_axes_values(stats)
    d0s = [d for d in d0s if d > 0]
    if not d0s:
        return
    cells = by_cell(stats, series)
    nc = min(4, len(omegas))
    nr = int(np.ceil(len(omegas) / nc))
    fig, axes = plt.subplots(nr, nc, figsize=(3.6 * nc, 2.8 * nr), sharey=True, squeeze=False, layout="constrained")
    colors = seq_colors(len(d0s), "Oranges", 0.35, 1.0)
    for k, w in enumerate(omegas):
        ax = axes[k // nc, k % nc]
        for d, col in zip(d0s, colors):
            if (w, d) not in cells:
                continue
            group = cells[(w, d)]
            N, sig = group[0][0]["N"], 100.0
            sig = group[0][1].get("sigma", sig)
            prof = np.mean([g[1]["late_prof"] for g in group], axis=0)
            delay = np.arange(1, N) * d / sig
            ax.plot(delay, prof, color=col, lw=1.3, label=f"d0 = {d:g}")
        ax.set_xscale("log")
        ax.set_ylim(-0.03, 1.03)
        ax.set_title(f"omega = {w:g}")
        ax.set_xlabel("delay |i - j|*d0 / $\\sigma_{STDP}$")
        if k % nc == 0:
            ax.set_ylabel("weight (last-quarter mean)")
    for k in range(len(omegas), nr * nc):
        axes[k // nc, k % nc].set_axis_off()
    axes[0, 0].legend(loc="best", fontsize=7)
    fig.suptitle("Does the delay alone set the profile? Late weight vs physical delay, all d0 overlaid", fontsize=11)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def grid_summary_figure(stats, path):
    omegas, d0s = grid_axes_values(stats)

    def cell_mean(key):
        M = np.full((len(omegas), len(d0s)), np.nan)
        for r, w in enumerate(omegas):
            for c, d in enumerate(d0s):
                v = [st[key] for st in stats if st["omega"] == w and st["d0"] == d and st[key] is not None]
                v = [x for x in v if np.isfinite(x)]
                if v:
                    M[r, c] = np.mean(v)
        return M

    def cell_label(key):
        L = [["" for _ in d0s] for _ in omegas]
        for r, w in enumerate(omegas):
            for c, d in enumerate(d0s):
                v = [st[key] for st in stats if st["omega"] == w and st["d0"] == d]
                if v:
                    L[r][c] = v[0] if len(set(v)) == 1 else "mixed"
        return L

    panels = [
        ("mean_w_final", "final mean weight", dict(vmin=0, vmax=1), None),
        ("profile_contrast", "profile contrast (max - min, late)", dict(vmin=0), None),
        ("late_rate_per_1e4", "late rate, spikes/neuron/10$^4$ steps", dict(norm="log"), "regime"),
        ("edge_corr_75_100", "edge correlation, W(75 %) vs W(100 %)", dict(vmin=0, vmax=1), "convergence"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(4 + 0.9 * len(d0s) * 2, 2.5 + 0.55 * len(omegas) * 2),
                             squeeze=False, layout="constrained")
    for ax, (key, title, kw, lab) in zip(axes.flat, panels):
        M = cell_mean(key)
        kw = dict(kw)
        if kw.get("norm") == "log":
            pos = M[np.isfinite(M) & (M > 0)]
            kw = dict(norm=mcolors.LogNorm(vmin=pos.min(), vmax=pos.max())) if pos.size else {}
        im = ax.imshow(M, origin="lower", aspect="auto", cmap="viridis", **kw)
        fig.colorbar(im, ax=ax, pad=0.01)
        L = cell_label(lab) if lab else None
        for r in range(len(omegas)):
            for c in range(len(d0s)):
                if not np.isfinite(M[r, c]):
                    continue
                txt = f"{M[r, c]:.2g}" + (f"\n{ABBR.get(L[r][c], L[r][c])}" if L else "")
                rgba = im.cmap(im.norm(M[r, c]))
                lum = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
                ax.text(c, r, txt, ha="center", va="center", fontsize=6.5, color="white" if lum < 0.5 else INK)
        ax.set_xticks(range(len(d0s)), [f"{d:g}" for d in d0s])
        ax.set_yticks(range(len(omegas)), [f"{w:g}" for w in omegas])
        ax.set_xlabel("d0 (nearest-neighbour delay)")
        ax.set_ylabel("initial omega")
        ax.set_title(title)
        ax.grid(False)
    fig.suptitle("Spontaneous sweep summary (cells averaged over seeds)", fontsize=11)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def write_summary_csv(stats, path):
    keys = list(stats[0].keys())
    with open(path, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=keys)
        wr.writeheader()
        for st in sorted(stats, key=lambda x: (x["omega"], x["d0"], x["seed"])):
            wr.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in st.items()})


# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--sweep", help="sweep folder produced by run_spontaneous_sweep.py")
    g.add_argument("--run", nargs="+", help="one or more single-run folders")
    ap.add_argument("--figdir", default=None, help="default: <sweep>/figures, or each run folder with --run")
    ap.add_argument("--no-per-run", action="store_true", help="skip the per-run summary figures")
    ap.add_argument("--tol-frozen", type=float, default=0.01, help="RMS change W(75 %%) -> W(100 %%) below which weights count as frozen")
    ap.add_argument("--tol-profile", type=float, default=0.02, help="profile / mean-weight drift below which the profile counts as stationary")
    args = ap.parse_args()

    if args.run:
        run_dirs = args.run
    else:
        run_dirs = sorted(d for d in glob.glob(os.path.join(args.sweep, "runs", "*"))
                          if os.path.exists(os.path.join(d, "meta.json")))
    if not run_dirs:
        raise SystemExit("no runs found")

    all_stats, all_series = [], []
    for rd in run_dirs:
        try:
            run = load_run(rd)
        except (OSError, ValueError, KeyError) as e:
            print(f"[skip] {rd}: {e}")
            continue
        if len(run["times"]) < 2:
            print(f"[skip] {rd}: fewer than 2 snapshots so far")
            continue
        st, se = analyse(run, args.tol_frozen, args.tol_profile)
        se["sigma"] = run["meta"]["sigma_stdp"]
        all_stats.append(st)
        all_series.append(se)
        print(f"{st['name']:24s} {st['status']:8s} regime={st['regime']:9s} conv={st['convergence']:10s} "
              f"mean_w={st['mean_w_final']:.3f} contrast={st['profile_contrast']:.3f} "
              f"drift={st['profile_drift_rms']:.4f} turnover={st['late_turnover_rms']:.3f} edge_corr={st['edge_corr_75_100']:.2f}")
        if not args.no_per_run:
            if args.run and not args.figdir:
                out = os.path.join(rd, "summary.png")
            else:
                figdir = args.figdir or os.path.join(args.sweep, "figures")
                os.makedirs(os.path.join(figdir, "per_run"), exist_ok=True)
                out = os.path.join(figdir, "per_run", st["name"] + ".png")
            per_run_figure(run, st, se, out)
        del run

    if not all_stats:
        raise SystemExit("nothing to analyse")
    figdir = args.figdir or (os.path.join(args.sweep, "figures") if args.sweep else os.path.dirname(os.path.normpath(run_dirs[0])))
    os.makedirs(figdir, exist_ok=True)
    write_summary_csv(all_stats, os.path.join(figdir, "sweep_summary.csv"))
    if len(all_stats) > 1:
        grid_profiles_figure(all_stats, all_series, os.path.join(figdir, "grid_profiles.png"))
        delay_collapse_figure(all_stats, all_series, os.path.join(figdir, "delay_collapse.png"))
        grid_summary_figure(all_stats, os.path.join(figdir, "grid_summary.png"))
    print(f"figures and sweep_summary.csv in {figdir}")


if __name__ == "__main__":
    main()
