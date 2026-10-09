"""
run_spontaneous_sweep.py

Motivation
----------
Luis's experiment: start a 1D array with every weight equal to omega, let it run spontaneously
(no external stimulus) for a long time and check whether the weights stabilise, then repeat for
several omega and several delays at fixed N. This script runs that grid with 1D_spontaneous.cpp.

The grid has to straddle the two scales that change the activity qualitatively (derived in the
docstring of 1D_spontaneous.cpp), otherwise every run lands in the same regime:
  * omega around 1/(N-1) (~0.01 for N = 100), where a spike stops causing on average less than
    one further spike and starts recruiting the array (no leak in v, all-to-all coupling);
  * d0 around refractory/(N-1) (~2 for N = 100), above which the far end of the array can
    re-excite a neuron after its refractory period, so cascades can sustain themselves.
d0 = 0 is the exact zero-delay limit (STDP should do nothing systematic with distance there) and
omega = 0 starts from an empty matrix.

What it measures
----------------
Nothing by itself. It compiles the driver if needed, runs every (omega, d0, seed) combination in
parallel, one process per run, skips runs whose meta.json already says "done" with the same
parameters (so an interrupted sweep resumes), and writes:
  <outdir>/runs/w<omega>_d<d0>_s<seed>/   one folder per run (see 1D_spontaneous.cpp)
  <outdir>/runs/.../run.log              stdout/stderr of that run
  <outdir>/sweep_index.csv               one row per run: parameters, status, wall time, spikes
Analyse afterwards with Plotting/plot_spontaneous.py.

Rough cost on one core with N = 100 (measured on a short pilot): about 5 us per step while the
activity is sparse or bursty, about 80 us per step once the array is self-sustained (large omega,
d0 >= 3). With tmax = 1e7 that is about 1 minute and about 15 minutes per run respectively.

Usage
-----
    python run_spontaneous_sweep.py --outdir sweep_spont
    python run_spontaneous_sweep.py --outdir sweep_spont --tmax 20000000 --seeds 1 2 3 --jobs 20
    python run_spontaneous_sweep.py --omegas 0.01 0.25 --d0s 0 1 5 --tmax 200000 --outdir pilot   # quick pilot
    python run_spontaneous_sweep.py --extra "--packet-sigma 1.5" --outdir sweep_packets
    python run_spontaneous_sweep.py --dry-run
"""

import argparse
import csv
import json
import os
import shlex
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__))
EXE_SUFFIX = ".exe" if os.name == "nt" else ""

DEFAULT_OMEGAS = [0.0, 0.005, 0.01, 0.02, 0.05, 0.1, 0.25, 0.5]
DEFAULT_D0S = [0.0, 0.1, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0]


def run_name(omega, d0, seed):
    return f"w{omega:.4f}_d{d0:.3f}_s{seed}"


def needs_build(exe, sources):
    if not os.path.exists(exe):
        return True
    t_exe = os.path.getmtime(exe)
    return any(os.path.getmtime(s) > t_exe for s in sources if os.path.exists(s))


def build(src, exe, cxx, flags):
    sources = [src, os.path.join(os.path.dirname(src), "rwNeuron.h")]
    if not needs_build(exe, sources):
        print(f"[build] {exe} is up to date")
        return
    cmd = [cxx] + shlex.split(flags) + ["-o", exe, src]
    print("[build] " + " ".join(cmd))
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        sys.exit("[build] compilation failed:\n" + res.stderr)


def is_done(run_dir, omega, d0, seed, args):
    meta_path = os.path.join(run_dir, "meta.json")
    if not os.path.exists(meta_path):
        return False
    try:
        with open(meta_path) as f:
            m = json.load(f)
    except (OSError, json.JSONDecodeError):
        return False
    same = (m.get("status") == "done" and m.get("N") == args.N and m.get("tmax") == args.tmax
            and abs(m.get("omega", -1) - omega) < 1e-12 and abs(m.get("d0", -1) - d0) < 1e-12
            and m.get("seed") == seed)
    return same


def run_one(exe, run_dir, omega, d0, seed, args):
    os.makedirs(run_dir, exist_ok=True)
    cmd = [exe, "--omega", repr(omega), "--d0", repr(d0), "--N", str(args.N),
           "--tmax", str(args.tmax), "--seed", str(seed), "--outdir", run_dir, "--quiet"]
    if args.snap_every:
        cmd += ["--snap-every", str(args.snap_every)]
    if args.extra:
        cmd += shlex.split(args.extra)
    t0 = time.time()
    with open(os.path.join(run_dir, "run.log"), "w") as log:
        log.write(" ".join(cmd) + "\n")
        log.flush()
        res = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
    return res.returncode, time.time() - t0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--outdir", default="sweep_spont")
    ap.add_argument("--omegas", type=float, nargs="+", default=DEFAULT_OMEGAS)
    ap.add_argument("--d0s", type=float, nargs="+", default=DEFAULT_D0S, help="nearest-neighbour delays")
    ap.add_argument("--seeds", type=int, nargs="+", default=[1])
    ap.add_argument("--N", type=int, default=100, help="kept fixed across the whole sweep")
    ap.add_argument("--tmax", type=int, default=10_000_000)
    ap.add_argument("--snap-every", type=int, default=0, help="0 -> driver default (tmax/400)")
    ap.add_argument("--extra", default="", help='extra driver arguments, e.g. "--packet-sigma 1.5"')
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--src", default=os.path.join(HERE, "1D_spontaneous.cpp"))
    ap.add_argument("--exe", default=os.path.join(HERE, "1D_spontaneous" + EXE_SUFFIX))
    ap.add_argument("--cxx", default="g++")
    ap.add_argument("--cxxflags", default="-O3 -std=c++17")
    ap.add_argument("--force", action="store_true", help="rerun even if a run is already done")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    runs_dir = os.path.join(args.outdir, "runs")
    grid = [(w, d, s) for w in args.omegas for d in args.d0s for s in args.seeds]
    # Longest runs first (self-sustained activity: large omega and large delays) for load balance.
    grid.sort(key=lambda x: (x[1] * (args.N - 1) > 200, x[0], x[1]), reverse=True)

    todo, skipped = [], []
    for w, d, s in grid:
        rd = os.path.join(runs_dir, run_name(w, d, s))
        if not args.force and is_done(rd, w, d, s, args):
            skipped.append((w, d, s))
        else:
            todo.append((w, d, s))

    print(f"[sweep] N={args.N} tmax={args.tmax:.3g} omegas={args.omegas} d0s={args.d0s} seeds={args.seeds}")
    print(f"[sweep] {len(grid)} runs: {len(todo)} to do, {len(skipped)} already done, {args.jobs} parallel jobs")
    if args.dry_run:
        for w, d, s in todo:
            print("   ", run_name(w, d, s))
        return

    build(args.src, args.exe, args.cxx, args.cxxflags)
    os.makedirs(runs_dir, exist_ok=True)

    t_start = time.time()
    failures = 0
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futs = {pool.submit(run_one, args.exe, os.path.join(runs_dir, run_name(w, d, s)), w, d, s, args): (w, d, s)
                for w, d, s in todo}
        for k, fut in enumerate(as_completed(futs), 1):
            w, d, s = futs[fut]
            code, wall = fut.result()
            failures += code != 0
            tag = "ok" if code == 0 else f"FAILED (exit {code}, see run.log)"
            print(f"[{k}/{len(todo)}] {run_name(w, d, s)}  {wall/60:.1f} min  {tag}   "
                  f"(elapsed {(time.time()-t_start)/60:.1f} min)", flush=True)

    # Index of the whole grid (done and not done)
    rows = []
    for w, d, s in sorted(grid):
        rd = os.path.join(runs_dir, run_name(w, d, s))
        row = dict(name=run_name(w, d, s), omega=w, d0=d, seed=s, N=args.N, tmax=args.tmax,
                   status="missing", wall_seconds="", spikes_per_neuron="")
        try:
            with open(os.path.join(rd, "meta.json")) as f:
                m = json.load(f)
            row.update(status=m.get("status"), wall_seconds=m.get("wall_seconds"),
                       spikes_per_neuron=m.get("spikes_per_neuron"))
        except (OSError, json.JSONDecodeError):
            pass
        rows.append(row)
    with open(os.path.join(args.outdir, "sweep_index.csv"), "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)
    print(f"[sweep] finished, {failures} failures. Index: {os.path.join(args.outdir, 'sweep_index.csv')}")
    print(f"[sweep] next: python Plotting/plot_spontaneous.py --sweep {args.outdir}")


if __name__ == "__main__":
    main()
