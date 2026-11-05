"""
benchmark_cpu_gpu.py
====================
CPU vs GPU FerQ Simulated Annealing — Google Colab benchmark.

Usage in Colab
--------------
1.  Runtime → Change runtime type → T4 GPU
2.  Upload ferq_gpu.py to /content/ (or your Drive)
3.  Run cells top-to-bottom

What this script measures
--------------------------
• Wall-clock solve time for CPU Numba SA vs GPU CUDA SA
  on synthetic random k-SAT HOBO problems.
• Varies: n_vars (50 → 1000) and num_reads (50, 256, 1024).
• Reports: solve time [ms], best energy, GPU speedup.
• Produces: a styled results table + a speedup heatmap.
"""

# ═══════════════════════════════════════════════════════════════════════════
# CELL 1 — Install / imports
# ═══════════════════════════════════════════════════════════════════════════
# @title ⚙️ Setup

# Uncomment if numba / matplotlib are not already installed:
# !pip install -q numba matplotlib

import sys, os, time, math, random, warnings
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from typing import Dict, List, Tuple

# ── load ferq_gpu ────────────────────────────────────────────────────────────
# Adjust the path below if ferq_gpu.py lives in a different directory.
FERQ_GPU_PATH = "/content"     # Colab default upload path
if FERQ_GPU_PATH not in sys.path:
    sys.path.insert(0, FERQ_GPU_PATH)

from ferq_gpu import (
    FERQOptimizerGPU,
    build_flat_arrays,
    warmup_gpu,
    gpu_info,
    _CUDA_AVAILABLE,
    _NUMBA_CPU,
)

print("=" * 60)
print("FerQ  CPU vs GPU  Benchmark")
print("=" * 60)
print(f"  CUDA available : {_CUDA_AVAILABLE}")
print(f"  Numba CPU JIT  : {_NUMBA_CPU}")
if _CUDA_AVAILABLE:
    print(f"  GPU device     : {gpu_info()}")
print("=" * 60)


# ═══════════════════════════════════════════════════════════════════════════
# CELL 2 — Problem generator  (no reducers dependency)
# ═══════════════════════════════════════════════════════════════════════════
# @title 🧩 Problem Generator

def _expand_clause_poly(clause: List[Tuple[int, bool]]) -> Dict[frozenset, float]:
    """
    Convert one k-SAT clause into a multilinear polynomial over {0,1}^n.

    For a clause  (x_{i1} ∨ ¬x_{i2} ∨ x_{i3})  the violated assignment is
    x_{i1}=0, x_{i2}=1, x_{i3}=0, which contributes +1 to the penalty.
    We expand  ∏_{j : negated} x_j · ∏_{j : not negated} (1-x_j)  and negate.

    Energy = 0  iff all clauses satisfied.
    Energy > 0  counts unsatisfied clauses (weighted).
    """
    # Start with the constant polynomial 1, then multiply factor by factor.
    poly: Dict[frozenset, float] = {frozenset(): 1.0}

    for var_idx, is_negated in clause:
        new_poly: Dict[frozenset, float] = {}
        if is_negated:
            # factor = x_{var_idx}
            for term, coeff in poly.items():
                new_term = frozenset(term | {var_idx})
                new_poly[new_term] = new_poly.get(new_term, 0.0) + coeff
        else:
            # factor = (1 - x_{var_idx})
            for term, coeff in poly.items():
                new_poly[term] = new_poly.get(term, 0.0) + coeff
                new_term = frozenset(term | {var_idx})
                new_poly[new_term] = new_poly.get(new_term, 0.0) - coeff
        poly = new_poly

    return poly   # this is the "violated" polynomial; penalty = -poly


def generate_ksat_hobo(
    n_vars: int,
    k: int = 3,
    clause_density: float = 4.0,
    seed: int = 0,
) -> Tuple[Dict[frozenset, float], int]:
    """
    Generate a random k-SAT instance and return its HOBO polynomial.

    The HOBO energy equals the number of *unsatisfied* clauses, so the
    global minimum is 0 (all clauses satisfied) when the instance is SAT.

    Parameters
    ----------
    n_vars        : number of boolean variables
    k             : clause width (default 3-SAT)
    clause_density: ratio m/n (default 4.0 — near phase transition for 3-SAT)
    seed          : RNG seed

    Returns
    -------
    hobo_terms : Dict[frozenset, float]
    n_clauses  : int
    """
    rng      = random.Random(seed)
    n_clauses = int(n_vars * clause_density)
    terms: Dict[frozenset, float] = {}

    for _ in range(n_clauses):
        vars_in_clause = rng.sample(range(n_vars), min(k, n_vars))
        clause         = [(v, rng.random() < 0.5) for v in vars_in_clause]

        # Clause penalty polynomial: +1 when clause is violated
        clause_poly = _expand_clause_poly(clause)
        # Sum violated-product into global terms
        for term, coeff in clause_poly.items():
            terms[term] = terms.get(term, 0.0) + coeff

    # Remove near-zero coefficients
    terms = {t: v for t, v in terms.items() if abs(v) > 1e-12}
    return terms, n_clauses


# Quick sanity check
_t, _m = generate_ksat_hobo(20, k=3, seed=42)
print(f"Sanity check — n_vars=20, n_clauses={_m}, n_hobo_terms={len(_t)}")


# ═══════════════════════════════════════════════════════════════════════════
# CELL 3 — Benchmark helpers
# ═══════════════════════════════════════════════════════════════════════════
# @title 🔬 Benchmark Functions

def run_once(
    hobo_terms: Dict,
    n_vars: int,
    num_reads: int,
    num_sweeps: int,
    beta_range: Tuple[float, float],
    seed: int,
    device: str,         # "cpu" or "gpu"
) -> Tuple[float, float]:
    """
    Run one solve and return (energy, wall_ms).

    wall_ms includes device transfers (GPU) but NOT JIT compilation.
    """
    opt = FERQOptimizerGPU(
        n_vars     = n_vars,
        num_reads  = num_reads,
        num_sweeps = num_sweeps,
        beta_range = beta_range,
        seed       = seed,
    )
    opt.set_terms(hobo_terms)

    if device == "gpu":
        _, energy, ms = opt.solve_gpu()
    else:
        _, energy, ms = opt.solve_cpu()
    return energy, ms


def benchmark_config(
    n_vars: int,
    num_reads: int,
    num_sweeps: int,
    k: int                        = 3,
    clause_density: float         = 4.0,
    beta_range: Tuple[float,float]= (0.1, 10.0),
    n_trials: int                 = 3,
    problem_seed: int             = 0,
    solver_seed: int              = 42,
) -> Dict:
    """
    Run n_trials on a single (n_vars, num_reads) configuration and return
    aggregated timing / quality statistics.
    """
    hobo_terms, n_clauses = generate_ksat_hobo(
        n_vars, k=k, clause_density=clause_density, seed=problem_seed
    )
    n_terms = len(hobo_terms)

    results = {"n_vars": n_vars, "num_reads": num_reads,
               "n_clauses": n_clauses, "n_terms": n_terms}

    for device in (["cpu", "gpu"] if _CUDA_AVAILABLE else ["cpu"]):
        times, energies = [], []
        for trial in range(n_trials):
            e, ms = run_once(
                hobo_terms, n_vars, num_reads, num_sweeps,
                beta_range, solver_seed + trial, device,
            )
            times.append(ms)
            energies.append(e)

        results[f"{device}_time_mean_ms"]   = float(np.mean(times))
        results[f"{device}_time_min_ms"]    = float(np.min(times))
        results[f"{device}_energy_mean"]    = float(np.mean(energies))
        results[f"{device}_energy_best"]    = float(np.min(energies))

    if _CUDA_AVAILABLE and "cpu_time_mean_ms" in results and "gpu_time_mean_ms" in results:
        results["speedup_mean"] = results["cpu_time_mean_ms"] / max(results["gpu_time_mean_ms"], 1e-9)
        results["speedup_min"]  = results["cpu_time_min_ms"]  / max(results["gpu_time_min_ms"],  1e-9)

    return results


def print_results_table(rows: List[Dict]) -> None:
    """Pretty-print a results table to stdout."""
    if not rows:
        return
    has_gpu = _CUDA_AVAILABLE

    hdr = (
        f"{'n_vars':>8} {'reads':>7} {'n_terms':>8} "
        f"{'CPU ms':>10} {'CPU E':>9}"
    )
    if has_gpu:
        hdr += f" {'GPU ms':>10} {'GPU E':>9} {'Speedup':>8}"
    print("\n" + "─" * len(hdr))
    print(hdr)
    print("─" * len(hdr))

    for r in rows:
        line = (
            f"{r['n_vars']:>8} {r['num_reads']:>7} {r['n_terms']:>8} "
            f"{r['cpu_time_mean_ms']:>10.1f} {r['cpu_energy_best']:>9.2f}"
        )
        if has_gpu:
            line += (
                f" {r.get('gpu_time_mean_ms', 0):>10.1f}"
                f" {r.get('gpu_energy_best', 0):>9.2f}"
                f" {r.get('speedup_mean', 0):>7.1f}×"
            )
        print(line)

    print("─" * len(hdr))


# ═══════════════════════════════════════════════════════════════════════════
# CELL 4 — Warm up CUDA JIT (must do BEFORE any timed run)
# ═══════════════════════════════════════════════════════════════════════════
# @title 🔥 GPU Warmup (compile CUDA kernel)

if _CUDA_AVAILABLE:
    print("Warming up CUDA kernel (one-time JIT compilation)…", end=" ", flush=True)
    t_wu = time.perf_counter()
    warmup_gpu(n_vars=30, num_reads=16, num_sweeps=200)
    print(f"done in {(time.perf_counter() - t_wu):.1f} s")
else:
    print("No GPU — skipping warmup.")

# Also trigger CPU Numba JIT compilation once
print("Warming up CPU Numba kernel…", end=" ", flush=True)
t_wu = time.perf_counter()
_dummy_terms = {frozenset({i, (i+1)%10, (i+2)%10}): 1.0 for i in range(8)}
_opt_warmup  = FERQOptimizerGPU(n_vars=10, num_reads=4, num_sweeps=100)
_opt_warmup.set_terms(_dummy_terms)
_opt_warmup.solve_cpu()
print(f"done in {(time.perf_counter() - t_wu):.1f} s")


# ═══════════════════════════════════════════════════════════════════════════
# CELL 5 — Run the benchmark
# ═══════════════════════════════════════════════════════════════════════════
# @title 🚀 Run Benchmark

# ── Configuration ─────────────────────────────────────────────────────────
N_VARS_LIST   = [50, 100, 200, 500, 1000]    # problem sizes to sweep
NUM_READS_LIST = [50, 256, 1024]             # reads per config
NUM_SWEEPS     = 2000                        # SA sweeps per read (fixed)
K              = 3                           # SAT clause width
CLAUSE_DENSITY = 4.0                         # clauses / variables
N_TRIALS       = 3                           # repeats per config (take mean)
BETA_RANGE     = (0.1, 10.0)
# ──────────────────────────────────────────────────────────────────────────

all_rows: List[Dict] = []
total_configs = len(N_VARS_LIST) * len(NUM_READS_LIST)
done = 0

print(f"\nRunning {total_configs} configurations "
      f"({N_TRIALS} trials each)…\n")

for n_vars in N_VARS_LIST:
    for num_reads in NUM_READS_LIST:
        done += 1
        print(f"  [{done:>2}/{total_configs}]  n_vars={n_vars:>4}  "
              f"num_reads={num_reads:>4} …", end=" ", flush=True)
        t0  = time.perf_counter()
        row = benchmark_config(
            n_vars        = n_vars,
            num_reads     = num_reads,
            num_sweeps    = NUM_SWEEPS,
            k             = K,
            clause_density= CLAUSE_DENSITY,
            beta_range    = BETA_RANGE,
            n_trials      = N_TRIALS,
        )
        elapsed = time.perf_counter() - t0
        spd = f"  speedup {row.get('speedup_mean', 0):.1f}×" if _CUDA_AVAILABLE else ""
        print(f"done in {elapsed:.1f}s{spd}")
        all_rows.append(row)

print_results_table(all_rows)


# ═══════════════════════════════════════════════════════════════════════════
# CELL 6 — Visualisation
# ═══════════════════════════════════════════════════════════════════════════
# @title 📊 Plots

def make_plots(rows: List[Dict]) -> None:
    """
    Produce two figures:

    Fig 1 — Runtime [ms] vs n_vars for each num_reads setting,
            CPU (solid) vs GPU (dashed).  Log-y scale.

    Fig 2 — GPU speedup heatmap: rows = n_vars, cols = num_reads.
    """
    n_vars_vals   = sorted(set(r["n_vars"]    for r in rows))
    num_reads_vals = sorted(set(r["num_reads"] for r in rows))

    # ── palette ─────────────────────────────────────────────────────────────
    colors = plt.cm.tab10.colors

    # ════════════════════════════════════════════════════════════════════════
    # Figure 1 — Runtime
    # ════════════════════════════════════════════════════════════════════════
    fig1, axes = plt.subplots(
        1, len(num_reads_vals),
        figsize=(5.5 * len(num_reads_vals), 5),
        sharey=True,
    )
    if len(num_reads_vals) == 1:
        axes = [axes]

    fig1.suptitle(
        "FerQ SA — Solve Time vs Problem Size\n"
        f"(num_sweeps={NUM_SWEEPS}, {N_TRIALS} trials, mean)",
        fontsize=13, fontweight="bold",
    )

    for ax_idx, num_reads in enumerate(num_reads_vals):
        ax = axes[ax_idx]
        sub = [r for r in rows if r["num_reads"] == num_reads]

        nv   = [r["n_vars"]             for r in sub]
        cpu  = [r["cpu_time_mean_ms"]   for r in sub]
        ax.plot(nv, cpu, "o-", color=colors[0],
                linewidth=2, markersize=7, label="CPU (Numba JIT)")

        if _CUDA_AVAILABLE:
            gpu = [r.get("gpu_time_mean_ms", 0) for r in sub]
            ax.plot(nv, gpu, "s--", color=colors[1],
                    linewidth=2, markersize=7, label="GPU (CUDA)")

        ax.set_title(f"num_reads = {num_reads}", fontsize=11)
        ax.set_xlabel("n_vars", fontsize=10)
        if ax_idx == 0:
            ax.set_ylabel("Time [ms]  (mean over 3 runs)", fontsize=10)
        ax.set_yscale("log")
        ax.set_xscale("log")
        ax.set_xticks(nv)
        ax.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
        ax.legend(fontsize=9)
        ax.grid(True, which="both", alpha=0.3)

    plt.tight_layout()
    plt.savefig("/content/ferq_runtime.png", dpi=150, bbox_inches="tight")
    print("Saved: /content/ferq_runtime.png")
    plt.show()

    # ════════════════════════════════════════════════════════════════════════
    # Figure 2 — Speedup heatmap  (only when GPU is available)
    # ════════════════════════════════════════════════════════════════════════
    if not _CUDA_AVAILABLE:
        print("Skipping speedup heatmap (no GPU).")
        return

    speedup_matrix = np.zeros((len(n_vars_vals), len(num_reads_vals)))
    for i, nv in enumerate(n_vars_vals):
        for j, nr in enumerate(num_reads_vals):
            matches = [r for r in rows if r["n_vars"] == nv and r["num_reads"] == nr]
            if matches:
                speedup_matrix[i, j] = matches[0].get("speedup_mean", 0)

    fig2, ax = plt.subplots(figsize=(6, 4.5))
    im = ax.imshow(speedup_matrix, aspect="auto", cmap="RdYlGn",
                   vmin=0, vmax=max(speedup_matrix.max(), 1))

    ax.set_xticks(range(len(num_reads_vals)))
    ax.set_xticklabels([str(x) for x in num_reads_vals])
    ax.set_yticks(range(len(n_vars_vals)))
    ax.set_yticklabels([str(x) for x in n_vars_vals])
    ax.set_xlabel("num_reads",  fontsize=11)
    ax.set_ylabel("n_vars",     fontsize=11)
    ax.set_title(
        f"GPU Speedup over CPU Numba\n"
        f"({gpu_info()}, num_sweeps={NUM_SWEEPS})",
        fontsize=11, fontweight="bold",
    )

    for i in range(len(n_vars_vals)):
        for j in range(len(num_reads_vals)):
            v = speedup_matrix[i, j]
            ax.text(j, i, f"{v:.1f}×",
                    ha="center", va="center", fontsize=11,
                    color="black" if v < speedup_matrix.max() * 0.7 else "white",
                    fontweight="bold")

    cbar = fig2.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Speedup (CPU ms / GPU ms)", fontsize=9)

    plt.tight_layout()
    plt.savefig("/content/ferq_speedup_heatmap.png", dpi=150, bbox_inches="tight")
    print("Saved: /content/ferq_speedup_heatmap.png")
    plt.show()

    # ── Figure 3 — Speedup vs num_reads for each n_vars ───────────────────
    fig3, ax = plt.subplots(figsize=(7, 4.5))
    for i, nv in enumerate(n_vars_vals):
        sub = [r for r in rows if r["n_vars"] == nv]
        nr  = [r["num_reads"]         for r in sub]
        sp  = [r.get("speedup_mean", 0) for r in sub]
        ax.plot(nr, sp, "o-", color=colors[i % len(colors)],
                linewidth=2, markersize=7, label=f"n_vars={nv}")

    ax.axhline(1.0, color="gray", linestyle="--", linewidth=1, alpha=0.6,
               label="No speedup (1×)")
    ax.set_xlabel("num_reads",          fontsize=11)
    ax.set_ylabel("GPU Speedup (×)",    fontsize=11)
    ax.set_title("GPU Speedup vs num_reads\n"
                 f"({gpu_info()})",
                 fontsize=11, fontweight="bold")
    ax.set_xscale("log")
    ax.legend(fontsize=9)
    ax.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig("/content/ferq_speedup_vs_reads.png", dpi=150, bbox_inches="tight")
    print("Saved: /content/ferq_speedup_vs_reads.png")
    plt.show()


make_plots(all_rows)


# ═══════════════════════════════════════════════════════════════════════════
# CELL 7 — Summary statistics
# ═══════════════════════════════════════════════════════════════════════════
# @title 📋 Summary

print("\n" + "=" * 60)
print("BENCHMARK SUMMARY")
print("=" * 60)

if _CUDA_AVAILABLE:
    max_speedup_row = max(all_rows, key=lambda r: r.get("speedup_mean", 0))
    min_speedup_row = min(all_rows, key=lambda r: r.get("speedup_mean", 0))
    all_speedups    = [r.get("speedup_mean", 0) for r in all_rows]

    print(f"  GPU device     : {gpu_info()}")
    print(f"  Max speedup    : {max_speedup_row['speedup_mean']:.1f}× "
          f"(n_vars={max_speedup_row['n_vars']}, num_reads={max_speedup_row['num_reads']})")
    print(f"  Min speedup    : {min_speedup_row['speedup_mean']:.1f}× "
          f"(n_vars={min_speedup_row['n_vars']}, num_reads={min_speedup_row['num_reads']})")
    print(f"  Median speedup : {np.median(all_speedups):.1f}×")
    print()
    print("  GPU under-utilised with num_reads=50 (few threads).")
    print("  Use num_reads ≥ 256 to fill GPU warps and maximise speedup.")
else:
    print("  No CUDA GPU detected.  Showing CPU-only timing.")
    cpu_times = [r["cpu_time_mean_ms"] for r in all_rows]
    print(f"  CPU time range : {min(cpu_times):.1f} – {max(cpu_times):.1f} ms")

print()
print("  Saved plots:")
print("    /content/ferq_runtime.png")
if _CUDA_AVAILABLE:
    print("    /content/ferq_speedup_heatmap.png")
    print("    /content/ferq_speedup_vs_reads.png")
print("=" * 60)


# ═══════════════════════════════════════════════════════════════════════════
# CELL 8 — (Optional) Integration test: verify GPU ≈ CPU energy
# ═══════════════════════════════════════════════════════════════════════════
# @title ✅ Correctness Check

print("\nRunning correctness check (GPU and CPU should find similar energy)…")

TEST_N_VARS    = 100
TEST_NUM_READS = 256
TEST_SWEEPS    = 3000
terms_test, _  = generate_ksat_hobo(TEST_N_VARS, k=3, seed=999)

opt_test = FERQOptimizerGPU(
    n_vars     = TEST_N_VARS,
    num_reads  = TEST_NUM_READS,
    num_sweeps = TEST_SWEEPS,
    seed       = 42,
)
opt_test.set_terms(terms_test)

x_cpu, e_cpu, ms_cpu = opt_test.solve_cpu()
print(f"  CPU  energy = {e_cpu:.4f}   time = {ms_cpu:.1f} ms")

if _CUDA_AVAILABLE:
    x_gpu, e_gpu, ms_gpu = opt_test.solve_gpu()
    print(f"  GPU  energy = {e_gpu:.4f}   time = {ms_gpu:.1f} ms")
    rel_diff = abs(e_cpu - e_gpu) / (abs(e_cpu) + 1e-9)
    status   = "✓ PASS" if rel_diff < 0.05 else "⚠ DIFFER"
    print(f"  Relative energy diff: {rel_diff * 100:.2f}%  →  {status}")
    print(f"  Speedup: {ms_cpu / ms_gpu:.1f}×")
else:
    print("  (No GPU — skipping GPU check)")