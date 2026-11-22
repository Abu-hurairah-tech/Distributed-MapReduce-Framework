# Stage 4: Benchmarking & Performance Analysis

Benchmarking harness and plotting scripts for the MPI MapReduce framework (`wordfreq` and `invindex` workloads).

## Scripts

| Script | Purpose |
| --- | --- |
| `benchmarks/run_benchmarks.py` | Runs strong scaling, weak scaling, fault recovery, and `cProfile` experiments; writes raw JSONs to `benchmarks/results/raw/` and appends per-run rows to `benchmarks/results/*.csv`. |
| `benchmarks/plot_metrics.py` | Aggregates repeated runs by median with `pandas`, computes $S_p$, $E_p$, $T_{\text{comm}}/T_{\text{comp}}$, Karp–Flatt serial fraction $e(p)$, Amdahl's Law curve, weak efficiency, and recovery overhead; renders PNG plots and `benchmarks/results/summary.md`. |

## Usage

### 1. Run all experiments (quick mode — finishes in minutes)

```powershell
python benchmarks/run_benchmarks.py --quick --repeats 3 --reset --plot
```

### 2. Run individual experiments (subcommands or flags)

```powershell
# Strong scaling: fixed dataset, sequential T1 then mpiexec -n 2,3,5,9 (1, 2, 4, 8 workers)
python benchmarks/run_benchmarks.py strong --quick --repeats 3

# Weak scaling: dataset size proportional to worker count (1x, 2x, 4x, 8x base size)
python benchmarks/run_benchmarks.py weak --quick --repeats 3

# Fault recovery: same run with and without --fail-rank 1 --fail-after 1
python benchmarks/run_benchmarks.py fault --quick --repeats 3

# Profiling: cProfile on sequential baseline and per-rank MPI processes
python benchmarks/run_benchmarks.py profile --quick
```

### 3. Run at proposal scale (500 MB+)

```powershell
python benchmarks/run_benchmarks.py all --size-mb 500 --repeats 3 --plot
```

### 4. Generate plots and markdown summary table

```powershell
python benchmarks/plot_metrics.py --results-dir benchmarks/results
```

## Metrics & Formulas

- **Speedup:** $S_p = \dfrac{T_1}{T_p}$, where $T_1$ is the sequential baseline runtime (`baseline/run_sequential.py`) and $T_p$ is the $p$-worker MPI runtime (`mpiexec -n (p+1) python mr_driver.py`). Self-relative MPI speedup $S_p^{(\text{mpi})} = T_{\text{mpi},1} / T_p$ is also reported.
- **Parallel Efficiency:** $E_p = \dfrac{S_p}{p}$.
- **Communication-to-Computation Ratio:** $\dfrac{T_{\text{comm}}}{T_{\text{comp}}} = \dfrac{\sum \text{comm\_seconds}}{\sum \text{comp\_seconds}}$ across workers.
- **Karp–Flatt Serial Fraction ($p > 1$):**
  $$e(p) = \frac{\frac{1}{S_p} - \frac{1}{p}}{1 - \frac{1}{p}}$$
- **Amdahl's Law Curve (fitted to $\hat{e}$):**
  $$S_{\text{Amdahl}}(p) = \frac{1}{\hat{e} + \frac{1 - \hat{e}}{p}}$$
- **Weak Scaling Efficiency:** $E_{\text{weak}}(p) = \dfrac{T_1}{T_p}$ with dataset size $M(p) = p \cdot M_{\text{base}}$.
- **Fault Recovery Overhead:** $\Delta T = T_{\text{fault}} - T_{\text{healthy}}$ and $\text{Overhead (\%)} = \dfrac{T_{\text{fault}} - T_{\text{healthy}}}{T_{\text{healthy}}} \times 100\%$, alongside master-measured `recovery_seconds` and `redispatched_tasks`.

## Outputs (`benchmarks/results/`)

- `strong_scaling.csv`, `weak_scaling.csv`, `fault_recovery.csv`, `profile_runs.csv`, `profile_summary.csv`, `all_runs.csv`
- `derived_strong_metrics.csv`, `derived_weak_metrics.csv`, `derived_fault_metrics.csv`
- `strong_scaling_speedup.png`, `strong_scaling_efficiency.png`, `phase_breakdown_comm_comp.png`, `weak_scaling.png`, `fault_recovery.png`, `cprofile_hotspots.png`
- `summary.md` — Markdown tables and analysis summary for the final report.
