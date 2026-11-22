"""Compute MapReduce scaling/fault metrics, generate PNG plots, and write summary.md.

Consumes the CSVs produced by `benchmarks/run_benchmarks.py` in `benchmarks/results/`
using `pandas` and `matplotlib`.

Computed Metrics
----------------
* Speedup:            S_p = T_1 / T_p
* Efficiency:         E_p = S_p / p
* Comm-to-Comp Ratio: T_comm / T_comp
* Karp-Flatt Metric:  e(p) = (1/S_p - 1/p) / (1 - 1/p)   for p > 1
* Amdahl's Law Fit:   S_Amdahl(p) = 1 / (e_fit + (1 - e_fit) / p)
* Weak Efficiency:    E_weak(p) = T_1 / T_p   (with dataset size proportional to p)
* Recovery Overhead:  Delta T = T_fault - T_healthy,  Overhead % = (Delta T / T_healthy) * 100%
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


JOB_LABELS = {
    "wordfreq": "Word Frequency",
    "invindex": "Inverted Index",
}

JOB_COLORS = {
    "wordfreq": "#1f77b4",
    "invindex": "#d62728",
}


def karp_flatt(sp: float, p: int) -> float:
    """Compute the Karp-Flatt experimentally determined serial fraction e(p) for p > 1."""
    if p <= 1 or sp <= 0 or math.isnan(sp):
        return float("nan")
    return ((1.0 / sp) - (1.0 / p)) / (1.0 - (1.0 / p))


def amdahl_speedup(p: float, e_serial: float) -> float:
    """Compute Amdahl's Law speedup S(p) = 1 / (e + (1 - e)/p)."""
    if p <= 0 or math.isnan(e_serial):
        return float("nan")
    e_clamped = max(0.0, min(1.0, e_serial))
    denom = e_clamped + (1.0 - e_clamped) / float(p)
    return (1.0 / denom) if denom > 0 else float("nan")


def fit_serial_fraction(e_values: List[float]) -> float:
    """Fit a single serial fraction e_hat from observed Karp-Flatt values across p > 1."""
    valid = [v for v in e_values if not math.isnan(v) and math.isfinite(v)]
    if not valid:
        return 0.1
    pos = [v for v in valid if v > 0.0]
    pool = pos if pos else valid
    s = sorted(pool)
    mid = len(s) // 2
    med = s[mid] if len(s) % 2 == 1 else 0.5 * (s[mid - 1] + s[mid])
    return max(0.001, min(0.999, med))


def clean_rank_str(val: Any) -> str:
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return ""
    s = str(val).strip()
    if s in ("", "[]", "nan", "None"):
        return ""
    if s.endswith(".0"):
        s = s[:-2]
    return s


def load_csv_if_exists(csv_path: Path, fallback_all: Optional[pd.DataFrame], exp_name: str) -> pd.DataFrame:
    if csv_path.exists() and csv_path.stat().st_size > 0:
        return pd.read_csv(csv_path)
    if fallback_all is not None and not fallback_all.empty and "experiment" in fallback_all.columns:
        sub = fallback_all[fallback_all["experiment"] == exp_name].copy()
        return sub
    return pd.DataFrame()


def aggregate_strong_scaling(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Dict[str, float]]]:
    """Aggregate strong scaling repeats by median and compute Sp, Ep, Tcomm/Tcomp, Karp-Flatt, Amdahl."""
    if df.empty:
        return pd.DataFrame(), {}

    numeric_cols = [
        "workers",
        "mpi_procs",
        "target_size_mb",
        "input_bytes",
        "split_size_bytes",
        "num_splits",
        "total_seconds",
        "map_seconds",
        "shuffle_seconds",
        "reduce_seconds",
        "comp_seconds",
        "comm_seconds",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    grouped = (
        df.groupby(["job", "engine", "mpi_procs", "workers"], as_index=False)
        .agg(
            repeats=("total_seconds", "count"),
            target_size_mb=("target_size_mb", "median"),
            input_bytes=("input_bytes", "median"),
            split_size_bytes=("split_size_bytes", "median"),
            num_splits=("num_splits", "median"),
            total_seconds=("total_seconds", "median"),
            total_min=("total_seconds", "min"),
            total_max=("total_seconds", "max"),
            map_seconds=("map_seconds", "median"),
            shuffle_seconds=("shuffle_seconds", "median"),
            reduce_seconds=("reduce_seconds", "median"),
            comp_seconds=("comp_seconds", "median"),
            comm_seconds=("comm_seconds", "median"),
        )
    )

    # Sort: wordfreq then invindex; sequential first, then mpi by workers ascending
    grouped["job_order"] = grouped["job"].map({"wordfreq": 0, "invindex": 1}).fillna(2)
    grouped["engine_order"] = grouped["engine"].map({"sequential": 0, "mpi": 1}).fillna(2)
    grouped = (
        grouped.sort_values(["job_order", "engine_order", "workers"])
        .drop(columns=["job_order", "engine_order"])
        .reset_index(drop=True)
    )

    out_frames: List[pd.DataFrame] = []
    fit_summary: Dict[str, Dict[str, float]] = {}

    for job, g in grouped.groupby("job", sort=False):
        g = g.copy()
        seq_rows = g[g["engine"] == "sequential"]
        mpi1_rows = g[(g["engine"] == "mpi") & (g["workers"] == 1)]

        t1_seq = float(seq_rows["total_seconds"].iloc[0]) if not seq_rows.empty else float(g["total_seconds"].iloc[0])
        t1_mpi = float(mpi1_rows["total_seconds"].iloc[0]) if not mpi1_rows.empty else t1_seq

        g["t1_seq_seconds"] = t1_seq
        g["t1_mpi_seconds"] = t1_mpi
        g["speedup_sp"] = t1_seq / g["total_seconds"]
        g["efficiency_ep"] = g["speedup_sp"] / g["workers"]
        g["speedup_mpi"] = t1_mpi / g["total_seconds"]
        g["efficiency_mpi"] = g["speedup_mpi"] / g["workers"]
        g["comm_comp_ratio"] = g.apply(
            lambda r: (float(r["comm_seconds"]) / float(r["comp_seconds"]))
            if float(r["comp_seconds"]) > 0
            else 0.0,
            axis=1,
        )
        g["karp_flatt_e"] = g.apply(
            lambda r: karp_flatt(float(r["speedup_sp"]), int(r["workers"]))
            if r["engine"] == "mpi" and int(r["workers"]) > 1
            else float("nan"),
            axis=1,
        )
        g["karp_flatt_e_mpi"] = g.apply(
            lambda r: karp_flatt(float(r["speedup_mpi"]), int(r["workers"]))
            if r["engine"] == "mpi" and int(r["workers"]) > 1
            else float("nan"),
            axis=1,
        )

        e_vals_seq = [float(x) for x in g["karp_flatt_e"].dropna().tolist()]
        e_vals_mpi = [float(x) for x in g["karp_flatt_e_mpi"].dropna().tolist()]
        e_fit_seq = fit_serial_fraction(e_vals_seq)
        e_fit_mpi = fit_serial_fraction(e_vals_mpi)
        raw_med_seq = float(pd.Series(e_vals_seq).median()) if e_vals_seq else float("nan")
        raw_med_mpi = float(pd.Series(e_vals_mpi).median()) if e_vals_mpi else float("nan")

        g["amdahl_fitted_e"] = e_fit_seq
        g["amdahl_speedup_sp"] = g["workers"].apply(lambda p: amdahl_speedup(float(p), e_fit_seq))
        g["amdahl_fitted_e_mpi"] = e_fit_mpi
        g["amdahl_speedup_mpi"] = g["workers"].apply(lambda p: amdahl_speedup(float(p), e_fit_mpi))

        fit_summary[str(job)] = {
            "t1_seq": t1_seq,
            "t1_mpi": t1_mpi,
            "e_fit_seq": e_fit_seq,
            "e_raw_median_seq": raw_med_seq,
            "e_fit_mpi": e_fit_mpi,
            "e_raw_median_mpi": raw_med_mpi,
            "max_amdahl_speedup_seq": 1.0 / e_fit_seq if e_fit_seq > 0 else float("nan"),
            "max_amdahl_speedup_mpi": 1.0 / e_fit_mpi if e_fit_mpi > 0 else float("nan"),
        }
        out_frames.append(g)

    return pd.concat(out_frames, ignore_index=True), fit_summary


def aggregate_weak_scaling(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate weak scaling repeats by median and compute weak efficiency & Gustafson speedup."""
    if df.empty:
        return pd.DataFrame()

    numeric_cols = [
        "workers",
        "mpi_procs",
        "target_size_mb",
        "input_bytes",
        "split_size_bytes",
        "num_splits",
        "total_seconds",
        "map_seconds",
        "shuffle_seconds",
        "reduce_seconds",
        "comp_seconds",
        "comm_seconds",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    grouped = (
        df.groupby(["job", "engine", "mpi_procs", "workers", "target_size_mb"], as_index=False)
        .agg(
            repeats=("total_seconds", "count"),
            input_bytes=("input_bytes", "median"),
            split_size_bytes=("split_size_bytes", "median"),
            num_splits=("num_splits", "median"),
            total_seconds=("total_seconds", "median"),
            total_min=("total_seconds", "min"),
            total_max=("total_seconds", "max"),
            map_seconds=("map_seconds", "median"),
            shuffle_seconds=("shuffle_seconds", "median"),
            reduce_seconds=("reduce_seconds", "median"),
            comp_seconds=("comp_seconds", "median"),
            comm_seconds=("comm_seconds", "median"),
        )
    )
    grouped["job_order"] = grouped["job"].map({"wordfreq": 0, "invindex": 1}).fillna(2)
    grouped["engine_order"] = grouped["engine"].map({"sequential": 0, "mpi": 1}).fillna(2)
    grouped = (
        grouped.sort_values(["job_order", "engine_order", "workers"])
        .drop(columns=["job_order", "engine_order"])
        .reset_index(drop=True)
    )

    out_frames: List[pd.DataFrame] = []
    for _job, g in grouped.groupby("job", sort=False):
        g = g.copy()
        seq_rows = g[g["engine"] == "sequential"]
        mpi1_rows = g[(g["engine"] == "mpi") & (g["workers"] == 1)]
        t1_seq = float(seq_rows["total_seconds"].iloc[0]) if not seq_rows.empty else float(g["total_seconds"].iloc[0])
        t1_mpi = float(mpi1_rows["total_seconds"].iloc[0]) if not mpi1_rows.empty else t1_seq

        g["weak_efficiency_seq"] = t1_seq / g["total_seconds"]
        g["weak_efficiency_mpi"] = t1_mpi / g["total_seconds"]
        g["scaled_speedup_seq"] = g["workers"] * g["weak_efficiency_seq"]
        g["scaled_speedup_mpi"] = g["workers"] * g["weak_efficiency_mpi"]
        g["throughput_mb_s"] = (g["input_bytes"] / (1024.0 * 1024.0)) / g["total_seconds"]
        g["comm_comp_ratio"] = g.apply(
            lambda r: (float(r["comm_seconds"]) / float(r["comp_seconds"]))
            if float(r["comp_seconds"]) > 0
            else 0.0,
            axis=1,
        )
        out_frames.append(g)

    return pd.concat(out_frames, ignore_index=True)


def aggregate_fault_recovery(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate fault recovery runs and compute absolute & relative recovery overhead."""
    if df.empty:
        return pd.DataFrame()

    df = df.copy()
    df["failed_ranks_str"] = df["failed_ranks"].apply(clean_rank_str)
    df["mode"] = df["failed_ranks_str"].apply(lambda s: "fault_injected" if s != "" else "healthy")

    numeric_cols = [
        "workers",
        "mpi_procs",
        "target_size_mb",
        "input_bytes",
        "num_splits",
        "total_seconds",
        "map_seconds",
        "shuffle_seconds",
        "reduce_seconds",
        "comp_seconds",
        "comm_seconds",
        "redispatched_tasks",
        "recovery_seconds",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    grouped = (
        df.groupby(["job", "mpi_procs", "workers", "mode", "failed_ranks_str"], as_index=False)
        .agg(
            repeats=("total_seconds", "count"),
            target_size_mb=("target_size_mb", "median"),
            input_bytes=("input_bytes", "median"),
            num_splits=("num_splits", "median"),
            total_seconds=("total_seconds", "median"),
            total_min=("total_seconds", "min"),
            total_max=("total_seconds", "max"),
            map_seconds=("map_seconds", "median"),
            shuffle_seconds=("shuffle_seconds", "median"),
            reduce_seconds=("reduce_seconds", "median"),
            comp_seconds=("comp_seconds", "median"),
            comm_seconds=("comm_seconds", "median"),
            redispatched_tasks=("redispatched_tasks", "median"),
            recovery_seconds=("recovery_seconds", "median"),
        )
    )

    out_frames: List[pd.DataFrame] = []
    for (_job, _procs), g in grouped.groupby(["job", "mpi_procs"], sort=False):
        g = g.copy()
        healthy = g[g["mode"] == "healthy"]
        t_healthy = float(healthy["total_seconds"].iloc[0]) if not healthy.empty else float("nan")
        g["healthy_baseline_seconds"] = t_healthy
        g["overhead_seconds"] = g["total_seconds"] - t_healthy
        g["overhead_pct"] = g["overhead_seconds"].apply(
            lambda d: (d / t_healthy * 100.0) if t_healthy > 0 and not math.isnan(t_healthy) else 0.0
        )
        out_frames.append(g)

    out_df = pd.concat(out_frames, ignore_index=True)
    out_df["job_order"] = out_df["job"].map({"wordfreq": 0, "invindex": 1}).fillna(2)
    out_df["mode_order"] = out_df["mode"].map({"healthy": 0, "fault_injected": 1}).fillna(2)
    out_df = (
        out_df.sort_values(["job_order", "mpi_procs", "mode_order"])
        .drop(columns=["job_order", "mode_order"])
        .reset_index(drop=True)
    )
    return out_df


def plot_strong_scaling_speedup(
    strong_df: pd.DataFrame,
    fit_summary: Dict[str, Dict[str, float]],
    out_path: Path,
) -> None:
    if strong_df.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.4), dpi=300)
    p_dense = [1 + 0.1 * i for i in range(0, 71)]  # 1.0 .. 8.0

    # Left panel: Speedup vs Sequential Baseline T1 (Proposal definition Sp = T1 / Tp)
    ax0 = axes[0]
    ax0.plot(
        [1, 2, 4, 8],
        [1, 2, 4, 8],
        linestyle=":",
        color="#666666",
        linewidth=1.5,
        label="Ideal Linear ($S_p = p$)",
    )
    for job, g in strong_df.groupby("job", sort=False):
        mpi_g = g[g["engine"] == "mpi"].sort_values("workers")
        if mpi_g.empty:
            continue
        color = JOB_COLORS.get(str(job), "#1f77b4")
        label = JOB_LABELS.get(str(job), str(job))
        ax0.plot(
            mpi_g["workers"],
            mpi_g["speedup_sp"],
            marker="o",
            linewidth=2.2,
            color=color,
            label=f"{label} Measured ($T_{{seq,1}} / T_p$)",
        )
        e_fit = fit_summary.get(str(job), {}).get("e_fit_seq", 0.1)
        amdahl_y = [amdahl_speedup(p, e_fit) for p in p_dense]
        ax0.plot(
            p_dense,
            amdahl_y,
            linestyle="--",
            linewidth=1.6,
            color=color,
            alpha=0.8,
            label=f"{label} Amdahl Fit ($\\hat{{e}}={e_fit:.3f}$)",
        )

    ax0.set_title("Strong Scaling Speedup vs. Sequential Baseline ($T_1$)", fontsize=11, fontweight="bold")
    ax0.set_xlabel("MPI Worker Count $p$ (ranks $1..N-1$)", fontsize=10)
    ax0.set_ylabel("Speedup $S_p = T_1 / T_p$", fontsize=10)
    ax0.set_xticks([1, 2, 4, 8])
    ax0.grid(True, linestyle="--", alpha=0.35)
    ax0.legend(fontsize=8.2, loc="upper left")

    # Right panel: Self-Relative MPI Speedup (T_{mpi,1} / T_p) & Amdahl Fit
    ax1 = axes[1]
    ax1.plot(
        [1, 2, 4, 8],
        [1, 2, 4, 8],
        linestyle=":",
        color="#666666",
        linewidth=1.5,
        label="Ideal Linear ($S_p = p$)",
    )
    for job, g in strong_df.groupby("job", sort=False):
        mpi_g = g[g["engine"] == "mpi"].sort_values("workers")
        if mpi_g.empty:
            continue
        color = JOB_COLORS.get(str(job), "#1f77b4")
        label = JOB_LABELS.get(str(job), str(job))
        ax1.plot(
            mpi_g["workers"],
            mpi_g["speedup_mpi"],
            marker="s",
            linewidth=2.2,
            color=color,
            label=f"{label} Measured ($T_{{mpi,1}} / T_p$)",
        )
        e_fit_mpi = fit_summary.get(str(job), {}).get("e_fit_mpi", 0.1)
        amdahl_y = [amdahl_speedup(p, e_fit_mpi) for p in p_dense]
        ax1.plot(
            p_dense,
            amdahl_y,
            linestyle="--",
            linewidth=1.6,
            color=color,
            alpha=0.8,
            label=f"{label} Amdahl Fit ($\\hat{{e}}_{{mpi}}={e_fit_mpi:.3f}$)",
        )

    ax1.set_title("Self-Relative MPI Scaling ($T_{mpi,1} / T_p$) & Amdahl Fit", fontsize=11, fontweight="bold")
    ax1.set_xlabel("MPI Worker Count $p$ (ranks $1..N-1$)", fontsize=10)
    ax1.set_ylabel("MPI Relative Speedup $S_p^{(mpi)} = T_{mpi,1} / T_p$", fontsize=10)
    ax1.set_xticks([1, 2, 4, 8])
    ax1.grid(True, linestyle="--", alpha=0.35)
    ax1.legend(fontsize=8.2, loc="upper left")

    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_strong_scaling_efficiency(strong_df: pd.DataFrame, out_path: Path) -> None:
    if strong_df.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), dpi=300)

    # Left: Efficiency Ep = Sp / p
    ax0 = axes[0]
    ax0.axhline(1.0, linestyle=":", color="#666666", linewidth=1.4, label="Ideal Efficiency ($E_p = 1.0$)")
    for job, g in strong_df.groupby("job", sort=False):
        mpi_g = g[g["engine"] == "mpi"].sort_values("workers")
        if mpi_g.empty:
            continue
        color = JOB_COLORS.get(str(job), "#1f77b4")
        label = JOB_LABELS.get(str(job), str(job))
        ax0.plot(
            mpi_g["workers"],
            mpi_g["efficiency_ep"],
            marker="o",
            linewidth=2.2,
            color=color,
            label=f"{label} $E_p$ (vs. Seq $T_1$)",
        )
        ax0.plot(
            mpi_g["workers"],
            mpi_g["efficiency_mpi"],
            marker="s",
            linestyle="--",
            linewidth=1.8,
            color=color,
            alpha=0.75,
            label=f"{label} $E_p^{{(mpi)}}$ (vs. MPI 1W)",
        )

    ax0.set_title("Parallel Efficiency $E_p = S_p / p$", fontsize=11, fontweight="bold")
    ax0.set_xlabel("MPI Worker Count $p$", fontsize=10)
    ax0.set_ylabel("Parallel Efficiency $E_p$", fontsize=10)
    ax0.set_xticks([1, 2, 4, 8])
    ax0.grid(True, linestyle="--", alpha=0.35)
    ax0.legend(fontsize=8.2)

    # Right: Karp-Flatt Serial Fraction e(p) for p in {2, 4, 8}
    ax1 = axes[1]
    for job, g in strong_df.groupby("job", sort=False):
        mpi_g = g[(g["engine"] == "mpi") & (g["workers"] > 1)].sort_values("workers")
        if mpi_g.empty:
            continue
        color = JOB_COLORS.get(str(job), "#1f77b4")
        label = JOB_LABELS.get(str(job), str(job))
        ax1.plot(
            mpi_g["workers"],
            mpi_g["karp_flatt_e"],
            marker="o",
            linewidth=2.2,
            color=color,
            label=f"{label} $e(p)$ (vs. Seq $T_1$)",
        )
        ax1.plot(
            mpi_g["workers"],
            mpi_g["karp_flatt_e_mpi"],
            marker="s",
            linestyle="--",
            linewidth=1.8,
            color=color,
            alpha=0.75,
            label=f"{label} $e_{{mpi}}(p)$ (vs. MPI 1W)",
        )

    ax1.set_title("Karp–Flatt Serial Fraction $e(p) = \\frac{1/S_p - 1/p}{1 - 1/p}$", fontsize=11, fontweight="bold")
    ax1.set_xlabel("MPI Worker Count $p > 1$", fontsize=10)
    ax1.set_ylabel("Experimental Serial Fraction $e(p)$", fontsize=10)
    ax1.set_xticks([2, 4, 8])
    ax1.grid(True, linestyle="--", alpha=0.35)
    ax1.legend(fontsize=8.2)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_phase_breakdown_and_comm_comp(strong_df: pd.DataFrame, out_path: Path) -> None:
    if strong_df.empty:
        return

    jobs = list(strong_df["job"].unique())
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.4), dpi=300)

    # Left: Stacked bar chart of Map / Shuffle / Reduce wall-clock seconds
    ax0 = axes[0]
    bar_labels: List[str] = []
    map_vals: List[float] = []
    shuf_vals: List[float] = []
    red_vals: List[float] = []

    for job in jobs:
        g = strong_df[strong_df["job"] == job]
        short_job = "WF" if job == "wordfreq" else "II"
        for _, row in g.iterrows():
            eng = "Seq" if row["engine"] == "sequential" else f"MPI-{int(row['workers'])}W"
            bar_labels.append(f"{short_job}\n{eng}")
            map_vals.append(float(row["map_seconds"]))
            shuf_vals.append(float(row["shuffle_seconds"]))
            red_vals.append(float(row["reduce_seconds"]))

    x_pos = list(range(len(bar_labels)))
    ax0.bar(x_pos, map_vals, label="Map Phase", color="#2b5c8f")
    ax0.bar(x_pos, shuf_vals, bottom=map_vals, label="Shuffle Phase (Alltoallv)", color="#e68a00")
    bottoms_red = [m + s for m, s in zip(map_vals, shuf_vals)]
    ax0.bar(x_pos, red_vals, bottom=bottoms_red, label="Reduce & Gather Phase", color="#2e8b57")

    ax0.set_title("Master Wall-Clock Phase Breakdown (Map / Shuffle / Reduce)", fontsize=11, fontweight="bold")
    ax0.set_ylabel("Wall-Clock Time (seconds)", fontsize=10)
    ax0.set_xticks(x_pos)
    ax0.set_xticklabels(bar_labels, fontsize=8)
    ax0.grid(True, axis="y", linestyle="--", alpha=0.35)
    ax0.legend(fontsize=8.5)

    # Right: Communication-to-Computation Ratio T_comm / T_comp vs workers
    ax1 = axes[1]
    for job, g in strong_df.groupby("job", sort=False):
        mpi_g = g[g["engine"] == "mpi"].sort_values("workers")
        if mpi_g.empty:
            continue
        color = JOB_COLORS.get(str(job), "#1f77b4")
        label = JOB_LABELS.get(str(job), str(job))
        ax1.plot(
            mpi_g["workers"],
            mpi_g["comm_comp_ratio"],
            marker="o",
            linewidth=2.2,
            color=color,
            label=f"{label} ($T_{{comm}} / T_{{comp}}$)",
        )
        for _, r in mpi_g.iterrows():
            ax1.annotate(
                f"{float(r['comm_comp_ratio']):.3f}",
                (float(r["workers"]), float(r["comm_comp_ratio"])),
                textcoords="offset points",
                xytext=(0, 7),
                ha="center",
                fontsize=8,
                color=color,
            )

    ax1.set_title("Communication-to-Computation Ratio ($T_{comm} / T_{comp}$)", fontsize=11, fontweight="bold")
    ax1.set_xlabel("MPI Worker Count $p$", fontsize=10)
    ax1.set_ylabel("Ratio $T_{comm} / T_{comp}$ (Worker Sum)", fontsize=10)
    ax1.set_xticks([1, 2, 4, 8])
    ax1.grid(True, linestyle="--", alpha=0.35)
    ax1.legend(fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_weak_scaling(weak_df: pd.DataFrame, out_path: Path) -> None:
    if weak_df.empty:
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.9), dpi=300)

    # Panel 1: Execution Time Tp vs Workers (with proportional dataset size)
    ax0 = axes[0]
    for job, g in weak_df.groupby("job", sort=False):
        mpi_g = g[g["engine"] == "mpi"].sort_values("workers")
        if mpi_g.empty:
            continue
        color = JOB_COLORS.get(str(job), "#1f77b4")
        label = JOB_LABELS.get(str(job), str(job))
        ax0.plot(
            mpi_g["workers"],
            mpi_g["total_seconds"],
            marker="o",
            linewidth=2.2,
            color=color,
            label=f"{label} (MPI)",
        )
        t1_mpi = float(mpi_g["total_seconds"].iloc[0])
        ax0.axhline(t1_mpi, linestyle="--", color=color, alpha=0.45, linewidth=1.3)

    ax0.set_title("Weak Scaling Runtime $T_p$ (Constant Work/Worker)", fontsize=10.5, fontweight="bold")
    ax0.set_xlabel("MPI Worker Count $p$ (Dataset $\\propto p$)", fontsize=9.5)
    ax0.set_ylabel("Total Execution Time $T_p$ (s)", fontsize=9.5)
    ax0.set_xticks([1, 2, 4, 8])
    ax0.grid(True, linestyle="--", alpha=0.35)
    ax0.legend(fontsize=8.5)

    # Panel 2: Weak Scaling Efficiency E_weak(p)
    ax1 = axes[1]
    ax1.axhline(1.0, linestyle=":", color="#666666", linewidth=1.5, label="Ideal Weak Efficiency (1.0)")
    for job, g in weak_df.groupby("job", sort=False):
        mpi_g = g[g["engine"] == "mpi"].sort_values("workers")
        if mpi_g.empty:
            continue
        color = JOB_COLORS.get(str(job), "#1f77b4")
        label = JOB_LABELS.get(str(job), str(job))
        ax1.plot(
            mpi_g["workers"],
            mpi_g["weak_efficiency_mpi"],
            marker="s",
            linewidth=2.2,
            color=color,
            label=f"{label} ($T_{{mpi,1}} / T_p$)",
        )
        ax1.plot(
            mpi_g["workers"],
            mpi_g["weak_efficiency_seq"],
            marker="o",
            linestyle="--",
            linewidth=1.7,
            color=color,
            alpha=0.75,
            label=f"{label} ($T_{{seq,1}} / T_p$)",
        )

    ax1.set_title("Weak Scaling Efficiency $E_{weak}(p) = T_1 / T_p$", fontsize=10.5, fontweight="bold")
    ax1.set_xlabel("MPI Worker Count $p$", fontsize=9.5)
    ax1.set_ylabel("Weak Efficiency $E_{weak}(p)$", fontsize=9.5)
    ax1.set_xticks([1, 2, 4, 8])
    ax1.grid(True, linestyle="--", alpha=0.35)
    ax1.legend(fontsize=8)

    # Panel 3: Aggregate Throughput (MiB/s)
    ax2 = axes[2]
    for job, g in weak_df.groupby("job", sort=False):
        mpi_g = g[g["engine"] == "mpi"].sort_values("workers")
        if mpi_g.empty:
            continue
        color = JOB_COLORS.get(str(job), "#1f77b4")
        label = JOB_LABELS.get(str(job), str(job))
        ax2.plot(
            mpi_g["workers"],
            mpi_g["throughput_mb_s"],
            marker="^",
            linewidth=2.2,
            color=color,
            label=f"{label} Throughput",
        )

    ax2.set_title("Weak Scaling Aggregate Throughput (MiB/s)", fontsize=10.5, fontweight="bold")
    ax2.set_xlabel("MPI Worker Count $p$", fontsize=9.5)
    ax2.set_ylabel("Throughput (MiB / s)", fontsize=9.5)
    ax2.set_xticks([1, 2, 4, 8])
    ax2.grid(True, linestyle="--", alpha=0.35)
    ax2.legend(fontsize=8.5)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_fault_recovery(fault_df: pd.DataFrame, out_path: Path) -> None:
    if fault_df.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2), dpi=300)

    # Left: Healthy vs Fault-Injected total wall time
    ax0 = axes[0]
    groups = list(fault_df.groupby(["job", "workers"], sort=False))
    x_pos = list(range(len(groups)))
    width = 0.34

    healthy_times: List[float] = []
    fault_times: List[float] = []
    recovery_times: List[float] = []
    x_labels: List[str] = []

    for (job, workers), g in groups:
        h_row = g[g["mode"] == "healthy"]
        f_row = g[g["mode"] == "fault_injected"]
        th = float(h_row["total_seconds"].iloc[0]) if not h_row.empty else 0.0
        tf = float(f_row["total_seconds"].iloc[0]) if not f_row.empty else 0.0
        tr = float(f_row["recovery_seconds"].iloc[0]) if not f_row.empty else 0.0
        healthy_times.append(th)
        fault_times.append(tf)
        recovery_times.append(tr)
        x_labels.append(f"{JOB_LABELS.get(str(job), str(job))}\n({int(workers)} workers)")

    bars_h = ax0.bar(
        [x - width / 2 for x in x_pos],
        healthy_times,
        width=width,
        label="Healthy Run (No Failure)",
        color="#2e8b57",
    )
    bars_f = ax0.bar(
        [x + width / 2 for x in x_pos],
        fault_times,
        width=width,
        label="Fault-Injected (--fail-rank 1)",
        color="#c0392b",
    )

    for b in list(bars_h) + list(bars_f):
        h = b.get_height()
        ax0.annotate(
            f"{h:.2f}s",
            (b.get_x() + b.get_width() / 2, h),
            textcoords="offset points",
            xytext=(0, 4),
            ha="center",
            fontsize=8.5,
        )

    ax0.set_title("Healthy vs. Fault-Injected Execution Time", fontsize=11, fontweight="bold")
    ax0.set_ylabel("Total Execution Time (seconds)", fontsize=10)
    ax0.set_xticks(x_pos)
    ax0.set_xticklabels(x_labels, fontsize=9)
    ax0.grid(True, axis="y", linestyle="--", alpha=0.35)
    ax0.legend(fontsize=9)

    # Right: Overhead & Re-dispatch Recovery Time
    ax1 = axes[1]
    overheads = [max(0.0, f - h) for h, f in zip(healthy_times, fault_times)]
    bars_ov = ax1.bar(
        [x - width / 2 for x in x_pos],
        overheads,
        width=width,
        label="Wall-Clock Overhead ($T_{fault} - T_{healthy}$)",
        color="#e67e22",
    )
    bars_rec = ax1.bar(
        [x + width / 2 for x in x_pos],
        recovery_times,
        width=width,
        label="Re-dispatch Recovery Window ($T_{recovery}$)",
        color="#8e44ad",
    )

    for b in list(bars_ov) + list(bars_rec):
        h = b.get_height()
        ax1.annotate(
            f"{h:.2f}s",
            (b.get_x() + b.get_width() / 2, h),
            textcoords="offset points",
            xytext=(0, 4),
            ha="center",
            fontsize=8.5,
        )

    ax1.set_title("Fault Detection & Task Re-dispatch Overhead", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Time (seconds)", fontsize=10)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(x_labels, fontsize=9)
    ax1.grid(True, axis="y", linestyle="--", alpha=0.35)
    ax1.legend(fontsize=8.8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_cprofile_hotspots(prof_df: pd.DataFrame, out_path: Path) -> None:
    if prof_df.empty:
        return

    # Plot top 6 hotspots by tottime for wordfreq (or first available job): sequential vs MPI worker rank 1
    job = "wordfreq" if "wordfreq" in prof_df["job"].values else str(prof_df["job"].iloc[0])
    sub = prof_df[prof_df["job"] == job].copy()

    seq_sub = sub[sub["role"] == "sequential"].sort_values("tottime", ascending=False).head(6).iloc[::-1]
    wrk_sub = (
        sub[(sub["role"] == "worker") & (sub["rank"] == 1)]
        .sort_values("tottime", ascending=False)
        .head(6)
        .iloc[::-1]
    )
    if wrk_sub.empty:
        wrk_sub = sub[sub["role"] == "worker"].sort_values("tottime", ascending=False).head(6).iloc[::-1]

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8), dpi=300)

    def short_label(row: pd.Series) -> str:
        fn = str(row["function"])
        mod = str(row["module"])
        if len(fn) > 34:
            fn = fn[:31] + "..."
        return f"{fn}\n({mod})"

    if not seq_sub.empty:
        labels0 = [short_label(r) for _, r in seq_sub.iterrows()]
        axes[0].barh(labels0, seq_sub["tottime"], color="#1f77b4")
        axes[0].set_title(
            f"Sequential Baseline cProfile Hotspots ({JOB_LABELS.get(job, job)})",
            fontsize=10.5,
            fontweight="bold",
        )
        axes[0].set_xlabel("Internal Time `tottime` (seconds)", fontsize=9.5)
        axes[0].grid(True, axis="x", linestyle="--", alpha=0.35)
        axes[0].tick_params(axis="y", labelsize=8)

    if not wrk_sub.empty:
        labels1 = [short_label(r) for _, r in wrk_sub.iterrows()]
        axes[1].barh(labels1, wrk_sub["tottime"], color="#d62728")
        axes[1].set_title(
            f"MPI Worker (Rank 1) cProfile Hotspots ({JOB_LABELS.get(job, job)})",
            fontsize=10.5,
            fontweight="bold",
        )
        axes[1].set_xlabel("Internal Time `tottime` (seconds)", fontsize=9.5)
        axes[1].grid(True, axis="x", linestyle="--", alpha=0.35)
        axes[1].tick_params(axis="y", labelsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def fmt_float(val: Any, digits: int = 4) -> str:
    try:
        f = float(val)
        if math.isnan(f) or not math.isfinite(f):
            return "—"
        return f"{f:.{digits}f}"
    except Exception:
        return "—"


def generate_summary_markdown(
    *,
    hw_meta: Dict[str, Any],
    strong_df: pd.DataFrame,
    fit_summary: Dict[str, Dict[str, float]],
    weak_df: pd.DataFrame,
    fault_df: pd.DataFrame,
    prof_df: pd.DataFrame,
    summary_path: Path,
) -> None:
    lines: List[str] = []
    lines.append("# Stage 4 Benchmarking Results & Performance Analysis")
    lines.append("")
    lines.append(
        "Empirical evaluation of the lightweight distributed MapReduce framework implemented from scratch in Python with `mpi4py`."
    )
    lines.append("")

    # 1. Hardware & Experimental Environment
    lines.append("## 1. Target Hardware & Experimental Environment")
    lines.append("")
    cpu_model = hw_meta.get("cpu_model", "13th Gen Intel(R) Core(TM) i7-1355U")
    phys_cores = hw_meta.get("physical_cores", 10)
    log_threads = hw_meta.get("logical_threads", 12)
    ram_gb = hw_meta.get("ram_gb", 7.53)
    py_ver = hw_meta.get("python_version", "3.13.5")
    size_mb = hw_meta.get("size_mb", "N/A")
    weak_base_mb = hw_meta.get("weak_base_mb", "N/A")
    repeats = hw_meta.get("repeats", 3)

    lines.append("| Property | Proposal Specification | Detected Local Machine |")
    lines.append("| --- | --- | --- |")
    lines.append(f"| **Processor** | Multi-core Intel Core i7 | `{cpu_model}` |")
    lines.append(
        f"| **Cores / Threads** | 4 physical cores / 8 threads | **{phys_cores} physical cores** (2 P-cores + 8 E-cores) / **{log_threads} logical threads** |"
    )
    lines.append(f"| **System Memory** | Local shared-memory RAM | **{ram_gb} GiB** physical RAM |")
    lines.append(f"| **Python & MPI** | Python 3.10+, `mpi4py` | Python `{py_ver}`, `mpi4py 4.1.2`, Intel MPI `2021.19.0` |")
    lines.append(
        f"| **MPI Topologies** | `-n {{2, 3, 5, 9}}` (1 master + 1, 2, 4, 8 workers) | `-n {{2, 3, 5, 9}}` (Rank 0 = Master; Ranks $1..N-1$ = $p \\in \\{{1, 2, 4, 8\\}}$ workers) |"
    )
    lines.append(
        f"| **Repetitions** | Report median across repeats | **{repeats} repeats** per configuration (median reported) |"
    )
    lines.append("")
    lines.append(
        f"> **Hardware & Dataset Scaling Note:** The local test machine has **{phys_cores} physical cores / {log_threads} logical threads** and **{ram_gb} GiB total physical RAM**. "
        f"Because the framework holds intermediate key–value lists and pickled `Alltoallv` shuffle buffers in memory across up to 9 concurrent OS processes (`-n 9`), "
        f"and executes single-threaded pure-Python regular-expression tokenization (`re.findall`), strong scaling, fault recovery, and profiling were executed with a "
        f"**{size_mb} MB** deterministic synthetic corpus (Zipf-distributed words + NASA HTTP access logs across multiple document files), and weak scaling scaled proportionally from "
        f"**{weak_base_mb} MB** ($p=1$) up to **{int(weak_base_mb) * 8 if isinstance(weak_base_mb, int) else '8x'} MB** ($p=8$) to avoid virtual-memory paging and complete all repeated multi-rank experiments cleanly in RAM."
    )
    lines.append("")

    # 2. Strong Scaling
    lines.append("## 2. Strong Scaling ($S_p$, $E_p$, $T_{\\text{comm}}/T_{\\text{comp}}$, Karp–Flatt & Amdahl's Law)")
    lines.append("")
    lines.append(
        "- **Speedup vs. Sequential Baseline ($T_1$):** $S_p = T_1 / T_p$, where $T_1$ is the single-process sequential baseline (`baseline/run_sequential.py`) and $T_p$ is the $p$-worker MPI runtime (`mr_driver.py` with `-n p+1`)."
    )
    lines.append(
        "- **Self-Relative MPI Speedup:** $S_p^{(\\text{mpi})} = T_{\\text{mpi},1} / T_p$, isolating multi-worker scaling inside the MPI engine by factoring out the fixed single-worker IPC/pickling cost between `run_sequential.py` and `mpiexec -n 2`."
    )
    lines.append("- **Parallel Efficiency:** $E_p = S_p / p$ and $E_p^{(\\text{mpi})} = S_p^{(\\text{mpi})} / p$.")
    lines.append(
        "- **Karp–Flatt Serial Fraction ($p > 1$):** $e(p) = \\frac{1/S_p - 1/p}{1 - 1/p}$, with Amdahl's Law curve $S_{\\text{Amdahl}}(p) = \\frac{1}{\\hat{e} + (1 - \\hat{e})/p}$."
    )
    lines.append("")

    if not strong_df.empty:
        for job, g in strong_df.groupby("job", sort=False):
            job_title = JOB_LABELS.get(str(job), str(job))
            jf = fit_summary.get(str(job), {})
            lines.append(f"### 2.{1 if job == 'wordfreq' else 2} {job_title} (`{job}`)")
            lines.append("")
            lines.append(
                "| Engine | MPI `-n` | Workers $p$ | Splits | $T_{\\text{total}}$ (s) | $T_{\\text{map}}$ (s) | $T_{\\text{shuffle}}$ (s) | $T_{\\text{reduce}}$ (s) | $\\sum T_{\\text{comp}}$ (s) | $\\sum T_{\\text{comm}}$ (s) | $T_{\\text{comm}}/T_{\\text{comp}}$ | $S_p = T_1/T_p$ | $E_p = S_p/p$ | $S_p^{(\\text{mpi})}$ | $E_p^{(\\text{mpi})}$ | Karp–Flatt $e(p)$ | Karp–Flatt $e_{\\text{mpi}}(p)$ | Amdahl $S(p)$ |"
            )
            lines.append(
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
            )
            for _, r in g.iterrows():
                lines.append(
                    f"| `{r['engine']}` | {int(r['mpi_procs'])} | {int(r['workers'])} | {int(r['num_splits'])} | "
                    f"**{fmt_float(r['total_seconds'])}** | {fmt_float(r['map_seconds'])} | {fmt_float(r['shuffle_seconds'])} | {fmt_float(r['reduce_seconds'])} | "
                    f"{fmt_float(r['comp_seconds'])} | {fmt_float(r['comm_seconds'])} | {fmt_float(r['comm_comp_ratio'])} | "
                    f"**{fmt_float(r['speedup_sp'], 3)}x** | {fmt_float(r['efficiency_ep'], 3)} | "
                    f"{fmt_float(r['speedup_mpi'], 3)}x | {fmt_float(r['efficiency_mpi'], 3)} | "
                    f"{fmt_float(r['karp_flatt_e'], 4)} | {fmt_float(r['karp_flatt_e_mpi'], 4)} | "
                    f"{fmt_float(r['amdahl_speedup_sp'], 3)}x |"
                )
            lines.append("")
            lines.append(
                f"- **Amdahl's Law Fit (`{job}`):** Fitted serial fraction vs. sequential $T_1$ is $\\hat{{e}} = {fmt_float(jf.get('e_fit_seq'), 4)}$ "
                f"(theoretical asymptotic ceiling $S_\\infty = 1/\\hat{{e}} = {fmt_float(jf.get('max_amdahl_speedup_seq'), 2)}\\times$); "
                f"self-relative MPI serial fraction is $\\hat{{e}}_{{\\text{{mpi}}}} = {fmt_float(jf.get('e_fit_mpi'), 4)}$ "
                f"($S_\\infty^{{(\\text{{mpi}})}} = {fmt_float(jf.get('max_amdahl_speedup_mpi'), 2)}\\times$)."
            )
            lines.append("")

    # 3. Weak Scaling
    lines.append("## 3. Weak Scaling (Dataset Size Proportional to Worker Count $p$)")
    lines.append("")
    if not weak_df.empty:
        lines.append(
            "| Job | Engine | MPI `-n` | Workers $p$ | Target Size (MB) | Input Bytes | Splits | $T_{\\text{total}}$ (s) | $T_{\\text{map}}$ (s) | $T_{\\text{shuffle}}$ (s) | $T_{\\text{reduce}}$ (s) | $T_{\\text{comm}}/T_{\\text{comp}}$ | Weak Eff. (vs Seq $T_1$) | Weak Eff. (vs MPI 1W) | Scaled Speedup $p \\cdot E_{\\text{weak}}$ | Throughput (MiB/s) |"
        )
        lines.append(
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
        )
        for _, r in weak_df.iterrows():
            lines.append(
                f"| `{r['job']}` | `{r['engine']}` | {int(r['mpi_procs'])} | {int(r['workers'])} | "
                f"{int(r['target_size_mb'])} | {int(r['input_bytes']):,} | {int(r['num_splits'])} | "
                f"**{fmt_float(r['total_seconds'])}** | {fmt_float(r['map_seconds'])} | {fmt_float(r['shuffle_seconds'])} | {fmt_float(r['reduce_seconds'])} | "
                f"{fmt_float(r['comm_comp_ratio'])} | {fmt_float(r['weak_efficiency_seq'], 3)} | "
                f"**{fmt_float(r['weak_efficiency_mpi'], 3)}** | {fmt_float(r['scaled_speedup_mpi'], 3)}x | "
                f"{fmt_float(r['throughput_mb_s'], 2)} |"
            )
        lines.append("")

    # 4. Fault Recovery Overhead
    lines.append("## 4. Fault-Tolerance & Recovery Overhead (`--fail-rank`)")
    lines.append("")
    if not fault_df.empty:
        lines.append(
            "| Job | MPI `-n` | Initial Workers | Mode | Failed Rank(s) | Redispatched Tasks | $T_{\\text{total}}$ (s) | $T_{\\text{map}}$ (s) | $T_{\\text{shuffle}}$ (s) | $T_{\\text{reduce}}$ (s) | Re-dispatch $T_{\\text{recovery}}$ (s) | Wall Overhead $\\Delta T$ (s) | Relative Overhead (%) |"
        )
        lines.append(
            "| --- | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
        )
        for _, r in fault_df.iterrows():
            failed_label = r["failed_ranks_str"] if r["failed_ranks_str"] else "None"
            lines.append(
                f"| `{r['job']}` | {int(r['mpi_procs'])} | {int(r['workers'])} | `{r['mode']}` | "
                f"`{failed_label}` | {int(r['redispatched_tasks'])} | **{fmt_float(r['total_seconds'])}** | "
                f"{fmt_float(r['map_seconds'])} | {fmt_float(r['shuffle_seconds'])} | {fmt_float(r['reduce_seconds'])} | "
                f"**{fmt_float(r['recovery_seconds'])}** | {fmt_float(r['overhead_seconds'], 4)} | "
                f"**{fmt_float(r['overhead_pct'], 2)}%** |"
            )
        lines.append("")

    # 5. cProfile Hotspots
    lines.append("## 5. `cProfile` Hotspot Summary (Sequential vs. Per-Rank MPI)")
    lines.append("")
    if not prof_df.empty:
        for job, g_job in prof_df.groupby("job", sort=False):
            job_title = JOB_LABELS.get(str(job), str(job))
            lines.append(f"### 5.{1 if job == 'wordfreq' else 2} {job_title} (`{job}`) — Top Functions by Internal Time (`tottime`)")
            lines.append("")
            # Show top 5 for sequential, master (rank 0), and worker (rank 1)
            subset_roles = [
                ("Sequential Baseline (`rank 0`)", g_job[g_job["role"] == "sequential"]),
                ("MPI Master (`rank 0`)", g_job[(g_job["role"] == "master") & (g_job["rank"] == 0)]),
                ("MPI Worker (`rank 1`)", g_job[(g_job["role"] == "worker") & (g_job["rank"] == 1)]),
            ]
            for title, sub in subset_roles:
                if sub.empty:
                    continue
                top5 = sub.sort_values("func_rank").head(5)
                tot_tt = float(top5["rank_total_tt"].iloc[0])
                lines.append(f"**{title}** (Total profiled CPU time: `{tot_tt:.4f}s`):")
                lines.append("")
                lines.append("| # | Module | Function | Calls | `tottime` (s) | `% tottime` | `cumtime` (s) | `% cumtime` |")
                lines.append("| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |")
                for _, r in top5.iterrows():
                    lines.append(
                        f"| {int(r['func_rank'])} | `{r['module']}:{int(r['line'])}` | `{r['function']}` | "
                        f"{int(r['ncalls']):,} | **{fmt_float(r['tottime'])}** | {fmt_float(r['pct_tottime'], 1)}% | "
                        f"{fmt_float(r['cumtime'])} | {fmt_float(r['pct_cumtime'], 1)}% |"
                    )
                lines.append("")

    # 6. Generated Plots Index
    lines.append("## 6. Generated Plots")
    lines.append("")
    lines.append("- `benchmarks/results/strong_scaling_speedup.png` — Strong scaling speedup $S_p$ vs. worker count $p$ with fitted Amdahl's Law curves.")
    lines.append("- `benchmarks/results/strong_scaling_efficiency.png` — Parallel efficiency $E_p = S_p/p$ and Karp–Flatt serial fraction $e(p)$.")
    lines.append("- `benchmarks/results/phase_breakdown_comm_comp.png` — Map/Shuffle/Reduce master wall-clock breakdown and $T_{\\text{comm}}/T_{\\text{comp}}$ ratio.")
    lines.append("- `benchmarks/results/weak_scaling.png` — Weak scaling execution time, weak efficiency, and aggregate throughput.")
    lines.append("- `benchmarks/results/fault_recovery.png` — Healthy vs. fault-injected execution time and task re-dispatch overhead.")
    lines.append("- `benchmarks/results/cprofile_hotspots.png` — Top hotspot functions from `cProfile` (`pstats`).")
    lines.append("")

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compute scaling/recovery metrics and plot Stage 4 MapReduce benchmark figures."
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default=str(REPO_ROOT / "benchmarks" / "results"),
        help="Path to benchmarks/results directory containing CSV files.",
    )
    args = parser.parse_args(argv)

    results_dir = Path(args.results_dir).resolve()
    results_dir.mkdir(parents=True, exist_ok=True)

    all_csv = results_dir / "all_runs.csv"
    all_df = pd.read_csv(all_csv) if all_csv.exists() and all_csv.stat().st_size > 0 else None

    strong_raw = load_csv_if_exists(results_dir / "strong_scaling.csv", all_df, "strong")
    weak_raw = load_csv_if_exists(results_dir / "weak_scaling.csv", all_df, "weak")
    fault_raw = load_csv_if_exists(results_dir / "fault_recovery.csv", all_df, "fault")
    prof_csv = results_dir / "profile_summary.csv"
    prof_df = pd.read_csv(prof_csv) if prof_csv.exists() and prof_csv.stat().st_size > 0 else pd.DataFrame()

    hw_meta_path = results_dir / "raw" / "hardware_info.json"
    hw_meta: Dict[str, Any] = {}
    if hw_meta_path.exists():
        try:
            hw_meta = json.loads(hw_meta_path.read_text(encoding="utf-8"))
        except Exception:
            pass
    if not hw_meta and not strong_raw.empty:
        first = strong_raw.iloc[-1]
        hw_meta = {
            "cpu_model": first.get("cpu_model", "Unknown CPU"),
            "physical_cores": int(first.get("physical_cores", 1)),
            "logical_threads": int(first.get("logical_threads", 1)),
            "ram_gb": float(first.get("ram_gb", 0.0)),
            "size_mb": int(first.get("target_size_mb", 0)),
        }

    strong_df, fit_summary = aggregate_strong_scaling(strong_raw)
    weak_df = aggregate_weak_scaling(weak_raw)
    fault_df = aggregate_fault_recovery(fault_raw)

    if not strong_df.empty:
        strong_df.to_csv(results_dir / "derived_strong_metrics.csv", index=False)
        plot_strong_scaling_speedup(
            strong_df, fit_summary, results_dir / "strong_scaling_speedup.png"
        )
        plot_strong_scaling_efficiency(
            strong_df, results_dir / "strong_scaling_efficiency.png"
        )
        plot_phase_breakdown_and_comm_comp(
            strong_df, results_dir / "phase_breakdown_comm_comp.png"
        )
        print("[plot_metrics] Saved strong scaling plots & derived_strong_metrics.csv")

    if not weak_df.empty:
        weak_df.to_csv(results_dir / "derived_weak_metrics.csv", index=False)
        plot_weak_scaling(weak_df, results_dir / "weak_scaling.png")
        print("[plot_metrics] Saved weak_scaling.png & derived_weak_metrics.csv")

    if not fault_df.empty:
        fault_df.to_csv(results_dir / "derived_fault_metrics.csv", index=False)
        plot_fault_recovery(fault_df, results_dir / "fault_recovery.png")
        print("[plot_metrics] Saved fault_recovery.png & derived_fault_metrics.csv")

    if not prof_df.empty:
        plot_cprofile_hotspots(prof_df, results_dir / "cprofile_hotspots.png")
        print("[plot_metrics] Saved cprofile_hotspots.png")

    summary_path = results_dir / "summary.md"
    generate_summary_markdown(
        hw_meta=hw_meta,
        strong_df=strong_df,
        fit_summary=fit_summary,
        weak_df=weak_df,
        fault_df=fault_df,
        prof_df=prof_df,
        summary_path=summary_path,
    )
    print(f"[plot_metrics] Wrote summary table to {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
