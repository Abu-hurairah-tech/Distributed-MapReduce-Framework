"""Stage 4 Benchmarking Suite for the Lightweight MPI MapReduce Framework.

Runs strong scaling, weak scaling, fault-injection recovery, and cProfile
profiling experiments across the sequential baseline (`baseline/run_sequential.py`)
and the MPI engine (`mr_driver.py`).

Examples
--------
Full suite in quick mode (finishes in a few minutes):
    python benchmarks/run_benchmarks.py --quick

Individual experiments (subcommand or flag):
    python benchmarks/run_benchmarks.py strong --quick --repeats 3
    python benchmarks/run_benchmarks.py weak --quick --repeats 3
    python benchmarks/run_benchmarks.py fault --quick --repeats 3
    python benchmarks/run_benchmarks.py profile --quick

Proposal-scale run (500 MB+):
    python benchmarks/run_benchmarks.py all --size-mb 500 --repeats 3
"""
from __future__ import annotations

import argparse
import csv
import ctypes
import datetime as dt
import json
import os
import platform
import pstats
import re
import shutil
import statistics
import struct
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

METRICS_KEYS = (
    "engine",
    "job",
    "workers",
    "input_bytes",
    "num_splits",
    "total_seconds",
    "map_seconds",
    "shuffle_seconds",
    "reduce_seconds",
    "comp_seconds",
    "comm_seconds",
    "failed_ranks",
    "redispatched_tasks",
    "recovery_seconds",
)

CSV_COLUMNS = (
    "experiment",
    "timestamp",
    "repeat",
    "job",
    "engine",
    "mpi_procs",
    "workers",
    "target_size_mb",
    "input_bytes",
    "split_size_bytes",
    "num_splits",
    "combiner",
    "injected_fail_rank",
    "injected_fail_after",
    "timeout_sec",
    "total_seconds",
    "map_seconds",
    "shuffle_seconds",
    "reduce_seconds",
    "comp_seconds",
    "comm_seconds",
    "failed_ranks",
    "redispatched_tasks",
    "recovery_seconds",
    "cpu_model",
    "physical_cores",
    "logical_threads",
    "ram_gb",
)

PROFILE_CSV_COLUMNS = (
    "job",
    "engine",
    "mpi_procs",
    "workers",
    "role",
    "rank",
    "rank_total_tt",
    "func_rank",
    "module",
    "line",
    "function",
    "ncalls",
    "primitive_calls",
    "tottime",
    "cumtime",
    "pct_tottime",
    "pct_cumtime",
)

VALID_EXPERIMENTS = ("strong", "weak", "fault", "profile")


def detect_hardware_info() -> Dict[str, Any]:
    """Detect actual CPU model, physical core count, logical thread count, and RAM."""
    logical_threads = os.cpu_count() or 1
    physical_cores = logical_threads
    cpu_model = platform.processor() or "Unknown CPU"
    ram_gb = 0.0

    if sys.platform == "win32":
        try:
            import winreg

            key = winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE,
                r"HARDWARE\DESCRIPTION\System\CentralProcessor\0",
            )
            val, _ = winreg.QueryValueEx(key, "ProcessorNameString")
            if val:
                cpu_model = str(val).strip()
        except Exception:
            pass

        try:
            k32 = ctypes.windll.kernel32
            length = ctypes.c_ulong(0)
            # RelationProcessorCore == 0
            k32.GetLogicalProcessorInformationEx(0, None, ctypes.byref(length))
            if length.value > 0:
                buf = (ctypes.c_byte * length.value)()
                if k32.GetLogicalProcessorInformationEx(0, buf, ctypes.byref(length)):
                    raw = bytes(buf)
                    offset = 0
                    cores = 0
                    while offset + 8 <= len(raw):
                        rel, sz = struct.unpack_from("<II", raw, offset)
                        if sz <= 0:
                            break
                        if rel == 0:
                            cores += 1
                        offset += sz
                    if cores > 0:
                        physical_cores = cores
        except Exception:
            pass

        try:

            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            mem = MEMORYSTATUSEX(ctypes.sizeof(MEMORYSTATUSEX))
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(mem)):
                ram_gb = round(mem.ullTotalPhys / (1024 ** 3), 2)
        except Exception:
            pass
    else:
        try:
            if Path("/proc/cpuinfo").exists():
                text = Path("/proc/cpuinfo").read_text(encoding="utf-8", errors="ignore")
                for line in text.splitlines():
                    if "model name" in line:
                        cpu_model = line.split(":", 1)[1].strip()
                        break
                core_ids = set(re.findall(r"core id\s*:\s*(\d+)", text))
                if core_ids:
                    physical_cores = len(core_ids)
        except Exception:
            pass

    return {
        "cpu_model": cpu_model,
        "physical_cores": physical_cores,
        "logical_threads": logical_threads,
        "ram_gb": ram_gb,
        "platform": platform.platform(),
        "python_version": platform.python_version(),
    }


def locate_python(explicit: Optional[str] = None) -> str:
    if explicit:
        return explicit
    return sys.executable


def locate_mpiexec(python_exe: str, explicit: Optional[str] = None) -> str:
    if explicit:
        p = Path(explicit)
        if p.exists():
            return str(p)
        found = shutil.which(explicit)
        if found:
            return found
        raise FileNotFoundError(f"Specified mpiexec not found: {explicit}")

    py_path = Path(python_exe).resolve()
    venv_root = py_path.parent.parent
    candidates = [
        venv_root / "Library" / "bin" / "mpiexec.exe",
        venv_root / "Scripts" / "mpiexec.exe",
        venv_root / "bin" / "mpiexec",
        REPO_ROOT / ".venv" / "Library" / "bin" / "mpiexec.exe",
        REPO_ROOT / ".venv" / "Scripts" / "mpiexec.exe",
        REPO_ROOT.parent.parent / ".venv" / "Library" / "bin" / "mpiexec.exe",
        REPO_ROOT.parent.parent / ".venv" / "Scripts" / "mpiexec.exe",
    ]
    for cand in candidates:
        if cand.exists():
            return str(cand)

    which_mpi = shutil.which("mpiexec") or shutil.which("mpirun")
    if which_mpi:
        return which_mpi

    raise FileNotFoundError(
        "Could not locate mpiexec. Checked .venv/Library/bin/mpiexec.exe and PATH."
    )


def choose_split_size(target_size_mb: int, explicit_split_size: Optional[int]) -> int:
    """Choose a split size that ensures enough splits for load balancing & fault recovery."""
    if explicit_split_size is not None and explicit_split_size > 0:
        return explicit_split_size
    total_bytes = max(1, target_size_mb) * 1024 * 1024
    if target_size_mb >= 256:
        return 16 * 1024 * 1024  # 16 MiB
    # Aim for roughly 16-24 splits so up to 8 workers get >= 2 tasks each
    raw = max(64 * 1024, total_bytes // 16)
    # Snap to nearest clean power-of-two multiple of 64 KiB up to 16 MiB
    power = 64 * 1024
    while power * 2 <= raw and power < 16 * 1024 * 1024:
        power *= 2
    return power


def dir_size_bytes(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    if not path.exists():
        return 0
    return sum(f.stat().st_size for f in path.iterdir() if f.is_file())


def ensure_dataset(
    python_exe: str,
    source: str,
    size_mb: int,
    seed: int,
    regenerate: bool = False,
) -> Path:
    """Prepare or reuse a benchmark dataset of `size_mb` MB using `data/prepare_datasets.py`."""
    out_dir = REPO_ROOT / "data" / "raw" / f"bench_{source}_{size_mb}mb_seed{seed}"
    expected_bytes = size_mb * 1000 * 1000
    current_bytes = dir_size_bytes(out_dir)

    if (
        not regenerate
        and out_dir.exists()
        and current_bytes >= int(expected_bytes * 0.85)
    ):
        print(
            f"[dataset] Reusing existing {source} corpus ({current_bytes / (1024*1024):.2f} MiB) at {out_dir}"
        )
        return out_dir

    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.parent.mkdir(parents=True, exist_ok=True)

    prepare_script = REPO_ROOT / "data" / "prepare_datasets.py"
    if not prepare_script.exists():
        raise FileNotFoundError(f"Missing dataset generator: {prepare_script}")

    cmd = [
        python_exe,
        str(prepare_script),
        "--source",
        source,
        "--size-mb",
        str(size_mb),
        "--out",
        str(out_dir),
        "--seed",
        str(seed),
    ]
    print(f"[dataset] Generating {size_mb} MB ({source}, seed={seed}) -> {out_dir} ...")
    t0 = time.perf_counter()
    proc = subprocess.run(
        cmd,
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"prepare_datasets.py failed (exit {proc.returncode}):\n"
            f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        )
    actual_bytes = dir_size_bytes(out_dir)
    elapsed = time.perf_counter() - t0
    print(
        f"[dataset] Ready in {elapsed:.2f}s ({actual_bytes:,} bytes = {actual_bytes / (1024*1024):.2f} MiB)"
    )
    return out_dir


def verify_against_reference(
    python_exe: str, ref_output: Path, candidate_output: Path
) -> None:
    verify_script = REPO_ROOT / "baseline" / "verify_output.py"
    if not verify_script.exists():
        return
    cmd = [python_exe, str(verify_script), str(ref_output), str(candidate_output)]
    proc = subprocess.run(
        cmd,
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"Output verification failed between {ref_output} and {candidate_output}:\n"
            f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        )


def append_csv_row(csv_path: Path, columns: Sequence[str], row: Dict[str, Any]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = csv_path.exists() and csv_path.stat().st_size > 0
    with open(csv_path, "a", encoding="utf-8", newline="\n") as f:
        writer = csv.DictWriter(f, fieldnames=list(columns))
        if not file_exists:
            writer.writeheader()
        writer.writerow({col: row.get(col, "") for col in columns})


def format_failed_ranks(val: Any) -> str:
    if not val:
        return ""
    if isinstance(val, (list, tuple)):
        return ";".join(str(x) for x in val)
    return str(val)


def run_single_invocation(
    *,
    experiment: str,
    repeat: int,
    python_exe: str,
    mpiexec_exe: str,
    engine: str,
    mpi_procs: int,
    job: str,
    input_paths: Sequence[Path],
    output_file: Path,
    raw_json_path: Path,
    target_size_mb: int,
    split_size_bytes: int,
    use_combiner: bool,
    seq_use_combiner: bool = False,
    fail_rank: Optional[int] = None,
    fail_after: Optional[int] = None,
    timeout_sec: Optional[float] = None,
    profile_path: Optional[Path] = None,
    profile_dir: Optional[Path] = None,
    hw_info: Dict[str, Any],
    csv_targets: Sequence[Path],
) -> Dict[str, Any]:
    """Execute one sequential or MPI run, load its metrics JSON, and append CSV rows."""
    output_file.parent.mkdir(parents=True, exist_ok=True)
    raw_json_path.parent.mkdir(parents=True, exist_ok=True)
    if raw_json_path.exists():
        raw_json_path.unlink()

    input_strs = [str(p) for p in input_paths]

    if engine == "sequential":
        seq_script = REPO_ROOT / "baseline" / "run_sequential.py"
        cmd = [
            python_exe,
            str(seq_script),
            "--job",
            job,
            "--input",
            *input_strs,
            "--output",
            str(output_file),
            "--split-size",
            str(split_size_bytes),
            "--metrics-json",
            str(raw_json_path),
        ]
        if seq_use_combiner and use_combiner:
            cmd.append("--use-combiner")
        if profile_path is not None:
            profile_path.parent.mkdir(parents=True, exist_ok=True)
            cmd.extend(["--profile", str(profile_path)])
    elif engine == "mpi":
        driver_script = REPO_ROOT / "mr_driver.py"
        cmd = [
            mpiexec_exe,
            "-n",
            str(mpi_procs),
            python_exe,
            str(driver_script),
            "--job",
            job,
            "--input",
            *input_strs,
            "--output",
            str(output_file),
            "--split-size",
            str(split_size_bytes),
            "--metrics-json",
            str(raw_json_path),
        ]
        if not use_combiner:
            cmd.append("--no-combiner")
        if fail_rank is not None:
            cmd.extend(["--fail-rank", str(fail_rank)])
        if fail_after is not None:
            cmd.extend(["--fail-after", str(fail_after)])
        if timeout_sec is not None:
            cmd.extend(["--timeout", str(timeout_sec)])
        if profile_dir is not None:
            if profile_dir.exists():
                shutil.rmtree(profile_dir)
            profile_dir.mkdir(parents=True, exist_ok=True)
            cmd.extend(["--profile-dir", str(profile_dir)])
    else:
        raise ValueError(f"Unknown engine: {engine}")

    t0 = time.perf_counter()
    proc = subprocess.run(
        cmd,
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    wall_elapsed = time.perf_counter() - t0
    if proc.returncode != 0:
        raise RuntimeError(
            f"Run failed ({engine}, job={job}, mpi_procs={mpi_procs}, exit={proc.returncode}):\n"
            f"Command: {' '.join(cmd)}\n"
            f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        )

    if not raw_json_path.exists():
        raise RuntimeError(
            f"Metrics JSON was not created at {raw_json_path} by command:\n{' '.join(cmd)}"
        )

    metrics = json.loads(raw_json_path.read_text(encoding="utf-8"))
    for key in METRICS_KEYS:
        if key not in metrics:
            raise KeyError(
                f"Metrics JSON {raw_json_path} missing required schema key '{key}'. "
                f"Found keys: {sorted(metrics.keys())}"
            )

    timestamp = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    combiner_active = (seq_use_combiner and use_combiner) if engine == "sequential" else use_combiner

    row: Dict[str, Any] = {
        "experiment": experiment,
        "timestamp": timestamp,
        "repeat": repeat,
        "job": metrics["job"],
        "engine": metrics["engine"],
        "mpi_procs": 1 if engine == "sequential" else mpi_procs,
        "workers": int(metrics["workers"]),
        "target_size_mb": target_size_mb,
        "input_bytes": int(metrics["input_bytes"]),
        "split_size_bytes": split_size_bytes,
        "num_splits": int(metrics["num_splits"]),
        "combiner": combiner_active,
        "injected_fail_rank": "" if fail_rank is None else fail_rank,
        "injected_fail_after": "" if fail_after is None else fail_after,
        "timeout_sec": "" if timeout_sec is None else timeout_sec,
        "total_seconds": round(float(metrics["total_seconds"]), 6),
        "map_seconds": round(float(metrics["map_seconds"]), 6),
        "shuffle_seconds": round(float(metrics["shuffle_seconds"]), 6),
        "reduce_seconds": round(float(metrics["reduce_seconds"]), 6),
        "comp_seconds": round(float(metrics["comp_seconds"]), 6),
        "comm_seconds": round(float(metrics["comm_seconds"]), 6),
        "failed_ranks": format_failed_ranks(metrics["failed_ranks"]),
        "redispatched_tasks": int(metrics["redispatched_tasks"]),
        "recovery_seconds": round(float(metrics["recovery_seconds"]), 6),
        "cpu_model": hw_info["cpu_model"],
        "physical_cores": hw_info["physical_cores"],
        "logical_threads": hw_info["logical_threads"],
        "ram_gb": hw_info["ram_gb"],
    }

    # Enrich raw JSON with harness metadata too
    enriched = dict(metrics)
    enriched["_harness"] = {
        "experiment": experiment,
        "repeat": repeat,
        "mpi_procs": 1 if engine == "sequential" else mpi_procs,
        "target_size_mb": target_size_mb,
        "split_size_bytes": split_size_bytes,
        "combiner": combiner_active,
        "injected_fail_rank": fail_rank,
        "injected_fail_after": fail_after,
        "timeout_sec": timeout_sec,
        "subprocess_wall_seconds": round(wall_elapsed, 6),
        "hardware": hw_info,
    }
    raw_json_path.write_text(json.dumps(enriched, indent=2), encoding="utf-8")

    for csv_target in csv_targets:
        append_csv_row(csv_target, CSV_COLUMNS, row)

    return row


def summarize_Configuration_repeats(rows: Sequence[Dict[str, Any]], label: str) -> float:
    totals = [float(r["total_seconds"]) for r in rows]
    maps = [float(r["map_seconds"]) for r in rows]
    shuffles = [float(r["shuffle_seconds"]) for r in rows]
    reduces = [float(r["reduce_seconds"]) for r in rows]
    med_total = statistics.median(totals)
    med_map = statistics.median(maps)
    med_shuf = statistics.median(shuffles)
    med_red = statistics.median(reduces)
    print(
        f"  -> {label}: median total={med_total:.4f}s "
        f"(map={med_map:.4f}s, shuffle={med_shuf:.4f}s, reduce={med_red:.4f}s) "
        f"across {len(rows)} repeat(s) {[round(t, 4) for t in totals]}"
    )
    return med_total


def run_strong_scaling(
    *,
    args: argparse.Namespace,
    python_exe: str,
    mpiexec_exe: str,
    hw_info: Dict[str, Any],
    results_dir: Path,
    raw_dir: Path,
    output_dir: Path,
) -> None:
    print("\n" + "=" * 72)
    print(f"[STRONG SCALING] jobs={args.jobs}, size_mb={args.size_mb}, repeats={args.repeats}")
    print("=" * 72)

    if args.input:
        input_paths = [Path(p).resolve() for p in args.input]
    else:
        dataset_dir = ensure_dataset(
            python_exe=python_exe,
            source=args.source,
            size_mb=args.size_mb,
            seed=args.seed,
            regenerate=args.regenerate_data,
        )
        input_paths = [dataset_dir]

    split_size = choose_split_size(args.size_mb, args.split_size)
    strong_csv = results_dir / "strong_scaling.csv"
    all_csv = results_dir / "all_runs.csv"
    run_tag = dt.datetime.now().strftime("%Y%m%d_%H%M%S")

    for job in args.jobs:
        print(f"\n--- Strong Scaling | Job: {job} (split_size={split_size:,} B) ---")
        ref_output = output_dir / f"strong_{job}_seq.tsv"
        seq_rows: List[Dict[str, Any]] = []
        for r in range(1, args.repeats + 1):
            raw_json = raw_dir / f"strong_{job}_seq_w1_r{r}_{run_tag}.json"
            row = run_single_invocation(
                experiment="strong",
                repeat=r,
                python_exe=python_exe,
                mpiexec_exe=mpiexec_exe,
                engine="sequential",
                mpi_procs=1,
                job=job,
                input_paths=input_paths,
                output_file=ref_output,
                raw_json_path=raw_json,
                target_size_mb=args.size_mb,
                split_size_bytes=split_size,
                use_combiner=not args.no_combiner,
                seq_use_combiner=args.seq_use_combiner,
                hw_info=hw_info,
                csv_targets=[strong_csv, all_csv],
            )
            seq_rows.append(row)
        t1_med = summarize_Configuration_repeats(seq_rows, f"{job} | sequential (T1, workers=1)")

        for n_procs in args.mpi_procs:
            workers = n_procs - 1
            mpi_output = output_dir / f"strong_{job}_mpi_n{n_procs}.tsv"
            mpi_rows: List[Dict[str, Any]] = []
            for r in range(1, args.repeats + 1):
                raw_json = raw_dir / f"strong_{job}_mpi_n{n_procs}_w{workers}_r{r}_{run_tag}.json"
                row = run_single_invocation(
                    experiment="strong",
                    repeat=r,
                    python_exe=python_exe,
                    mpiexec_exe=mpiexec_exe,
                    engine="mpi",
                    mpi_procs=n_procs,
                    job=job,
                    input_paths=input_paths,
                    output_file=mpi_output,
                    raw_json_path=raw_json,
                    target_size_mb=args.size_mb,
                    split_size_bytes=split_size,
                    use_combiner=not args.no_combiner,
                    hw_info=hw_info,
                    csv_targets=[strong_csv, all_csv],
                )
                if r == 1 and not args.no_verify:
                    verify_against_reference(python_exe, ref_output, mpi_output)
                mpi_rows.append(row)
            tp_med = summarize_Configuration_repeats(
                mpi_rows, f"{job} | mpi -n {n_procs} (workers={workers})"
            )
            sp = t1_med / tp_med if tp_med > 0 else 0.0
            ep = sp / workers if workers > 0 else 0.0
            print(f"     Speedup Sp = {sp:.3f}x, Efficiency Ep = {ep:.3f} ({ep*100:.1f}%)")


def run_weak_scaling(
    *,
    args: argparse.Namespace,
    python_exe: str,
    mpiexec_exe: str,
    hw_info: Dict[str, Any],
    results_dir: Path,
    raw_dir: Path,
    output_dir: Path,
) -> None:
    base_mb = args.weak_base_mb
    print("\n" + "=" * 72)
    print(
        f"[WEAK SCALING] jobs={args.jobs}, base_mb_per_worker={base_mb}, repeats={args.repeats}"
    )
    print("=" * 72)

    # Keep split size fixed across weak scaling steps so number of splits scales linearly with p
    split_size = choose_split_size(base_mb * 2, args.split_size)
    weak_csv = results_dir / "weak_scaling.csv"
    all_csv = results_dir / "all_runs.csv"
    run_tag = dt.datetime.now().strftime("%Y%m%d_%H%M%S")

    # Prepare datasets for each worker count
    worker_counts = sorted({1} | {n - 1 for n in args.mpi_procs if n > 1})
    datasets: Dict[int, Path] = {}
    for w in worker_counts:
        scaled_mb = max(1, base_mb * w)
        datasets[w] = ensure_dataset(
            python_exe=python_exe,
            source=args.source,
            size_mb=scaled_mb,
            seed=args.seed,
            regenerate=args.regenerate_data,
        )

    for job in args.jobs:
        print(f"\n--- Weak Scaling | Job: {job} (base={base_mb} MB/worker, split_size={split_size:,} B) ---")
        # Sequential baseline on 1x base dataset
        seq_mb = max(1, base_mb * 1)
        seq_output = output_dir / f"weak_{job}_seq_1w.tsv"
        seq_rows: List[Dict[str, Any]] = []
        for r in range(1, args.repeats + 1):
            raw_json = raw_dir / f"weak_{job}_seq_w1_{seq_mb}mb_r{r}_{run_tag}.json"
            row = run_single_invocation(
                experiment="weak",
                repeat=r,
                python_exe=python_exe,
                mpiexec_exe=mpiexec_exe,
                engine="sequential",
                mpi_procs=1,
                job=job,
                input_paths=[datasets[1]],
                output_file=seq_output,
                raw_json_path=raw_json,
                target_size_mb=seq_mb,
                split_size_bytes=split_size,
                use_combiner=not args.no_combiner,
                seq_use_combiner=args.seq_use_combiner,
                hw_info=hw_info,
                csv_targets=[weak_csv, all_csv],
            )
            seq_rows.append(row)
        t_seq1 = summarize_Configuration_repeats(
            seq_rows, f"{job} | sequential (workers=1, size={seq_mb} MB)"
        )

        t_mpi1: Optional[float] = None
        for n_procs in args.mpi_procs:
            workers = n_procs - 1
            scaled_mb = max(1, base_mb * workers)
            mpi_output = output_dir / f"weak_{job}_mpi_n{n_procs}_{scaled_mb}mb.tsv"
            mpi_rows: List[Dict[str, Any]] = []
            for r in range(1, args.repeats + 1):
                raw_json = (
                    raw_dir
                    / f"weak_{job}_mpi_n{n_procs}_w{workers}_{scaled_mb}mb_r{r}_{run_tag}.json"
                )
                row = run_single_invocation(
                    experiment="weak",
                    repeat=r,
                    python_exe=python_exe,
                    mpiexec_exe=mpiexec_exe,
                    engine="mpi",
                    mpi_procs=n_procs,
                    job=job,
                    input_paths=[datasets[workers]],
                    output_file=mpi_output,
                    raw_json_path=raw_json,
                    target_size_mb=scaled_mb,
                    split_size_bytes=split_size,
                    use_combiner=not args.no_combiner,
                    hw_info=hw_info,
                    csv_targets=[weak_csv, all_csv],
                )
                if r == 1 and workers == 1 and not args.no_verify:
                    verify_against_reference(python_exe, seq_output, mpi_output)
                mpi_rows.append(row)
            tp_med = summarize_Configuration_repeats(
                mpi_rows,
                f"{job} | mpi -n {n_procs} (workers={workers}, size={scaled_mb} MB)",
            )
            if workers == 1:
                t_mpi1 = tp_med
            ref_t = t_mpi1 if t_mpi1 is not None else t_seq1
            weak_eff_seq = t_seq1 / tp_med if tp_med > 0 else 0.0
            weak_eff_mpi = ref_t / tp_med if tp_med > 0 else 0.0
            print(
                f"     Weak Efficiency (vs seq T1) = {weak_eff_seq:.3f}, "
                f"(vs MPI 1-worker) = {weak_eff_mpi:.3f}"
            )


def run_fault_recovery(
    *,
    args: argparse.Namespace,
    python_exe: str,
    mpiexec_exe: str,
    hw_info: Dict[str, Any],
    results_dir: Path,
    raw_dir: Path,
    output_dir: Path,
) -> None:
    print("\n" + "=" * 72)
    print(
        f"[FAULT RECOVERY] jobs={args.jobs}, size_mb={args.size_mb}, "
        f"procs={args.fault_procs}, fail_rank={args.fail_rank}, "
        f"fail_after={args.fail_after}, timeout={args.timeout}s, repeats={args.repeats}"
    )
    print("=" * 72)

    if args.input:
        input_paths = [Path(p).resolve() for p in args.input]
    else:
        dataset_dir = ensure_dataset(
            python_exe=python_exe,
            source=args.source,
            size_mb=args.size_mb,
            seed=args.seed,
            regenerate=args.regenerate_data,
        )
        input_paths = [dataset_dir]

    # Ensure enough splits so the failed rank has both completed and in-flight work
    split_size = choose_split_size(args.size_mb, args.split_size)
    fault_csv = results_dir / "fault_recovery.csv"
    all_csv = results_dir / "all_runs.csv"
    run_tag = dt.datetime.now().strftime("%Y%m%d_%H%M%S")

    for job in args.jobs:
        for n_procs in args.fault_procs:
            workers = n_procs - 1
            if args.fail_rank >= n_procs or args.fail_rank <= 0:
                raise ValueError(
                    f"--fail-rank {args.fail_rank} must be a valid worker rank in 1..{n_procs - 1}"
                )
            print(
                f"\n--- Fault Recovery | Job: {job} | mpi -n {n_procs} (workers={workers}) ---"
            )
            healthy_output = output_dir / f"fault_{job}_n{n_procs}_healthy.tsv"
            healthy_rows: List[Dict[str, Any]] = []
            for r in range(1, args.repeats + 1):
                raw_json = (
                    raw_dir / f"fault_{job}_mpi_n{n_procs}_healthy_r{r}_{run_tag}.json"
                )
                row = run_single_invocation(
                    experiment="fault",
                    repeat=r,
                    python_exe=python_exe,
                    mpiexec_exe=mpiexec_exe,
                    engine="mpi",
                    mpi_procs=n_procs,
                    job=job,
                    input_paths=input_paths,
                    output_file=healthy_output,
                    raw_json_path=raw_json,
                    target_size_mb=args.size_mb,
                    split_size_bytes=split_size,
                    use_combiner=not args.no_combiner,
                    timeout_sec=args.timeout,
                    hw_info=hw_info,
                    csv_targets=[fault_csv, all_csv],
                )
                healthy_rows.append(row)
            t_healthy = summarize_Configuration_repeats(
                healthy_rows, f"{job} | healthy (-n {n_procs}, no failure)"
            )

            fault_output = (
                output_dir
                / f"fault_{job}_n{n_procs}_failrank{args.fail_rank}_after{args.fail_after}.tsv"
            )
            fault_rows: List[Dict[str, Any]] = []
            for r in range(1, args.repeats + 1):
                raw_json = (
                    raw_dir
                    / f"fault_{job}_mpi_n{n_procs}_fail{args.fail_rank}_after{args.fail_after}_r{r}_{run_tag}.json"
                )
                row = run_single_invocation(
                    experiment="fault",
                    repeat=r,
                    python_exe=python_exe,
                    mpiexec_exe=mpiexec_exe,
                    engine="mpi",
                    mpi_procs=n_procs,
                    job=job,
                    input_paths=input_paths,
                    output_file=fault_output,
                    raw_json_path=raw_json,
                    target_size_mb=args.size_mb,
                    split_size_bytes=split_size,
                    use_combiner=not args.no_combiner,
                    fail_rank=args.fail_rank,
                    fail_after=args.fail_after,
                    timeout_sec=args.timeout,
                    hw_info=hw_info,
                    csv_targets=[fault_csv, all_csv],
                )
                if r == 1 and not args.no_verify:
                    verify_against_reference(python_exe, healthy_output, fault_output)
                fault_rows.append(row)
            t_fault = summarize_Configuration_repeats(
                fault_rows,
                f"{job} | fault-injected (-n {n_procs}, fail_rank={args.fail_rank}, fail_after={args.fail_after})",
            )
            med_recovery = statistics.median(
                float(r["recovery_seconds"]) for r in fault_rows
            )
            med_redispatched = statistics.median(
                int(r["redispatched_tasks"]) for r in fault_rows
            )
            delta_s = t_fault - t_healthy
            pct_overhead = (delta_s / t_healthy * 100.0) if t_healthy > 0 else 0.0
            print(
                f"     Redispatched tasks={med_redispatched}, recovery_seconds={med_recovery:.4f}s, "
                f"wall overhead={delta_s:+.4f}s ({pct_overhead:+.1f}%)"
            )


def extract_top_functions_from_prof(
    prof_path: Path, top_k: int = 12
) -> Tuple[float, List[Dict[str, Any]]]:
    """Parse a cProfile `.prof` file with `pstats.Stats` and return total time & top functions."""
    stats = pstats.Stats(str(prof_path))
    total_tt = float(getattr(stats, "total_tt", 0.0))
    entries = []
    # stats.stats maps (filename, line, func_name) -> (cc, nc, tt, ct, callers)
    raw_stats: Dict[Tuple[str, int, str], Tuple[int, int, float, float, Any]] = getattr(
        stats, "stats", {}
    )
    for (filename, line, func_name), (cc, nc, tt, ct, _callers) in raw_stats.items():
        mod_path = Path(filename)
        if mod_path.name in ("~", "<string>"):
            mod_label = filename
        else:
            try:
                mod_label = str(mod_path.resolve().relative_to(REPO_ROOT)).replace("\\", "/")
            except Exception:
                mod_label = mod_path.name
        entries.append(
            {
                "module": mod_label,
                "line": int(line),
                "function": func_name,
                "ncalls": int(nc),
                "primitive_calls": int(cc),
                "tottime": float(tt),
                "cumtime": float(ct),
                "pct_tottime": (float(tt) / total_tt * 100.0) if total_tt > 0 else 0.0,
                "pct_cumtime": (float(ct) / total_tt * 100.0) if total_tt > 0 else 0.0,
            }
        )
    # Sort by internal time (tottime) descending, then cumtime descending
    entries.sort(key=lambda e: (e["tottime"], e["cumtime"]), reverse=True)
    top_entries = entries[:top_k]
    for idx, item in enumerate(top_entries, start=1):
        item["func_rank"] = idx
    return total_tt, top_entries


def parse_rank_from_prof_name(prof_path: Path) -> int:
    m = re.search(r"(\d+)", prof_path.stem)
    return int(m.group(1)) if m else 0


def run_profiling(
    *,
    args: argparse.Namespace,
    python_exe: str,
    mpiexec_exe: str,
    hw_info: Dict[str, Any],
    results_dir: Path,
    raw_dir: Path,
    output_dir: Path,
) -> None:
    print("\n" + "=" * 72)
    print(
        f"[CPROFILE PROFILING] jobs={args.jobs}, size_mb={args.size_mb}, mpi_procs={args.profile_procs}"
    )
    print("=" * 72)

    if args.input:
        input_paths = [Path(p).resolve() for p in args.input]
    else:
        dataset_dir = ensure_dataset(
            python_exe=python_exe,
            source=args.source,
            size_mb=args.size_mb,
            seed=args.seed,
            regenerate=args.regenerate_data,
        )
        input_paths = [dataset_dir]

    split_size = choose_split_size(args.size_mb, args.split_size)
    profiles_root = raw_dir / "profiles"
    profiles_root.mkdir(parents=True, exist_ok=True)
    profile_runs_csv = results_dir / "profile_runs.csv"
    profile_summary_csv = results_dir / "profile_summary.csv"
    all_csv = results_dir / "all_runs.csv"
    run_tag = dt.datetime.now().strftime("%Y%m%d_%H%M%S")

    # Reset profile_summary.csv on a fresh profile run so top-k isn't duplicated
    if profile_summary_csv.exists():
        profile_summary_csv.unlink()

    for job in args.jobs:
        print(f"\n--- cProfile | Job: {job} ---")
        seq_prof = profiles_root / f"seq_{job}.prof"
        seq_out = output_dir / f"profile_{job}_seq.tsv"
        seq_json = raw_dir / f"profile_{job}_seq_{run_tag}.json"
        run_single_invocation(
            experiment="profile",
            repeat=1,
            python_exe=python_exe,
            mpiexec_exe=mpiexec_exe,
            engine="sequential",
            mpi_procs=1,
            job=job,
            input_paths=input_paths,
            output_file=seq_out,
            raw_json_path=seq_json,
            target_size_mb=args.size_mb,
            split_size_bytes=split_size,
            use_combiner=not args.no_combiner,
            seq_use_combiner=args.seq_use_combiner,
            profile_path=seq_prof,
            hw_info=hw_info,
            csv_targets=[profile_runs_csv, all_csv],
        )

        if seq_prof.exists():
            total_tt, top_funcs = extract_top_functions_from_prof(seq_prof, top_k=12)
            print(f"  [Sequential cProfile] total_tt={total_tt:.4f}s | Top 5 by tottime:")
            for tf in top_funcs[:5]:
                print(
                    f"    #{tf['func_rank']} {tf['module']}:{tf['line']}({tf['function']}) "
                    f"tottime={tf['tottime']:.4f}s ({tf['pct_tottime']:.1f}%), "
                    f"cumtime={tf['cumtime']:.4f}s ({tf['pct_cumtime']:.1f}%), ncalls={tf['ncalls']}"
                )
            for tf in top_funcs:
                row = {
                    "job": job,
                    "engine": "sequential",
                    "mpi_procs": 1,
                    "workers": 1,
                    "role": "sequential",
                    "rank": 0,
                    "rank_total_tt": round(total_tt, 6),
                    "func_rank": tf["func_rank"],
                    "module": tf["module"],
                    "line": tf["line"],
                    "function": tf["function"],
                    "ncalls": tf["ncalls"],
                    "primitive_calls": tf["primitive_calls"],
                    "tottime": round(tf["tottime"], 6),
                    "cumtime": round(tf["cumtime"], 6),
                    "pct_tottime": round(tf["pct_tottime"], 2),
                    "pct_cumtime": round(tf["pct_cumtime"], 2),
                }
                append_csv_row(profile_summary_csv, PROFILE_CSV_COLUMNS, row)

        n_procs = args.profile_procs
        workers = n_procs - 1
        mpi_prof_dir = profiles_root / f"mpi_{job}_n{n_procs}"
        mpi_out = output_dir / f"profile_{job}_mpi_n{n_procs}.tsv"
        mpi_json = raw_dir / f"profile_{job}_mpi_n{n_procs}_{run_tag}.json"
        run_single_invocation(
            experiment="profile",
            repeat=1,
            python_exe=python_exe,
            mpiexec_exe=mpiexec_exe,
            engine="mpi",
            mpi_procs=n_procs,
            job=job,
            input_paths=input_paths,
            output_file=mpi_out,
            raw_json_path=mpi_json,
            target_size_mb=args.size_mb,
            split_size_bytes=split_size,
            use_combiner=not args.no_combiner,
            profile_dir=mpi_prof_dir,
            hw_info=hw_info,
            csv_targets=[profile_runs_csv, all_csv],
        )

        prof_files = sorted(
            mpi_prof_dir.glob("*.prof"), key=parse_rank_from_prof_name
        )
        for pf in prof_files:
            rank = parse_rank_from_prof_name(pf)
            role = "master" if rank == 0 else "worker"
            total_tt, top_funcs = extract_top_functions_from_prof(pf, top_k=12)
            if rank in (0, 1):
                print(
                    f"  [MPI Rank {rank} ({role}) cProfile] total_tt={total_tt:.4f}s | Top 5 by tottime:"
                )
                for tf in top_funcs[:5]:
                    print(
                        f"    #{tf['func_rank']} {tf['module']}:{tf['line']}({tf['function']}) "
                        f"tottime={tf['tottime']:.4f}s ({tf['pct_tottime']:.1f}%), "
                        f"cumtime={tf['cumtime']:.4f}s ({tf['pct_cumtime']:.1f}%), ncalls={tf['ncalls']}"
                    )
            for tf in top_funcs:
                row = {
                    "job": job,
                    "engine": "mpi",
                    "mpi_procs": n_procs,
                    "workers": workers,
                    "role": role,
                    "rank": rank,
                    "rank_total_tt": round(total_tt, 6),
                    "func_rank": tf["func_rank"],
                    "module": tf["module"],
                    "line": tf["line"],
                    "function": tf["function"],
                    "ncalls": tf["ncalls"],
                    "primitive_calls": tf["primitive_calls"],
                    "tottime": round(tf["tottime"], 6),
                    "cumtime": round(tf["cumtime"], 6),
                    "pct_tottime": round(tf["pct_tottime"], 2),
                    "pct_cumtime": round(tf["pct_cumtime"], 2),
                }
                append_csv_row(profile_summary_csv, PROFILE_CSV_COLUMNS, row)


def parse_jobs_list(raw_jobs: Sequence[str]) -> List[str]:
    out: List[str] = []
    for item in raw_jobs:
        for part in item.split(","):
            p = part.strip()
            if p and p not in out:
                if p not in ("wordfreq", "invindex"):
                    raise ValueError(
                        f"Unsupported job '{p}'. Choose from 'wordfreq', 'invindex'."
                    )
                out.append(p)
    return out or ["wordfreq", "invindex"]


def parse_int_list(raw_vals: Sequence[str]) -> List[int]:
    out: List[int] = []
    for item in raw_vals:
        for part in str(item).split(","):
            p = part.strip()
            if p:
                out.append(int(p))
    return out


def resolve_experiments(args: argparse.Namespace) -> List[str]:
    selected: List[str] = []

    def add_exp(token: str) -> None:
        for part in token.split(","):
            t = part.strip().lower()
            if not t:
                continue
            if t == "all":
                for e in VALID_EXPERIMENTS:
                    if e not in selected:
                        selected.append(e)
            elif t in VALID_EXPERIMENTS:
                if t not in selected:
                    selected.append(t)
            else:
                raise ValueError(
                    f"Unknown experiment '{t}'. Valid choices: all, {', '.join(VALID_EXPERIMENTS)}"
                )

    if args.strong:
        add_exp("strong")
    if args.weak:
        add_exp("weak")
    if args.fault:
        add_exp("fault")
    if args.profile:
        add_exp("profile")
    if args.experiments:
        for item in args.experiments:
            add_exp(item)
    if args.experiment:
        for item in args.experiment:
            add_exp(item)

    if not selected:
        return list(VALID_EXPERIMENTS)
    return selected


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Stage 4 MapReduce MPI Benchmark Runner (strong/weak scaling, fault recovery, cProfile)."
    )
    parser.add_argument(
        "experiment",
        nargs="*",
        default=[],
        help="Experiment subcommand(s) to run: {all, strong, weak, fault, profile}. Defaults to 'all'.",
    )
    parser.add_argument(
        "--experiments",
        "--experiment",
        nargs="+",
        default=[],
        help="Experiment(s) to run via flag (e.g. --experiments strong weak).",
    )
    parser.add_argument(
        "--strong",
        action="store_true",
        help="Run the strong scaling experiment (fixed dataset, sequential T1 then -n 2,3,5,9).",
    )
    parser.add_argument(
        "--weak",
        action="store_true",
        help="Run the weak scaling experiment (dataset size proportional to worker count).",
    )
    parser.add_argument(
        "--fault",
        action="store_true",
        help="Run the fault recovery experiment (with and without --fail-rank).",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Run cProfile on sequential baseline and per-rank MPI processes.",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick benchmark mode that finishes in minutes (defaults to 24 MB strong/fault/profile, 4 MB/worker weak).",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="Number of repetitions per configuration (reports median). Default: 3.",
    )
    parser.add_argument(
        "--size-mb",
        type=int,
        default=None,
        help="Dataset size in MB (default: 500 per proposal, or 24 in --quick mode).",
    )
    parser.add_argument(
        "--weak-base-mb",
        type=int,
        default=None,
        help="Base dataset size in MB per worker for weak scaling (default: size_mb // 8, or 4 MB in --quick mode).",
    )
    parser.add_argument(
        "--jobs",
        nargs="+",
        default=["wordfreq", "invindex"],
        help="Jobs to benchmark: wordfreq, invindex (space- or comma-separated).",
    )
    parser.add_argument(
        "--source",
        choices=["synthetic", "gutenberg", "nasa"],
        default="synthetic",
        help="Dataset source for data/prepare_datasets.py (default: synthetic).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for dataset generation (default: 42).",
    )
    parser.add_argument(
        "--input",
        nargs="+",
        default=None,
        help="Optional existing input directory/files to use for strong/fault/profile instead of prepare_datasets.py.",
    )
    parser.add_argument(
        "--split-size",
        type=int,
        default=None,
        help="Explicit split size in bytes (default: auto-scaled to give >=16 splits, 16 MiB for >=256 MB).",
    )
    parser.add_argument(
        "--mpi-procs",
        nargs="+",
        default=["2", "3", "5", "9"],
        help="MPI process counts (-n) for scaling runs (default: 2 3 5 9 -> 1, 2, 4, 8 workers).",
    )
    parser.add_argument(
        "--fault-procs",
        nargs="+",
        default=["5"],
        help="MPI process count(s) (-n) for fault recovery comparison (default: 5 -> 4 workers).",
    )
    parser.add_argument(
        "--fail-rank",
        type=int,
        default=1,
        help="Worker rank to inject fault into during fault recovery experiment (default: 1).",
    )
    parser.add_argument(
        "--fail-after",
        type=int,
        default=1,
        help="Number of tasks completed by --fail-rank before going silent (default: 1).",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=3.0,
        help="Master heartbeat timeout tau in seconds (default: 3.0).",
    )
    parser.add_argument(
        "--profile-procs",
        type=int,
        default=5,
        help="MPI process count (-n) for cProfile experiment (default: 5 -> 1 master + 4 workers).",
    )
    parser.add_argument(
        "--no-combiner",
        action="store_true",
        help="Disable map-side combiner.",
    )
    parser.add_argument(
        "--seq-use-combiner",
        action="store_true",
        help="Also enable --use-combiner on baseline/run_sequential.py.",
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip baseline/verify_output.py output equivalence check.",
    )
    parser.add_argument(
        "--regenerate-data",
        action="store_true",
        help="Force regeneration of benchmark datasets even if cached under data/raw/.",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete existing CSVs in benchmarks/results/ before starting.",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Automatically run benchmarks/plot_metrics.py after benchmarks complete.",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default=str(REPO_ROOT / "benchmarks" / "results"),
        help="Directory for output CSVs, raw JSONs, and plots.",
    )
    parser.add_argument(
        "--python",
        type=str,
        default=None,
        help="Path to Python interpreter.",
    )
    parser.add_argument(
        "--mpiexec",
        type=str,
        default=None,
        help="Path to mpiexec launcher.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_arg_parser()
    if hasattr(parser, "parse_intermixed_args"):
        args = parser.parse_intermixed_args(argv)
    else:
        args = parser.parse_args(argv)

    experiments = resolve_experiments(args)
    args.jobs = parse_jobs_list(args.jobs)
    args.mpi_procs = parse_int_list(args.mpi_procs)
    args.fault_procs = parse_int_list(args.fault_procs)

    if args.size_mb is None:
        args.size_mb = 24 if args.quick else 500

    if args.weak_base_mb is None:
        if args.quick:
            args.weak_base_mb = max(1, min(4, args.size_mb // 4))
        else:
            args.weak_base_mb = max(1, args.size_mb // 8)

    python_exe = locate_python(args.python)
    mpiexec_exe = locate_mpiexec(python_exe, args.mpiexec)
    hw_info = detect_hardware_info()

    results_dir = Path(args.results_dir).resolve()
    raw_dir = results_dir / "raw"
    output_dir = REPO_ROOT / "output" / "benchmarks"
    results_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.reset:
        for csv_file in results_dir.glob("*.csv"):
            csv_file.unlink()

    # Save detected hardware metadata for plot_metrics.py / summary.md
    hw_meta_path = raw_dir / "hardware_info.json"
    hw_meta_path.write_text(
        json.dumps(
            {
                **hw_info,
                "quick_mode": bool(args.quick),
                "size_mb": args.size_mb,
                "weak_base_mb": args.weak_base_mb,
                "repeats": args.repeats,
                "source": args.source,
                "seed": args.seed,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print("=" * 72)
    print("MapReduce MPI Benchmarking Harness (Stage 4)")
    print("=" * 72)
    print(f"Detected CPU     : {hw_info['cpu_model']}")
    print(
        f"Cores / Threads  : {hw_info['physical_cores']} physical cores / {hw_info['logical_threads']} logical threads"
    )
    print(f"System RAM       : {hw_info['ram_gb']} GiB")
    print(f"Python           : {python_exe} ({hw_info['python_version']})")
    print(f"MPI Launcher     : {mpiexec_exe}")
    print(f"Experiments      : {', '.join(experiments)}")
    print(f"Jobs             : {', '.join(args.jobs)}")
    print(
        f"Dataset Size     : {args.size_mb} MB (weak scaling base: {args.weak_base_mb} MB/worker, quick={args.quick})"
    )
    print(f"Repeats          : {args.repeats} (reporting median)")
    print(f"Results Dir      : {results_dir}")

    t_suite_start = time.perf_counter()

    if "strong" in experiments:
        run_strong_scaling(
            args=args,
            python_exe=python_exe,
            mpiexec_exe=mpiexec_exe,
            hw_info=hw_info,
            results_dir=results_dir,
            raw_dir=raw_dir,
            output_dir=output_dir,
        )

    if "weak" in experiments:
        run_weak_scaling(
            args=args,
            python_exe=python_exe,
            mpiexec_exe=mpiexec_exe,
            hw_info=hw_info,
            results_dir=results_dir,
            raw_dir=raw_dir,
            output_dir=output_dir,
        )

    if "fault" in experiments:
        run_fault_recovery(
            args=args,
            python_exe=python_exe,
            mpiexec_exe=mpiexec_exe,
            hw_info=hw_info,
            results_dir=results_dir,
            raw_dir=raw_dir,
            output_dir=output_dir,
        )

    if "profile" in experiments:
        run_profiling(
            args=args,
            python_exe=python_exe,
            mpiexec_exe=mpiexec_exe,
            hw_info=hw_info,
            results_dir=results_dir,
            raw_dir=raw_dir,
            output_dir=output_dir,
        )

    suite_elapsed = time.perf_counter() - t_suite_start
    print("\n" + "=" * 72)
    print(f"Benchmark run finished in {suite_elapsed:.2f}s. CSVs saved to {results_dir}")
    print("=" * 72)

    if args.plot:
        plot_script = REPO_ROOT / "benchmarks" / "plot_metrics.py"
        if plot_script.exists():
            print("[plot] Running benchmarks/plot_metrics.py ...")
            subprocess.run(
                [python_exe, str(plot_script), "--results-dir", str(results_dir)],
                cwd=str(REPO_ROOT),
                check=True,
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
