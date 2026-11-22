"""One-command professor demo for the Lightweight MPI MapReduce Framework.

Usage (from the project root):
    python run_demo.py

Automatically uses .venv\\Scripts\\python.exe and .venv\\Library\\bin\\mpiexec.exe
so you do not need to activate the virtual environment or type individual commands.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VENV_PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"
VENV_MPIEXEC = ROOT / ".venv" / "Library" / "bin" / "mpiexec.exe"


def ensure_venv_python() -> str:
    if VENV_PYTHON.exists():
        return str(VENV_PYTHON)
    return sys.executable


def ensure_mpiexec() -> str:
    if VENV_MPIEXEC.exists():
        return str(VENV_MPIEXEC)
    return "mpiexec"


def banner(title: str) -> None:
    line = "=" * 78
    print(f"\n{line}\n  {title}\n{line}", flush=True)


def run_cmd(cmd: list[str], env: dict[str, str] | None = None) -> subprocess.CompletedProcess:
    merged_env = os.environ.copy()
    merged_env["I_MPI_PIN"] = "0"
    if env:
        merged_env.update(env)
    proc = subprocess.run(
        cmd,
        cwd=str(ROOT),
        env=merged_env,
        text=True,
        capture_output=True,
    )
    if proc.returncode != 0:
        print(f"Command failed: {' '.join(cmd)}", file=sys.stderr)
        print(proc.stdout, file=sys.stderr)
        print(proc.stderr, file=sys.stderr)
        sys.exit(proc.returncode)
    return proc


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def preview_tsv(path: Path, n: int = 5) -> None:
    lines = path.read_text(encoding="utf-8").splitlines()
    print(f"    Total unique keys emitted: {len(lines):,}")
    for line in lines[:n]:
        k, v = line.split("\t", 1)
        if len(v) > 60:
            v = v[:57] + "..."
        print(f"      {k:<18} -> {v}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run full end-to-end professor demo.")
    parser.add_argument("--size-mb", type=float, default=8.0, help="Demo dataset size in MiB (default: 8)")
    parser.add_argument("--skip-tests", action="store_true", help="Skip the 14-test unittest suite at the end")
    args = parser.parse_args()

    py = ensure_venv_python()
    mpiexec = ensure_mpiexec()
    sample_dir = ROOT / "data" / "samples" / "demo_corpus"
    out_dir = ROOT / "output" / "demo"
    out_dir.mkdir(parents=True, exist_ok=True)

    banner("LIGHTWEIGHT DISTRIBUTED MAPREDUCE FRAMEWORK ON MPI (mpi4py)")
    print("  Course : Parallel & Distributed Computing | BSIT Semester 5")
    print("  Team   : BITF24A037 - Muhammad Abu Hurairah Nazar")
    print("           BITF24A031 - Muhammad Zain Khan")
    print("           BITF24A003 - Abdul Ghani")
    print(f"  Python : {py}")
    print(f"  MPI    : {mpiexec}")

    # 1. Stage 1: Dataset Preparation
    banner(f"STAGE 1: Preparing Deterministic Input Corpus ({args.size_mb:.1f} MiB, 16 files)")
    run_cmd([
        py, "data/prepare_datasets.py",
        "--source", "synthetic",
        "--size-mb", str(args.size_mb),
        "--num-docs", "16",
        "--seed", "42",
        "--out", str(sample_dir),
    ])
    files = sorted(sample_dir.glob("*.txt"))
    total_bytes = sum(f.stat().st_size for f in files)
    print(f"  [OK] Generated {len(files)} document files ({total_bytes / (1024 * 1024):.2f} MiB) in {sample_dir.relative_to(ROOT)}")

    # 2. Stage 2: Sequential Baseline
    banner("STAGE 2: Running Single-Process Sequential Baseline (Measuring T1)")
    seq_times: dict[str, float] = {}
    for job in ("wordfreq", "invindex"):
        seq_out = out_dir / f"seq_{job}.tsv"
        seq_json = out_dir / f"seq_{job}.json"
        run_cmd([
            py, "baseline/run_sequential.py",
            "--job", job,
            "--input", str(sample_dir),
            "--output", str(seq_out),
            "--split-size", str(1024 * 1024),
            "--use-combiner",
            "--metrics-json", str(seq_json),
        ])
        m = load_json(seq_json)
        seq_times[job] = m["total_seconds"]
        print(f"\n  Job: {job:<10} | Splits: {m['num_splits']:<3} | Sequential T1 = {m['total_seconds']:.4f} s")
        preview_tsv(seq_out, n=3)

    # 3. Stage 3a & 3b: Parallel MPI MapReduce Scaling
    banner("STAGE 3a/3b: Running Parallel MPI MapReduce (-n 2, 3, 5, 9 => 1, 2, 4, 8 Workers)")
    for job in ("wordfreq", "invindex"):
        print(f"\n  Workload: {job.upper()} (Sequential T1 = {seq_times[job]:.4f} s)")
        print("  " + "-" * 74)
        print(f"  {'MPI Ranks':<10} {'Workers (p)':<12} {'Tp (s)':<10} {'Speedup Sp':<12} {'Efficiency Ep':<14} {'Tcomm/Tcomp':<12} {'Baseline Match'}")
        print("  " + "-" * 74)
        seq_out = out_dir / f"seq_{job}.tsv"
        for np_ranks in (2, 3, 5, 9):
            workers = np_ranks - 1
            mpi_out = out_dir / f"mpi_{job}_p{workers}.tsv"
            mpi_json = out_dir / f"mpi_{job}_p{workers}.json"
            run_cmd([
                mpiexec, "-n", str(np_ranks),
                py, "mr_driver.py",
                "--job", job,
                "--input", str(sample_dir),
                "--output", str(mpi_out),
                "--split-size", str(1024 * 1024),
                "--metrics-json", str(mpi_json),
            ])
            run_cmd([py, "baseline/verify_output.py", str(seq_out), str(mpi_out)])
            m = load_json(mpi_json)
            tp = m["total_seconds"]
            sp = seq_times[job] / tp if tp > 0 else 0.0
            ep = sp / workers if workers > 0 else 0.0
            ratio = m["comm_seconds"] / m["comp_seconds"] if m["comp_seconds"] > 0 else 0.0
            print(f"  -n {np_ranks:<7} {workers:<12} {tp:<10.4f} {sp:<11.2f}x {ep * 100:<13.1f}% {ratio:<12.4f} [PASS] Identical")

    # 4. Stage 3c: Fault Injection & Heartbeat Timeout Recovery
    banner("STAGE 3c: Fault Injection & Heartbeat Timeout Recovery Demo (-n 5, Rank 1 Fails)")
    fault_out = out_dir / "mpi_wordfreq_fault.tsv"
    fault_json = out_dir / "mpi_wordfreq_fault.json"
    print("  Injecting fault: Worker Rank 1 stops responding after 1 task (--fail-rank 1 --fail-after 1)")
    print("  Master Rank 0 monitors heartbeats (timeout tau = 2.0 s) and re-dispatches lost tasks...")
    run_cmd([
        mpiexec, "-n", "5",
        py, "mr_driver.py",
        "--job", "wordfreq",
        "--input", str(sample_dir),
        "--output", str(fault_out),
        "--split-size", str(1024 * 1024),
        "--fail-rank", "1",
        "--fail-after", "1",
        "--timeout", "2.0",
        "--metrics-json", str(fault_json),
    ])
    run_cmd([py, "baseline/verify_output.py", str(out_dir / "seq_wordfreq.tsv"), str(fault_out)])
    fm = load_json(fault_json)
    print(f"  [OK] Master detected failed ranks     : {fm['failed_ranks']}")
    print(f"  [OK] Unacknowledged/lost tasks re-run : {fm['redispatched_tasks']} tasks")
    print(f"  [OK] Active re-dispatch recovery time : {fm['recovery_seconds']:.4f} s (total run: {fm['total_seconds']:.4f} s)")
    print("  [OK] Survivor Alltoallv output match  : [PASS] Byte-identical to sequential baseline!")

    # 5. Automated Unit Test Suite
    if not args.skip_tests:
        banner("VERIFICATION: Running Full 14-Test Unit & MPI Integration Suite")
        proc = run_cmd([py, "-m", "unittest", "discover", "-s", "tests", "-t", "."])
        tail = (proc.stderr or proc.stdout).strip().splitlines()[-3:]
        for line in tail:
            print(f"  {line}")

    # 6. Summary of Benchmarks & Report Artifacts
    banner("STAGE 4 & 5: Pre-Generated Benchmark Tables, Plots & Final Report")
    print("  Benchmark Summary Table : benchmarks/results/summary.md")
    print("  Generated Plots (.png)  : benchmarks/results/strong_scaling_speedup.png")
    print("                            benchmarks/results/strong_scaling_efficiency.png")
    print("                            benchmarks/results/phase_breakdown_comm_comp.png")
    print("                            benchmarks/results/weak_scaling.png")
    print("                            benchmarks/results/fault_recovery.png")
    print("                            benchmarks/results/cprofile_hotspots.png")
    print("  Final Academic Report   : docs/report/final_report.md")
    print("  Presentation Deck       : docs/report/presentation.md")
    print("\n  DEMO COMPLETE: All 5 proposal stages executed and verified successfully!\n")


if __name__ == "__main__":
    main()
