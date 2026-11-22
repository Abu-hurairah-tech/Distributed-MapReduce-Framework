"""MPI entry point for the lightweight distributed MapReduce framework (Stage 3)."""
from __future__ import annotations

import argparse
import cProfile
import json
import os
import sys
from pathlib import Path
from typing import List, Optional, Sequence

_REPO_ROOT = Path(__file__).resolve().parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from mpi4py import MPI  # noqa: E402

from jobs import JOB_NAMES, get_job  # noqa: E402
from mapreduce.master import Master  # noqa: E402
from mapreduce.protocol import MASTER, TIMEOUT_TAU  # noqa: E402
from mapreduce.worker import Worker  # noqa: E402

DEFAULT_SPLIT_SIZE = 16 * 1024 * 1024  # 16 MiB


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Distributed MapReduce driver over MPI processes (rank 0 = master)."
    )
    parser.add_argument(
        "--job",
        required=True,
        choices=JOB_NAMES,
        help="Workload to execute.",
    )
    parser.add_argument(
        "--input",
        required=True,
        nargs="+",
        help="Input file(s) or directory(ies).",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output TSV file path.",
    )
    parser.add_argument(
        "--split-size",
        type=int,
        default=DEFAULT_SPLIT_SIZE,
        help="Approximate split size in bytes (default: 16 MiB).",
    )
    parser.add_argument(
        "--no-combiner",
        action="store_true",
        help="Disable the map-side combiner on workers.",
    )
    parser.add_argument(
        "--fail-rank",
        type=int,
        default=None,
        help="Simulated failure: worker rank that stops responding.",
    )
    parser.add_argument(
        "--fail-after",
        type=int,
        default=0,
        help="Number of tasks the failing rank completes before going silent (default: 0).",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=TIMEOUT_TAU,
        help=f"Heartbeat silence timeout tau in seconds (default: {TIMEOUT_TAU}).",
    )
    parser.add_argument(
        "--metrics-json",
        default=None,
        help="Optional path to write master timing and execution metrics as JSON.",
    )
    parser.add_argument(
        "--profile-dir",
        default=None,
        help="Optional directory to write one cProfile .prof file per MPI rank.",
    )
    return parser.parse_args(argv)


def run_mpi_job(
    job_name: str,
    input_paths: Sequence[str],
    output_path: str,
    split_size: int = DEFAULT_SPLIT_SIZE,
    use_combiner: bool = True,
    fail_rank: Optional[int] = None,
    fail_after: int = 0,
    timeout: float = TIMEOUT_TAU,
    metrics_json: Optional[str] = None,
    profile_dir: Optional[str] = None,
    comm: Optional[MPI.Comm] = None,
) -> Optional[dict]:
    """Execute the MPI MapReduce job across all ranks in ``comm``."""
    comm = comm or MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    if size < 2:
        raise ValueError("mr_driver.py requires at least 2 MPI ranks (1 master + 1 worker)")

    job = get_job(job_name)
    profiler = cProfile.Profile() if profile_dir else None
    if profiler is not None:
        profiler.enable()

    metrics_dict: Optional[dict] = None
    try:
        if rank == MASTER:
            out_dir = os.path.dirname(os.path.abspath(output_path))
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)
            master = Master(
                comm=comm,
                paths=input_paths,
                split_size=split_size,
                timeout=timeout,
            )
            metrics = master.run(output_path=output_path)
            metrics_dict = metrics.to_dict(job_name=job.name, num_workers=size - 1)
            if metrics_json:
                m_dir = os.path.dirname(os.path.abspath(metrics_json))
                if m_dir:
                    os.makedirs(m_dir, exist_ok=True)
                with open(metrics_json, "w", encoding="utf-8", newline="\n") as f:
                    json.dump(metrics_dict, f, indent=2)
                    f.write("\n")
        else:
            worker = Worker(
                comm=comm,
                job=job,
                use_combiner=use_combiner,
                fail_rank=fail_rank,
                fail_after=fail_after,
            )
            worker.run()
    finally:
        if profiler is not None and profile_dir is not None:
            profiler.disable()
            os.makedirs(profile_dir, exist_ok=True)
            prof_path = os.path.join(profile_dir, f"rank_{rank}.prof")
            profiler.dump_stats(prof_path)

    return metrics_dict


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    run_mpi_job(
        job_name=args.job,
        input_paths=args.input,
        output_path=args.output,
        split_size=args.split_size,
        use_combiner=not args.no_combiner,
        fail_rank=args.fail_rank,
        fail_after=args.fail_after,
        timeout=args.timeout,
        metrics_json=args.metrics_json,
        profile_dir=args.profile_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
