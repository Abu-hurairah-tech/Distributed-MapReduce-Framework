"""Single-process sequential MapReduce runner (Stage 2 baseline, T1)."""
from __future__ import annotations

import argparse
import cProfile
import json
import os
import sys
import time
from pathlib import Path
from typing import Iterable, Optional

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from jobs import JOB_NAMES, get_job  # noqa: E402
from mapreduce.api import JobContext, run_reducer, write_output  # noqa: E402
from mapreduce.splitter import input_size, make_splits, read_split  # noqa: E402

DEFAULT_SPLIT_SIZE = 16 * 1024 * 1024  # 16 MiB


def run_sequential(
    job_name: str,
    input_paths: Iterable[str],
    output_path: str,
    split_size: int = DEFAULT_SPLIT_SIZE,
    use_combiner: bool = False,
    metrics_json: Optional[str] = None,
    profile_path: Optional[str] = None,
) -> dict:
    """Execute a MapReduce job sequentially in a single process and return metrics."""
    job = get_job(job_name)
    paths = list(input_paths)

    profiler = cProfile.Profile() if profile_path else None
    if profiler is not None:
        profiler.enable()

    try:
        t_total_start = time.perf_counter()

        total_bytes = input_size(paths)
        splits = make_splits(paths, split_size)

        t_map_start = time.perf_counter()
        intermediate = []
        for split in splits:
            doc_id = os.path.basename(split.path)
            ctx = JobContext(job_name=job.name, rank=0, num_partitions=1, split=split)
            mapper = job.mapper()
            mapper.setup(ctx)
            split_pairs = []
            for offset, line in read_split(split):
                split_pairs.extend(mapper.map((doc_id, offset), line, ctx))
            split_pairs.extend(mapper.cleanup(ctx))
            if use_combiner and job.combiner is not None:
                split_pairs = run_reducer(job.combiner(), split_pairs, ctx)
            intermediate.extend(split_pairs)
        map_seconds = time.perf_counter() - t_map_start

        t_reduce_start = time.perf_counter()
        reduce_ctx = JobContext(job_name=job.name, rank=0, num_partitions=1)
        results = run_reducer(job.reducer(), intermediate, reduce_ctx)
        reduce_seconds = time.perf_counter() - t_reduce_start

        out_dir = os.path.dirname(os.path.abspath(output_path))
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        write_output(results, output_path)

        total_seconds = time.perf_counter() - t_total_start
    finally:
        if profiler is not None:
            profiler.disable()
            prof_dir = os.path.dirname(os.path.abspath(profile_path))
            if prof_dir:
                os.makedirs(prof_dir, exist_ok=True)
            profiler.dump_stats(profile_path)

    comp_seconds = map_seconds + reduce_seconds
    metrics = {
        "engine": "sequential",
        "job": job.name,
        "workers": 1,
        "input_bytes": total_bytes,
        "num_splits": len(splits),
        "total_seconds": total_seconds,
        "map_seconds": map_seconds,
        "shuffle_seconds": 0,
        "reduce_seconds": reduce_seconds,
        "comp_seconds": comp_seconds,
        "comm_seconds": 0,
        "failed_ranks": [],
        "redispatched_tasks": 0,
        "recovery_seconds": 0,
    }

    if metrics_json:
        metrics_dir = os.path.dirname(os.path.abspath(metrics_json))
        if metrics_dir:
            os.makedirs(metrics_dir, exist_ok=True)
        with open(metrics_json, "w", encoding="utf-8", newline="\n") as f:
            json.dump(metrics, f, indent=2)
            f.write("\n")

    return metrics


def parse_args(argv: Optional[list] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sequential single-process MapReduce baseline runner."
    )
    parser.add_argument(
        "--job",
        required=True,
        choices=JOB_NAMES,
        help="Workload to run.",
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
        "--use-combiner",
        action="store_true",
        help="Run the job's map-side combiner on each split before reducing.",
    )
    parser.add_argument(
        "--metrics-json",
        default=None,
        help="Optional path to write timing and execution metrics as JSON.",
    )
    parser.add_argument(
        "--profile",
        default=None,
        help="Optional path to write cProfile stats (.prof).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[list] = None) -> int:
    args = parse_args(argv)
    run_sequential(
        job_name=args.job,
        input_paths=args.input,
        output_path=args.output,
        split_size=args.split_size,
        use_combiner=args.use_combiner,
        metrics_json=args.metrics_json,
        profile_path=args.profile,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
