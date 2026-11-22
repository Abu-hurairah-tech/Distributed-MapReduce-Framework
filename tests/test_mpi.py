"""Unit tests for the Stage 3 MPI MapReduce engine and fault-tolerance recovery."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from typing import List, Optional, Tuple

from baseline.run_sequential import run_sequential
from baseline.verify_output import compare_outputs

_REPO_ROOT = Path(__file__).resolve().parent.parent


def _find_mpiexec() -> str:
    """Locate mpiexec in the active virtual environment or system PATH."""
    py_path = Path(sys.executable).resolve()
    candidates = [
        py_path.parent.parent / "Library" / "bin" / "mpiexec.exe",
        py_path.parent / "mpiexec.exe",
        py_path.parent.parent / "bin" / "mpiexec",
    ]
    for c in candidates:
        if c.is_file():
            return str(c)
    found = shutil.which("mpiexec") or shutil.which("mpirun")
    if found:
        return found
    raise RuntimeError("Could not find mpiexec executable")


class TestMPIMapReduce(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.mpiexec = _find_mpiexec()
        cls.driver = str(_REPO_ROOT / "mr_driver.py")

    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.input_dir = os.path.join(self.tmpdir.name, "corpus")
        os.makedirs(self.input_dir, exist_ok=True)

        for doc_idx in range(4):
            doc_path = os.path.join(self.input_dir, f"doc_{doc_idx}.txt")
            lines = []
            for line_idx in range(60):
                lines.append(
                    f"Parallel distributed MapReduce over MPI rank doc {doc_idx} "
                    f"line {line_idx} token_{line_idx % 19} shared_{(doc_idx + line_idx) % 11}"
                )
            with open(doc_path, "w", encoding="utf-8", newline="\n") as f:
                f.write("\n".join(lines) + "\n")

        self.seq_wordfreq = os.path.join(self.tmpdir.name, "seq_wordfreq.tsv")
        self.seq_invindex = os.path.join(self.tmpdir.name, "seq_invindex.tsv")
        run_sequential(
            job_name="wordfreq",
            input_paths=[self.input_dir],
            output_path=self.seq_wordfreq,
            split_size=512,
            use_combiner=True,
        )
        run_sequential(
            job_name="invindex",
            input_paths=[self.input_dir],
            output_path=self.seq_invindex,
            split_size=512,
            use_combiner=True,
        )

    def tearDown(self) -> None:
        self.tmpdir.cleanup()

    def _run_mpi(
        self,
        num_ranks: int,
        job: str,
        output_path: str,
        split_size: int = 512,
        no_combiner: bool = False,
        fail_rank: Optional[int] = None,
        fail_after: int = 0,
        timeout: float = 0.6,
        metrics_json: Optional[str] = None,
        profile_dir: Optional[str] = None,
    ) -> Tuple[subprocess.CompletedProcess, Optional[dict]]:
        cmd: List[str] = [
            self.mpiexec,
            "-n",
            str(num_ranks),
            sys.executable,
            self.driver,
            "--job",
            job,
            "--input",
            self.input_dir,
            "--output",
            output_path,
            "--split-size",
            str(split_size),
            "--timeout",
            str(timeout),
        ]
        if no_combiner:
            cmd.append("--no-combiner")
        if fail_rank is not None:
            cmd.extend(["--fail-rank", str(fail_rank), "--fail-after", str(fail_after)])
        if metrics_json is not None:
            cmd.extend(["--metrics-json", metrics_json])
        if profile_dir is not None:
            cmd.extend(["--profile-dir", profile_dir])

        proc = subprocess.run(
            cmd,
            cwd=str(_REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=60,
        )
        metrics = None
        if proc.returncode == 0 and metrics_json and os.path.isfile(metrics_json):
            with open(metrics_json, "r", encoding="utf-8") as f:
                metrics = json.load(f)
        return proc, metrics

    def test_wordfreq_and_invindex_equivalence_across_worker_counts(self) -> None:
        expected_keys = {
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
        }
        for job, seq_ref in (("wordfreq", self.seq_wordfreq), ("invindex", self.seq_invindex)):
            for n in (2, 3, 5, 9):
                with self.subTest(job=job, num_ranks=n):
                    out_path = os.path.join(self.tmpdir.name, f"mpi_{job}_n{n}.tsv")
                    m_path = os.path.join(self.tmpdir.name, f"metrics_{job}_n{n}.json")
                    proc, metrics = self._run_mpi(
                        num_ranks=n,
                        job=job,
                        output_path=out_path,
                        split_size=512,
                        metrics_json=m_path,
                    )
                    self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
                    ok, diffs = compare_outputs(seq_ref, out_path)
                    self.assertTrue(ok, f"Mismatch for {job} at -n {n}: {diffs}")
                    self.assertIsNotNone(metrics)
                    assert metrics is not None
                    self.assertEqual(set(metrics.keys()), expected_keys)
                    self.assertEqual(metrics["engine"], "mpi")
                    self.assertEqual(metrics["job"], job)
                    self.assertEqual(metrics["workers"], n - 1)
                    self.assertEqual(metrics["failed_ranks"], [])
                    self.assertEqual(metrics["redispatched_tasks"], 0)
                    with open(seq_ref, "rb") as fa, open(out_path, "rb") as fb:
                        self.assertEqual(fa.read(), fb.read())

    def test_no_combiner_and_profile_dir(self) -> None:
        out_path = os.path.join(self.tmpdir.name, "mpi_no_comb.tsv")
        prof_dir = os.path.join(self.tmpdir.name, "profiles")
        proc, _ = self._run_mpi(
            num_ranks=3,
            job="wordfreq",
            output_path=out_path,
            split_size=512,
            no_combiner=True,
            profile_dir=prof_dir,
        )
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        ok, diffs = compare_outputs(self.seq_wordfreq, out_path)
        self.assertTrue(ok, f"Mismatch with --no-combiner: {diffs}")
        for r in range(3):
            self.assertTrue(os.path.isfile(os.path.join(prof_dir, f"rank_{r}.prof")))

    def test_fault_injection_recovery_matches_baseline(self) -> None:
        # Test both fail_after > 0 (re-queues completed + in-flight tasks) and fail_after == 0
        scenarios = [
            ("wordfreq", self.seq_wordfreq, 5, 2, 2),
            ("invindex", self.seq_invindex, 5, 1, 0),
        ]
        for job, seq_ref, n, fail_rank, fail_after in scenarios:
            with self.subTest(job=job, fail_rank=fail_rank, fail_after=fail_after):
                out_path = os.path.join(
                    self.tmpdir.name, f"mpi_fault_{job}_r{fail_rank}_k{fail_after}.tsv"
                )
                m_path = os.path.join(
                    self.tmpdir.name, f"metrics_fault_{job}_r{fail_rank}_k{fail_after}.json"
                )
                proc, metrics = self._run_mpi(
                    num_ranks=n,
                    job=job,
                    output_path=out_path,
                    split_size=512,
                    fail_rank=fail_rank,
                    fail_after=fail_after,
                    timeout=0.5,
                    metrics_json=m_path,
                )
                self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
                ok, diffs = compare_outputs(seq_ref, out_path)
                self.assertTrue(
                    ok,
                    f"Fault recovery output mismatch for {job} (fail_rank={fail_rank}, fail_after={fail_after}): {diffs}",
                )
                with open(seq_ref, "rb") as fa, open(out_path, "rb") as fb:
                    self.assertEqual(fa.read(), fb.read())

                self.assertIsNotNone(metrics)
                assert metrics is not None
                self.assertEqual(metrics["failed_ranks"], [fail_rank])
                self.assertGreaterEqual(metrics["redispatched_tasks"], fail_after + 1)
                self.assertGreater(metrics["recovery_seconds"], 0.0)


if __name__ == "__main__":
    unittest.main()
