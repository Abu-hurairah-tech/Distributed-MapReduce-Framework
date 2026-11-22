"""Unit tests for Word Frequency, Inverted Index, sequential runner, and verifier."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from baseline.run_sequential import run_sequential
from baseline.verify_output import compare_outputs
from jobs import JOB_NAMES, get_job

_REPO_ROOT = Path(__file__).resolve().parent.parent


class TestJobsAndBaseline(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.input_dir = os.path.join(self.tmpdir.name, "corpus")
        os.makedirs(self.input_dir, exist_ok=True)

        self.docs = {
            "doc1.txt": (
                "Hello, world! Hello MPI MapReduce.\n"
                "MapReduce counts words: hello again, world 123.\n"
            ),
            "doc2.txt": (
                "Distributed systems and parallel computing with MPI.\n"
                "World of parallel MapReduce systems 123!\n"
            ),
            "doc3.txt": (
                "Hello from doc3. Repeating words: test test test, mpi mpi.\n"
            ),
        }
        for name, text in self.docs.items():
            with open(os.path.join(self.input_dir, name), "w", encoding="utf-8", newline="\n") as f:
                f.write(text)

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_job_registry(self):
        self.assertEqual(set(JOB_NAMES), {"wordfreq", "invindex"})
        for name in JOB_NAMES:
            job = get_job(name)
            self.assertEqual(job.name, name)
            self.assertIsNotNone(job.combiner)
        with self.assertRaises(ValueError):
            get_job("nonexistent")

    def test_word_frequency_with_and_without_combiner(self):
        out_no_comb = os.path.join(self.tmpdir.name, "wf_no_comb.tsv")
        out_comb = os.path.join(self.tmpdir.name, "wf_comb.tsv")
        metrics_path = os.path.join(self.tmpdir.name, "wf_metrics.json")

        m1 = run_sequential(
            job_name="wordfreq",
            input_paths=[self.input_dir],
            output_path=out_no_comb,
            split_size=25,
            use_combiner=False,
            metrics_json=metrics_path,
        )
        m2 = run_sequential(
            job_name="wordfreq",
            input_paths=[self.input_dir],
            output_path=out_comb,
            split_size=25,
            use_combiner=True,
        )

        ok, diffs = compare_outputs(out_no_comb, out_comb)
        self.assertTrue(ok, f"Combiner changed wordfreq output: {diffs}")

        # Verify exact counts on known tokens
        counts = {}
        with open(out_comb, "r", encoding="utf-8") as f:
            for line in f:
                k, v = line.rstrip("\n").split("\t")
                counts[k] = int(v)

        self.assertEqual(counts["hello"], 4)
        self.assertEqual(counts["world"], 3)
        self.assertEqual(counts["mapreduce"], 3)
        self.assertEqual(counts["mpi"], 4)
        self.assertEqual(counts["123"], 2)
        self.assertEqual(counts["test"], 3)

        # Verify shared metrics JSON schema
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
        self.assertEqual(set(m1.keys()), expected_keys)
        self.assertEqual(set(m2.keys()), expected_keys)
        with open(metrics_path, "r", encoding="utf-8") as f:
            loaded = json.load(f)
        self.assertEqual(set(loaded.keys()), expected_keys)
        self.assertEqual(loaded["engine"], "sequential")
        self.assertEqual(loaded["job"], "wordfreq")
        self.assertEqual(loaded["workers"], 1)

    def test_inverted_index_with_and_without_combiner(self):
        out_no_comb = os.path.join(self.tmpdir.name, "ii_no_comb.tsv")
        out_comb = os.path.join(self.tmpdir.name, "ii_comb.tsv")

        run_sequential(
            job_name="invindex",
            input_paths=[self.input_dir],
            output_path=out_no_comb,
            split_size=20,
            use_combiner=False,
        )
        run_sequential(
            job_name="invindex",
            input_paths=[self.input_dir],
            output_path=out_comb,
            split_size=20,
            use_combiner=True,
        )

        ok, diffs = compare_outputs(out_no_comb, out_comb)
        self.assertTrue(ok, f"Combiner changed invindex output: {diffs}")

        index = {}
        with open(out_comb, "r", encoding="utf-8") as f:
            for line in f:
                k, v = line.rstrip("\n").split("\t")
                index[k] = v

        self.assertEqual(index["mpi"], "doc1.txt,doc2.txt,doc3.txt")
        self.assertEqual(index["hello"], "doc1.txt,doc3.txt")
        self.assertEqual(index["world"], "doc1.txt,doc2.txt")
        self.assertEqual(index["distributed"], "doc2.txt")
        self.assertEqual(index["test"], "doc3.txt")

    def test_verify_output_detects_differences(self):
        file_a = os.path.join(self.tmpdir.name, "a.tsv")
        file_b = os.path.join(self.tmpdir.name, "b.tsv")
        with open(file_a, "w", encoding="utf-8", newline="\n") as f:
            f.write("alpha\t1\nbeta\t2\n")
        with open(file_b, "w", encoding="utf-8", newline="\n") as f:
            f.write("alpha\t1\nbeta\t3\ngamma\t1\n")

        ok, diffs = compare_outputs(file_a, file_b)
        self.assertFalse(ok)
        self.assertEqual(len(diffs), 2)

    def test_cli_run_sequential_and_verify_output(self):
        out_a = os.path.join(self.tmpdir.name, "cli_a.tsv")
        out_b = os.path.join(self.tmpdir.name, "cli_b.tsv")
        metrics_file = os.path.join(self.tmpdir.name, "cli_metrics.json")
        prof_file = os.path.join(self.tmpdir.name, "cli.prof")

        runner = str(_REPO_ROOT / "baseline" / "run_sequential.py")
        verifier = str(_REPO_ROOT / "baseline" / "verify_output.py")

        res1 = subprocess.run(
            [
                sys.executable,
                runner,
                "--job",
                "wordfreq",
                "--input",
                self.input_dir,
                "--output",
                out_a,
                "--split-size",
                "32",
                "--metrics-json",
                metrics_file,
                "--profile",
                prof_file,
            ],
            cwd=str(_REPO_ROOT),
            capture_output=True,
            text=True,
        )
        self.assertEqual(res1.returncode, 0, res1.stderr)
        self.assertTrue(os.path.isfile(metrics_file))
        self.assertTrue(os.path.isfile(prof_file))

        res2 = subprocess.run(
            [
                sys.executable,
                runner,
                "--job",
                "wordfreq",
                "--input",
                self.input_dir,
                "--output",
                out_b,
                "--split-size",
                "32",
                "--use-combiner",
            ],
            cwd=str(_REPO_ROOT),
            capture_output=True,
            text=True,
        )
        self.assertEqual(res2.returncode, 0, res2.stderr)

        verify_ok = subprocess.run(
            [sys.executable, verifier, out_a, out_b],
            cwd=str(_REPO_ROOT),
            capture_output=True,
            text=True,
        )
        self.assertEqual(verify_ok.returncode, 0, verify_ok.stdout + verify_ok.stderr)


if __name__ == "__main__":
    unittest.main()
