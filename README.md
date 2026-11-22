# Lightweight Distributed MapReduce on MPI Processes

Term project for Parallel & Distributed Computing (BSIT Semester 5), Topic 15:
a simplified MapReduce framework written from scratch in Python with `mpi4py`.

**Team:** BITF24A037 Muhammad Abu Hurairah Nazar · BITF24A031 Muhammad Zain Khan · BITF24A003 Abdul Ghani

## How it works

- **Rank 0 (master)** splits the input into line-aligned byte ranges and hands
  them out from a task queue.
- **Ranks 1..N-1 (workers)** run the Map phase, apply a combiner, and exchange
  intermediate pairs by hash partitioning with `Alltoallv` (Shuffle). Each worker
  then reduces its own partition (Reduce), and the master gathers the results.
- **Fault tolerance:** workers send heartbeats. If a worker goes quiet for
  longer than τ, the master re-dispatches its unacknowledged tasks. Failure is
  simulated by fault injection, because standard MPI aborts the whole job when
  a rank really dies.

## Layout

| Path | Purpose |
| --- | --- |
| `mapreduce/` | Framework: `api.py` (Mapper/Reducer/Combiner/JobContext), `protocol.py`, `splitter.py`, `master.py`, `worker.py` |
| `jobs/` | Workloads: Word Frequency, Inverted Index |
| `baseline/` | Sequential baseline (T1) and output verification |
| `mr_driver.py` | MPI entry point |
| `data/` | Dataset preparation (Project Gutenberg, NASA HTTP logs) |
| `tests/` | Correctness and fault-injection tests |
| `benchmarks/` | Strong/weak scaling, cProfile, plots, result tables |
| `docs/` | Protocol/API spec, architecture diagram, final report |

## Setup

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

You also need an MPI runtime: Microsoft MPI on Windows, or OpenMPI/MPICH on Linux.

## Run

```powershell
python data/prepare_datasets.py --help
python baseline/run_sequential.py --job wordfreq --input data/samples --output output/seq_wordfreq.tsv
mpiexec -n 5 python mr_driver.py --job wordfreq --input data/samples --output output/mpi_wordfreq.tsv
python baseline/verify_output.py output/seq_wordfreq.tsv output/mpi_wordfreq.tsv
```

`-n {2, 3, 5, 9}` gives 1, 2, 4 or 8 workers, because rank 0 is the master.

## Milestones

| Tag | Stage | Output |
| --- | --- | --- |
| `v0.1-design` | 1. Design (week 1) | API spec, protocol, architecture diagram, datasets |
| `v0.2-baseline` | 2. Sequential baseline (week 2) | Splitter, both jobs, T1, reference output |
| `v0.3-mpi` | 3. Parallel implementation (weeks 3–5) | MPI framework, output identical to baseline, fault-injection tests |
| `v0.4-bench` | 4. Benchmarking (weeks 6–7) | Sp, Ep, Tcomm/Tcomp, recovery overhead tables and plots |
| `v1.0` | 5. Report (week 8) | Final report and presentation |
