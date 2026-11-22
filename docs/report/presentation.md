# Final Project Presentation & Live Demo Script

**Title:** Design and Implementation of a Lightweight Distributed MapReduce Framework Using MPI Processes  
**Course:** Parallel & Distributed Computing (BSIT Semester 5 — Topic 15)  
**Team:**
- **BITF24A037** — Muhammad Abu Hurairah Nazar
- **BITF24A031** — Muhammad Zain Khan
- **BITF24A003** — Abdul Ghani

---

## Slide 1 — Title & Project Overview

- **Goal:** Build a simplified, fault-tolerant, in-memory MapReduce framework from scratch in Python using MPI (`mpi4py`).
- **Why Not Single-Core Python?** Bottlenecked by serial regex tokenization, hash-table sorting, I/O, and CPython's Global Interpreter Lock (GIL).
- **Why Not Hadoop / Spark?** Heavy JVM memory overhead, YARN/HDFS cluster daemons, and operational complexity unsuitable for lightweight OS-process parallelism.
- **Workloads Validated:**
  1. **Word Frequency (`wordfreq`)** — Global term frequency count across documents and HTTP logs.
  2. **Inverted Index (`invindex`)** — Term-to-sorted-document-list index (`term -> doc_001.txt,doc_002.txt,...`).

---

## Slide 2 — System Architecture: `1 Master + (N-1) Worker` MPI Processes

- **Process Topology (`mpiexec -n {2, 3, 5, 9}`):**
  - **Rank 0 (`Master`):** Non-copying input splitter, dynamic task queue, `Iprobe` non-blocking event loop, `in_flight` & `completed` ledgers, heartbeat timeout monitor ($\tau = 3.0\text{ s}$), and global TSV output writer.
  - **Ranks `1..N-1` (`p = 1, 2, 4, 8` Workers):** Execute Map, local `Combiner`, deterministic `CRC32` partitioning, `Alltoallv` collective shuffle, and Reduce.
- **No Threads, No GPU Kernels:**
  - Single-threaded per rank completely avoids Python GIL contention and thread race conditions.
  - Each rank runs in an isolated virtual address space communicating over Intel MPI's shared-memory fabric (`shm`).
- **Theoretical Classification:**
  - **Flynn's Taxonomy:** Asymmetric Master/Worker **MIMD (Multiple Instruction, Multiple Data)**.
  - **Memory Architecture:** **Distributed-Memory Message Passing** (Hwang, Fox & Dongarra) with fine-grained dynamic task scheduling ($k \gg p$ splits, Kuzmiakova).

---

## Slide 3 — Line-Aligned Byte Splitter & Deterministic CRC32 Shuffle

- **Zero-Copy Line-Ownership Rule (`mapreduce/splitter.py`):**
  - A byte range `InputSplit(split_id, path, start, end)` owns every line whose **first byte** lies in `[start, end)`.
  - If `start > 0`, the worker seeks in binary mode to `start - 1` and discards one `readline()`, aligning to the exact start of the next line and reading until `f.tell() >= end`.
  - Guarantees **zero duplicate and zero dropped lines** across any split size (verified from 1-byte splits up to full files).
- **Map-Side `Combiner` for Zipfian Key Skew:**
  - Locally aggregates high-frequency Zipf words (`"the"`, `"of"`, `"and"`) per split before the network shuffle.
- **Why `zlib.crc32(key) % R` Instead of Built-in `hash(key)`?**
  - Python randomizes `hash(str)` per OS process via `PYTHONHASHSEED`; different MPI ranks would map the same key to different reducers!
  - `zlib.crc32(key.encode("utf-8")) % R` is 100% deterministic across all ranks and runs.

---

## Slide 4 — Fault-Tolerance Protocol (Tanenbaum & van Steen, Ch. 8)

- **Simulated Fail-Silent Failure (`--fail-rank R --fail-after K`):**
  - Because standard MPI (`MPI_ERRORS_ARE_FATAL`) aborts the entire job if an OS rank exits abnormally, a fault-injected worker goes silent after $K$ completed map tasks and waits for `Shutdown`.
- **Push Heartbeats & Timeout Detection:**
  - Workers emit `Heartbeat` (`TAG_HEARTBEAT`) every `0.5 s` inline from their map loop.
  - Master declares a worker failed after $\tau = 3.0\text{ s}$ of silence (`TIMEOUT_TAU`).
- **Dual-Ledger Re-Dispatch (`in_flight` + `completed`):**
  - Because intermediate map outputs reside in volatile worker RAM until the shuffle, the master re-queues **both** the dead worker's `in_flight` tasks **and** its previously `completed` tasks.
  - `ShufflePlan.accepted` enforces idempotency so re-executed tasks are counted once.
- **Surviving Sub-Communicator (`MPI.Comm.Create_group`):**
  - Master and surviving workers call `comm.Create_group(world_group.Incl([0, *alive]))`.
  - Only members of `alive` participate in `Create_group`, preventing the silent worker from hanging `Alltoallv` or `gather`.

---

## Slide 5 — Strong Scaling, Karp–Flatt Serial Fraction & Amdahl's Law

**Testbed:** 13th Gen Intel Core i7-1355U (10 physical cores: 2 P-cores + 8 E-cores / 12 threads, 7.53 GiB RAM, Intel MPI 2021.19.0, 24 MiB corpus, 32 splits, median of 3 repeats).

| Workload | Seq $T_1$ (s) | MPI $p=1$ (s) | MPI $p=2$ (s) | MPI $p=4$ (s) | MPI $p=8$ (s) | Peak Speedup $S_8$ | Self-Rel $S_8^{(\text{mpi})}$ | Fitted Serial $\hat{e}$ | Amdahl Ceiling $S_\infty$ | Max $T_{\text{comm}}/T_{\text{comp}}$ |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **`wordfreq`** | 9.9034 | 9.8483 | 4.3575 ($2.27\times$) | 2.9249 ($3.39\times$) | **1.0520** | **$9.413\times$** ($E_8=1.18$) | $9.361\times$ | $0.0605$ | $16.54\times$ | $0.0167$ ($1.67\%$) |
| **`invindex`** | 5.2466 | 6.4293 | 2.8034 ($1.87\times$) | 1.6307 ($3.22\times$) | **1.2458** | **$4.211\times$** ($E_8=0.53$) | **$5.161\times$** | $0.0811$ | $12.33\times$ | $0.0230$ ($2.30\%$) |

- **Key Findings:**
  - `wordfreq` exhibits **cache/heap super-linear speedup** ($S_8 = 9.413\times, E_8 = 117.7\%$) because splitting sorting/grouping dictionaries across 8 isolated heaps avoids single-process L3 cache thrashing and CPython GC overhead.
  - `invindex` scales to $4.211\times$ vs. sequential ($5.161\times$ self-relative MPI), with Karp–Flatt $e(p)$ rising from $0.0687 \to 0.1285$ as workers spill from the 2 high-clock P-cores onto the 8 lower-clock E-cores.
  - Communication overhead is under **$2.3\%$** of compute time ($T_{\text{comm}}/T_{\text{comp}} \le 0.023$).

---

## Slide 6 — Weak Scaling (Gustafson's Law), Fault Recovery & `cProfile` Hotspots

- **Weak Scaling (4 MiB / Worker $\to$ 4, 8, 16, 32 MiB across $p = 1, 2, 4, 8$):**
  - `wordfreq` throughput scales from **$4.44\text{ MiB/s} \to 23.12\text{ MiB/s}$** (**$5.21\times$** Gustafson scaled speedup).
  - `invindex` throughput scales from **$6.31\text{ MiB/s} \to 35.70\text{ MiB/s}$** (**$5.66\times$** Gustafson scaled speedup).
- **Fault Recovery Overhead (`-n 5`, `--fail-rank 1 --fail-after 1`, $\tau = 3.0\text{ s}$):**
  - `wordfreq`: Healthy $1.0626\text{ s} \to$ Fault-injected $3.2671\text{ s}$ ($\Delta T = +2.2045\text{ s}$); **active re-dispatch & re-computation of the 2 lost tasks takes only $T_{\text{recovery}} = 0.0995\text{ s}$** ($<5\%$ of $\Delta T$; the rest is the passive $\tau = 3.0\text{ s}$ detection timeout).
  - `invindex`: Healthy $1.2394\text{ s} \to$ Fault-injected $3.2445\text{ s}$ ($\Delta T = +2.0051\text{ s}$); **active re-dispatch takes only $T_{\text{recovery}} = 0.1040\text{ s}$**.
- **`cProfile` Hotspot Breakdown (`-n 5`):**
  - **Master (`Rank 0`):** Spends **$91.1\%\text{--}92.0\%$** of time sleeping inside its non-blocking `Iprobe` loop (`<9%` active CPU).
  - **Workers (`Ranks 1..4`):** Purely compute-bound in `group_by_key` ($20.1\%\text{--}20.8\%$), `re.Pattern.findall` ($11.6\%\text{--}12.1\%$), `list.extend` ($11.3\%\text{--}11.4\%$), `dict.setdefault` ($10.4\%\text{--}10.9\%$), and `Mapper.map` ($8.7\%\text{--}8.8\%$).

---

## Slide 7 — Live Demonstration Script (End-to-End Commands)

Run the following self-contained commands from the repository root in PowerShell to demonstrate dataset generation, sequential baseline execution, multi-rank MPI execution, live fault-injection recovery, and byte-for-byte output verification:

```powershell
# Use the project virtual environment and Intel MPI launcher
$PY = ".\.venv\Scripts\python.exe"
$MPIEXEC = ".\.venv\Library\bin\mpiexec.exe"
$env:I_MPI_PIN = "0"

# 1. Generate a deterministic 20 MiB synthetic corpus (32 docs: Zipf words + NASA HTTP logs)
& $PY data/prepare_datasets.py --source synthetic --size-mb 20 --out data/samples --seed 42

# 2. Run the Stage 2 Sequential Baseline (T1) for Word Frequency & Inverted Index
& $PY baseline/run_sequential.py --job wordfreq --input data/samples --output output/seq_wordfreq.tsv --split-size 1048576 --use-combiner --metrics-json output/seq_wordfreq.json
& $PY baseline/run_sequential.py --job invindex --input data/samples --output output/seq_invindex.tsv --split-size 1048576 --use-combiner --metrics-json output/seq_invindex.json

# 3. Run the Stage 3 Parallel MPI Engine with 4 Workers (-n 5: 1 Master + 4 Workers)
& $MPIEXEC -n 5 $PY mr_driver.py --job wordfreq --input data/samples --output output/mpi_wordfreq.tsv --split-size 1048576 --metrics-json output/mpi_wordfreq.json
& $MPIEXEC -n 5 $PY mr_driver.py --job invindex --input data/samples --output output/mpi_invindex.tsv --split-size 1048576 --metrics-json output/mpi_invindex.json

# 4. Verify byte-for-byte identical outputs against the sequential baseline
& $PY baseline/verify_output.py output/seq_wordfreq.tsv output/mpi_wordfreq.tsv
& $PY baseline/verify_output.py output/seq_invindex.tsv output/mpi_invindex.tsv

# 5. Live Fault-Injection Demo: Worker Rank 1 fails after 1 task; Master detects after tau=3.0s,
#    re-queues both completed + in-flight tasks, and shuffles over surviving Ranks [2, 3, 4]
& $MPIEXEC -n 5 $PY mr_driver.py --job wordfreq --input data/samples --output output/mpi_fault_wordfreq.tsv --split-size 1048576 --fail-rank 1 --fail-after 1 --timeout 3.0 --metrics-json output/mpi_fault_wordfreq.json
& $PY baseline/verify_output.py output/seq_wordfreq.tsv output/mpi_fault_wordfreq.tsv

# 6. Run the full 14-test automated unit + MPI fault-tolerance test suite
& $PY -m unittest discover -s tests -t . -v
```
