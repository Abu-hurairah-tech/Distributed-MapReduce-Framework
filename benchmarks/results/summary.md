# Stage 4 Benchmarking Results & Performance Analysis

Empirical evaluation of the lightweight distributed MapReduce framework implemented from scratch in Python with `mpi4py`.

## 1. Target Hardware & Experimental Environment

| Property | Proposal Specification | Detected Local Machine |
| --- | --- | --- |
| **Processor** | Multi-core Intel Core i7 | `13th Gen Intel(R) Core(TM) i7-1355U` |
| **Cores / Threads** | 4 physical cores / 8 threads | **10 physical cores** (2 P-cores + 8 E-cores) / **12 logical threads** |
| **System Memory** | Local shared-memory RAM | **7.53 GiB** physical RAM |
| **Python & MPI** | Python 3.10+, `mpi4py` | Python `3.13.5`, `mpi4py 4.1.2`, Intel MPI `2021.19.0` |
| **MPI Topologies** | `-n {2, 3, 5, 9}` (1 master + 1, 2, 4, 8 workers) | `-n {2, 3, 5, 9}` (Rank 0 = Master; Ranks $1..N-1$ = $p \in \{1, 2, 4, 8\}$ workers) |
| **Repetitions** | Report median across repeats | **3 repeats** per configuration (median reported) |

> **Hardware & Dataset Scaling Note:** The local test machine has **10 physical cores / 12 logical threads** and **7.53 GiB total physical RAM**. Because the framework holds intermediate key–value lists and pickled `Alltoallv` shuffle buffers in memory across up to 9 concurrent OS processes (`-n 9`), and executes single-threaded pure-Python regular-expression tokenization (`re.findall`), strong scaling, fault recovery, and profiling were executed with a **24 MB** deterministic synthetic corpus (Zipf-distributed words + NASA HTTP access logs across multiple document files), and weak scaling scaled proportionally from **4 MB** ($p=1$) up to **32 MB** ($p=8$) to avoid virtual-memory paging and complete all repeated multi-rank experiments cleanly in RAM.

## 2. Strong Scaling ($S_p$, $E_p$, $T_{\text{comm}}/T_{\text{comp}}$, Karp–Flatt & Amdahl's Law)

- **Speedup vs. Sequential Baseline ($T_1$):** $S_p = T_1 / T_p$, where $T_1$ is the single-process sequential baseline (`baseline/run_sequential.py`) and $T_p$ is the $p$-worker MPI runtime (`mr_driver.py` with `-n p+1`).
- **Self-Relative MPI Speedup:** $S_p^{(\text{mpi})} = T_{\text{mpi},1} / T_p$, isolating multi-worker scaling inside the MPI engine by factoring out the fixed single-worker IPC/pickling cost between `run_sequential.py` and `mpiexec -n 2`.
- **Parallel Efficiency:** $E_p = S_p / p$ and $E_p^{(\text{mpi})} = S_p^{(\text{mpi})} / p$.
- **Karp–Flatt Serial Fraction ($p > 1$):** $e(p) = \frac{1/S_p - 1/p}{1 - 1/p}$, with Amdahl's Law curve $S_{\text{Amdahl}}(p) = \frac{1}{\hat{e} + (1 - \hat{e})/p}$.

### 2.1 Word Frequency (`wordfreq`)

| Engine | MPI `-n` | Workers $p$ | Splits | $T_{\text{total}}$ (s) | $T_{\text{map}}$ (s) | $T_{\text{shuffle}}$ (s) | $T_{\text{reduce}}$ (s) | $\sum T_{\text{comp}}$ (s) | $\sum T_{\text{comm}}$ (s) | $T_{\text{comm}}/T_{\text{comp}}$ | $S_p = T_1/T_p$ | $E_p = S_p/p$ | $S_p^{(\text{mpi})}$ | $E_p^{(\text{mpi})}$ | Karp–Flatt $e(p)$ | Karp–Flatt $e_{\text{mpi}}(p)$ | Amdahl $S(p)$ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `sequential` | 1 | 1 | 32 | **9.9034** | 9.8516 | 0.0000 | 0.0128 | 9.8617 | 0.0000 | 0.0000 | **1.000x** | 1.000 | 0.994x | 0.994 | — | — | 1.000x |
| `mpi` | 2 | 1 | 32 | **9.8483** | 9.7613 | 0.0419 | 0.0437 | 7.5587 | 0.0482 | 0.0064 | **1.006x** | 1.006 | 1.000x | 1.000 | — | — | 1.000x |
| `mpi` | 3 | 2 | 32 | **4.3575** | 4.3128 | 0.0300 | 0.0150 | 6.6495 | 0.0635 | 0.0096 | **2.273x** | 1.136 | 2.260x | 1.130 | -0.1200 | -0.1151 | 1.886x |
| `mpi` | 5 | 4 | 32 | **2.9249** | 2.8755 | 0.0241 | 0.0100 | 8.8044 | 0.1468 | 0.0167 | **3.386x** | 0.846 | 3.367x | 0.842 | 0.0605 | 0.0627 | 3.386x |
| `mpi` | 9 | 8 | 32 | **1.0520** | 1.0211 | 0.0123 | 0.0056 | 6.2328 | 0.0995 | 0.0160 | **9.413x** | 1.177 | 9.361x | 1.170 | -0.0215 | -0.0208 | 5.621x |

- **Amdahl's Law Fit (`wordfreq`):** Fitted serial fraction vs. sequential $T_1$ is $\hat{e} = 0.0605$ (theoretical asymptotic ceiling $S_\infty = 1/\hat{e} = 16.54\times$); self-relative MPI serial fraction is $\hat{e}_{\text{mpi}} = 0.0627$ ($S_\infty^{(\text{mpi})} = 15.96\times$).

### 2.2 Inverted Index (`invindex`)

| Engine | MPI `-n` | Workers $p$ | Splits | $T_{\text{total}}$ (s) | $T_{\text{map}}$ (s) | $T_{\text{shuffle}}$ (s) | $T_{\text{reduce}}$ (s) | $\sum T_{\text{comp}}$ (s) | $\sum T_{\text{comm}}$ (s) | $T_{\text{comm}}/T_{\text{comp}}$ | $S_p = T_1/T_p$ | $E_p = S_p/p$ | $S_p^{(\text{mpi})}$ | $E_p^{(\text{mpi})}$ | Karp–Flatt $e(p)$ | Karp–Flatt $e_{\text{mpi}}(p)$ | Amdahl $S(p)$ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `sequential` | 1 | 1 | 32 | **5.2466** | 5.2075 | 0.0000 | 0.0056 | 5.2135 | 0.0000 | 0.0000 | **1.000x** | 1.000 | 1.225x | 1.225 | — | — | 1.000x |
| `mpi` | 2 | 1 | 32 | **6.4293** | 6.3135 | 0.0318 | 0.0302 | 5.0023 | 0.0487 | 0.0097 | **0.816x** | 0.816 | 1.000x | 1.000 | — | — | 1.000x |
| `mpi` | 3 | 2 | 32 | **2.8034** | 2.7387 | 0.0173 | 0.0146 | 4.2391 | 0.0409 | 0.0096 | **1.872x** | 0.936 | 2.293x | 1.147 | 0.0687 | -0.1279 | 1.850x |
| `mpi` | 5 | 4 | 32 | **1.6307** | 1.5982 | 0.0166 | 0.0092 | 4.8921 | 0.0605 | 0.0124 | **3.217x** | 0.804 | 3.943x | 0.986 | 0.0811 | 0.0048 | 3.217x |
| `mpi` | 9 | 8 | 32 | **1.2458** | 1.1877 | 0.0122 | 0.0052 | 6.5972 | 0.1520 | 0.0230 | **4.211x** | 0.526 | 5.161x | 0.645 | 0.1285 | 0.0786 | 5.103x |

- **Amdahl's Law Fit (`invindex`):** Fitted serial fraction vs. sequential $T_1$ is $\hat{e} = 0.0811$ (theoretical asymptotic ceiling $S_\infty = 1/\hat{e} = 12.33\times$); self-relative MPI serial fraction is $\hat{e}_{\text{mpi}} = 0.0417$ ($S_\infty^{(\text{mpi})} = 23.97\times$).

## 3. Weak Scaling (Dataset Size Proportional to Worker Count $p$)

| Job | Engine | MPI `-n` | Workers $p$ | Target Size (MB) | Input Bytes | Splits | $T_{\text{total}}$ (s) | $T_{\text{map}}$ (s) | $T_{\text{shuffle}}$ (s) | $T_{\text{reduce}}$ (s) | $T_{\text{comm}}/T_{\text{comp}}$ | Weak Eff. (vs Seq $T_1$) | Weak Eff. (vs MPI 1W) | Scaled Speedup $p \cdot E_{\text{weak}}$ | Throughput (MiB/s) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `wordfreq` | `sequential` | 1 | 1 | 4 | 4,194,304 | 32 | **0.8286** | 0.8004 | 0.0000 | 0.0038 | 0.0000 | 1.000 | **1.087** | 1.087x | 4.83 |
| `wordfreq` | `mpi` | 2 | 1 | 4 | 4,194,304 | 32 | **0.9007** | 0.8716 | 0.0183 | 0.0073 | 0.0303 | 0.920 | **1.000** | 1.000x | 4.44 |
| `wordfreq` | `mpi` | 3 | 2 | 8 | 8,388,608 | 32 | **0.6092** | 0.5894 | 0.0116 | 0.0047 | 0.0284 | 1.360 | **1.478** | 2.957x | 13.13 |
| `wordfreq` | `mpi` | 5 | 4 | 16 | 16,777,216 | 32 | **0.7056** | 0.6942 | 0.0062 | 0.0030 | 0.0100 | 1.174 | **1.277** | 5.106x | 22.68 |
| `wordfreq` | `mpi` | 9 | 8 | 32 | 33,554,432 | 64 | **1.3841** | 1.3592 | 0.0097 | 0.0054 | 0.0221 | 0.599 | **0.651** | 5.206x | 23.12 |
| `invindex` | `sequential` | 1 | 1 | 4 | 4,194,304 | 32 | **0.6142** | 0.5652 | 0.0000 | 0.0038 | 0.0000 | 1.000 | **1.032** | 1.032x | 6.51 |
| `invindex` | `mpi` | 2 | 1 | 4 | 4,194,304 | 32 | **0.6340** | 0.6111 | 0.0140 | 0.0068 | 0.0365 | 0.969 | **1.000** | 1.000x | 6.31 |
| `invindex` | `mpi` | 3 | 2 | 8 | 8,388,608 | 32 | **0.4926** | 0.4735 | 0.0093 | 0.0063 | 0.0252 | 1.247 | **1.287** | 2.574x | 16.24 |
| `invindex` | `mpi` | 5 | 4 | 16 | 16,777,216 | 32 | **0.6070** | 0.5917 | 0.0063 | 0.0038 | 0.0115 | 1.012 | **1.044** | 4.178x | 26.36 |
| `invindex` | `mpi` | 9 | 8 | 32 | 33,554,432 | 64 | **0.8963** | 0.8735 | 0.0079 | 0.0056 | 0.0169 | 0.685 | **0.707** | 5.658x | 35.70 |

## 4. Fault-Tolerance & Recovery Overhead (`--fail-rank`)

| Job | MPI `-n` | Initial Workers | Mode | Failed Rank(s) | Redispatched Tasks | $T_{\text{total}}$ (s) | $T_{\text{map}}$ (s) | $T_{\text{shuffle}}$ (s) | $T_{\text{reduce}}$ (s) | Re-dispatch $T_{\text{recovery}}$ (s) | Wall Overhead $\Delta T$ (s) | Relative Overhead (%) |
| --- | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `wordfreq` | 5 | 4 | `healthy` | `None` | 0 | **1.0626** | 1.0505 | 0.0064 | 0.0036 | **0.0000** | 0.0000 | **0.00%** |
| `wordfreq` | 5 | 4 | `fault_injected` | `1` | 2 | **3.2671** | 3.2517 | 0.0103 | 0.0035 | **0.0995** | 2.2045 | **207.46%** |
| `invindex` | 5 | 4 | `healthy` | `None` | 0 | **1.2394** | 1.2249 | 0.0065 | 0.0062 | **0.0000** | 0.0000 | **0.00%** |
| `invindex` | 5 | 4 | `fault_injected` | `1` | 2 | **3.2445** | 3.2287 | 0.0086 | 0.0050 | **0.1040** | 2.0051 | **161.77%** |

## 5. `cProfile` Hotspot Summary (Sequential vs. Per-Rank MPI)

### 5.1 Word Frequency (`wordfreq`) — Top Functions by Internal Time (`tottime`)

**Sequential Baseline (`rank 0`)** (Total profiled CPU time: `9.6973s`):

| # | Module | Function | Calls | `tottime` (s) | `% tottime` | `cumtime` (s) | `% cumtime` |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | `mapreduce/api.py:118` | `group_by_key` | 15,512 | **2.2088** | 22.8% | 4.2633 | 44.0% |
| 2 | `~:0` | `<method 'extend' of 'list' objects>` | 335,692 | **1.3298** | 13.7% | 3.9577 | 40.8% |
| 3 | `~:0` | `<method 'findall' of 're.Pattern' objects>` | 320,149 | **1.2567** | 13.0% | 1.2567 | 13.0% |
| 4 | `~:0` | `<method 'setdefault' of 'dict' objects>` | 4,710,049 | **1.1453** | 11.8% | 1.1453 | 11.8% |
| 5 | `jobs/word_frequency.py:16` | `map` | 5,015,858 | **0.9883** | 10.2% | 2.5898 | 26.7% |

**MPI Master (`rank 0`)** (Total profiled CPU time: `3.9025s`):

| # | Module | Function | Calls | `tottime` (s) | `% tottime` | `cumtime` (s) | `% cumtime` |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | `~:0` | `<built-in method time.sleep>` | 3,207 | **3.5555** | 91.1% | 3.5555 | 91.1% |
| 2 | `mapreduce/master.py:148` | `_poll_messages` | 3,256 | **0.1810** | 4.6% | 0.1812 | 4.6% |
| 3 | `mapreduce/master.py:228` | `run_map_phase` | 1 | **0.0369** | 0.9% | 3.8452 | 98.5% |
| 4 | `mapreduce/master.py:184` | `_check_timeouts` | 3,255 | **0.0322** | 0.8% | 0.0558 | 1.4% |
| 5 | `mapreduce/master.py:256` | `run_shuffle_and_reduce` | 1 | **0.0299** | 0.8% | 0.0302 | 0.8% |

**MPI Worker (`rank 1`)** (Total profiled CPU time: `3.8985s`):

| # | Module | Function | Calls | `tottime` (s) | `% tottime` | `cumtime` (s) | `% cumtime` |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | `mapreduce/api.py:118` | `group_by_key` | 3,877 | **0.8120** | 20.8% | 1.5807 | 40.5% |
| 2 | `~:0` | `<method 'findall' of 're.Pattern' objects>` | 79,719 | **0.4712** | 12.1% | 0.4712 | 12.1% |
| 3 | `~:0` | `<method 'extend' of 'list' objects>` | 83,591 | **0.4427** | 11.3% | 1.3955 | 35.8% |
| 4 | `~:0` | `<method 'setdefault' of 'dict' objects>` | 1,172,265 | **0.4263** | 10.9% | 0.4263 | 10.9% |
| 5 | `jobs/word_frequency.py:16` | `map` | 1,248,323 | **0.3449** | 8.8% | 0.9398 | 24.1% |

### 5.2 Inverted Index (`invindex`) — Top Functions by Internal Time (`tottime`)

**Sequential Baseline (`rank 0`)** (Total profiled CPU time: `8.1745s`):

| # | Module | Function | Calls | `tottime` (s) | `% tottime` | `cumtime` (s) | `% cumtime` |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | `mapreduce/api.py:118` | `group_by_key` | 15,512 | **1.9239** | 23.5% | 3.6381 | 44.5% |
| 2 | `~:0` | `<method 'extend' of 'list' objects>` | 335,692 | **1.1020** | 13.5% | 3.3690 | 41.2% |
| 3 | `~:0` | `<method 'findall' of 're.Pattern' objects>` | 320,149 | **1.0478** | 12.8% | 1.0478 | 12.8% |
| 4 | `~:0` | `<method 'setdefault' of 'dict' objects>` | 4,710,049 | **0.9611** | 11.8% | 0.9611 | 11.8% |
| 5 | `jobs/inverted_index.py:13` | `map` | 5,015,858 | **0.8304** | 10.2% | 2.1879 | 26.8% |

**MPI Master (`rank 0`)** (Total profiled CPU time: `2.9606s`):

| # | Module | Function | Calls | `tottime` (s) | `% tottime` | `cumtime` (s) | `% cumtime` |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | `~:0` | `<built-in method time.sleep>` | 2,554 | **2.7237** | 92.0% | 2.7237 | 92.0% |
| 2 | `mapreduce/master.py:148` | `_poll_messages` | 2,596 | **0.0962** | 3.2% | 0.0963 | 3.2% |
| 3 | `mapreduce/master.py:228` | `run_map_phase` | 1 | **0.0238** | 0.8% | 2.8827 | 97.4% |
| 4 | `mapreduce/master.py:256` | `run_shuffle_and_reduce` | 1 | **0.0221** | 0.7% | 0.0222 | 0.8% |
| 5 | `~:0` | `<built-in method _io.open>` | 2 | **0.0179** | 0.6% | 0.0179 | 0.6% |

**MPI Worker (`rank 1`)** (Total profiled CPU time: `2.9342s`):

| # | Module | Function | Calls | `tottime` (s) | `% tottime` | `cumtime` (s) | `% cumtime` |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | `mapreduce/api.py:118` | `group_by_key` | 3,877 | **0.5907** | 20.1% | 1.1309 | 38.5% |
| 2 | `~:0` | `<method 'findall' of 're.Pattern' objects>` | 79,990 | **0.3401** | 11.6% | 0.3401 | 11.6% |
| 3 | `~:0` | `<method 'extend' of 'list' objects>` | 83,862 | **0.3355** | 11.4% | 1.0455 | 35.6% |
| 4 | `~:0` | `<method 'setdefault' of 'dict' objects>` | 1,176,904 | **0.3048** | 10.4% | 0.3048 | 10.4% |
| 5 | `jobs/inverted_index.py:13` | `map` | 1,253,233 | **0.2557** | 8.7% | 0.6869 | 23.4% |

## 6. Generated Plots

- `benchmarks/results/strong_scaling_speedup.png` — Strong scaling speedup $S_p$ vs. worker count $p$ with fitted Amdahl's Law curves.
- `benchmarks/results/strong_scaling_efficiency.png` — Parallel efficiency $E_p = S_p/p$ and Karp–Flatt serial fraction $e(p)$.
- `benchmarks/results/phase_breakdown_comm_comp.png` — Map/Shuffle/Reduce master wall-clock breakdown and $T_{\text{comm}}/T_{\text{comp}}$ ratio.
- `benchmarks/results/weak_scaling.png` — Weak scaling execution time, weak efficiency, and aggregate throughput.
- `benchmarks/results/fault_recovery.png` — Healthy vs. fault-injected execution time and task re-dispatch overhead.
- `benchmarks/results/cprofile_hotspots.png` — Top hotspot functions from `cProfile` (`pstats`).
