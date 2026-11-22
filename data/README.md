# Dataset Preparation (`data/`)

This directory holds dataset preparation scripts and local benchmark corpora for the Lightweight Distributed MapReduce framework.

> **Note:** Large generated/downloaded datasets (`500 MB – 2 GB`) under `data/raw/` and sample outputs under `data/samples/` are ignored by Git (`.gitignore`) and generated locally on demand.

---

## Quick Start

Run `data/prepare_datasets.py` from the repository root:

```powershell
# 1. Offline deterministic synthetic generator (default: 500 MiB, 32 doc files, seed 42)
python data/prepare_datasets.py --source synthetic --size-mb 500 --out data/raw/synthetic --seed 42

# 2. Quick 20 MiB synthetic corpus for smoke tests
python data/prepare_datasets.py --source synthetic --size-mb 20 --out data/samples --seed 42

# 3. Project Gutenberg plain-text books (requires internet access)
python data/prepare_datasets.py --source gutenberg --size-mb 500 --out data/raw/gutenberg

# 4. NASA Kennedy Space Center HTTP access log (Jul 1995 gzip, requires internet access)
python data/prepare_datasets.py --source nasa --size-mb 500 --out data/raw/nasa
```

---

## CLI Options

| Flag | Default | Description |
| :--- | :--- | :--- |
| `--source {gutenberg,nasa,synthetic}` | `synthetic` | Dataset source to download or generate. |
| `--size-mb SIZE_MB` | `500` | Target cumulative corpus size in MiB (proposal range: `500` to `2048` MiB). |
| `--out DIR` | `data/raw/<source>` | Destination directory where document/log files are written. |
| `--seed SEED` | `42` | PRNG seed for `--source synthetic` (guarantees byte-identical reproducibility). |
| `--num-docs N` | `32` | Number of separate document/shard files created in `--out`. |

---

## Dataset Sources & Workload Suitability

1. **`--source synthetic` (Offline, Deterministic)**:
   - Produces $N$ separate document files (`doc_001.txt` .. `doc_032.txt`) totaling exact `--size-mb` MiB.
   - Combines **Zipf-distributed words** ($P(r) \propto r^{-1.08}$ over a shared core vocabulary plus topic-clustered vocabularies) with **NASA-style Common Log Format HTTP access lines** (`<host> - - [dd/Jul/1995:hh:mm:ss -0400] "GET <path> HTTP/1.0" <status> <bytes>`).
   - **Word Frequency (`wordfreq`)**: High-rank Zipfian stop words (`"the"`, `"of"`, `"and"`) create realistic key skew to exercise map-side `Combiner` reduction.
   - **Inverted Index (`invindex`)**: Topic-partitioned vocabularies and per-document markers produce selective posting lists across the $N$ document basenames (`doc_001.txt`, `doc_002.txt`, ...).
2. **`--source gutenberg` (Online)**:
   - Downloads 10 public-domain UTF-8 books from Project Gutenberg, normalizes line endings to LF (`\n`), and replicates them across `gutenberg_001_<book>.txt .. gutenberg_032_<book>.txt` up to `--size-mb`.
3. **`--source nasa` (Online)**:
   - Downloads `NASA_access_log_Jul95.gz`, streams decompression, and splits/replicates the HTTP log records across `nasa_jul95_001.log .. nasa_jul95_032.log` up to `--size-mb`.
