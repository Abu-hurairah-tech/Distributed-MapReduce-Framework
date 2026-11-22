#!/usr/bin/env python3
"""Dataset preparation utility for the Lightweight MPI MapReduce project.

Supports three data sources (500 MB - 2 GB target range by default):
  1. ``gutenberg`` : Downloads classic Project Gutenberg plain-text books and
                     replicates them across separate document files up to
                     ``--size-mb`` so Inverted Index has meaningful document IDs.
  2. ``nasa``      : Downloads the NASA Kennedy Space Center HTTP access log
                     (July 1995, gzip), decompresses, and truncates/replicates
                     it across log shards up to ``--size-mb``.
  3. ``synthetic`` : Fast, deterministic offline generator producing Zipf-
                     distributed text lines and NASA-style Common Log Format
                     HTTP access log lines across N document files.

Usage examples:
  python data/prepare_datasets.py --source synthetic --size-mb 500 --out data/raw/synthetic --seed 42
  python data/prepare_datasets.py --source gutenberg --size-mb 500 --out data/raw/gutenberg
  python data/prepare_datasets.py --source nasa --size-mb 500 --out data/raw/nasa
"""
from __future__ import annotations

import argparse
import gzip
import itertools
import os
import random
import shutil
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import List, Sequence, Tuple

MIB = 1024 * 1024

# Curated public-domain Project Gutenberg UTF-8 plain-text books
GUTENBERG_BOOKS: Sequence[Tuple[str, str]] = (
    ("pg1342_pride_and_prejudice", "https://www.gutenberg.org/cache/epub/1342/pg1342.txt"),
    ("pg2701_moby_dick", "https://www.gutenberg.org/cache/epub/2701/pg2701.txt"),
    ("pg11_alice_in_wonderland", "https://www.gutenberg.org/cache/epub/11/pg11.txt"),
    ("pg84_frankenstein", "https://www.gutenberg.org/cache/epub/84/pg84.txt"),
    ("pg1661_sherlock_holmes", "https://www.gutenberg.org/cache/epub/1661/pg1661.txt"),
    ("pg98_tale_of_two_cities", "https://www.gutenberg.org/cache/epub/98/pg98.txt"),
    ("pg174_dorian_gray", "https://www.gutenberg.org/cache/epub/174/pg174.txt"),
    ("pg345_dracula", "https://www.gutenberg.org/cache/epub/345/pg345.txt"),
    ("pg1400_great_expectations", "https://www.gutenberg.org/cache/epub/1400/pg1400.txt"),
    ("pg2600_war_and_peace", "https://www.gutenberg.org/cache/epub/2600/pg2600.txt"),
)

# NASA HTTP Jul95 access log mirrors (Internet Traffic Archive)
NASA_LOG_URLS: Sequence[str] = (
    "https://ita.ee.lbl.gov/traces/NASA_access_log_Jul95.gz",
    "http://ita.ee.lbl.gov/traces/NASA_access_log_Jul95.gz",
)

# Common English + parallel-computing core vocabulary for Zipfian synthetic text
CORE_WORDS: Sequence[str] = (
    "the", "of", "and", "to", "in", "a", "is", "that", "for", "it",
    "as", "was", "with", "be", "by", "on", "not", "he", "i", "this",
    "are", "or", "his", "from", "at", "which", "but", "have", "an", "had",
    "they", "you", "were", "their", "one", "all", "we", "can", "her", "has",
    "there", "been", "if", "more", "when", "will", "would", "who", "so", "no",
    "out", "up", "said", "what", "its", "about", "into", "than", "them", "some",
    "could", "time", "only", "other", "new", "two", "may", "then", "do", "first",
    "any", "my", "now", "such", "like", "our", "over", "man", "me", "even",
    "most", "made", "after", "also", "did", "many", "before", "must", "through", "back",
    "years", "where", "much", "your", "way", "well", "down", "should", "because", "each",
    "parallel", "distributed", "mapreduce", "mpi", "worker", "master", "shuffle", "reduce",
    "combiner", "partition", "heartbeat", "split", "cluster", "latency", "throughput", "speedup",
    "efficiency", "amdahl", "flynn", "mimd", "process", "rank", "message", "collective",
    "alltoallv", "gather", "buffer", "pickle", "socket", "memory", "bandwidth", "scaling",
    "strong", "weak", "fault", "tolerance", "timeout", "ledger", "dispatch", "baseline",
    "sequential", "token", "index", "inverted", "frequency", "corpus", "document", "offset",
    "stream", "compute", "network", "protocol", "barrier", "group", "communicator", "kernel",
    "thread", "core", "cache", "pipeline", "schedule", "queue", "payload", "checksum",
)

WORD_PREFIXES: Sequence[str] = (
    "astro", "cyber", "data", "eigen", "flux", "geo", "hyper", "inter",
    "kilo", "logic", "macro", "micro", "nano", "omni", "para", "quant",
    "radio", "servo", "tele", "ultra", "vect", "wave", "xeno", "zeta",
)

WORD_STEMS: Sequence[str] = (
    "node", "link", "grid", "mesh", "port", "frame", "block", "shard",
    "task", "job", "stage", "phase", "cycle", "pulse", "probe", "trace",
    "graph", "tree", "hash", "table", "list", "array", "stack", "heap",
    "orbit", "lunar", "solar", "probe", "relay", "radar", "craft", "field",
)

NASA_HOSTS: Sequence[str] = (
    "199.72.81.55", "unicomp6.unicomp.net", "199.120.110.21", "burger.letters.com",
    "199.33.32.50", "d104.aa.net", "129.94.144.152", "alyssa.prodigy.com",
    "205.189.154.54", "piweba3y.prodigy.com", "198.116.142.34", "gw1.att.com",
    "163.206.89.4", "titan.ksc.nasa.gov", "apollo.msfc.nasa.gov", "voyager.jpl.nasa.gov",
)

NASA_ENDPOINTS: Sequence[str] = (
    "/history/apollo/apollo.html",
    "/history/apollo/apollo-11/apollo-11.html",
    "/history/apollo/apollo-13/apollo-13.html",
    "/shuttle/missions/sts-71/mission-sts-71.html",
    "/shuttle/missions/sts-71/images/images.html",
    "/shuttle/missions/sts-70/mission-sts-70.html",
    "/shuttle/countdown/countdown.html",
    "/shuttle/countdown/liftoff.html",
    "/shuttle/technology/sts-newsref/stsref-toc.html",
    "/facilities/lc39a.html",
    "/images/ksclogo-medium.gif",
    "/images/NASA-logosmall.gif",
    "/images/KSC-logosmall.gif",
    "/images/launch-logo.gif",
    "/htbin/cdt_main.pl",
    "/ksc.html",
)

NASA_STATUS_CODES: Sequence[Tuple[int, int]] = (
    (200, 6245),
    (200, 12840),
    (200, 3985),
    (200, 48210),
    (304, 0),
    (302, 184),
    (404, 0),
)


def _clean_output_dir(out_dir: Path) -> None:
    """Ensure the output directory exists and remove previous generated files."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for child in out_dir.iterdir():
        if child.is_file() and child.name != ".gitkeep":
            child.unlink()


def _download_bytes(url: str, timeout: float = 30.0) -> bytes:
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "MapReduce-MPI-TermProject/1.0 (Academic Benchmark)"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def _write_exact_lines(path: Path, chunks: Sequence[bytes], target_bytes: int, rng: random.Random) -> int:
    """Write deterministically sampled chunks to ``path`` up to ``target_bytes`` (LF-terminated)."""
    if target_bytes <= 0:
        return 0
    written = 0
    num_chunks = len(chunks)
    with open(path, "wb") as f:
        while written < target_bytes:
            chunk = chunks[rng.randrange(num_chunks)]
            remaining = target_bytes - written
            if len(chunk) <= remaining:
                f.write(chunk)
                written += len(chunk)
            else:
                if remaining == 1:
                    f.write(b"\n")
                    written += 1
                else:
                    slice_part = chunk[: remaining - 1].replace(b"\n", b" ")
                    f.write(slice_part)
                    f.write(b"\n")
                    written += remaining
                break
    return written


def prepare_gutenberg(out_dir: Path, size_mb: float, num_docs: int) -> Tuple[int, int]:
    """Download Project Gutenberg books and replicate across separate doc files."""
    _clean_output_dir(out_dir)
    target_bytes = int(round(size_mb * MIB))
    downloaded: List[Tuple[str, bytes]] = []

    for slug, url in GUTENBERG_BOOKS:
        try:
            print(f"[gutenberg] Downloading {slug} from {url} ...", flush=True)
            raw = _download_bytes(url)
            # Normalize CRLF -> LF so byte splits see uniform LF lines
            normalized = raw.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
            if not normalized.endswith(b"\n"):
                normalized += b"\n"
            downloaded.append((slug, normalized))
        except (urllib.error.URLError, OSError) as exc:
            print(f"[gutenberg] Warning: failed to fetch {url}: {exc}", file=sys.stderr)

    if not downloaded:
        raise RuntimeError(
            "Unable to download any Project Gutenberg books (check network connection, "
            "or use '--source synthetic' for offline dataset generation)."
        )

    doc_count = max(num_docs, len(downloaded))
    base_per_doc = target_bytes // doc_count
    remainder = target_bytes % doc_count

    total_written = 0
    files_created = 0
    for idx in range(doc_count):
        doc_target = base_per_doc + (1 if idx < remainder else 0)
        if doc_target <= 0:
            continue
        slug, content = downloaded[idx % len(downloaded)]
        doc_path = out_dir / f"gutenberg_{idx + 1:03d}_{slug}.txt"
        written = 0
        with open(doc_path, "wb") as f:
            while written < doc_target:
                rem = doc_target - written
                if len(content) <= rem:
                    f.write(content)
                    written += len(content)
                else:
                    if rem == 1:
                        f.write(b"\n")
                        written += 1
                    else:
                        head = content[: rem - 1]
                        last_nl = head.rfind(b"\n")
                        if last_nl != -1 and last_nl > 0:
                            f.write(head[: last_nl + 1])
                            pad = rem - (last_nl + 1)
                            if pad > 1:
                                f.write(b"a" * (pad - 1) + b"\n")
                            elif pad == 1:
                                f.write(b"\n")
                        else:
                            f.write(head.replace(b"\n", b" ") + b"\n")
                        written += rem
                    break
        total_written += written
        files_created += 1

    return files_created, total_written


def prepare_nasa(out_dir: Path, size_mb: float, num_docs: int) -> Tuple[int, int]:
    """Download NASA HTTP Jul95 access log (gzip) and decompress/truncate to size_mb."""
    _clean_output_dir(out_dir)
    target_bytes = int(round(size_mb * MIB))
    gz_cache = out_dir / ".nasa_jul95.gz.tmp"

    last_err: Exception | None = None
    for url in NASA_LOG_URLS:
        try:
            print(f"[nasa] Downloading {url} ...", flush=True)
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "MapReduce-MPI-TermProject/1.0 (Academic Benchmark)"},
            )
            with urllib.request.urlopen(req, timeout=60.0) as resp, open(gz_cache, "wb") as out_f:
                shutil.copyfileobj(resp, out_f, length=MIB)
            last_err = None
            break
        except (urllib.error.URLError, OSError) as exc:
            last_err = exc
            print(f"[nasa] Warning: failed to fetch {url}: {exc}", file=sys.stderr)

    if last_err is not None or not gz_cache.exists():
        raise RuntimeError(
            f"Unable to download NASA HTTP access log ({last_err}). "
            "Use '--source synthetic' for offline NASA log + Zipf generation."
        )

    try:
        # Decompress up to one pass (or target_bytes) into shard files
        doc_count = max(1, num_docs)
        base_per_doc = target_bytes // doc_count
        remainder = target_bytes % doc_count

        total_written = 0
        files_created = 0
        with gzip.open(gz_cache, "rb") as gz_in:
            for idx in range(doc_count):
                doc_target = base_per_doc + (1 if idx < remainder else 0)
                if doc_target <= 0:
                    continue
                shard_path = out_dir / f"nasa_jul95_{idx + 1:03d}.log"
                written = 0
                with open(shard_path, "wb") as out_f:
                    while written < doc_target:
                        line = gz_in.readline()
                        if not line:
                            gz_in.seek(0)
                            line = gz_in.readline()
                            if not line:
                                break
                        line = line.replace(b"\r\n", b"\n")
                        if not line.endswith(b"\n"):
                            line += b"\n"
                        rem = doc_target - written
                        if len(line) <= rem:
                            out_f.write(line)
                            written += len(line)
                        else:
                            if rem == 1:
                                out_f.write(b"\n")
                            else:
                                out_f.write(line[: rem - 1].replace(b"\n", b" ") + b"\n")
                            written += rem
                            break
                total_written += written
                files_created += 1
        return files_created, total_written
    finally:
        if gz_cache.exists():
            gz_cache.unlink()


def _build_vocabulary() -> Tuple[List[str], List[str]]:
    """Return (core_words, topic_words) for deterministic Zipf generation."""
    core = list(CORE_WORDS)
    topic: List[str] = []
    for idx, (pref, stem) in enumerate(itertools.product(WORD_PREFIXES, WORD_STEMS), start=1):
        topic.append(f"{pref}{stem}{idx:03d}")
    return core, topic


def prepare_synthetic(
    out_dir: Path,
    size_mb: float,
    num_docs: int,
    seed: int,
    zipf_s: float = 1.08,
) -> Tuple[int, int]:
    """Deterministically generate Zipf-distributed words + NASA HTTP log lines across N doc files."""
    _clean_output_dir(out_dir)
    target_bytes = int(round(size_mb * MIB))
    if target_bytes <= 0:
        return 0, 0

    rng = random.Random(seed)
    core_words, topic_words = _build_vocabulary()

    # Zipf cumulative weights: w_r = 1 / r^s
    core_cum = list(
        itertools.accumulate(1.0 / ((r + 1) ** zipf_s) for r in range(len(core_words)))
    )

    doc_count = max(1, min(num_docs, target_bytes // 64 or 1))
    num_topic_groups = max(4, min(16, doc_count))
    group_size = max(16, len(topic_words) // num_topic_groups)

    # Pre-sample global Zipf prose lines (12-16 words per line)
    global_prose_lines: List[bytes] = []
    sampled_core = rng.choices(core_words, cum_weights=core_cum, k=256 * 14)
    for i in range(0, len(sampled_core), 14):
        line_str = " ".join(sampled_core[i : i + 14]) + "\n"
        global_prose_lines.append(line_str.encode("utf-8"))

    # Pre-sample NASA-style Common Log Format HTTP access lines
    nasa_lines: List[bytes] = []
    for i in range(128):
        host = rng.choice(NASA_HOSTS)
        day = rng.randint(1, 28)
        hour = rng.randint(0, 23)
        minute = rng.randint(0, 59)
        second = rng.randint(0, 59)
        endpoint = rng.choice(NASA_ENDPOINTS)
        status, byte_cnt = rng.choice(NASA_STATUS_CODES)
        if byte_cnt > 0:
            byte_cnt += rng.randint(0, 512)
        log_line = (
            f'{host} - - [{day:02d}/Jul/1995:{hour:02d}:{minute:02d}:{second:02d} -0400] '
            f'"GET {endpoint} HTTP/1.0" {status} {byte_cnt}\n'
        )
        nasa_lines.append(log_line.encode("utf-8"))

    # Pre-sample topic-specific Zipf lines so Inverted Index has selective posting lists
    topic_group_lines: List[List[bytes]] = []
    for g in range(num_topic_groups):
        start_i = (g * group_size) % len(topic_words)
        group_vocab = topic_words[start_i : start_i + group_size]
        if len(group_vocab) < 16:
            group_vocab = topic_words[:32]
        group_cum = list(
            itertools.accumulate(1.0 / ((r + 1) ** zipf_s) for r in range(len(group_vocab)))
        )
        sampled_topic = rng.choices(group_vocab, cum_weights=group_cum, k=64 * 6)
        sampled_mix_core = rng.choices(core_words, cum_weights=core_cum, k=64 * 6)
        g_lines: List[bytes] = []
        for row in range(64):
            words = sampled_topic[row * 6 : (row + 1) * 6] + sampled_mix_core[row * 6 : (row + 1) * 6]
            rng.shuffle(words)
            g_lines.append((" ".join(words) + "\n").encode("utf-8"))
        topic_group_lines.append(g_lines)

    base_per_doc = target_bytes // doc_count
    remainder = target_bytes % doc_count

    total_written = 0
    files_created = 0

    for doc_idx in range(doc_count):
        doc_target = base_per_doc + (1 if doc_idx < remainder else 0)
        if doc_target <= 0:
            continue

        doc_path = out_dir / f"doc_{doc_idx + 1:03d}.txt"
        primary_topic = topic_group_lines[doc_idx % num_topic_groups]
        secondary_topic = topic_group_lines[(doc_idx * 3 + 1) % num_topic_groups]
        unique_doc_marker = f"docmarker{doc_idx + 1:03d} ".encode("utf-8")

        # Build 16 reusable multi-line chunks (~4-8 KB each) tailored to this document
        doc_chunks: List[bytes] = []
        for c_idx in range(16):
            buf = bytearray()
            if c_idx == 0:
                buf.extend(unique_doc_marker)
                buf.extend(primary_topic[0])
            for _ in range(40):
                buf.extend(rng.choice(global_prose_lines))
            for _ in range(16):
                buf.extend(rng.choice(primary_topic))
            for _ in range(6):
                buf.extend(rng.choice(secondary_topic))
            for _ in range(12):
                buf.extend(rng.choice(nasa_lines))
            doc_chunks.append(bytes(buf))

        written = _write_exact_lines(doc_path, doc_chunks, doc_target, rng)
        total_written += written
        files_created += 1

    return files_created, total_written


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare text and log datasets (500 MB - 2 GB) for MPI MapReduce benchmarks."
    )
    parser.add_argument(
        "--source",
        choices=("gutenberg", "nasa", "synthetic"),
        default="synthetic",
        help="Dataset source: 'gutenberg', 'nasa', or offline 'synthetic' (default: synthetic).",
    )
    parser.add_argument(
        "--size-mb",
        type=float,
        default=500.0,
        help="Target corpus size in MiB (default: 500; proposal range: 500 MB - 2 GB).",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output directory (default: data/raw/<source>).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Deterministic PRNG seed for '--source synthetic' (default: 42).",
    )
    parser.add_argument(
        "--num-docs",
        type=int,
        default=32,
        help="Number of separate document files to generate (default: 32).",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.size_mb <= 0:
        print("Error: --size-mb must be positive.", file=sys.stderr)
        return 1
    if args.num_docs <= 0:
        print("Error: --num-docs must be positive.", file=sys.stderr)
        return 1

    out_dir = Path(args.out) if args.out else Path("data") / "raw" / args.source
    t0 = time.perf_counter()

    if args.source == "synthetic":
        files_created, total_bytes = prepare_synthetic(
            out_dir=out_dir,
            size_mb=args.size_mb,
            num_docs=args.num_docs,
            seed=args.seed,
        )
    elif args.source == "gutenberg":
        files_created, total_bytes = prepare_gutenberg(
            out_dir=out_dir,
            size_mb=args.size_mb,
            num_docs=args.num_docs,
        )
    elif args.source == "nasa":
        files_created, total_bytes = prepare_nasa(
            out_dir=out_dir,
            size_mb=args.size_mb,
            num_docs=args.num_docs,
        )
    else:
        raise ValueError(f"Unsupported source: {args.source}")

    elapsed = time.perf_counter() - t0
    actual_mb = total_bytes / MIB
    rate_mb_s = actual_mb / elapsed if elapsed > 0 else 0.0
    print(
        f"[{args.source}] Generated {files_created} files ({total_bytes:,} bytes / "
        f"{actual_mb:.2f} MiB) in '{out_dir}' in {elapsed:.2f}s ({rate_mb_s:.1f} MiB/s)."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
