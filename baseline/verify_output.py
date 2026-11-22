"""Verify that two MapReduce TSV output files are identical."""
from __future__ import annotations

import argparse
import sys
from typing import Dict, List, Optional, Tuple


def _read_tsv(path: str) -> Tuple[List[Tuple[str, str]], Dict[str, str]]:
    records: List[Tuple[str, str]] = []
    by_key: Dict[str, str] = {}
    with open(path, "r", encoding="utf-8") as f:
        for lineno, raw in enumerate(f, start=1):
            line = raw.rstrip("\r\n")
            key, sep, value = line.partition("\t")
            if not sep:
                key = f"<line {lineno} malformed>"
                value = line
            records.append((key, value))
            if key in by_key:
                by_key[key] = f"{by_key[key]} (duplicate: {value})"
            else:
                by_key[key] = value
    return records, by_key


def compare_outputs(path_a: str, path_b: str, max_diffs: int = 10) -> Tuple[bool, List[str]]:
    """Compare two TSV output files.

    Returns ``(ok, diff_messages)``.
    """
    records_a, map_a = _read_tsv(path_a)
    records_b, map_b = _read_tsv(path_b)

    if records_a == records_b:
        return True, []

    diffs: List[str] = []
    all_keys = sorted(set(map_a) | set(map_b))
    for key in all_keys:
        val_a = map_a.get(key)
        val_b = map_b.get(key)
        if val_a != val_b:
            diffs.append(f"key {key!r}: A={val_a!r} vs B={val_b!r}")
            if len(diffs) >= max_diffs:
                break

    if not diffs:
        # Same key/value map, but different line ordering or duplicate count.
        limit = max(len(records_a), len(records_b))
        for idx in range(limit):
            rec_a = records_a[idx] if idx < len(records_a) else None
            rec_b = records_b[idx] if idx < len(records_b) else None
            if rec_a != rec_b:
                key = rec_a[0] if rec_a is not None else rec_b[0]
                diffs.append(f"line {idx + 1} (key {key!r}): A={rec_a!r} vs B={rec_b!r}")
                if len(diffs) >= max_diffs:
                    break

    return False, diffs


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Verify that two MapReduce output files are identical."
    )
    parser.add_argument("path_a", metavar="A", help="First output file (e.g. reference).")
    parser.add_argument("path_b", metavar="B", help="Second output file (e.g. candidate).")
    args = parser.parse_args(argv)

    try:
        ok, diffs = compare_outputs(args.path_a, args.path_b)
    except OSError as exc:
        print(f"error reading output files: {exc}", file=sys.stderr)
        return 1

    if ok:
        return 0

    print(f"Outputs differ between {args.path_a} and {args.path_b}:")
    for line in diffs:
        print(f"  {line}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
