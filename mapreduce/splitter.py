"""Line-aligned input splitter.

A split is a byte range ``[start, end)`` of one file (:class:`InputSplit`).
Splits are cut at fixed byte offsets without reading the file, so the master
can plan them cheaply. Line alignment is enforced when a split is *read*:
a split owns every line whose first byte lies in ``[start, end)``. A line that
crosses ``end`` is read to its end by the split that owns it, and skipped by
the next split. Every line is therefore seen exactly once.
"""
from __future__ import annotations

import os
from typing import Iterable, Iterator, List, Tuple, Union

from .api import InputSplit

PathArg = Union[str, "os.PathLike[str]"]


def _as_list(paths: Union[PathArg, Iterable[PathArg]]) -> List[str]:
    if isinstance(paths, (str, os.PathLike)):
        paths = [paths]
    return [os.fspath(p) for p in paths]


def list_input_files(paths: Union[PathArg, Iterable[PathArg]]) -> List[str]:
    """Expand ``paths`` into a sorted list of input files.

    Directories are expanded non-recursively; hidden entries (names starting
    with ``.``, such as ``.gitkeep``) and subdirectories inside them are
    skipped. Duplicates are removed.
    """
    files = set()
    for p in _as_list(paths):
        if os.path.isdir(p):
            for name in os.listdir(p):
                full = os.path.join(p, name)
                if not name.startswith(".") and os.path.isfile(full):
                    files.add(os.path.normpath(full))
        elif os.path.isfile(p):
            files.add(os.path.normpath(p))
        else:
            raise FileNotFoundError(f"input path not found: {p}")
    return sorted(files)


def make_splits(paths: Union[PathArg, Iterable[PathArg]], split_size: int) -> List[InputSplit]:
    """Cut every input file into byte ranges of ``split_size`` bytes.

    Each non-empty file yields ``ceil(size / split_size)`` splits (the last
    one may be shorter); empty files yield none. Split ids run 0..k-1 in file
    order.
    """
    if split_size <= 0:
        raise ValueError("split_size must be positive")
    splits: List[InputSplit] = []
    for path in list_input_files(paths):
        size = os.path.getsize(path)
        for start in range(0, size, split_size):
            end = min(start + split_size, size)
            splits.append(InputSplit(len(splits), path, start, end))
    return splits


def _strip_newline(raw: bytes) -> bytes:
    if raw.endswith(b"\r\n"):
        return raw[:-2]
    if raw.endswith(b"\n"):
        return raw[:-1]
    return raw


def read_split(split: InputSplit) -> Iterator[Tuple[int, str]]:
    """Yield ``(line_start_offset, line)`` for every line the split owns.

    The file is read in binary so offsets are exact byte positions. Lines are
    decoded as UTF-8 (``errors="replace"``) without the trailing newline.
    """
    with open(split.path, "rb") as f:
        if split.start > 0:
            # If byte start-1 is '\n', this consumes only it and we land on
            # ``start``; otherwise it skips the rest of a line owned by the
            # previous split.
            f.seek(split.start - 1)
            f.readline()
        pos = f.tell()
        end = split.end
        readline = f.readline
        while pos < end:
            raw = readline()
            if not raw:
                break
            yield pos, _strip_newline(raw).decode("utf-8", errors="replace")
            pos += len(raw)


def input_size(paths: Union[PathArg, Iterable[PathArg]]) -> int:
    """Total size in bytes of all input files."""
    return sum(os.path.getsize(p) for p in list_input_files(paths))
