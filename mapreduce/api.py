"""Public API of the lightweight MapReduce framework (Stage 1 contract).

Every job (Word Frequency, Inverted Index) and both engines (the sequential
baseline and the MPI engine) are written against this module only.

Data model
----------
* Input is a list of text files. A file is cut into byte ranges
  (:class:`InputSplit`) by ``mapreduce.splitter``.
* A record handed to ``Mapper.map`` is ``key=(doc_id, offset)`` and
  ``value=line`` where ``doc_id`` is the file's basename, ``offset`` is the
  byte offset of the line start in that file and ``line`` is the decoded line
  without its trailing newline.
* Mappers and combiners emit ``(key, value)`` pairs. Keys are ``str``.
* Reducers emit final ``(key, value)`` pairs; ``value`` must be ``str`` or
  ``int`` so it can be written as ``key<TAB>value``.
* Final output is sorted by key (see :func:`write_output`), so the sequential
  and parallel engines produce byte-identical files.
"""
from __future__ import annotations

import zlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Iterator, Optional, Tuple

KeyValue = Tuple[str, Any]


@dataclass(frozen=True)
class InputSplit:
    """A line-aligned byte range ``[start, end)`` of one input file.

    A split owns every line whose first byte lies in ``[start, end)``.
    """

    split_id: int
    path: str
    start: int
    end: int

    @property
    def length(self) -> int:
        return self.end - self.start


@dataclass
class JobContext:
    """Per-task context passed to user code.

    ``rank`` is the MPI rank running the task (0 for the sequential engine),
    ``num_partitions`` is the number of reduce partitions R.
    """

    job_name: str
    rank: int = 0
    num_partitions: int = 1
    split: Optional[InputSplit] = None
    counters: dict = field(default_factory=dict)

    def increment(self, name: str, amount: int = 1) -> None:
        self.counters[name] = self.counters.get(name, 0) + amount


class Mapper(ABC):
    def setup(self, ctx: JobContext) -> None:
        """Called once per task before the first record."""

    @abstractmethod
    def map(self, key: Tuple[str, int], value: str, ctx: JobContext) -> Iterable[KeyValue]:
        """Turn one input line into zero or more intermediate pairs."""

    def cleanup(self, ctx: JobContext) -> Iterable[KeyValue]:
        """Called once per task after the last record; may emit pairs."""
        return ()


class Reducer(ABC):
    def setup(self, ctx: JobContext) -> None:
        """Called once per partition before the first key."""

    @abstractmethod
    def reduce(self, key: str, values: Iterable[Any], ctx: JobContext) -> Iterable[KeyValue]:
        """Aggregate all values of one key."""


class Combiner(Reducer):
    """Map-side reducer used for skew handling.

    Runs on the worker over that worker's map output before the shuffle.
    It must emit pairs of the *same shape as the mapper output* and must be
    safe to apply zero, one or many times.
    """


@dataclass(frozen=True)
class Job:
    """A MapReduce job definition: factories for the user classes."""

    name: str
    mapper: Callable[[], Mapper]
    reducer: Callable[[], Reducer]
    combiner: Optional[Callable[[], Combiner]] = None


def partition(key: str, num_partitions: int) -> int:
    """Stable hash partitioner: ``hash(key) % R``.

    Python's built-in ``hash`` of ``str`` is salted per process
    (PYTHONHASHSEED), so different MPI ranks would disagree. CRC32 is stable
    across processes and runs.
    """
    if num_partitions <= 0:
        raise ValueError("num_partitions must be positive")
    return zlib.crc32(key.encode("utf-8")) % num_partitions


def group_by_key(pairs: Iterable[KeyValue]) -> Iterator[Tuple[str, list]]:
    """Group pairs by key, yielding keys in sorted order."""
    groups: dict = {}
    for k, v in pairs:
        groups.setdefault(k, []).append(v)
    for k in sorted(groups):
        yield k, groups[k]


def run_reducer(reducer: Reducer, pairs: Iterable[KeyValue], ctx: JobContext) -> list:
    """Apply a reducer (or combiner) to unsorted pairs; returns emitted pairs."""
    reducer.setup(ctx)
    out: list = []
    for k, vs in group_by_key(pairs):
        out.extend(reducer.reduce(k, vs, ctx))
    return out


def write_output(pairs: Iterable[KeyValue], path: str) -> None:
    """Write final pairs as ``key<TAB>value`` lines sorted by key (UTF-8, LF)."""
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for k, v in sorted(pairs, key=lambda kv: kv[0]):
            f.write(f"{k}\t{v}\n")
