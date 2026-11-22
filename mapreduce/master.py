"""Master process implementation for the MPI MapReduce framework (Stage 3)."""
from __future__ import annotations

import os
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Deque, Dict, List, Optional, Sequence, Set, Tuple

from mpi4py import MPI

from . import api, protocol
from .protocol import (
    Heartbeat,
    MapTask,
    ShufflePlan,
    Shutdown,
    TAG_CONTROL,
    TAG_DONE,
    TAG_HEARTBEAT,
    TAG_TASK,
    TIMEOUT_TAU,
    TaskDone,
)


@dataclass
class MasterMetrics:
    """Timing and execution metrics collected on the master (rank 0)."""

    input_bytes: int = 0
    num_splits: int = 0
    total_seconds: float = 0.0
    map_seconds: float = 0.0
    shuffle_seconds: float = 0.0
    reduce_seconds: float = 0.0
    comp_seconds: float = 0.0
    comm_seconds: float = 0.0
    failed_ranks: List[int] = field(default_factory=list)
    redispatched_tasks: int = 0
    recovery_seconds: float = 0.0


def _resolve_splitter(
    make_splits_fn: Optional[Callable[[Sequence[str], int], List[api.InputSplit]]],
    input_size_fn: Optional[Callable[[Sequence[str]], int]],
) -> Tuple[Callable[[Sequence[str], int], List[api.InputSplit]], Callable[[Sequence[str]], int]]:
    if make_splits_fn is not None and input_size_fn is not None:
        return make_splits_fn, input_size_fn
    from . import splitter  # imported lazily so master can be unit-tested with custom splitters

    return (
        make_splits_fn or splitter.make_splits,
        input_size_fn or splitter.input_size,
    )


class Master:
    """Rank-0 coordinator that partitions input, schedules Map tasks, and tracks workers."""

    def __init__(
        self,
        comm: MPI.Comm,
        paths: Sequence[str],
        split_size: int = 16 * 1024 * 1024,
        timeout: float = TIMEOUT_TAU,
        make_splits_fn: Optional[Callable[[Sequence[str], int], List[api.InputSplit]]] = None,
        input_size_fn: Optional[Callable[[Sequence[str]], int]] = None,
    ) -> None:
        self.comm = comm
        self.paths = list(paths)
        self.split_size = split_size
        self.timeout = timeout

        num_ranks = comm.Get_size()
        if num_ranks < 2:
            raise ValueError("MPI MapReduce requires at least 2 ranks (1 master + 1 worker)")

        self.all_workers: List[int] = list(range(1, num_ranks))
        self.alive_workers: Set[int] = set(self.all_workers)
        self.idle_workers: Deque[int] = deque(self.all_workers)

        make_splits_impl, input_size_impl = _resolve_splitter(make_splits_fn, input_size_fn)
        self.splits: List[api.InputSplit] = list(make_splits_impl(self.paths, self.split_size))
        self.input_bytes: int = int(input_size_impl(self.paths))

        self.pending_tasks: Deque[MapTask] = deque(
            MapTask(task_id=i, split=split) for i, split in enumerate(self.splits)
        )
        self.splits_by_id: Dict[int, api.InputSplit] = {
            i: split for i, split in enumerate(self.splits)
        }

        # Ledgers: task_id -> (worker_rank, dispatch_timestamp) and task_id -> worker_rank
        self.in_flight: Dict[int, Tuple[int, float]] = {}
        self.worker_in_flight: Dict[int, int] = {}
        self.completed: Dict[int, int] = {}
        self.task_results: Dict[int, TaskDone] = {}

        now = time.perf_counter()
        self.last_heard: Dict[int, float] = {w: now for w in self.all_workers}
        self.pending_requests: List[MPI.Request] = []
        self.metrics = MasterMetrics(
            input_bytes=self.input_bytes,
            num_splits=len(self.splits),
        )

    def _send_async(self, obj: Any, dest: int, tag: int) -> None:
        req = self.comm.isend(obj, dest=dest, tag=tag)
        self.pending_requests.append(req)

    def _dispatch_available(self) -> None:
        while self.pending_tasks and self.idle_workers:
            worker = self.idle_workers.popleft()
            if worker not in self.alive_workers:
                continue
            task = self.pending_tasks.popleft()
            now = time.perf_counter()
            self.in_flight[task.task_id] = (worker, now)
            self.worker_in_flight[worker] = task.task_id
            # Reset timer on assignment so an idle period doesn't cause immediate timeout
            self.last_heard[worker] = now
            self._send_async(task, dest=worker, tag=TAG_TASK)

    def _poll_messages(self) -> bool:
        """Drain all waiting point-to-point messages with Iprobe. Returns True if any arrived."""
        status = MPI.Status()
        received_any = False
        while self.comm.Iprobe(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status):
            received_any = True
            src = status.Get_source()
            tag = status.Get_tag()
            msg = self.comm.recv(source=src, tag=tag)
            now = time.perf_counter()

            if src in self.alive_workers:
                self.last_heard[src] = now

            if tag == TAG_HEARTBEAT and isinstance(msg, Heartbeat):
                continue

            if tag == TAG_DONE and isinstance(msg, TaskDone):
                if src not in self.alive_workers:
                    continue
                if self.worker_in_flight.get(src) == msg.task_id:
                    self.worker_in_flight.pop(src, None)
                    self.idle_workers.append(src)
                if msg.task_id not in self.completed:
                    self.in_flight.pop(msg.task_id, None)
                    self.completed[msg.task_id] = src
                    self.task_results[msg.task_id] = msg
        return received_any

    def run_map_phase(self) -> Dict[int, int]:
        """Run the dynamic Map task queue until all splits are acknowledged."""
        t0 = time.perf_counter()
        total_tasks = len(self.splits)
        now = time.perf_counter()
        for w in self.alive_workers:
            self.last_heard[w] = now

        while len(self.completed) < total_tasks:
            self._dispatch_available()
            got_msg = self._poll_messages()
            if len(self.completed) < total_tasks and not got_msg:
                time.sleep(0.0005)

        self.metrics.map_seconds = time.perf_counter() - t0
        return dict(self.completed)

    def shutdown_workers(self, reason: str = "done") -> None:
        """Send Shutdown to all worker ranks."""
        for w in self.all_workers:
            self._send_async(Shutdown(reason=reason), dest=w, tag=TAG_CONTROL)
