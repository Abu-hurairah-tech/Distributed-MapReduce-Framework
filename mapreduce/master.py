"""Master process implementation for the MPI MapReduce framework (Stage 3)."""
from __future__ import annotations

import array
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Deque, Dict, List, Optional, Sequence, Set, Tuple

from mpi4py import MPI

from . import api
from .protocol import (
    Heartbeat,
    MASTER,
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

    def to_dict(self, job_name: str, num_workers: int) -> Dict[str, Any]:
        return {
            "engine": "mpi",
            "job": job_name,
            "workers": num_workers,
            "input_bytes": self.input_bytes,
            "num_splits": self.num_splits,
            "total_seconds": round(self.total_seconds, 6),
            "map_seconds": round(self.map_seconds, 6),
            "shuffle_seconds": round(self.shuffle_seconds, 6),
            "reduce_seconds": round(self.reduce_seconds, 6),
            "comp_seconds": round(self.comp_seconds, 6),
            "comm_seconds": round(self.comm_seconds, 6),
            "failed_ranks": sorted(self.failed_ranks),
            "redispatched_tasks": self.redispatched_tasks,
            "recovery_seconds": round(self.recovery_seconds, 6),
        }


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
    """Rank-0 coordinator that partitions input, schedules Map tasks, and drives Shuffle/Reduce."""

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

        # Fault-tolerance recovery tracking
        self.recovery_start_time: Optional[float] = None
        self.redispatched_pending: Set[int] = set()

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
                    if msg.task_id in self.redispatched_pending:
                        self.redispatched_pending.discard(msg.task_id)
                        if (
                            not self.redispatched_pending
                            and self.recovery_start_time is not None
                        ):
                            self.metrics.recovery_seconds = now - self.recovery_start_time
        return received_any

    def _check_timeouts(self) -> None:
        """Detect workers silent for longer than timeout tau and re-queue their tasks."""
        now = time.perf_counter()
        timed_out: List[int] = []
        for w in list(self.alive_workers):
            if now - self.last_heard.get(w, now) > self.timeout:
                timed_out.append(w)

        for w in sorted(timed_out):
            self._handle_worker_failure(w, now)

    def _handle_worker_failure(self, worker: int, detected_at: float) -> None:
        """Mark worker as failed, remove from alive set, and re-queue in-flight + completed tasks."""
        if worker not in self.alive_workers:
            return
        self.alive_workers.remove(worker)
        if not self.alive_workers:
            raise RuntimeError("All MPI workers failed; cannot complete MapReduce job")

        self.metrics.failed_ranks.append(worker)
        self.idle_workers = deque(w for w in self.idle_workers if w != worker)
        self.worker_in_flight.pop(worker, None)

        to_requeue: List[int] = []
        for tid, (w_rank, _) in list(self.in_flight.items()):
            if w_rank == worker:
                self.in_flight.pop(tid, None)
                to_requeue.append(tid)

        for tid, w_rank in list(self.completed.items()):
            if w_rank == worker:
                self.completed.pop(tid, None)
                self.task_results.pop(tid, None)
                to_requeue.append(tid)

        to_requeue.sort()
        if to_requeue and self.recovery_start_time is None:
            self.recovery_start_time = detected_at

        for tid in to_requeue:
            self.pending_tasks.append(MapTask(task_id=tid, split=self.splits_by_id[tid]))
            self.redispatched_pending.add(tid)
            self.metrics.redispatched_tasks += 1

    def run_map_phase(self) -> Dict[int, int]:
        """Run the dynamic Map task queue until all splits are acknowledged by alive workers."""
        t0 = time.perf_counter()
        total_tasks = len(self.splits)
        now = time.perf_counter()
        for w in self.alive_workers:
            self.last_heard[w] = now

        while len(self.completed) < total_tasks:
            self._dispatch_available()
            got_msg = self._poll_messages()
            self._check_timeouts()
            if len(self.completed) < total_tasks and not got_msg:
                time.sleep(0.0005)

        self.metrics.map_seconds = time.perf_counter() - t0
        return dict(self.completed)

    def build_shuffle_plan(self) -> ShufflePlan:
        """Build the ShufflePlan listing sorted alive workers and accepted task_ids per worker."""
        alive_sorted = tuple(sorted(self.alive_workers))
        by_worker: Dict[int, List[int]] = {w: [] for w in alive_sorted}
        for task_id, worker_rank in sorted(self.completed.items()):
            if worker_rank in by_worker:
                by_worker[worker_rank].append(task_id)
        accepted = {w: tuple(tids) for w, tids in by_worker.items()}
        return ShufflePlan(alive=alive_sorted, accepted=accepted)

    def run_shuffle_and_reduce(self) -> List[api.KeyValue]:
        """Coordinate the sub-communicator Shuffle (Alltoallv) and gather Reduce outputs."""
        t_shuf0 = time.perf_counter()
        plan = self.build_shuffle_plan()
        for w in plan.alive:
            self._send_async(plan, dest=w, tag=TAG_CONTROL)

        members = [MASTER] + list(plan.alive)
        world_group = self.comm.Get_group()
        sub_group = world_group.Incl(members)
        sub_comm = self.comm.Create_group(sub_group, tag=TAG_CONTROL)
        sub_group.Free()
        world_group.Free()

        try:
            sub_size = sub_comm.Get_size()
            send_counts = array.array("i", [0] * sub_size)
            recv_counts = array.array("i", [0] * sub_size)
            sub_comm.Alltoall([send_counts, MPI.INT], [recv_counts, MPI.INT])

            send_displs = array.array("i", [0] * sub_size)
            recv_displs = array.array("i", [0] * sub_size)
            send_buf = b""
            recv_buf = bytearray(sum(recv_counts))
            sub_comm.Alltoallv(
                [send_buf, send_counts, send_displs, MPI.BYTE],
                [recv_buf, recv_counts, recv_displs, MPI.BYTE],
            )
            sub_comm.Barrier()
            self.metrics.shuffle_seconds = time.perf_counter() - t_shuf0

            t_red0 = time.perf_counter()
            gathered = sub_comm.gather(None, root=0)
            self.metrics.reduce_seconds = time.perf_counter() - t_red0

            final_pairs: List[api.KeyValue] = []
            total_comp = 0.0
            total_comm = 0.0
            for item in gathered or []:
                if item is None:
                    continue
                pairs, w_comp, w_comm = item
                final_pairs.extend(pairs)
                total_comp += float(w_comp)
                total_comm += float(w_comm)

            self.metrics.comp_seconds = total_comp
            self.metrics.comm_seconds = total_comm
            return final_pairs
        finally:
            if sub_comm != MPI.COMM_NULL:
                sub_comm.Free()

    def shutdown_workers(self, reason: str = "done") -> None:
        """Send Shutdown to all worker ranks."""
        for w in self.all_workers:
            self._send_async(Shutdown(reason=reason), dest=w, tag=TAG_CONTROL)

    def run(self, output_path: str) -> MasterMetrics:
        """Execute full Map -> Shuffle -> Reduce -> Write pipeline on rank 0."""
        t0 = time.perf_counter()
        self.run_map_phase()
        final_pairs = self.run_shuffle_and_reduce()
        self.shutdown_workers(reason="done")
        # Drain any trailing messages that were in flight before Shutdown
        self._poll_messages()
        api.write_output(final_pairs, output_path)
        self.metrics.total_seconds = time.perf_counter() - t0
        return self.metrics
