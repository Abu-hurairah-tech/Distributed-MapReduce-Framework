"""Worker process implementation for the MPI MapReduce framework (Stage 3)."""
from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Callable, Dict, Iterator, List, Optional, Tuple

from mpi4py import MPI

from . import api
from .protocol import (
    HEARTBEAT_INTERVAL,
    Heartbeat,
    MASTER,
    MapTask,
    ShufflePlan,
    Shutdown,
    TAG_CONTROL,
    TAG_DONE,
    TAG_HEARTBEAT,
    TAG_TASK,
    TaskDone,
)


@dataclass
class WorkerStats:
    """Cumulative computation and communication timers on one worker."""

    comp_seconds: float = 0.0
    comm_seconds: float = 0.0
    tasks_completed: int = 0


def _resolve_read_split(
    read_split_fn: Optional[Callable[[api.InputSplit], Iterator[Tuple[int, str]]]],
) -> Callable[[api.InputSplit], Iterator[Tuple[int, str]]]:
    if read_split_fn is not None:
        return read_split_fn
    from . import splitter

    return splitter.read_split


class Worker:
    """Rank 1..N-1 worker that executes Map tasks, runs the local Combiner, and stores output per task_id."""

    def __init__(
        self,
        comm: MPI.Comm,
        job: api.Job,
        use_combiner: bool = True,
        heartbeat_interval: float = HEARTBEAT_INTERVAL,
        read_split_fn: Optional[Callable[[api.InputSplit], Iterator[Tuple[int, str]]]] = None,
    ) -> None:
        self.comm = comm
        self.rank = comm.Get_rank()
        self.job = job
        self.use_combiner = use_combiner
        self.heartbeat_interval = heartbeat_interval
        self.read_split_fn = _resolve_read_split(read_split_fn)

        # Stored intermediate output per task_id: task_id -> list[(key, value)]
        self.task_outputs: Dict[int, List[api.KeyValue]] = {}
        self.stats = WorkerStats()
        self.last_heartbeat: float = time.perf_counter()
        self.pending_requests: List[MPI.Request] = []
        self._buffered_control: Optional[object] = None

    def _send_async(self, obj: object, dest: int, tag: int) -> None:
        t0 = time.perf_counter()
        req = self.comm.isend(obj, dest=dest, tag=tag)
        self.stats.comm_seconds += time.perf_counter() - t0
        self.pending_requests.append(req)

    def _maybe_send_heartbeat(self, task_id: Optional[int]) -> None:
        now = time.perf_counter()
        if now - self.last_heartbeat >= self.heartbeat_interval:
            self.last_heartbeat = now
            self._send_async(
                Heartbeat(rank=self.rank, task_id=task_id, sent_at=now),
                dest=MASTER,
                tag=TAG_HEARTBEAT,
            )

    def execute_map_task(self, task: MapTask) -> TaskDone:
        """Execute a single MapTask (and local Combiner if enabled) and store output under task.task_id."""
        split = task.split
        doc_id = os.path.basename(split.path)
        num_workers = max(1, self.comm.Get_size() - 1)
        ctx = api.JobContext(
            job_name=self.job.name,
            rank=self.rank,
            num_partitions=num_workers,
            split=split,
        )

        comp_elapsed = 0.0
        t_comp = time.perf_counter()
        mapper = self.job.mapper()
        mapper.setup(ctx)
        comp_elapsed += time.perf_counter() - t_comp

        raw_pairs: List[api.KeyValue] = []
        records_in = 0

        for idx, (offset, line) in enumerate(self.read_split_fn(split)):
            records_in += 1
            t_comp = time.perf_counter()
            emitted = mapper.map((doc_id, offset), line, ctx)
            if emitted:
                raw_pairs.extend(emitted)
            comp_elapsed += time.perf_counter() - t_comp

            if (idx & 0x3FF) == 0:
                self._maybe_send_heartbeat(task.task_id)

        t_comp = time.perf_counter()
        cleanup_pairs = mapper.cleanup(ctx)
        if cleanup_pairs:
            raw_pairs.extend(cleanup_pairs)

        if self.use_combiner and self.job.combiner is not None and raw_pairs:
            combiner = self.job.combiner()
            final_pairs = api.run_reducer(combiner, raw_pairs, ctx)
        else:
            final_pairs = raw_pairs
        comp_elapsed += time.perf_counter() - t_comp

        self.task_outputs[task.task_id] = final_pairs
        self.stats.comp_seconds += comp_elapsed
        self.stats.tasks_completed += 1

        return TaskDone(
            task_id=task.task_id,
            rank=self.rank,
            records_in=records_in,
            pairs_out=len(final_pairs),
            compute_seconds=comp_elapsed,
        )

    def run_map_loop(self) -> Optional[object]:
        """Receive MapTasks from master until a control message (ShufflePlan or Shutdown) arrives."""
        status = MPI.Status()
        while True:
            if self.comm.Iprobe(source=MASTER, tag=MPI.ANY_TAG, status=status):
                tag = status.Get_tag()
                t_comm = time.perf_counter()
                msg = self.comm.recv(source=MASTER, tag=tag)
                self.stats.comm_seconds += time.perf_counter() - t_comm

                if tag == TAG_TASK and isinstance(msg, MapTask):
                    done = self.execute_map_task(msg)
                    self._send_async(done, dest=MASTER, tag=TAG_DONE)
                elif tag == TAG_CONTROL:
                    self._buffered_control = msg
                    return msg
            else:
                self._maybe_send_heartbeat(task_id=None)
                time.sleep(0.0005)
