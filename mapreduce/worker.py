"""Worker process implementation for the MPI MapReduce framework (Stage 3)."""
from __future__ import annotations

import array
import os
import pickle
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
    """Rank 1..N-1 worker that executes Map tasks, runs Shuffle (Alltoallv), and reduces its partition."""

    def __init__(
        self,
        comm: MPI.Comm,
        job: api.Job,
        use_combiner: bool = True,
        heartbeat_interval: float = HEARTBEAT_INTERVAL,
        fail_rank: Optional[int] = None,
        fail_after: int = 0,
        read_split_fn: Optional[Callable[[api.InputSplit], Iterator[Tuple[int, str]]]] = None,
    ) -> None:
        self.comm = comm
        self.rank = comm.Get_rank()
        self.job = job
        self.use_combiner = use_combiner
        self.heartbeat_interval = heartbeat_interval
        self.should_fail = (fail_rank is not None) and (self.rank == fail_rank)
        self.fail_after = max(0, int(fail_after))
        self.failed_silently = False
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
        if self.failed_silently:
            return
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
        if self.should_fail and self.fail_after == 0:
            self.failed_silently = True
            return self.wait_for_shutdown()

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
                    if self.should_fail and self.stats.tasks_completed >= self.fail_after:
                        self.failed_silently = True
                        return self.wait_for_shutdown()
                elif tag == TAG_CONTROL:
                    self._buffered_control = msg
                    return msg
            else:
                self._maybe_send_heartbeat(task_id=None)
                time.sleep(0.0005)

    def run_shuffle_and_reduce(self, plan: ShufflePlan) -> List[api.KeyValue]:
        """Participate in the sub-communicator Alltoallv shuffle, run Reducer, and gather to master."""
        if self.rank not in plan.alive:
            return []

        num_partitions = len(plan.alive)
        if self.rank in plan.accepted:
            accepted_tids = plan.accepted[self.rank]
        else:
            accepted_tids = tuple(sorted(self.task_outputs.keys()))

        buckets: List[List[api.KeyValue]] = [[] for _ in range(num_partitions)]
        for tid in accepted_tids:
            for k, v in self.task_outputs.get(tid, ()):
                p = api.partition(k, num_partitions)
                buckets[p].append((k, v))

        t_comm = time.perf_counter()
        members = [MASTER] + list(plan.alive)
        world_group = self.comm.Get_group()
        sub_group = world_group.Incl(members)
        sub_comm = self.comm.Create_group(sub_group, tag=TAG_CONTROL)
        sub_group.Free()
        world_group.Free()

        try:
            sub_size = sub_comm.Get_size()
            # Sub-communicator rank 0 is master (empty buffer); ranks 1..R are alive workers 0..R-1
            chunks: List[bytes] = [b""]
            for b in buckets:
                chunks.append(pickle.dumps(b, protocol=pickle.HIGHEST_PROTOCOL) if b else b"")

            send_counts = array.array("i", (len(c) for c in chunks))
            recv_counts = array.array("i", [0] * sub_size)
            sub_comm.Alltoall([send_counts, MPI.INT], [recv_counts, MPI.INT])

            send_displs = array.array("i", [0] * sub_size)
            recv_displs = array.array("i", [0] * sub_size)
            for i in range(1, sub_size):
                send_displs[i] = send_displs[i - 1] + send_counts[i - 1]
                recv_displs[i] = recv_displs[i - 1] + recv_counts[i - 1]

            send_buf = b"".join(chunks)
            recv_buf = bytearray(sum(recv_counts))
            sub_comm.Alltoallv(
                [send_buf, send_counts, send_displs, MPI.BYTE],
                [recv_buf, recv_counts, recv_displs, MPI.BYTE],
            )
            sub_comm.Barrier()

            received_pairs: List[api.KeyValue] = []
            mv = memoryview(recv_buf)
            for i in range(sub_size):
                cnt = recv_counts[i]
                if cnt > 0:
                    disp = recv_displs[i]
                    received_pairs.extend(pickle.loads(mv[disp : disp + cnt]))
            self.stats.comm_seconds += time.perf_counter() - t_comm

            t_comp = time.perf_counter()
            ctx = api.JobContext(
                job_name=self.job.name,
                rank=self.rank,
                num_partitions=num_partitions,
            )
            reducer = self.job.reducer()
            reduced_pairs = api.run_reducer(reducer, received_pairs, ctx)
            self.stats.comp_seconds += time.perf_counter() - t_comp

            t_comm = time.perf_counter()
            _ = len(pickle.dumps(reduced_pairs, protocol=pickle.HIGHEST_PROTOCOL))
            self.stats.comm_seconds += time.perf_counter() - t_comm
            payload = (reduced_pairs, self.stats.comp_seconds, self.stats.comm_seconds)
            sub_comm.gather(payload, root=0)
            return reduced_pairs
        finally:
            if sub_comm != MPI.COMM_NULL:
                sub_comm.Free()

    def wait_for_shutdown(self) -> Shutdown:
        """Block until the master sends a Shutdown message, ignoring any queued MapTasks."""
        if isinstance(self._buffered_control, Shutdown):
            return self._buffered_control
        status = MPI.Status()
        while True:
            msg = self.comm.recv(source=MASTER, tag=MPI.ANY_TAG, status=status)
            if status.Get_tag() == TAG_CONTROL and isinstance(msg, Shutdown):
                return msg

    def run(self) -> None:
        """Execute full Worker lifecycle: Map loop -> Shuffle & Reduce -> Wait for Shutdown."""
        ctrl = self.run_map_loop()
        if isinstance(ctrl, ShufflePlan):
            self.run_shuffle_and_reduce(ctrl)
            self.wait_for_shutdown()
        elif isinstance(ctrl, Shutdown):
            return
