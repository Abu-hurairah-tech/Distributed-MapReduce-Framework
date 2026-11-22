"""Master/worker message protocol (Stage 1 contract).

All point-to-point messages are pickled Python objects sent with mpi4py's
lowercase API (``comm.isend`` / ``comm.recv``) on ``MPI.COMM_WORLD``.
Rank 0 is the master; ranks 1..N-1 are workers. No threads are used: workers
send heartbeats from inside their map loop, and the master polls with
``Iprobe``.

Phases
------
1. MAP     master -> worker  TAG_TASK      MapTask
           worker -> master  TAG_HEARTBEAT Heartbeat (every HEARTBEAT_INTERVAL s)
           worker -> master  TAG_DONE      TaskDone (ack; master updates ledger)
   The master keeps an in-flight ledger task_id -> (worker, dispatch time)
   and a completed ledger task_id -> worker. A worker silent for longer than
   TIMEOUT_TAU seconds is declared failed: its in-flight AND completed tasks
   are re-queued (its map output lived only in its memory) and it is excluded
   from later phases.
2. SHUFFLE master -> workers TAG_CONTROL   ShufflePlan(alive ranks)
           The master and alive workers build a sub-communicator with
           ``comm.Create_group`` (only members participate, so a silent
           worker cannot hang it). Workers hash-partition combined map output
           with ``api.partition(key, R)`` (R = number of alive workers) and
           exchange it with ``Alltoallv`` on pickled byte buffers; the master
           contributes empty buffers.
3. REDUCE  each alive worker reduces its partition; results are gathered to
           the master over the sub-communicator, which writes the output.
4. STOP    master -> every worker TAG_CONTROL Shutdown.

Fault injection (simulated failure, see the proposal's scope note): a worker
told to fail stops sending heartbeats and acks after ``fail_after`` tasks and
waits for Shutdown. Standard MPI would abort the job if a rank really died.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

from .api import InputSplit

MASTER = 0

TAG_TASK = 1
TAG_DONE = 2
TAG_HEARTBEAT = 3
TAG_CONTROL = 4

HEARTBEAT_INTERVAL = 0.5  # seconds between worker heartbeats
TIMEOUT_TAU = 3.0  # seconds of silence before the master declares a worker failed


@dataclass(frozen=True)
class MapTask:
    task_id: int
    split: InputSplit


@dataclass(frozen=True)
class TaskDone:
    task_id: int
    rank: int
    records_in: int
    pairs_out: int
    compute_seconds: float


@dataclass(frozen=True)
class Heartbeat:
    rank: int
    task_id: Optional[int]
    sent_at: float


@dataclass(frozen=True)
class ShufflePlan:
    alive: Tuple[int, ...]  # world ranks of alive workers, sorted
    accepted: Dict[int, Tuple[int, ...]] = field(default_factory=dict)
    # rank -> task_ids whose output that worker must contribute to the shuffle
    # (a re-dispatched task may have run twice; only the accepted copy counts)


@dataclass(frozen=True)
class Shutdown:
    reason: str = "done"
