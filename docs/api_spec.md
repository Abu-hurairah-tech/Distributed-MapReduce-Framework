# MapReduce Framework Public API Specification (`mapreduce.api`)

This document specifies the frozen Stage 1 public programming interface (`mapreduce/api.py`) for the lightweight distributed MapReduce framework. Both execution engines—the single-process sequential baseline (`baseline/run_sequential.py`) and the multi-process MPI engine (`mr_driver.py`)—as well as all user workloads (`jobs/word_frequency.py`, `jobs/inverted_index.py`) program exclusively against this contract.

---

## 1. Data and Record Model

The framework processes collections of plain-text files through four data transformations:

| Stage | Input Type | Output Type | Description |
| :--- | :--- | :--- | :--- |
| **Input Splitting** | File paths (`list[str]`) | `list[InputSplit]` | Cuts input files into contiguous, half-open byte intervals `[start, end)`. |
| **Map (`Mapper.map`)** | `key: (doc_id, offset)`<br>`value: line` | `Iterable[KeyValue]`<br>(`Tuple[str, Any]`) | Transforms one decoded input line into zero or more intermediate `(key, value)` pairs. |
| **Combine (`Combiner.reduce`)** | `key: str`<br>`values: Iterable[Any]` | `Iterable[KeyValue]`<br>(`Tuple[str, Any]`) | Locally aggregates a worker's map output per key prior to network shuffle (same pair shape as `Mapper` output). |
| **Reduce (`Reducer.reduce`)** | `key: str`<br>`values: Iterable[Any]` | `Iterable[KeyValue]`<br>(`Tuple[str, str \| int]`) | Aggregates all intermediate values for `key` across all mappers into final output records. |

### Map Input Record Format
For every line owned by an `InputSplit`, the split reader (`mapreduce.splitter.read_split`) invokes `Mapper.map(key, value, ctx)` with:
- **`key = (doc_id, offset)`** (`Tuple[str, int]`):
  - `doc_id` (`str`): The basename of the source file (`os.path.basename(split.path)`), enabling document-aware workloads such as Inverted Index without passing full filesystem paths into outputs.
  - `offset` (`int`): The zero-based byte offset of the first byte of the line within that file.
- **`value = line`** (`str`): The UTF-8 decoded text line (`errors="replace"`) with its trailing line terminator (`\n` or `\r\n`) stripped.

---

## 2. `InputSplit` and the Line-Ownership Rule

```python
@dataclass(frozen=True)
class InputSplit:
    split_id: int
    path: str
    start: int
    end: int

    @property
    def length(self) -> int:
        return self.end - self.start
```

### Fields
- **`split_id`** (`int`): Globally unique, zero-based sequential identifier (`0 .. k-1`) across all splits in a job.
- **`path`** (`str`): Filesystem path to the input file containing this byte range.
- **`start`** (`int`): Inclusive starting byte offset (`0 <= start < end`).
- **`end`** (`int`): Exclusive ending byte offset (`start < end <= file_size`).
- **`length`** (`int`): Byte span of the split (`end - start`).

### Line-Ownership Rule
Because fixed-size byte chunks (e.g., 16 MiB) routinely cut across the middle of text lines, the splitter and split reader enforce a deterministic **line-ownership rule** without copying or pre-scanning entire files:

> **Rule:** An `InputSplit(split_id, path, start, end)` owns every line in `path` whose **first byte** lies within the half-open interval `[start, end)`.

Concretely, when a worker reads `split` in binary mode (`"rb"`):
1. **Left Boundary (`start`)**:
   - If `start == 0`, reading starts directly at byte `0` (which is always the first byte of the first line).
   - If `start > 0`, byte `start` might fall in the middle of a line that began at some offset `< start` (and is therefore owned by the preceding split) or immediately after a `\n` at `start - 1`. The reader seeks to `start - 1` and discards one `readline()`.
     - If byte `start - 1` was `\n`, discarding that single `readline()` consumes only the single `\n` byte at `start - 1`, leaving the file pointer at `start` (the exact start of a new line).
     - If byte `start - 1` was not `\n`, discarding `readline()` skips the remainder of the partial line that started before `start`, advancing the file pointer to the first byte of the next complete line.
2. **Right Boundary (`end`)**:
   - Before reading each subsequent line, the reader checks the current byte offset `pos = f.tell()`.
   - As long as `pos < end`, the line starts inside `[start, end)` and is owned by this split; the reader calls `readline()`—reading past `end` if the line extends across the boundary—and yields `(pos, decoded_line)`.
   - Once `pos >= end`, the reader stops immediately, because any line starting at or after `end` belongs to the next split.

This invariant guarantees that for any split size (from 1 byte up to the full file size) and any line ending (`\n` or `\r\n`), **every line in every file is processed by exactly one split—never duplicated and never dropped**.

---

## 3. Core Classes and Job Lifecycle

### `JobContext`
```python
@dataclass
class JobContext:
    job_name: str
    rank: int = 0
    num_partitions: int = 1
    split: Optional[InputSplit] = None
    counters: dict = field(default_factory=dict)

    def increment(self, name: str, amount: int = 1) -> None:
        self.counters[name] = self.counters.get(name, 0) + amount
```
`JobContext` passes runtime metadata and instrumentation hooks into user `Mapper`, `Combiner`, and `Reducer` instances:
- **`job_name`**: Logical name of the executing workload (e.g., `"wordfreq"`, `"invindex"`).
- **`rank`**: MPI world rank executing the current task (`0` in the sequential engine; `1 .. N-1` on MPI workers).
- **`num_partitions`**: Total number of reduce partitions $R$ ($R = 1$ in the sequential baseline; $R = |\text{alive workers}|$ in the MPI shuffle/reduce phases).
- **`split`**: The `InputSplit` currently being mapped during a Map task, or `None` during Combine/Reduce phases.
- **`counters` / `increment(name, amount=1)`**: Per-task dictionary for recording user or framework telemetry counters.

### `Mapper`
```python
class Mapper(ABC):
    def setup(self, ctx: JobContext) -> None: ...

    @abstractmethod
    def map(self, key: Tuple[str, int], value: str, ctx: JobContext) -> Iterable[KeyValue]: ...

    def cleanup(self, ctx: JobContext) -> Iterable[KeyValue]:
        return ()
```
- **`setup(ctx)`**: Invoked once per `InputSplit` task before the first record is passed to `map`.
- **`map(key, value, ctx)`**: Abstract method called once per line owned by the split. Yields or returns zero or more `(intermediate_key, intermediate_value)` pairs where `intermediate_key` is a `str`.
- **`cleanup(ctx)`**: Invoked once per `InputSplit` task after all records in the split have been mapped. May emit flushed intermediate pairs (defaults to an empty tuple `()`).

### `Reducer`
```python
class Reducer(ABC):
    def setup(self, ctx: JobContext) -> None: ...

    @abstractmethod
    def reduce(self, key: str, values: Iterable[Any], ctx: JobContext) -> Iterable[KeyValue]: ...
```
- **`setup(ctx)`**: Invoked once per partition before the first grouped key is reduced.
- **`reduce(key, values, ctx)`**: Abstract method invoked once for each unique intermediate `key` in lexicographical order, receiving all intermediate `values` associated with `key`. Emits final `(key, value)` pairs where `value` is `str` or `int`.

### `Combiner`
```python
class Combiner(Reducer):
    """Map-side reducer used for skew handling."""
```
`Combiner` subclasses `Reducer` and executes on the worker over that worker's local map output before the network shuffle:
- **Purpose**: Mitigates key skew (e.g., high-frequency Zipfian stop words like `"the"`, `"of"`, `"and"`) and cuts `Alltoallv` shuffle payload volume by collapsing repeated keys locally within each map task.
- **Type & Algebraic Invariant**: Because combiner outputs become the input to the downstream `Reducer` (or a second combining pass), a `Combiner` **must emit `(key, value)` pairs of the exact same shape and value domain as `Mapper.map`**, and its aggregation operation must be associative, commutative, and idempotent to applying the combiner zero, one, or multiple times:
  - *Word Frequency*: Mapper emits `(word, 1)`; Combiner sums local counts and emits `(word, partial_sum)`; Reducer sums partial counts and emits `(word, total_sum)`.
  - *Inverted Index*: Mapper emits `(term, doc_id)`; Combiner deduplicates per document and emits each unique `(term, doc_id)` once; Reducer collects unique `doc_id` strings across all splits/documents and emits `(term, "doc1,doc2,...")`.

### `Job`
```python
@dataclass(frozen=True)
class Job:
    name: str
    mapper: Callable[[], Mapper]
    reducer: Callable[[], Reducer]
    combiner: Optional[Callable[[], Combiner]] = None
```
Encapsulates a complete MapReduce workload definition using zero-argument factory callables (`mapper`, `reducer`, and optional `combiner`) so the execution engine can instantiate fresh, isolated state for every map task and reduce partition.

---

## 4. Stable CRC32 Partitioner vs. Built-in `hash()`

```python
def partition(key: str, num_partitions: int) -> int:
    if num_partitions <= 0:
        raise ValueError("num_partitions must be positive")
    return zlib.crc32(key.encode("utf-8")) % num_partitions
```

During the Shuffle phase, every worker routes each intermediate pair `(key, value)` to one of $R$ alive reduce workers using `partition(key, R) -> r` where $r \in \{0, \dots, R - 1\}$.

### Why Python's Built-in `hash()` is Unsafe Across MPI Ranks
Since Python 3.3 (PEP 456), Python enables **SipHash randomization** by default to protect dictionaries against hash-flooding Denial-of-Service attacks:
1. Unless `PYTHONHASHSEED` is explicitly pinned to a fixed integer in the environment *before* every Python interpreter starts, each OS process spawned by `mpiexec -n N python mr_driver.py` initializes with an independent, randomly generated 128-bit secret seed from OS entropy.
2. Consequently, `hash("parallel") % R` evaluated on Worker Rank 1 will generally yield a **different partition index** than `hash("parallel") % R` evaluated on Worker Rank 2.
3. If two workers send intermediate values for the same key `"parallel"` to different reduce partitions, the global grouping invariant of MapReduce is violated: multiple reducers will emit partial results for `"parallel"`, corrupting both Word Frequency sums and Inverted Index posting lists.

### Why `zlib.crc32` Guarantees Correctness
`zlib.crc32(key.encode("utf-8"))` computes the IEEE 802.3 32-bit Cyclic Redundancy Check over the UTF-8 byte representation of `key` and returns an unsigned 32-bit integer in $[0, 2^{32}-1]$:
- It is **purely deterministic** across all OS processes, machines, architectures, and Python invocations regardless of `PYTHONHASHSEED`.
- It runs in C via `zlib` at memory-bandwidth speeds with uniform distribution across modulo buckets `0 .. R-1`.

---

## 5. Grouping, Reduction Helpers, and Output Format

### `group_by_key` and `run_reducer`
```python
def group_by_key(pairs: Iterable[KeyValue]) -> Iterator[Tuple[str, list]]: ...
def run_reducer(reducer: Reducer, pairs: Iterable[KeyValue], ctx: JobContext) -> list: ...
```
- `group_by_key(pairs)` accumulates values for identical keys into a `list` (preserving arrival order within each key) and yields `(key, values_list)` in **sorted key order** (`sorted(groups)`).
- `run_reducer(reducer, pairs, ctx)` invokes `reducer.setup(ctx)`, streams sorted groups from `group_by_key(pairs)` into `reducer.reduce(k, vs, ctx)`, and returns the collected list of emitted `(key, value)` pairs. Both the map-side `Combiner` and the reduce-side `Reducer` execute through `run_reducer`.

### `write_output` and Deterministic TSV Output Format
```python
def write_output(pairs: Iterable[KeyValue], path: str) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for k, v in sorted(pairs, key=lambda kv: kv[0]):
            f.write(f"{k}\t{v}\n")
```
The final job output file conforms to a strict canonical format:
- **Encoding**: UTF-8 without BOM (`encoding="utf-8"`).
- **Line Termination**: Unix Line Feed (`\n`, enforced across Windows and Linux via `newline="\n"`).
- **Record Layout**: Tab-separated `key<TAB>value\n` (`f"{k}\t{v}\n"`).
- **Ordering**: Globally sorted by `key` in ascending Unicode code-point order (`sorted(pairs, key=lambda kv: kv[0])`).

Because keys are sorted globally before writing and reducers produce deterministic values per key (e.g., integer sum for Word Frequency; comma-joined sorted unique `doc_id`s for Inverted Index), the parallel MPI engine produces **byte-for-byte identical output files** to the sequential baseline for any worker count $p \in \{1, 2, 4, 8\}$ and any split size.
