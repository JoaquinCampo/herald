# Performance idioms

Code-quality rubric entries for **performance idioms** (16 entries). See `README.md` for the full topic index.

## Keep CPU-GPU sync points out of the hot loop (.item(), .cpu(), print on tensors)  ·  `high`

Inside a training/encoding loop, do not call `.item()`, `.cpu()`, `.tolist()`, `print(tensor)`, or branch on a CUDA tensor's value every step. Each forces a device->host synchronization that stalls the GPU. Accumulate metrics on-device (e.g. sum a running loss tensor) and pull a single scalar with `.item()` only at logging boundaries (end of epoch / every N steps).


**Why:** The PyTorch tuning guide lists `.item()`, `cuda_tensor.cpu()`, `print(cuda_tensor)`, and 'python control flow which depends on results of operations performed on CUDA tensors' as synchronizations to avoid, so 'the CPU can run ahead of the accelerator.' A per-step `loss.item()` serializes CPU and GPU and can dominate step time. No linter sees this; it looks like ordinary logging.


**Avoid:**
```python
for batch in loader:
    loss = step(batch)
    running += loss.item()   # device->host sync every step
```

**Prefer:**
```python
running = torch.zeros((), device=device)
for batch in loader:
    running += step(batch).detach()   # stays on device
epoch_loss = (running / len(loader)).item()   # one sync at the end
```

Source: [PyTorch Tutorials — Performance Tuning Guide](https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html)

## Bound `@cache`/`@lru_cache(maxsize=None)` on long-lived functions  ·  `high`

`@functools.cache` is exactly `lru_cache(maxsize=None)` and never evicts: in a long-running process it grows without bound and pins every argument AND return value alive forever. Use `@lru_cache(maxsize=N)` (a sensible bound) for anything keyed on open-ended or large inputs, reserving unbounded `@cache` for small fixed key spaces (recursion, a handful of config keys).


**Why:** The docs state the cache 'keeps references to the arguments and return values until they age out of the cache or until the cache is cleared' — with `maxsize=None` they never age out, so big tensors/strings cached by, say, sentence text are leaked for the process lifetime. Note B019 (enforced here) only flags this on instance methods; module-level functions slip past the linter entirely, making this a reviewer-only catch.


**Avoid:**
```python
@cache  # = lru_cache(maxsize=None): never evicts
def embed_sentence(text: str) -> np.ndarray:
    return model.encode(text)  # one entry per distinct sentence, forever
```

**Prefer:**
```python
@lru_cache(maxsize=4096)
def embed_sentence(text: str) -> np.ndarray:
    return model.encode(text)  # bounded; old entries evicted, refs released
```

Source: [Python docs: functools (cache / lru_cache)](https://docs.python.org/3/library/functools.html)

## Do not put @cached_property on a slots=True dataclass  ·  `high`

functools.cached_property stores its result in the instance __dict__, which a class with __slots__ (including @dataclass(slots=True)) does not have. Either drop slots=True on that class, or add "__dict__" to the slots, or cache manually into a declared slot.


**Why:** Accessing the cached_property raises TypeError: No '__dict__' attribute on '<X>' instance to cache '<prop>' property the first time it is read. Because slots=True and cached_property are both popular optimizations, they get combined reflexively and the failure only surfaces at the first property access, not at class definition.


**Avoid:**
```python
@dataclass(slots=True)
class Encoder:
    weights: np.ndarray

    @cached_property
    def norm(self) -> float:
        return float(np.linalg.norm(self.weights))
# enc.norm -> TypeError: No '__dict__' attribute ... to cache 'norm'
```

**Prefer:**
```python
@dataclass  # no slots=True, so instances keep a __dict__
class Encoder:
    weights: np.ndarray

    @cached_property
    def norm(self) -> float:
        return float(np.linalg.norm(self.weights))
```

Source: [Python docs - functools.cached_property (requires a mutable __dict__; will not work with __slots__ that omit __dict__)](https://docs.python.org/3/library/functools.html#functools.cached_property)

## Hoist set/dict membership out of loops; never test membership against a list  ·  `high`

If you test `x in collection` inside a loop, the collection must be a set or dict (O(1)), never a list or tuple (O(n)). Build the lookup set once before the loop, not on each iteration.


**Why:** `x in list` is a linear scan; doing it inside a loop is a silent O(n*m) trap that looks innocent and only bites at scale (e.g. filtering surviving concept IDs against the full vocabulary while streaming a corpus). The Python wiki TimeComplexity table gives list `in` as O(n) and set/dict `in` as O(1) average. No enabled ruff rule flags this; it is purely a reviewer-caught quadratic.


**Avoid:**
```python
surviving = [c.id for c in pruned]  # a list
kept = [s for s in sentences if s.concept_id in surviving]  # O(n*m) scan
```

**Prefer:**
```python
surviving = {c.id for c in pruned}  # set, built once
kept = [s for s in sentences if s.concept_id in surviving]  # O(1) per check
```

Source: [Python Wiki: TimeComplexity (list vs set/dict membership)](https://wiki.python.org/moin/TimeComplexity)

## itertools.groupby groups only CONSECUTIVE keys; pre-sort or it silently splits groups  ·  `high`

Before calling itertools.groupby(data, key=f), ensure data is sorted by the same key f. If you only need grouping (not ordering) of unsorted data, prefer a defaultdict(list); reach for groupby when input is already sorted.


**Why:** groupby starts a new group every time the key value changes, so on unsorted input it produces multiple fragmented groups for the same key. The docs warn this 'differs from SQL's GROUP BY which aggregates common elements regardless of their input order.' This is a correctness bug that masquerades as working code on small/accidentally-sorted samples. Also: because groups share the source iterator, materialize each group with list(g) before advancing if you need it later.


**Avoid:**
```python
from itertools import groupby
# concepts NOT sorted by language -> 'en' group appears several times
by_lang = {k: list(g) for k, g in groupby(concepts, key=lambda c: c.lang)}
```

**Prefer:**
```python
from itertools import groupby
ordered = sorted(concepts, key=lambda c: c.lang)
by_lang = {k: list(g) for k, g in groupby(ordered, key=lambda c: c.lang)}
# or, if order is irrelevant, skip groupby entirely and use defaultdict(list)
```

Source: [Python docs: itertools.groupby](https://docs.python.org/3/library/itertools.html#itertools.groupby)

## Defer heavy/optional imports (torch, transformers) to function scope, not module top  ·  `medium`

When a heavy or optional dependency (torch, transformers, a large submodule) is used only inside a few functions or a single code path, import it inside those functions rather than at module top. Module-top imports of heavy packages tax every importer of the module, including the CLI and any caller that never touches that path.


**Why:** Eager top-level importing of large submodules introduces unacceptable startup slowdowns; SPEC 1 notes this is exactly why developers historically moved imports inside functions. A `minicoil --help` invocation that transitively imports a module which imports torch at module top pays the full torch import cost for nothing. Function-level imports are cheap on repeat (sys.modules cache) and only fire when the path runs.


**Avoid:**
```python
# qdrant_store.py — paid by every importer, even those never touching tensors
import torch

def upload(points): ...  # does not use torch

def to_tensor(vecs):
    return torch.tensor(vecs)
```

**Prefer:**
```python
# qdrant_store.py — torch cost only when to_tensor() actually runs
def upload(points): ...

def to_tensor(vecs):
    import torch
    return torch.tensor(vecs)
```

Source: [Scientific Python SPEC 1 — Lazy Loading of Submodules and Functions](https://scientific-python.org/specs/spec-0001/)

## Build TypeAdapter once at module scope, never per call  ·  `medium`

Instantiate a TypeAdapter once at module level (or cache it) and reuse it. Never construct a TypeAdapter inside a function or loop that runs repeatedly, because each instantiation rebuilds a fresh validator and serializer.


**Why:** Constructing a TypeAdapter is expensive: it compiles a new pydantic-core validator and serializer every time. Doing it per-call silently throws away that compiled core on every invocation, turning a one-time cost into a hot-path cost. Ruff cannot see this because the code is syntactically fine.


**Avoid:**
```python
from pydantic import TypeAdapter

def parse_rows(raw: list[bytes]) -> list[dict]:
    adapter = TypeAdapter(list[dict[str, float]])  # rebuilt every call
    return [adapter.validate_json(r) for r in raw]
```

**Prefer:**
```python
from pydantic import TypeAdapter

_ROWS = TypeAdapter(list[dict[str, float]])  # built once at import

def parse_rows(raw: list[bytes]) -> list[dict]:
    return [_ROWS.validate_json(r) for r in raw]
```

Source: [Pydantic docs - Performance](https://pydantic.dev/docs/validation/latest/concepts/performance/)

## Do not assume cached_property is thread-safe (the lock was removed in 3.12)  ·  `medium`

Before 3.12, cached_property held an (undocumented) per-property lock; in Python 3.12+ that lock is gone. The getter can now run more than once concurrently on the same instance. If the computation is expensive, non-idempotent, or has side effects, add explicit locking inside the getter.


**Why:** Code that worked under 3.11 because of the implicit lock can start double-computing under 3.12 with no error, just duplicated work or a side effect firing twice (e.g. loading a model or opening a connection). It is a silent behavior change tied to the interpreter version, not the code.


**Avoid:**
```python
class Model:
    @cached_property
    def session(self) -> Session:
        # heavy, side-effecting; assumed to run once
        return load_onnx_session(self.path)  # may run twice under 3.12 threads
```

**Prefer:**
```python
class Model:
    _lock = threading.Lock()

    @cached_property
    def session(self) -> Session:
        with self._lock:
            return load_onnx_session(self.path)  # guaranteed once
```

Source: [Python docs - functools.cached_property (Changed in 3.12: locking removed; getter may run more than once)](https://docs.python.org/3/library/functools.html#functools.cached_property)

## Build lists with a comprehension, not an append loop  ·  `medium`

When constructing a new list (or dict/set) by transforming or filtering an iterable, use a comprehension instead of preallocating an empty list and calling .append() in a loop. Reserve the explicit loop for cases with side effects or multiple statements per item.


**Why:** A comprehension emits the dedicated LIST_APPEND bytecode and never re-resolves the list name or its .append method on each iteration, so it is reliably as fast or faster than the manual loop. Ruff's C4/PERF rules (PERF401) that flag this are NOT enabled in this repo (only E,F,I,UP,B,SIM), so the linter will not catch it. It is also less code and signals intent (this produces a list) at a glance.


**Avoid:**
```python
concept_ids = []
for c in concepts:
    if c.count_en > 0:
        concept_ids.append(c.id)
```

**Prefer:**
```python
concept_ids = [c.id for c in concepts if c.count_en > 0]
```

Source: [Ruff rule docs: manual-list-comprehension (PERF401)](https://docs.astral.sh/ruff/rules/manual-list-comprehension/)

## Stream a generator into reducing/short-circuiting consumers; do not materialize first  ·  `medium`

Pass a generator expression directly into a consumer that reduces (sum, min, max) or short-circuits (any, all) rather than building a list comprehension first. Do NOT claim a memory win for join(), sorted(), or list(): those materialize the sequence internally regardless, so the only benefit there is avoiding a named intermediate.


**Why:** sum/min/max stream one item at a time and never hold the whole sequence; any/all additionally stop at the first decisive element, so a list comprehension wastes both memory and (for any/all) compute by building every element up front. This matters in this repo's streaming-Wikipedia hot paths. Knowing which consumers materialize (join, sorted, list, tuple) versus stream prevents cargo-culting a non-existent optimization.


**Avoid:**
```python
if any([c.count_en > THRESHOLD for c in concepts]):  # builds full list, no short-circuit
    ...
total = sum([s.n_tokens for s in sentences])  # materializes a throwaway list
```

**Prefer:**
```python
if any(c.count_en > THRESHOLD for c in concepts):  # stops at first True
    ...
total = sum(s.n_tokens for s in sentences)  # streams, never materializes
```

Source: [Python docs: Functional Programming HOWTO (generator expressions)](https://docs.python.org/3/howto/functional.html)

## Join strings; never accumulate with += in a loop  ·  `medium`

Build a string from many pieces with ''.join(parts) (collecting parts in a list or generator), not by repeatedly doing s += piece inside a loop.


**Why:** Python strings are immutable, so s += piece allocates a fresh string and copies all prior content each iteration, giving O(n^2) total work; the official wiki calls this 'a very common and catastrophic mistake when building large strings.' join() allocates the result once. CPython sometimes optimizes simple in-place += on a local, but that is fragile and not guaranteed, so do not rely on it.


**Avoid:**
```python
text = ""
for sent in sentences:
    text += sent + " "
```

**Prefer:**
```python
text = " ".join(sentences)
```

Source: [Python Wiki: PythonSpeed/PerformanceTips (string concatenation)](https://wiki.python.org/moin/PythonSpeed/PerformanceTips)

## Group and count with defaultdict / Counter, not manual key-existence checks  ·  `medium`

For grouping use collections.defaultdict(list) and append directly; for tallying use collections.Counter. Avoid the `if k not in d: d[k] = ...` pattern and avoid dict.setdefault inside hot loops.


**Why:** defaultdict resolves missing keys via __missing__ in C and the docs state it is 'simpler and faster than an equivalent technique using dict.setdefault()' (setdefault constructs the default value on every call even when unused). Counter handles missing keys as zero, removing the existence-check boilerplate. Ruff's PERF403 (manual-dict-comprehension / defaultdict) is not enabled here, so neither the slowness nor the verbosity is linted.


**Avoid:**
```python
groups = {}
for sent in sentences:
    if sent.concept_id not in groups:
        groups[sent.concept_id] = []
    groups[sent.concept_id].append(sent)
```

**Prefer:**
```python
from collections import defaultdict
groups = defaultdict(list)
for sent in sentences:
    groups[sent.concept_id].append(sent)
```

Source: [Python docs: collections (defaultdict, Counter)](https://docs.python.org/3/library/collections.html)

## Batch with itertools.batched (3.12) instead of hand-rolled index slicing  ·  `medium`

To process an iterable in fixed-size chunks (e.g. batching encoder inputs or Qdrant upserts), use itertools.batched(iterable, n) on Python 3.12+. Avoid manual range(0, len(x), n) slicing, which requires a materialized sequence and is easy to get wrong on the final partial batch.


**Why:** batched is new in 3.12 (this repo targets py312), consumes the input lazily one batch at a time so it works on any iterator without materializing the whole sequence, and correctly yields a shorter final tuple. Manual slicing forces the input to be a list and re-implements off-by-one edge cases. Note batched yields tuples, and the strict= flag only arrives in 3.13.


**Avoid:**
```python
for i in range(0, len(docs), 256):
    batch = docs[i : i + 256]
    encode(batch)
```

**Prefer:**
```python
from itertools import batched
for batch in batched(docs, 256):  # batch is a tuple; lazy over any iterable
    encode(batch)
```

Source: [Python docs: itertools.batched](https://docs.python.org/3/library/itertools.html#itertools.batched)

## Slice and concatenate iterators lazily with islice / chain.from_iterable  ·  `medium`

Use itertools.islice to take a prefix/window of a lazy stream instead of list(stream)[:n], and itertools.chain.from_iterable to flatten one level of nesting instead of summing lists or nested append loops.


**Why:** list(stream)[:n] materializes the entire (possibly huge or infinite) stream just to discard most of it; islice pulls only what is needed, which is the right tool for capping a Wikipedia stream during a smoke run. chain.from_iterable is 'roughly equivalent to: for it in iterables: yield from it' and flattens without building intermediate lists, unlike sum(lists, []) which is accidentally O(n^2).


**Avoid:**
```python
first_1k = list(stream_wiki())[:1000]  # consumes the whole stream
all_tokens = sum(sentence_token_lists, [])  # O(n^2) list concatenation
```

**Prefer:**
```python
from itertools import islice, chain
first_1k = list(islice(stream_wiki(), 1000))  # pulls only 1000
all_tokens = list(chain.from_iterable(sentence_token_lists))  # lazy flatten
```

Source: [Python docs: itertools (islice, chain.from_iterable)](https://docs.python.org/3/library/itertools.html#itertools.islice)

## Measure before micro-optimizing; the 3.11+ interpreter already caches name/method lookups  ·  `medium`

Do not apply legacy micro-optimizations like binding `append = obj.append` or `_local = some.attr` purely to dodge lookup cost, and do not hand-roll C-level tricks without a profile showing a real hot path. The one still-worthwhile manual hoist is pulling a genuinely global/module-level name into a local inside a proven hot loop.


**Why:** PEP 659's specializing adaptive interpreter (3.11+, which this py312 repo runs on) caches global, attribute, and method lookups inline so 'method loading now has no namespace lookups even for classes with long inheritance chains.' The old method-reference-caching trick is now mostly dead weight: it hurts readability for negligible gain and is an anti-idiom a current reviewer flags. Specialization only kicks in on hot code, and CPU-bound pure Python; I/O- or NumPy/torch-bound code (most of this repo) sees no benefit, so profile before optimizing.


**Avoid:**
```python
def upper_all(words):
    out = []
    append = out.append          # obsolete post-3.11 ceremony
    upper = str.upper
    for w in words:
        append(upper(w))
    return out
```

**Prefer:**
```python
def upper_all(words):
    return [w.upper() for w in words]  # clearer; interpreter caches the lookups
```

Source: [Python docs: What's New in 3.11 (PEP 659 specializing adaptive interpreter)](https://docs.python.org/3/whatsnew/3.11.html#faster-cpython)

## Use bisect for sorted-list search, but know insort is O(n)  ·  `low`

To find a position in an already-sorted list, use bisect.bisect_left/bisect_right (O(log n)) rather than a linear scan or re-sorting. Do not reach for bisect.insort to build a sorted collection by repeated insertion, and for pure membership/lookup prefer a dict or set.


**Why:** bisect_left gives O(log n) search on sorted data, but the docs are explicit that 'the insort() functions are O(n) because the logarithmic search step is dominated by the linear time insertion step,' so insort-in-a-loop is a hidden O(n^2). The docs also note 'for locating specific values, dictionaries are more performant.' The key= parameter (added 3.10) lets you bisect on a computed key without precomputing, though it may re-call the key function per comparison.


**Avoid:**
```python
thresholds = []
for t in incoming:  # O(n^2): each insort is O(n)
    bisect.insort(thresholds, t)
```

**Prefer:**
```python
thresholds = sorted(incoming)  # one O(n log n) sort
idx = bisect.bisect_left(thresholds, target)  # O(log n) lookups thereafter
```

Source: [Python docs: bisect](https://docs.python.org/3/library/bisect.html)
