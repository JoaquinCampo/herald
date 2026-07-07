# Functions & API design

Code-quality rubric entries for **functions & api design** (5 entries). See `README.md` for the full topic index.

## Make options and flags keyword-only with `*`  ·  `high`

Put a bare `*` before option/flag parameters so callers must name them. This covers booleans (which FBT flags) but also two same-typed positionals and magic literals that FBT cannot see, and it lets you add or reorder options later without breaking existing call sites.


**Why:** A call like `encode(text, True, 4)` is unreadable and silently breaks if you reorder params; the enforced rule set here does not include FBT, and even FBT only catches `bool`-typed positionals, not ambiguous same-type positionals (`resize(img, 800, 600)`) or non-bool magic literals. Keyword-only params force `encode(text, normalize=True, dim=4)` and keep positional args reserved for the few that have obvious order, preserving forward-compat as the signature grows.


**Avoid:**
```python
def encode(text: str, normalize: bool = True, batch: bool = False) -> list[float]:
    ...

encode("hola", True, False)  # which flag is which?
```

**Prefer:**
```python
def encode(text: str, *, normalize: bool = True, batch: bool = False) -> list[float]:
    ...

encode("hola", normalize=True, batch=False)  # self-documenting, reorder-safe
```

Source: [Ruff rule docs: boolean-type-hint-positional-argument (FBT001)](https://docs.astral.sh/ruff/rules/boolean-type-hint-positional-argument/)

## Use a sentinel default, not `None`, when `None` is a valid argument  ·  `medium`

When a parameter must distinguish 'caller passed nothing' from 'caller explicitly passed None', use a private module-level sentinel object as the default and test it with `is`. Do not overload `None` to mean both 'use the default' and a legitimate value.


**Why:** If `None` is itself meaningful (e.g. `timeout=None` means 'no timeout' while 'not given' means 'use the configured default'), a `None` default cannot tell the two apart and you ship a silent bug. A dedicated sentinel (used throughout the stdlib and attrs) makes the distinction explicit and `is`-checkable; PEP 661 documents the rationale, but since it was never accepted the real idiom is `object()` or a tiny class, not a stdlib builtin.


**Avoid:**
```python
def search(query, limit=None):
    if limit is None:
        limit = DEFAULT_LIMIT  # can't express 'limit=None means unlimited'
```

**Prefer:**
```python
_UNSET = object()

def search(query, limit=_UNSET):
    if limit is _UNSET:
        limit = DEFAULT_LIMIT
    # now limit=None can legitimately mean 'unlimited'
```

Source: [death and gravity: Python sentinel objects, type hints, and PEP 661](https://death.andgravity.com/sentinels)

## Match return type to the iteration contract: generator vs list  ·  `medium`

Return a generator only when the caller will iterate exactly once and streaming is the intended contract; return a concrete `list` when callers need to iterate twice, take `len()`, index, or re-scan. Do not return a one-shot generator where the call site reasonably expects a reusable collection.


**Why:** The dangerous half is not memory, it is the silent single-use trap: a generator yields nothing on a second pass and raises `TypeError` on `len()`/indexing, so a caller that loops over results twice gets empty output with no error. Conversely, eagerly building a giant list (e.g. all Wikipedia sentences) where the contract is a stream defeats the point and can OOM. Choose the type from how it will be consumed, not by habit; generators keep locals alive until exhausted or GC'd, which also matters for large held state.


**Avoid:**
```python
def parsed_rows(path):
    return (parse(line) for line in open(path))

rows = parsed_rows(p)
total = sum(r.n for r in rows)
for r in rows:  # already exhausted -> silently does nothing
    index(r)
```

**Prefer:**
```python
def parsed_rows(path) -> list[Row]:        # reusable: caller scans twice
    with open(path) as f:
        return [parse(line) for line in f]

def stream_rows(path) -> Iterator[Row]:    # explicit one-shot stream
    with open(path) as f:
        yield from (parse(line) for line in f)
```

Source: [Google Python Style Guide (2.13 Generators)](https://google.github.io/styleguide/pyguide.html)

## Keep each function single-purpose; split orchestration from low-level work  ·  `medium`

A function should do one thing. When a function both orchestrates a workflow and performs the low-level steps inline (mixing abstraction levels), extract the steps into named helpers so the top-level function reads as a sequence of intentions. Google §3.18: if a function exceeds ~40 lines, ask whether it can be broken up.


**Why:** Mixed abstraction levels are the reviewer-visible symptom of a function doing too much: high-level intent (load, encode, score) interleaved with byte-fiddling, retries, and index math forces the reader to hold every layer at once and is where 'someone modifying it in a few months adds behavior and introduces hard-to-find bugs' (Google §3.18). Extracting each concern into a named function makes the orchestrator self-documenting and the pieces independently testable.


**Avoid:**
```python
def run_eval(cfg):
    # opens files, decodes, batches, encodes, computes ndcg, writes csv,
    # logs, retries on http error... 120 lines, every layer interleaved
    ...
```

**Prefer:**
```python
def run_eval(cfg):
    corpus = load_corpus(cfg.path)
    vectors = encode_corpus(corpus, cfg.model)
    scores = score_retrieval(vectors, cfg.queries)
    write_report(scores, cfg.out)
# each helper is one abstraction level and unit-testable
```

Source: [Google Python Style Guide (3.18 Function length)](https://google.github.io/styleguide/pyguide.html)

## Use @overload when the return type depends on the argument type, not a union return  ·  `medium`

When a function's return type is determined by its input type (e.g. passing a str yields a str, passing a list yields a list), declare @overload stubs for each input-to-output mapping followed by a single untyped-overload implementation. Do not annotate it as a broad union return that forces every caller to narrow.


**Why:** A union return type (str | bytes) makes the type checker treat every call site as ambiguous, so callers must add casts or isinstance guards even when the type is statically obvious from the argument. @overload encodes the actual correspondence so the checker infers the precise return per call. This is invisible to a linter, which sees a valid signature either way.


**Avoid:**
```python
def normalize(x: str | bytes) -> str | bytes:  # caller of normalize("hi") still gets str | bytes
    return x.strip() if isinstance(x, str) else x.strip()
```

**Prefer:**
```python
from typing import overload

@overload
def normalize(x: str) -> str: ...
@overload
def normalize(x: bytes) -> bytes: ...
def normalize(x: str | bytes) -> str | bytes:
    return x.strip()  # normalize("hi") is now inferred as str, normalize(b"hi") as bytes
```

Source: [Python docs — typing.overload](https://docs.python.org/3/library/typing.html#typing.overload)
