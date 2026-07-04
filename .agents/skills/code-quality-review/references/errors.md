# Errors & control flow

Code-quality rubric entries for **errors & control flow** (10 entries). See `README.md` for the full topic index.

## Use `from None` only to hide genuine noise, never reflexively  ·  `high`

Ruff B904 already forces a `from` clause when you re-raise inside an except block, so the open question is which one. Default to `raise NewError(...) from exc` to preserve the cause chain. Use `from None` only when the original exception is true implementation noise the caller must never see (e.g. wrapping a third-party library's internal error behind your own API). Reaching for `from None` to quiet a traceback you find verbose destroys the debugging trail.


**Why:** `from exc` sets `__cause__` ("direct cause of"); `from None` suppresses the implicit `__context__` chain entirely. B904 enforces that a from-clause exists but cannot judge whether erasing context is appropriate, and a careless `from None` silently discards the root cause during incident debugging.


**Avoid:**
```python
try:
    cfg = json.loads(raw)
except json.JSONDecodeError:
    raise ConfigError("bad config") from None  # root cause gone
```

**Prefer:**
```python
try:
    cfg = json.loads(raw)
except json.JSONDecodeError as exc:
    raise ConfigError("bad config") from exc
```

Source: [Python 3 Tutorial - Errors and Exceptions (raise from / from None)](https://docs.python.org/3/tutorial/errors.html)

## Never catch `BaseException` (and avoid blanket `except Exception`)  ·  `high`

Bare `except:` is caught by ruff E722, but `except BaseException:` is not, and it swallows `KeyboardInterrupt`, `SystemExit`, and `GeneratorExit`, making the process impossible to Ctrl-C or shut down cleanly. Catch the narrowest type that the protected line can actually raise. Only catch `Exception` when you immediately re-raise, or at a deliberate top-level isolation boundary that logs and records the error.


**Why:** BaseException sits above Exception specifically so that interpreter-shutdown and interrupt signals are not swallowed by ordinary error handling. The enforced rule set (E722) only flags the truly bare form; the broad-typed forms slip through and hide both bugs and shutdown signals. Google's style guide forbids catch-all and Exception catches except when re-raising or isolating.


**Avoid:**
```python
try:
    run_training_step()
except BaseException:  # eats KeyboardInterrupt during a long run
    logger.error("step failed")
```

**Prefer:**
```python
try:
    run_training_step()
except (RuntimeError, ValueError) as exc:
    logger.error("step failed: {}", exc)
    raise
```

Source: [Google Python Style Guide (Exceptions section)](https://google.github.io/styleguide/pyguide.html)

## Never use `assert` to validate inputs or enforce runtime invariants  ·  `high`

`assert` statements are stripped from the bytecode when Python runs under `-O`/`-OO` or with `PYTHONOPTIMIZE`, so any check that must always run, argument validation, precondition enforcement, security/auth checks, must raise an explicit exception (`ValueError`, `TypeError`, etc.) instead. Reserve assert for internal sanity checks that could be deleted without changing program behaviour.


**Why:** Under optimized mode the assertion simply is not there, so input validation written as assert silently disappears in production and lets invalid or malicious data through. The flake8-bandit S101 rule that flags this is not in the enforced E/F/I/UP/B/SIM set, so a reviewer must catch it. Google's style guide: an assert "could be removed without breaking the code."


**Avoid:**
```python
def set_temperature(celsius: float) -> None:
    assert celsius >= -273.15, "below absolute zero"  # vanishes under -O
```

**Prefer:**
```python
def set_temperature(celsius: float) -> None:
    if celsius < -273.15:
        raise ValueError("temperature below absolute zero")
```

Source: [Google Python Style Guide (assert / -O)](https://google.github.io/styleguide/pyguide.html)

## Prefer EAFP, and never LBYL across a check/use gap  ·  `medium`

Reach for try/except (EAFP) rather than guarding every access with an if-test (LBYL). When you do test a precondition, never let state change between the test and the use: an `if key in mapping` followed by `mapping[key]` (or `os.path.exists` then `open`) is a TOCTOU race under threads or concurrent filesystem access. Catch the exception instead, or hold a lock.


**Why:** LBYL duplicates the work the operation already does, drifts out of sync with it, and opens a race window between the look and the leap. The official glossary explicitly calls EAFP the clean, fast, characteristically-Pythonic style and names the LBYL race as a real hazard. Ruff does not reason about control-flow style or TOCTOU.


**Avoid:**
```python
if key in cache:
    return cache[key]  # another thread can evict key here
return compute(key)
```

**Prefer:**
```python
try:
    return cache[key]
except KeyError:
    return compute(key)
```

Source: [Python 3 documentation - Glossary (EAFP / LBYL)](https://docs.python.org/3/glossary.html)

## Keep the try body minimal so it cannot hide an unexpected error  ·  `medium`

Wrap only the single statement that can raise the exception you intend to handle. Pulling unrelated setup, parsing, or follow-up calls into the same try block means an exception from a line you never thought about gets caught and misattributed to the expected failure.


**Why:** A large try body turns a precise handler into a blanket one: a bug three lines down gets silently treated as the anticipated error. Google's style guide states this directly ("Minimize the amount of code in a try/except block... the try/except block hides a real error"). No linter measures try-body breadth.


**Avoid:**
```python
try:
    row = parse(line)            # could raise too
    score = model.predict(row)   # the call we meant to guard
    cache[row.id] = score        # could raise too
except ValueError:
    score = 0.0
```

**Prefer:**
```python
row = parse(line)
try:
    score = model.predict(row)
except ValueError:
    score = 0.0
cache[row.id] = score
```

Source: [Google Python Style Guide (Exceptions section)](https://google.github.io/styleguide/pyguide.html)

## Watch the multi-statement trap in `contextlib.suppress`  ·  `medium`

Ruff SIM105 auto-rewrites a single-statement try/except/pass to `contextlib.suppress`, so the residual judgment is the multi-statement case it does not touch. A `with suppress(...)` body aborts at the first matching exception, so any later statements in that block are silently skipped. Put exactly one fallible statement under suppress; if you need several independent best-effort operations, give each its own suppress (or its own try). Also avoid suppress in hot loops, where it is measurably slower than try/except.


**Why:** Unlike per-statement handling, suppress over a block jumps straight out of the with-statement on the first raise, so `b()` after a failing `a()` never runs, a data-loss bug that reads as innocent. SIM105 only fires on the `pass` form and cannot see this semantic difference.


**Avoid:**
```python
with contextlib.suppress(OSError):
    os.remove(tmp_path)      # if this raises, the next line is skipped
    os.remove(lock_path)
```

**Prefer:**
```python
with contextlib.suppress(OSError):
    os.remove(tmp_path)
with contextlib.suppress(OSError):
    os.remove(lock_path)
```

Source: [Ruff SIM105 - suppressible-exception](https://docs.astral.sh/ruff/rules/suppressible-exception/)

## Raise `TypeError` for wrong type, `ValueError` for wrong value  ·  `medium`

When validating arguments, distinguish the two failure modes: a value of the right type but out of range or otherwise unacceptable is a `ValueError`; an argument of the wrong type entirely is a `TypeError`. Do not raise `ValueError` from an `isinstance` check.


**Why:** Python's own conventions and built-ins use TypeError for inappropriate types so callers can branch on it, and downstream `except TypeError` handlers rely on that contract. Ruff's TRY004 covers this but lives in the tryceratops family, not in the enforced E/F/I/UP/B/SIM set, so it must be caught by review.


**Avoid:**
```python
def scale(vec: np.ndarray, factor: float) -> np.ndarray:
    if not isinstance(factor, (int, float)):
        raise ValueError("factor must be numeric")
    return vec * factor
```

**Prefer:**
```python
def scale(vec: np.ndarray, factor: float) -> np.ndarray:
    if not isinstance(factor, (int, float)):
        raise TypeError("factor must be numeric")
    return vec * factor
```

Source: [Ruff TRY004 - type-check-without-type-error](https://docs.astral.sh/ruff/rules/type-check-without-type-error/)

## Define a project exception base; suffix names with `Error`, inherit from a builtin  ·  `medium`

Give the package one root exception (e.g. `MiniCoilError(Exception)`) and derive specific errors from it, so callers can catch the whole family or a precise member. Every custom exception name should end in `Error` and ultimately inherit from an appropriate built-in exception (don't end at a bare class or inherit nothing meaningful).


**Why:** A shared base lets a caller write one `except MiniCoilError` boundary; the `Error` suffix is the established convention that signals 'this is an exception' at a glance. The pep8-naming N818 rule (error-suffix-on-exception-name) enforces the suffix but is not in the enforced E/F/I/UP/B/SIM set. Google's guide: custom exceptions "must inherit from an existing exception class" and "names should end in Error."


**Avoid:**
```python
class VocabFault(Exception): ...      # no Error suffix
class BadPrune(VocabFault): ...        # no shared, catchable root
```

**Prefer:**
```python
class MiniCoilError(Exception): ...
class VocabError(MiniCoilError): ...
class PruneError(VocabError): ...
```

Source: [Google Python Style Guide (custom exceptions)](https://google.github.io/styleguide/pyguide.html)

## Raise instead of returning None as an error sentinel  ·  `medium`

When a function cannot produce a result, raise a documented exception rather than returning None (or a falsy sentinel) to mean failure. None collides with legitimately falsy results, 0, 0.0, empty string/array, so callers that test `if not result:` misclassify a valid value as an error. Reserve None for the genuine 'no value, and that is normal' case, and document which exceptions the function raises since Python has no checked-exception mechanism.


**Why:** A None-as-error return invites the caller to write `if not score:` which fires on a perfectly valid 0.0, a classic silent bug Effective Python (Item 32) calls out specifically. Exceptions force the caller to handle failure explicitly and keep the success path clean. No linter can tell an error-sentinel None from a legitimate one.


**Avoid:**
```python
def cosine(a, b):
    if not a.any() or not b.any():
        return None  # caller's `if not cosine(a,b)` also trips on a real 0.0
    return float(a @ b / (norm(a) * norm(b)))
```

**Prefer:**
```python
def cosine(a, b):
    if not a.any() or not b.any():
        raise ValueError("cosine undefined for a zero vector")
    return float(a @ b / (norm(a) * norm(b)))
```

Source: [Effective Python, 3rd ed. (Brett Slatkin) - Item 32: Prefer Raising Exceptions to Returning None](https://www.informit.com/articles/article.aspx?p=3203546&seqNum=3)

## Treat switching to ExceptionGroup/`except*` as an API-breaking change; use `add_note()` for context  ·  `low`

On 3.11+, `ExceptionGroup` plus `except*` is for surfacing several genuinely-concurrent failures at once (e.g. parallel tasks), not a default replacement for try/except. Because an existing caller's plain `except SomeError` will no longer match once you wrap failures in a group, changing a function to raise an ExceptionGroup is an API-breaking change, introduce it via new API, not by mutating an existing one. When you only need to attach context to a single in-flight exception, prefer `exc.add_note(...)` over wrapping or string-munging the message.


**Why:** `except*` always matches against a group and re-wraps, so flipping a function to raise groups silently breaks every existing single-exception handler downstream. `add_note()` (also 3.11+) is the idiomatic way to enrich a traceback without losing the original type or cause. These are recent features ruff does not police for misuse.


**Avoid:**
```python
# was: raise ValueError(...)  callers do `except ValueError`
def load(paths):
    raise ExceptionGroup("load failed", errs)  # breaks every existing handler
```

**Prefer:**
```python
try:
    cfg = parse(raw)
except ValueError as exc:
    exc.add_note(f"while loading {path}")
    raise
```

Source: [PEP 654 - Exception Groups and except*](https://peps.python.org/pep-0654/)
