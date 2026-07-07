# API design

Python APIs should be boring, explicit, and hard to misuse.

## Prefer the smallest current API

Add parameters, hooks, callback strategies, and configuration only when current
callers need them. Unused flexibility has a cost: tests, docs, compatibility,
and cognitive load.

## Make options keyword-only

Use `*` before option and flag parameters so call sites name them.

Avoid:

```python
def encode(text: str, normalize: bool = True, batch_size: int = 32) -> list[float]:
    ...

encode("hello", False, 8)
```

Prefer:

```python
def encode(text: str, *, normalize: bool = True, batch_size: int = 32) -> list[float]:
    ...

encode("hello", normalize=False, batch_size=8)
```

## Use a sentinel when `None` is meaningful

If `None` is a valid value, do not also use it to mean omitted.

```python
class _Unset:
    pass


_UNSET = _Unset()

def fetch(url: str, timeout: float | None | _Unset = _UNSET) -> bytes:
    if timeout is _UNSET:
        timeout = 10.0
    ...
```

For public libraries, a small private sentinel class gives clearer repr and
typing than a bare `object()`.

## Match return type to the consumption contract

- Return `list` or another concrete collection when callers need length,
  indexing, reuse, or multiple passes.
- Return an iterator when streaming and one-pass consumption are intentional.
- Name streaming functions accordingly, for example `iter_rows` or
  `stream_events`.

## Avoid tuple-shaped public results

For multiple related return values, prefer a dataclass or other named shape over
a bare tuple. Named fields make callers clearer and allow fields to be added or
reordered with less risk.

## Design dependency boundaries

Pass dependencies in when they affect IO, time, randomness, network calls, or
external services. Use small protocols or adapters for seams you test heavily.
Do not expose third-party client internals throughout your codebase.

## Sources

- Ruff FBT rule family: https://docs.astral.sh/ruff/rules/#flake8-boolean-trap-fbt
- Python typing overload docs: https://docs.python.org/3/library/typing.html#typing.overload
- Martin Fowler on YAGNI: https://martinfowler.com/bliki/Yagni.html
- Brett Cannon on named tuples in APIs: https://snarky.ca/dont-use-named-tuples-in-new-apis/
