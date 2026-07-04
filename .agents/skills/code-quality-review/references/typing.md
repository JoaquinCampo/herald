# Typing & annotations

Code-quality rubric entries for **typing & annotations** (8 entries). See `README.md` for the full topic index.

## Be aware that the future import stringifies annotations and breaks naive runtime introspection  ·  `high`

When 'from __future__ import annotations' IS present, every annotation becomes a string. Any code that reads __annotations__ directly (custom decorators, registries, simple dataclass-style metaprogramming) must instead call typing.get_type_hints(), or it will see raw strings. Annotations referencing function-local names or names imported only under 'if TYPE_CHECKING' cannot be resolved at runtime at all.


**Why:** Under PEP 563, raw __annotations__ holds 'int' (a str), not int. Naive runtime introspection silently misbehaves, and resolution fails for locally-scoped or TYPE_CHECKING-only names because they no longer exist when the string is later evaluated. Pydantic v2 and FastAPI resolve hints themselves so they mostly cope, but hand-rolled introspection and some plugins do not. A linter cannot see this cross-cutting semantic change.


**Avoid:**
```python
from __future__ import annotations
import typing

class Field:
    def __set_name__(self, owner, name):
        ann = owner.__annotations__[name]
        assert ann is int  # FAILS: ann == 'int' (a string), not the int type
```

**Prefer:**
```python
from __future__ import annotations
import typing

class Field:
    def __set_name__(self, owner, name):
        ann = typing.get_type_hints(owner)[name]
        assert ann is int  # OK: get_type_hints evaluates the stringized annotation
```

Source: [PEP 563 — Postponed Evaluation of Annotations (runtime-access caveats)](https://peps.python.org/pep-0563/)

## On 3.12, use PEP 695 syntax instead of hand-written _co/_contra TypeVars  ·  `medium`

On Python 3.12 (this repo's target), declare generics with class C[T]: / def f[T](): and aliases with the type statement. Do not hand-write TypeVar("T_co", covariant=True) or the _co/_contra naming convention; the checker infers variance from usage.


**Why:** PEP 695 eliminates the need to specify variance and the cumbersome _co/_contra naming. Carrying the legacy TypeVar+Generic boilerplate on 3.12 is verbose, error-prone, and the manual variance flags can disagree with inferred usage. This supersedes the older PEP 8 TypeVar-suffix advice.


**Avoid:**
```python
from typing import Generic, TypeVar
_T_co = TypeVar("_T_co", covariant=True, bound=str)
class Cache(Generic[_T_co]):
    def get(self) -> _T_co: ...
```

**Prefer:**
```python
class Cache[T: str]:
    def get(self) -> T: ...

type ListOrSet[T] = list[T] | set[T]
```

Source: [PEP 695 Type Parameter Syntax](https://peps.python.org/pep-0695/)

## Do not add 'from __future__ import annotations' just to use modern syntax  ·  `medium`

On Python 3.12, do not add 'from __future__ import annotations' merely to write list[int], dict[str, X], or X | None. Those already evaluate fine at runtime since 3.9/3.10. Add the future import only for genuine forward references (a name used in an annotation before it is defined) or when you deliberately want all annotations left unevaluated.


**Why:** PEP 563 was deferred and ultimately cancelled, then superseded by PEP 649/749; the import was never made mandatory and will eventually be deprecated. Adding it as cargo-cult boilerplate signals a misunderstanding and, more importantly, changes module semantics (it stringifies every annotation) for no benefit when the only goal was builtin-generic or union syntax that already works. Note: ruff's FA rules (FA100/FA102) touch this, but FA is outside the E/F/I/UP/B/SIM set this repo enforces, so it is not auto-handled here.


**Avoid:**
```python
from __future__ import annotations

def prune(counts: dict[str, int], keep: list[str] | None = None) -> list[str]:
    ...  # the future import does nothing useful here; dict[...]/list[...]/X | None already work on 3.12
```

**Prefer:**
```python
def prune(counts: dict[str, int], keep: list[str] | None = None) -> list[str]:
    ...  # no future import needed on 3.12 for builtin generics or PEP 604 unions
```

Source: [Python docs — __future__ (annotations feature, status note pointing to PEP 649/749)](https://docs.python.org/3/library/__future__.html)

## Use assert_never with Literal/enum for compiler-checked exhaustiveness  ·  `medium`

When branching over a closed set (a Literal union, an Enum, or a tagged set of cases), add a final 'case _:' / else that calls typing.assert_never(value). The type checker then errors at the assert_never call the moment a new variant is added but left unhandled. Do not silently fall through or raise a bare ValueError.


**Why:** A plain else/default branch handles unknown values at runtime but gives no compile-time guarantee that every case is covered. assert_never narrows the value to Never only when all cases are exhausted, so forgetting a new Literal member becomes a static type error instead of a latent runtime bug, exactly the kind of regression a reviewer wants caught in CI, not in production. assert_never has been in typing since 3.11.


**Avoid:**
```python
Mode = Literal["en", "es"]

def tokenizer_for(mode: Mode) -> Tokenizer:
    if mode == "en":
        return EN
    return ES  # add "fr" to Mode later and this wrongly returns ES with no error
```

**Prefer:**
```python
from typing import assert_never

Mode = Literal["en", "es"]

def tokenizer_for(mode: Mode) -> Tokenizer:
    match mode:
        case "en":
            return EN
        case "es":
            return ES
        case _:
            assert_never(mode)  # adding "fr" to Mode now triggers a type-check error here
```

Source: [Python docs — typing.assert_never (exhaustiveness checking)](https://docs.python.org/3/library/typing.html#typing.assert_never)

## Return typing.Self instead of the concrete class or a hand-rolled self-TypeVar  ·  `medium`

For methods that return the receiver (builders, fluent setters, alternative constructors, __enter__, copy/clone), annotate the return as typing.Self, not the concrete class name and not a private bound TypeVar on self. Use Self only when the method actually returns an instance of the same (possibly sub-) class.


**Why:** Annotating the return as the concrete class loses subclass precision: a subclass's chained call is inferred as the base type. The old fix was a per-hierarchy `TypeVar('S', bound='C')` declared on self, which is verbose and easy to get wrong. Self (added in 3.11) expresses this in one token. Ruff's PYI019 flags the self-TypeVar pattern, but it lives in the PYI (stub-file) ruleset, outside the E/F/I/UP/B/SIM set this repo enforces, so it is not auto-handled here.


**Avoid:**
```python
from typing import TypeVar

_S = TypeVar("_S", bound="EncoderConfig")

class EncoderConfig:
    def with_dim(self: _S, dim: int) -> _S:  # verbose; or worse, -> "EncoderConfig" which loses subclass type
        self.dim = dim
        return self
```

**Prefer:**
```python
from typing import Self

class EncoderConfig:
    def with_dim(self, dim: int) -> Self:  # subclass.with_dim(...) is correctly typed as the subclass
        self.dim = dim
        return self
```

Source: [PEP 673 — Self Type](https://peps.python.org/pep-0673/)

## Accept the broadest abstract type a function needs; return a concrete type  ·  `medium`

Annotate parameters with the most general collections.abc type that suffices (Iterable if you only loop, Sequence if you index/len, Mapping if you only read a dict) rather than a concrete list/dict. Annotate return types with the concrete type you actually produce (list[X], dict[K, V]) so callers can use the full interface.


**Why:** This is a design judgment a linter cannot make: UP035 will move Sequence from typing to collections.abc, but nothing picks the right abstraction for you. Demanding list[str] when you only iterate forces callers to materialize tuples/generators needlessly and blocks lazy pipelines (relevant when streaming Wikipedia sentences). Conversely, returning Iterable hides len/indexing the caller legitimately wants. Be liberal in what you accept, specific in what you return.


**Avoid:**
```python
from collections.abc import Iterable

def count_concepts(sentences: list[str]) -> Iterable[tuple[str, int]]:  # too strict in, too vague out
    ...
```

**Prefer:**
```python
from collections.abc import Iterable

def count_concepts(sentences: Iterable[str]) -> dict[str, int]:  # accepts any iterable; returns a usable dict
    ...
```

Source: [mypy docs — Type hints cheat sheet (Iterable/Sequence for arguments)](https://mypy.readthedocs.io/en/stable/cheat_sheet_py3.html)

## Do not use NamedTuple for an opaque record you compare or care about identity of  ·  `medium`

A typing.NamedTuple is a tuple subclass: it compares equal to any plain tuple with the same values, unpacks positionally, and is indexable by integer. Use it when sequence behavior is wanted; use a frozen dataclass when you want a distinct record type whose equality is type-aware and that cannot be accidentally unpacked or index-confused.


**Why:** Point(1, 2) == (1, 2) is True and Color(1, 2) == Point(1, 2) is True for same-valued NamedTuples, so two semantically different records compare equal, and a/b = some_record silently unpacks. These are subtle correctness traps; a frozen dataclass refuses all of them.


**Avoid:**
```python
class Span(NamedTuple):
    start: int
    end: int
class Range(NamedTuple):
    lo: int
    hi: int
Span(0, 5) == Range(0, 5)   # True -- unrelated records compare equal
Span(0, 5) == (0, 5)        # True
```

**Prefer:**
```python
@dataclass(frozen=True, slots=True)
class Span:
    start: int
    end: int
Span(0, 5) == (0, 5)  # False -- type-aware equality, no positional unpack
```

Source: [Python docs - typing.NamedTuple (class syntax over collections.namedtuple; instances are tuple subclasses)](https://docs.python.org/3/library/typing.html#typing.NamedTuple)

## Model external JSON/config shapes with TypedDict (and NotRequired), not bare dict or a dataclass you never instantiate  ·  `low`

For dict-shaped data you receive or produce as plain dicts (API/JSON payloads, kwargs blobs, parsed config), declare a TypedDict so keys and value types are checked, using NotRequired[...] / Required[...] (or total=False) for optional keys. Use a dataclass only when you actually construct and pass around objects with attribute access and behavior; use a TypedDict when the value stays a dict.


**Why:** Annotating such data as dict[str, Any] discards all key/value information; reaching for a dataclass forces a conversion step and loses the natural dict shape for JSON round-trips. TypedDict gives static key checking with zero runtime overhead and no instance construction. The optionality keywords (NotRequired/Required since 3.11) let you mark per-key optionality precisely instead of all-or-nothing. A linter cannot infer the intended key schema.


**Avoid:**
```python
def parse_pair(d: dict[str, Any]) -> None:
    q = d["qeury"]  # typo not caught; value type unknown
```

**Prefer:**
```python
from typing import TypedDict, NotRequired

class EvalPair(TypedDict):
    query: str
    corpus_id: str
    lang: NotRequired[str]  # may be absent in some payloads

def parse_pair(d: EvalPair) -> None:
    q = d["query"]  # mistyped key now flagged statically
```

Source: [Python docs — typing.TypedDict (Required/NotRequired/total)](https://docs.python.org/3/library/typing.html#typing.TypedDict)
