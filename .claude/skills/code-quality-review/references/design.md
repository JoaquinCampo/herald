# Design / Zen of Python

Code-quality rubric entries for **design / zen of python** (8 entries). See `README.md` for the full topic index.

## Return a small dataclass, not a NamedTuple, from new APIs  ·  `medium`

When a function returns 2+ related values, return a `@dataclass` (or `@dataclass(frozen=True)`), not a `tuple` or `NamedTuple`. A bare tuple forces positional unpacking at every call site; a NamedTuple permanently commits you to supporting both index-based and attribute-based access.


**Why:** Brett Cannon: returning a named tuple 'doubles the data access API surface for your return type as you have to now support index-based and attribute-based data access forever' — callers writing `result[0]` or iterating make tuple semantics part of your contract that you can never remove. A dataclass exposes only named fields, so you can add or reorder fields later without silently breaking positional unpacking, and it reads clearly at the call site.


**Avoid:**
```python
def evaluate(...) -> tuple[float, float, int]:
    return ndcg, recall, n

ndcg, recall, n = evaluate(...)  # positional; add a metric and every caller breaks
```

**Prefer:**
```python
@dataclass(frozen=True)
class EvalResult:
    ndcg: float
    recall: float
    n_queries: int

def evaluate(...) -> EvalResult: ...
res = evaluate(...); res.ndcg  # named; extensible without breaking callers
```

Source: [Brett Cannon: Don't return named tuples in new APIs](https://snarky.ca/dont-use-named-tuples-in-new-apis/)

## Don't over-parameterize for hypothetical futures (YAGNI)  ·  `medium`

Add a parameter only when a current caller needs it. Resist 'configurable for later' knobs, unused hooks, and generic strategy/callback args with a single hard-coded value. Delete parameters that no caller varies.


**Why:** Fowler: a presumptive feature carries a 'cost of carry' because 'the code for the presumptive feature adds some complexity to the software, this complexity makes it harder to modify and debug,' and speculative parameters are a classic 'speculative generality' smell. Each unused knob widens the API surface you must test, document, and keep backward-compatible, while your guess at the future signature is usually wrong anyway — it's cheaper to add the param when a real need arrives.


**Avoid:**
```python
def build_vocab(
    src, *, clustering="bfs", distance_fn=None, salvage=True,
    max_depth=None, seed=None, normalize_fn=None,   # all defaults, one caller
): ...
```

**Prefer:**
```python
def build_vocab(src, *, salvage: bool = True):
    ...  # add distance_fn / max_depth only when a caller actually varies them
```

Source: [Martin Fowler – Bliki: Yagni](https://martinfowler.com/bliki/Yagni.html)

## Prefer Protocol for boundary interfaces; don't trust @runtime_checkable for structural validation  ·  `medium`

Type a dependency by the behavior it must have (a typing.Protocol) rather than forcing callers to subclass an ABC, especially for third-party or test-double objects you cannot make inherit. Reserve ABCs for when you share implementation via inheritance or need a registry. If you add @runtime_checkable, know that isinstance against a protocol only checks member existence, never signatures, types, or return values.


**Why:** ABCs impose nominal inheritance, which is intrusive and impossible for objects you don't own (e.g. a torch module, a fake in a test). Protocols give the same static guarantees structurally. The trap is @runtime_checkable: isinstance(obj, MyProto) reduces to roughly hasattr checks, so an object with a method of the wrong arity or return type passes, giving false confidence. The typing spec explicitly calls these checks opt-in and not statically reliable.


**Avoid:**
```python
@runtime_checkable
class Embedder(Protocol):
    def encode(self, text: str) -> np.ndarray: ...

def run(e: object) -> None:
    assert isinstance(e, Embedder)  # passes for ANY object with an `encode` attr, even encode(self) -> None
```

**Prefer:**
```python
class Embedder(Protocol):
    def encode(self, text: str) -> np.ndarray: ...

def run(e: Embedder) -> None:  # checked statically by mypy/pyright; signature mismatches are caught
    vec = e.encode("hola")
```

Source: [Python Typing Spec — Protocols (runtime_checkable limitations)](https://typing.python.org/en/latest/spec/protocol.html)

## Prefer Enum/StrEnum over IntEnum; reach for IntEnum only for true integer interop  ·  `medium`

For new code use plain Enum (or StrEnum, 3.11+) rather than IntEnum/IntFlag. IntEnum members compare equal to bare ints and, by transitivity, equal to members of unrelated IntEnums with the same value. Use IntEnum only when you must interoperate with a system that genuinely needs the int (e.g. a C API or wire protocol).


**Why:** IntEnum breaks the semantic promise that an enum member is distinct from everything outside its enum. Shape.CIRCLE == Request.POST returning True is a silent correctness bug that no linter catches; the docs explicitly steer new code to Enum/Flag for this reason.


**Avoid:**
```python
class Shape(IntEnum):
    CIRCLE = 1
class Request(IntEnum):
    POST = 1
Shape.CIRCLE == Request.POST  # True  -- unrelated enums compare equal
```

**Prefer:**
```python
class Shape(Enum):
    CIRCLE = auto()
class Request(Enum):
    POST = auto()
Shape.CIRCLE == Request.POST  # False  -- type-safe
```

Source: [Python docs - Enum HOWTO (Enum/Flag strongly recommended; IntEnum/IntFlag break semantic promises, use only for interop)](https://docs.python.org/3/howto/enum.html)

## Know that StrEnum + auto() yields the lowercased member name, not the literal  ·  `medium`

With StrEnum (3.11+), auto() generates the lowercased version of the member name as the value. If a serialized value must match an external contract (JSON field, DB column, API enum), spell the string out explicitly instead of trusting auto().


**Why:** Developers reach for auto() to avoid typos, but the value silently becomes the lowercased name. A member written as Lang.EN serializes to "en", and HTTP_GET to "http_get", which may not match the wire format an API or stored data expects. Wrong but plausible-looking serialized values are hard to spot in review.


**Avoid:**
```python
class Lang(StrEnum):
    EN = auto()      # value == "en"
    ES_419 = auto()  # value == "es_419", maybe not what the API wants
json.dumps({"lang": Lang.ES_419})  # '{"lang": "es_419"}'
```

**Prefer:**
```python
class Lang(StrEnum):
    EN = "en"
    ES_419 = "es-419"  # explicit; matches the external contract
```

Source: [Python docs - Enum HOWTO (for StrEnum, auto() returns the lowercased member name)](https://docs.python.org/3/howto/enum.html)

## Keep @property cheap and side-effect-free; expose expensive work as a method  ·  `medium`

A @property must behave like plain attribute access: cheap, deterministic, no surprising side effects. If reading it does I/O, a network/DB call, heavy computation, or mutates state, make it an explicit get_x()/compute_x() method (or a cached_property if memoizing a pure result) so callers can see the cost.


**Why:** Code that looks like attribute access (obj.score) but secretly runs an ONNX forward pass or hits the network leads to accidental repeated work in loops and to confusing subclasses. The cost is invisible at the call site, which is exactly what the Google style guide warns against.


**Avoid:**
```python
class Doc:
    @property
    def embedding(self) -> np.ndarray:
        # silent heavy work on every access
        return self.model.encode(self.text)
vecs = [d.embedding for d in docs]  # re-encodes every time, looks free
```

**Prefer:**
```python
class Doc:
    def encode(self) -> np.ndarray:        # explicit: caller sees the cost
        return self.model.encode(self.text)
    @cached_property
    def embedding(self) -> np.ndarray:     # or memoize a pure result
        return self.model.encode(self.text)
```

Source: [Google Python Style Guide - 2.13 Properties (must be cheap, straightforward, unsurprising; like regular attribute access)](https://google.github.io/styleguide/pyguide.html)

## Use positional-only `/` to keep internal parameter names refactorable  ·  `low`

For public functions where a leading parameter's name carries no meaning for callers (the obvious first arg, or a name you may want to rename later), mark it positional-only with `/`. This frees you to rename it without breaking keyword callers and prevents accidental keyword coupling.


**Why:** PEP 570: without positional-only params, every parameter name becomes part of the API the moment any caller passes it by keyword, so renaming `def tokenize(s, /)` -> `def tokenize(text, /)` is safe, whereas `def tokenize(s)` locks `s=` in forever. It also lets you accept `**kwargs` keys that collide with a parameter name. Reserve `/` for args where the name genuinely doesn't help readability; use keyword-only `*` where names do help.


**Avoid:**
```python
def stem(word: str) -> str:
    ...
stem(word="running")  # now 'word' is locked into your public API
```

**Prefer:**
```python
def stem(word: str, /) -> str:
    ...
stem("running")  # name is internal; you can rename the param freely later
```

Source: [PEP 570 – Python Positional-Only Parameters](https://peps.python.org/pep-0570/)

## Use a module-level function or closure, not a class, for stateless single-method logic  ·  `low`

A class that holds no real state and exists only to wrap one method (a configure-then-run() object, or a bag of @staticmethods) should be a plain module-level function, or a closure when it must capture a value. Do not nest functions/classes purely to hide them; module privacy via a leading underscore is the idiom.


**Why:** Single-method 'manager'/'runner' classes add ceremony (instantiate, then call) without encapsulating anything, and obscure that the operation is just a function. The Google style guide explicitly says to avoid nested functions/classes except to close over a non-self/cls local, and not to nest merely to hide from module users.


**Avoid:**
```python
class TripletSampler:
    def __init__(self, vocab: Vocab) -> None:
        self.vocab = vocab
    def sample(self, concept_id: int) -> Triplet:
        ...
TripletSampler(vocab).sample(cid)  # class adds nothing over a function
```

**Prefer:**
```python
def sample_triplet(vocab: Vocab, concept_id: int) -> Triplet:
    ...
sample_triplet(vocab, cid)
# or, when a captured value is genuinely needed, return a closure
```

Source: [Google Python Style Guide - 2.6 Nested/Local/Inner Classes and Functions (avoid nesting except to close over a local; don't nest just to hide)](https://google.github.io/styleguide/pyguide.html)
