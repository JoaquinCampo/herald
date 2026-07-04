# Classes & data modeling

Code-quality rubric entries for **classes & data modeling** (3 entries). See `README.md` for the full topic index.

## Add frozen=True to make a dataclass usable as a dict key or set member  ·  `high`

A plain @dataclass (the default eq=True, frozen=False) has __hash__ set to None, so it is unhashable. If instances need to go in a set or be used as dict keys, declare @dataclass(frozen=True) (which regenerates __hash__) rather than reaching for unsafe_hash=True.


**Why:** The default dataclass is silently unhashable: putting one in a set or dict key raises TypeError: unhashable type at runtime, often far from the definition. frozen=True both restores hashability and guarantees the object cannot mutate out from under the hash, which is the actual invariant a hashable record needs.


**Avoid:**
```python
@dataclass
class ConceptKey:
    concept_id: int
    lang: str

seen: set[ConceptKey] = set()
seen.add(ConceptKey(1, "en"))  # TypeError: unhashable type: 'ConceptKey'
```

**Prefer:**
```python
@dataclass(frozen=True, slots=True)
class ConceptKey:
    concept_id: int
    lang: str

seen: set[ConceptKey] = set()
seen.add(ConceptKey(1, "en"))  # works; hashable and immutable
```

Source: [Python docs - dataclasses (hash generation rules: eq=True, frozen=False sets __hash__ to None)](https://docs.python.org/3/library/dataclasses.html)

## Use kw_only to fix default-ordering when extending a dataclass, not field reordering  ·  `medium`

When a subclass or a combined dataclass adds a required field after an inherited field that has a default, you get 'non-default argument follows default argument'. Reach for @dataclass(kw_only=True) (or the KW_ONLY sentinel, 3.10+) so all fields become keyword-only, instead of reordering fields or assigning fake defaults.


**Why:** The naive fixes (reordering fields, or giving the new field a sentinel default) corrupt the public field order or weaken the type. kw_only=True keeps the natural declaration order and the required-ness intact while making __init__ valid. It is the intended 3.10+ solution and is easy to miss.


**Avoid:**
```python
@dataclass
class Base:
    name: str = ""
@dataclass
class Job(Base):
    priority: int  # TypeError: non-default argument 'priority' follows default argument
```

**Prefer:**
```python
@dataclass(kw_only=True)
class Base:
    name: str = ""
@dataclass(kw_only=True)
class Job(Base):
    priority: int  # ok; Job(priority=3) required, name optional
```

Source: [Python docs - dataclasses (kw_only parameter and KW_ONLY sentinel, added in 3.10)](https://docs.python.org/3/library/dataclasses.html)

## Set frozen-dataclass derived fields in __post_init__ via object.__setattr__  ·  `medium`

Inside __post_init__ of a frozen=True dataclass, normal assignment self.x = ... raises FrozenInstanceError. Use object.__setattr__(self, "x", value) to set computed/derived fields. This is the documented mechanism the generated __init__ itself uses.


**Why:** People add a __post_init__ to compute a derived field on a frozen dataclass and hit FrozenInstanceError at construction, then often abandon frozen entirely (losing immutability and hashability). object.__setattr__ is the supported escape hatch for one-time initialization without un-freezing the class.


**Avoid:**
```python
@dataclass(frozen=True)
class Vec:
    raw: tuple[float, ...]
    norm: float = 0.0
    def __post_init__(self) -> None:
        self.norm = math.sqrt(sum(v * v for v in self.raw))  # FrozenInstanceError
```

**Prefer:**
```python
@dataclass(frozen=True)
class Vec:
    raw: tuple[float, ...]
    norm: float = 0.0
    def __post_init__(self) -> None:
        object.__setattr__(self, "norm", math.sqrt(sum(v * v for v in self.raw)))
```

Source: [Python docs - dataclasses (frozen=True: __init__ must use object.__setattr__ instead of simple assignment)](https://docs.python.org/3/library/dataclasses.html)
