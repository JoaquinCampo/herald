# Library-specific

Code-quality rubric entries for **library-specific** (6 entries). See `README.md` for the full topic index.

## Use pydantic v2 validator semantics: classmethod order and return self  ·  `high`

On @field_validator, place @classmethod directly below the decorator (the v1 cls-first habit is wrong). For cross-field checks use @model_validator(mode='after') as an instance method that operates on self and returns self; a 'before' validator instead receives the raw input and runs prior to coercion.


**Why:** These signatures changed from v1 and a mistake fails at import or, worse, validates incorrectly at runtime. A model_validator(mode='after') that forgets `return self` returns None as the validated model; using mode='before' when you wanted post-coercion fields means you operate on un-coerced raw input. Ruff does not check pydantic decorator semantics.


**Avoid:**
```python
from pydantic import BaseModel, model_validator

class Cfg(BaseModel):
    lo: float
    hi: float
    @model_validator(mode="after")
    def check(self):
        if self.lo > self.hi:
            raise ValueError("lo>hi")
        # no return -> validated model becomes None
```

**Prefer:**
```python
from typing import Self
from pydantic import BaseModel, model_validator

class Cfg(BaseModel):
    lo: float
    hi: float
    @model_validator(mode="after")
    def check(self) -> Self:
        if self.lo > self.hi:
            raise ValueError("lo>hi")
        return self
```

Source: [Pydantic docs - Validators](https://pydantic.dev/docs/validation/latest/concepts/validators/)

## Set TOKENIZERS_PARALLELISM=false to avoid fast-tokenizer fork deadlocks  ·  `high`

If a fast (Rust) tokenizer is used before a process fork (DataLoader workers, multiprocessing.Pool, subprocess-based parallelism), set TOKENIZERS_PARALLELISM=false explicitly (env var, before tokenizers is imported). Otherwise the tokenizer's internal thread pool plus fork can deadlock or hang.


**Why:** The Rust tokenizers backend spins up worker threads; forking a process that already used them leaves the child with locks held by threads that no longer exist, which can hang the whole job with no error. The library disables parallelism and warns, but relying on that auto-fallback is fragile. This is a runtime hang a linter cannot see.


**Avoid:**
```python
# fast tokenizer used in main process, then DataLoader forks workers -> possible hang
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("intfloat/multilingual-e5-small")
loader = DataLoader(ds, num_workers=8)  # no TOKENIZERS_PARALLELISM set
```

**Prefer:**
```python
import os
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")  # before tokenizers import
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("intfloat/multilingual-e5-small")
loader = DataLoader(ds, num_workers=8)
```

Source: [huggingface/transformers issue #5486 (TOKENIZERS_PARALLELISM)](https://github.com/huggingface/transformers/issues/5486)

## Declare nested BaseSettings with default_factory, not a bare instance  ·  `medium`

For a nested pydantic-settings sub-model field, prefer Field(default_factory=QdrantSettings) over a bare default instance evaluated at class-definition time, when the sub-model itself reads environment/state. A bare `= QdrantSettings()` is constructed once at import and frozen, so later env changes (or test monkeypatching of env vars) do not re-read.


**Why:** A bare default instance is built at import time and shared as the class default; its env-derived fields are snapshotted at import, so per-process or per-test environment overrides for the nested model are silently ignored. default_factory defers construction to instance creation, so each Settings() re-reads the environment for the sub-model.


**Avoid:**
```python
class EmbedSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="MINICOIL_EMBED_")
    qdrant: QdrantSettings = QdrantSettings()  # env read once, at import
```

**Prefer:**
```python
from pydantic import Field

class EmbedSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="MINICOIL_EMBED_")
    qdrant: QdrantSettings = Field(default_factory=QdrantSettings)  # re-read per instance
```

Source: [Pydantic Settings docs - Nested model default settings](https://pydantic.dev/docs/validation/latest/concepts/pydantic_settings/)

## For expensive log args in hot paths, use logger.opt(lazy=True), not just {}  ·  `medium`

Loguru's '{}' formatting defers only the string formatting, not the argument evaluation: the arguments are still computed eagerly before the log call. To skip computing an expensive value when the level is filtered out, pass it as a callable under logger.opt(lazy=True).debug('{x}', x=lambda: expensive()).


**Why:** Reviewers often think replacing an f-string with '{}' makes a debug log free. It does not: the expensive expression still runs even when the sink rejects DEBUG. Only opt(lazy=True) with callables defers the work to after the level check. This is a correctness/perf nuance ruff cannot catch.


**Avoid:**
```python
# expensive_stats() runs even if DEBUG is filtered out
logger.debug("layer norms: {}", expensive_stats(weights))
```

**Prefer:**
```python
# lambda only runs if the sink accepts DEBUG
logger.opt(lazy=True).debug("layer norms: {x}", x=lambda: expensive_stats(weights))
```

Source: [Loguru docs - logger API (opt lazy)](https://loguru.readthedocs.io/en/stable/api/logger.html)

## Bulk-ingest with upload_points/upload_collection, not a hand-rolled upsert loop  ·  `medium`

For bulk ingestion into Qdrant, prefer client.upload_points / upload_collection (with batch_size, parallel, max_retries) over a manual loop calling client.upsert on chunks. The upload_* methods provide parallelization, a retry mechanism, and lazy batching out of the box.


**Why:** A hand-rolled upsert loop is single-threaded with no retry, so it under-saturates the network and dies on a transient error mid-ingest. The documented upload_* helpers parallelize batches and retry, and can stream from disk without holding all points in RAM. The manual loop is functionally correct so no linter flags it.


**Avoid:**
```python
for i in range(0, len(points), batch_size):
    client.upsert(collection_name=name, points=points[i : i + batch_size], wait=wait)
```

**Prefer:**
```python
client.upload_points(
    collection_name=name,
    points=points,        # or an iterator/generator
    batch_size=1000,
    parallel=4,
    max_retries=3,
)
```

Source: [Qdrant docs - Points (upload_points / upload_collection)](https://qdrant.tech/documentation/manage-data/points/)

## return_offsets_mapping requires a fast tokenizer; assert it  ·  `medium`

Code that calls tokenizer(..., return_offsets_mapping=True) (e.g. for token-pooling by char span) depends on a fast tokenizer; a slow Python tokenizer raises NotImplementedError. Load with AutoTokenizer.from_pretrained(name) and guard with assert tok.is_fast, since use_fast can silently fall back to a slow tokenizer when no fast variant exists.


**Why:** Offset mapping is a fast-tokenizer-only alignment feature; if the model ever ships only a slow tokenizer (or use_fast=False is passed), the call raises at runtime deep in the encode path. An explicit is_fast assertion turns a confusing downstream NotImplementedError into an immediate, clear failure. Ruff has no model-level knowledge to flag this.


**Avoid:**
```python
tok = AutoTokenizer.from_pretrained(model_name)
enc = tok(texts, return_offsets_mapping=True)  # NotImplementedError if slow tokenizer
```

**Prefer:**
```python
tok = AutoTokenizer.from_pretrained(model_name)
assert tok.is_fast, f"{model_name} needs a fast tokenizer for offset_mapping"
enc = tok(texts, return_offsets_mapping=True)
```

Source: [Hugging Face docs - Tokenizer (return_offsets_mapping / fast tokenizers)](https://huggingface.co/docs/transformers/main_classes/tokenizer)
