# Imports & packaging

Code-quality rubric entries for **imports & packaging** (7 entries). See `README.md` for the full topic index.

## Keep imports that pydantic / dataclasses resolve at runtime OUT of TYPE_CHECKING  ·  `high`

Do not move a type into `if TYPE_CHECKING:` if that type is resolved at runtime: pydantic model fields, anything passed through `get_type_hints()`, or a type used in a default value, isinstance, cast, or as a base class. `from __future__ import annotations` defers evaluation but pydantic still rebuilds and resolves field annotations at runtime and will raise on a name it cannot import.


**Why:** A type-checking block is skipped at runtime, so a symbol defined only there is absent when the code runs; for a pydantic field annotation this surfaces as a model-build/validation error, and for runtime uses as a NameError. Ruff's own TC002 docs carve this out: Pydantic and SQLAlchemy require annotations available at runtime. This is the inverse failure of the type-only-import rule and is easy to get backwards.


**Avoid:**
```python
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from mypkg.config import EncoderConfig  # pydantic resolves this at runtime

class Settings(BaseModel):
    encoder: EncoderConfig  # model build fails: name not importable at runtime
```

**Prefer:**
```python
from mypkg.config import EncoderConfig  # stays a real runtime import

class Settings(BaseModel):
    encoder: EncoderConfig
```

Source: [Ruff TC002 — typing-only-third-party-import (runtime-evaluated caveat)](https://docs.astral.sh/ruff/rules/typing-only-third-party-import/)

## Never hide a runtime-needed import inside a TYPE_CHECKING block  ·  `high`

If a name is referenced in executable code (a call, attribute access, isinstance, a default argument value), its import must be at normal runtime scope, never only under `if TYPE_CHECKING:`. Quoting a name in an annotation does not make a runtime use safe.


**Why:** The TYPE_CHECKING block does not execute at runtime, so an import placed only there leaves the name undefined and the first real call raises NameError. Ruff's TC004 exists precisely for this, but it is not in the enforced rule set here, so it falls to review. This bites most when someone over-zealously applies the type-only-import rule above to a symbol that is actually used at runtime.


**Avoid:**
```python
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    import foo

def bar() -> None:
    foo.run()  # NameError: name 'foo' is not defined
```

**Prefer:**
```python
import foo

def bar() -> None:
    foo.run()
```

Source: [Ruff TC004 — runtime-import-in-type-checking-block](https://docs.astral.sh/ruff/rules/runtime-import-in-type-checking-block/)

## Move type-only imports into a TYPE_CHECKING block (with future-annotations already enabled)  ·  `medium`

An import used only in annotations (e.g. transformers' PreTrainedModel / PreTrainedTokenizerBase used only in function signatures) should live under `if TYPE_CHECKING:`, not at runtime import scope. This is safe and free when `from __future__ import annotations` is present, since annotations are never evaluated at runtime. Ruff's TC001/TC002/TC003 flag this but are NOT in the enforced E/F/I/UP/B/SIM set, so a reviewer must catch it.


**Why:** Type-only imports add runtime import overhead and can drag a heavy dependency (transformers) into module-load time purely to satisfy a static checker. Hoisting them under TYPE_CHECKING removes the runtime cost and breaks import cycles, while mypy/pyright still see the names.


**Avoid:**
```python
from __future__ import annotations
from transformers import PreTrainedModel, PreTrainedTokenizerBase  # imported at runtime, used only in signatures

def load(name: str) -> tuple[PreTrainedTokenizerBase, PreTrainedModel]: ...
```

**Prefer:**
```python
from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizerBase

def load(name: str) -> tuple[PreTrainedTokenizerBase, PreTrainedModel]: ...
```

Source: [Ruff TC002 — typing-only-third-party-import](https://docs.astral.sh/ruff/rules/typing-only-third-party-import/)

## Place module-level dunders after the docstring, before imports  ·  `low`

Put module dunders (__all__, __version__, __author__) immediately after the module docstring and before any imports, with the sole exception of from __future__ imports, which must come first.


**Why:** PEP 8 prescribes this exact ordering. Putting __all__ below the imports (a common drift) is legal but non-idiomatic and harder to locate; the __future__ exception is a real subtlety because __future__ imports must precede everything including dunders. Ruff's import sorter (I) orders imports among themselves but does not enforce dunder-vs-docstring placement.


**Avoid:**
```python
"""Sparse encoder."""
import numpy as np
from .constants import DIM
__all__ = ["MiniCoilEncoder"]
__version__ = "2.0"
```

**Prefer:**
```python
"""Sparse encoder."""
from __future__ import annotations

__all__ = ["MiniCoilEncoder"]
__version__ = "2.0"

import numpy as np
from .constants import DIM
```

Source: [PEP 8, Module Level Dunder Names](https://peps.python.org/pep-0008/)

## Use module-level __getattr__ for lazy submodule exposure and deprecation, not import-time work  ·  `low`

When a package __init__ should expose subpackages for convenient access without paying their import cost up front (or needs to warn on a renamed/removed name), define a module-level `__getattr__` (PEP 562) that imports lazily on attribute access, and pair it with `__dir__`/`__all__`. For a library-shaped package, `lazy_loader.attach(...)` generates all three. Avoid doing the eager imports or other expensive work at __init__ import time.


**Why:** PEP 562 lets `import pkg` stay cheap while `pkg.heavy_submod` still resolves on first use, which is how scikit-image/NetworkX/SciPy keep import time low. The same hook gives a clean single place for deprecation warnings. Caveat: a name looked up as a module global bypasses __getattr__ (intentional, to protect builtin lookups), and you should define __dir__ so the lazy names show up in tab-completion and dir().


**Avoid:**
```python
# minicoil_v2/__init__.py
from minicoil_v2.train_concept_layers import train  # pulls torch in at every `import minicoil_v2`
from minicoil_v2.encoder import MiniCoilEncoder
```

**Prefer:**
```python
# minicoil_v2/__init__.py
import importlib
__all__ = ["train", "MiniCoilEncoder"]

def __getattr__(name: str):
    if name == "train":
        return importlib.import_module(".train_concept_layers", __name__).train
    if name == "MiniCoilEncoder":
        return importlib.import_module(".encoder", __name__).MiniCoilEncoder
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

def __dir__() -> list[str]:
    return sorted(__all__)
```

Source: [PEP 562 — Module __getattr__ and __dir__](https://peps.python.org/pep-0562/)

## Keep imports at the top of the file; treat any deferred import as a documented exception  ·  `low`

Default to placing imports at the top of the module, after the docstring and before any module globals. When you deliberately defer an import into a function (to dodge a heavy dependency, optional dependency, or import cycle), make that intent obvious, ideally with a short comment, so the next reader does not 'tidy' it back to the top and reintroduce the cost or the cycle.


**Why:** PEP 8 mandates top-of-file imports for readability and to avoid re-running the import machinery in each scope; an unexplained function-level import reads like an accident and tends to get hoisted. A one-line reason (heavy dep / breaks cycle / optional) preserves the deliberate choice and prevents regressions, since neither ruff nor a type checker will defend it.


**Avoid:**
```python
def encode(text: str):
    import torch  # no context: looks like a mistake, gets moved to module top in a cleanup
    ...
```

**Prefer:**
```python
def encode(text: str):
    import torch  # local: keep torch out of module import for CLI startup
    ...
```

Source: [PEP 8 — Style Guide for Python Code (Imports)](https://peps.python.org/pep-0008/)

## Prefer absolute imports; explicit relative is acceptable, implicit relative never, and don't mix within a package  ·  `low`

Use absolute imports (`from minicoil_v2.encoder import ...`) as the default; explicit relative imports (`from .encoder import ...`) are an acceptable alternative inside the package, but pick one style per package and stay consistent. Never use implicit relative imports (bare `import encoder`). Be aware a module that uses relative imports cannot be run as a loose script (`python src/minicoil_v2/foo.py`); run it as `python -m minicoil_v2.foo`.


**Why:** PEP 8 recommends absolute imports because they are more readable and give better error messages when the import system is misconfigured, while allowing explicit relative imports for complex layouts. Mixing the two within one package makes the public/internal boundary fuzzy, and the script-vs-`-m` gotcha (PEP 366) produces confusing 'attempted relative import with no known parent package' errors that ruff will not warn about.


**Avoid:**
```python
# inside src/minicoil_v2/cli.py
import encoder            # implicit relative: breaks under src layout
from encoder import App   # same problem
```

**Prefer:**
```python
# inside src/minicoil_v2/cli.py
from minicoil_v2.encoder import MiniCoilEncoder   # absolute, runs under -m and as installed pkg
```

Source: [PEP 8 — Style Guide for Python Code (Imports)](https://peps.python.org/pep-0008/)
