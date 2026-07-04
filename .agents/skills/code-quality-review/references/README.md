# Code-quality rubric — reference catalogue

Source-cited, adversarially-verified best-practice entries (Python 3.12), one file per topic. Open only the file you need. The skill's `SKILL.md` is the skimmable summary.

| Topic | File | Entries | Scope |
|-------|------|--------:|-------|
| Typing & annotations | [typing.md](typing.md) | 8 | modern syntax, future-import, PEP 695 generics, Protocol/TypedDict/Self |
| Errors & control flow | [errors.md](errors.md) | 10 | EAFP, narrow excepts, chaining, no silent failures, assert caveats |
| Functions & API design | [functions.md](functions.md) | 5 | kw-only, defaults, single responsibility, caching, generators |
| Classes & data modeling | [classes.md](classes.md) | 3 | class-vs-function, dataclass(slots), NamedTuple, enums |
| I/O & filesystem | [io.md](io.md) | 11 | pathlib, context managers, encoding, streaming |
| Imports & packaging | [imports.md](imports.md) | 7 | hygiene, lazy/TYPE_CHECKING imports, __all__, package data |
| Performance idioms | [perf.md](perf.md) | 16 | comprehensions/itertools, avoiding O(n^2), batched (3.12) |
| Naming, docstrings, comments | [naming.md](naming.md) | 6 | PEP 8 naming, builtin shadowing, PEP 257, comment discipline |
| numpy & torch idioms | [numpy-torch.md](numpy-torch.md) | 7 | vectorize, device/dtype, no_grad, Generator RNG |
| Library-specific | [libraries.md](libraries.md) | 6 | pydantic v2, typer, loguru, qdrant-client, transformers |
| Testing patterns | [testing.md](testing.md) | 10 | pytest fixtures/parametrize, real-code-over-mocks, hypothesis |
| Design / Zen of Python | [design.md](design.md) | 8 | decomposition, abstraction levels, readability, YAGNI |
