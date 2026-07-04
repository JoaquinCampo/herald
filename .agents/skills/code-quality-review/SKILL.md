---
name: code-quality-review
description: "Code-quality rubric for this repo (miniCOIL v2): what 'good' Python looks like here, focused on the judgment-level and subtle/version-aware issues that the linter cannot catch. Use whenever reviewing, refactoring, or cleaning up Python; when asked to follow 'best practices', 'modern Python', the 'Zen of Python', or to make code less ugly; and as the standard that the code-quality review workflow loads. Targets Python 3.12."
---

# Code Quality Review (miniCOIL v2, Python 3.12)

The standard for what good code looks like in this repo. It deliberately does **not**
re-list what Ruff already enforces — it captures the stuff that slips past the linter:
version-aware subtleties and human-judgment design issues.

**Full cited catalogue:** `references/` — 97 source-cited, adversarially-verified entries
(PEPs, docs.python.org, ruff/astral, library docs), split one file per topic and indexed in
`references/README.md` (typing, errors, functions, classes, io, imports, perf, naming,
numpy-torch, libraries, testing, design). Open only the topic you need; each entry has
rule · why · bad→good · citation. The sections below are the high-signal summary.

## Division of labour (don't duplicate the linter)

Ruff is already configured with `E, F, I, UP, B, SIM` (line-length 100, target py312)
and auto-fixes most mechanical modern-Python issues. **Assume those run.** A review's
value is everything below, which Ruff misses. For the mechanical gaps Ruff *could* catch
but isn't currently selecting, a reviewer may run, per file, without changing config:

```
ruff check --select N,C4,RET,PTH,RUF,PERF,PIE,TC,ARG,FBT --no-cache <file>
```

and fold the output in — but the linter never sees the §Subtle or §Design items.

## §Repo conventions (from AGENTS.md — non-negotiable)

- **Modern typing only.** `list`/`dict`/`tuple`/`set`, `X | None`, `X | Y`. Never
  `List`/`Dict`/`Optional`/`Union`/`Tuple` from `typing`. Drop the `typing` import when
  it's only those.
- **No fancy pydantic** unless genuinely necessary. Plain settings, simple models.
- **loguru** for logging (not stdlib `logging`/`print`). **typer** for CLI. **uv**
  (`uv run`, never bare `python`). **ruff** for lint/format. **pytest** for tests.
- Prefer a **function** over a class that is just `__init__` + one method.

## §Subtle / version-aware — the easy-to-miss stuff (py312)

These pass Ruff yet are wrong/redundant here:

- **`from __future__ import annotations`** — PEP 563 was deferred then cancelled (superseded
  by PEP 649/749); it was never made mandatory. On py312 it's **redundant** for `list[int]`/
  `X | None` (those already evaluate at runtime since 3.9/3.10). Keep it ONLY for genuine
  forward references or when you deliberately want annotations left unevaluated — and note it
  then *stringifies every annotation*, so naive `__annotations__` introspection breaks (use
  `typing.get_type_hints()`). Ruff `FA100/FA102` flag it, but `FA` isn't in the enforced set
  here, so it's a manual call. (19 files carry it; most don't need it.)
- **Lingering `typing.Optional`/`List`/`Dict`/`Union`/`Tuple`** → modern builtins/`|`.
- **Mutable default args** (`def f(x=[])`/`={}`) — classic footgun.
- **`os.path` / `os.getcwd` / `open(...)` string paths** → `pathlib.Path`; always
  `open()` inside a `with` (or `Path.read_text`/`write_text`).
- **`== None` / `!= None`** → `is None` / `is not None`. `== True/False` → truthiness.
- **`type(x) == T`** → `isinstance(x, T)`.
- **Broad/silent excepts** — bare `except:` or `except Exception:` that swallows; catch the
  specific type, and never `except: pass` without a logged reason.
- **`assert` for runtime validation** — stripped under `python -O`; use a real check + raise.
- **loguru in hot loops** — `logger.debug(f"{x}")` formats eagerly; prefer lazy
  `logger.debug("{}", x)` when the level is usually off and the path is hot.
- **Redundant `list()`/`dict()`/`.keys()`** — iterate the mapping directly; comprehend
  instead of `for ... append` (Ruff `C4` is off here).
- **`zip(a, b)` without `strict=`** in code that assumes equal length.
- **Naming (PEP 8, Ruff `N` is off here)** — `snake_case` funcs/vars, `CapWords` classes,
  `UPPER` constants; no single-char `l/O/I`; don't shadow builtins (`id`, `type`, `list`).
- **Magic numbers / stringly-typed flags** → named constants or `enum.Enum`/`StrEnum`.
- **Plain data carried as dicts/tuples** that would read better as a `@dataclass(slots=True)`
  or `NamedTuple`.
- **`else` after `return`/`raise`/`continue`** — drop it, dedent (Ruff `RET` is off).

## §Design / Zen of Python — what the linter can NEVER see

Score against PEP 20. Flag, with a concrete fix:

- **God functions / long functions** doing many things at mixed abstraction levels →
  decompose. (`train_concept_layers.py` is 960 lines — prime target.) *Flat is better
  than nested; simple is better than complex.*
- **Deep nesting** (`if` pyramids) → guard clauses / early returns.
- **Needless classes** — state-free class, or `__init__`+one method → function/closure.
- **Premature/speculative abstraction** (params/hooks/strategies with one caller) — YAGNI.
  *Special cases aren't special enough to break the rules.*
- **Duplicated logic** → extract one well-named helper.
- **Unclear names**; **comments that restate the code**; **dead or commented-out code**.
- **Hidden side effects / implicit behaviour** — *explicit is better than implicit.*
- **Errors passing silently** (swallowed, or returning `None`/`-1` sentinels) — surface them.
- **Hard to explain** = bad design. *If the implementation is hard to explain, it's a bad
  idea.* Readability counts.

## §Severity & discipline

- **high** = correctness footgun / bug / API misuse / silent failure.
- **medium** = maintainability / clarity / wrong abstraction.
- **low** = style nit (only if it genuinely aids reading).

Every finding must name a concrete fix and a reason. Don't bikeshed; don't invent
problems to look thorough; prefer the smallest change that removes the smell. When a
"fix" would change behaviour, say so explicitly (it needs a test, per TDD).

## §Finding output contract (for the review workflow)

Each finding: `{file, line, category (repo|subtle|design), severity (high|medium|low),
title, why, fix}`. One issue per finding; no raw file dumps back to the caller.
