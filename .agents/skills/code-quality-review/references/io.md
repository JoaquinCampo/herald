# I/O & filesystem

Code-quality rubric entries for **i/o & filesystem** (11 entries). See `README.md` for the full topic index.

## Pass encoding='utf-8' to every text open, including Path.read_text/write_text  ·  `high`

Whenever you open a text file, or call Path.read_text/write_text/open or io.open, pass encoding='utf-8' explicitly. Omitting it uses the locale encoding, which is cp1252 on Windows, not UTF-8. Path.read_text() is NOT safer than open() here: it has the exact same locale default.


**Why:** PEP 597 found that among the 4000 most-downloaded PyPI packages, 82 fail to install on non-UTF-8 locales purely because they omitted encoding. The locale default produces silent mojibake or UnicodeDecodeError when the file contains non-ASCII (e.g. Spanish 'gato'/accented text in this EN-ES project) and the runtime locale is not UTF-8. On Python 3.12 the locale is still the default; UTF-8-by-default (PEP 686) is not active, so the pathlib convenience methods inherit the same trap.


**Avoid:**
```python
from pathlib import Path
text = Path("vocab.txt").read_text()          # locale encoding
with open("out.csv", "w") as f:               # locale encoding
    f.write(spanish_concepts)
```

**Prefer:**
```python
from pathlib import Path
text = Path("vocab.txt").read_text(encoding="utf-8")
with open("out.csv", "w", encoding="utf-8") as f:
    f.write(spanish_concepts)
```

Source: [PEP 597 - Add optional EncodingWarning (peps.python.org)](https://peps.python.org/pep-0597/)

## Create temp files with mkstemp/NamedTemporaryFile, never a hand-built /tmp path  ·  `high`

For scratch files use tempfile.mkstemp() or tempfile.NamedTemporaryFile()/TemporaryDirectory(), never a predictable path like Path('/tmp') / f'{name}.tmp' and never the deprecated tempfile.mktemp(). The high-level helpers create the file atomically with random names and owner-only permissions.


**Why:** The tempfile docs say mkstemp 'creates a temporary file in the most secure manner possible' with 'no race conditions,' while mktemp 'may introduce a security hole' because between getting the name and creating the file 'someone else may have beaten you to the punch' (symlink/TOCTOU attack in a shared temp dir). A hand-built path in /tmp has the same predictable-name vulnerability. TemporaryDirectory also gives guaranteed cleanup via the context manager.


**Avoid:**
```python
from pathlib import Path
tmp = Path("/tmp") / f"minicoil_{os.getpid()}.pt"   # predictable, racy
tmp.write_bytes(blob)
```

**Prefer:**
```python
import tempfile
with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
    tmp.write(blob)
    path = tmp.name      # random name, owner-only, no race
```

Source: [tempfile - Generate temporary files and directories (docs.python.org)](https://docs.python.org/3/library/tempfile.html)

## Never build a path with the / operator from an untrusted or absolute right operand  ·  `medium`

Remember that Path / other silently discards the left side when other is an absolute path, mirroring os.path.join. Path('/data/wiki') / user_segment yields '/etc/passwd' if user_segment is '/etc/passwd'. Validate or relativize segments before joining.


**Why:** This is the real, non-obvious trap behind 'do not concatenate paths.' The pathlib docs state plainly: 'If the argument is an absolute path, the previous path is ignored.' Code that looks like safe sandboxing (base / name) is a path-escape vulnerability when name can start with a slash, and the linter cannot see it.


**Avoid:**
```python
from pathlib import Path
base = Path("/data/corpora")
target = base / requested_name          # '/etc/passwd' escapes base
target.read_bytes()
```

**Prefer:**
```python
from pathlib import Path
base = Path("/data/corpora").resolve()
target = (base / requested_name).resolve()
if not target.is_relative_to(base):     # 3.9+
    raise ValueError(f"path escapes base: {requested_name}")
```

Source: [pathlib - Object-oriented filesystem paths (docs.python.org)](https://docs.python.org/3/library/pathlib.html)

## Treat glob/rglob/iterdir/walk as one-shot iterators, not lists  ·  `medium`

Path.glob, Path.rglob, Path.iterdir and Path.walk return generators that exhaust after one pass. Do not len() them, do not iterate twice, and do not branch on truthiness. Materialize with list(...) once if you need random access, a count, or multiple passes.


**Why:** Because they are lazy iterators (not lists like a naive os.listdir mental model), a second for-loop over the same handle silently yields nothing and len(p.glob(...)) is a TypeError. The bug is invisible to ruff and easy to introduce when refactoring a function to also report a count. Materializing once also avoids re-walking a large tree.


**Avoid:**
```python
matches = corpus.rglob("*.jsonl")
if len(matches):                 # TypeError: generator has no len
    for m in matches: process(m)
for m in matches: archive(m)     # empty: already exhausted
```

**Prefer:**
```python
matches = list(corpus.rglob("*.jsonl"))
if matches:
    for m in matches: process(m)
for m in matches: archive(m)     # works: it is a list
```

Source: [pathlib - Object-oriented filesystem paths (docs.python.org)](https://docs.python.org/3/library/pathlib.html)

## Open CSV files with newline='' and explicit encoding  ·  `medium`

Files handed to csv.reader/csv.writer must be opened with newline='' (plus encoding='utf-8'). The csv module does its own universal-newline handling; without newline='' it can mis-split quoted fields containing newlines and emit a spurious extra \r on \r\n platforms.


**Why:** The csv docs state outright: 'If csvfile is a file object, it should be opened with newline=\'\''. The failure modes are subtle and platform-specific: an embedded newline inside a quoted field gets parsed wrong, and on Windows writes produce blank rows between records. The doc footnote says it is always safe to specify newline='', so there is no downside to making it a hard rule.


**Avoid:**
```python
import csv
with open("pairs.csv", "w", encoding="utf-8") as f:
    csv.writer(f).writerows(rows)   # blank rows on Windows, bad quoted fields
```

**Prefer:**
```python
import csv
with open("pairs.csv", "w", newline="", encoding="utf-8") as f:
    csv.writer(f).writerows(rows)
```

Source: [csv - CSV File Reading and Writing (docs.python.org)](https://docs.python.org/3/library/csv.html)

## Dump JSON with ensure_ascii=False for non-ASCII (EN-ES) data  ·  `medium`

When serializing data that contains non-ASCII text (Spanish accents, concept names like 'niño', 'gestión'), pass ensure_ascii=False to json.dump/json.dumps. The default True escapes every non-ASCII char to \uXXXX, bloating the file and making it unreadable in diffs and logs.


**Why:** The json docs confirm the default 'is guaranteed to have all incoming non-ASCII ... characters escaped.' For a bilingual EN-ES corpus this turns readable vocabulary into walls of ñ escapes, hurting diffs, manual inspection, and file size. ensure_ascii=False emits real UTF-8, which is exactly what a utf-8-opened file expects. This is on-domain and a linter never flags it.


**Avoid:**
```python
import json
json.dump(concepts, fp)            # 'niño' -> 'niño' in the file
```

**Prefer:**
```python
import json
json.dump(concepts, fp, ensure_ascii=False, indent=2)  # 'niño' stays 'niño'
```

Source: [json - JSON encoder and decoder (docs.python.org)](https://docs.python.org/3/library/json.html)

## Read packaged data with importlib.resources.files(), not __file__ path math  ·  `medium`

To read data shipped inside the package (dictionaries, JSON configs, vocab files), use `importlib.resources.files("minicoil_v2").joinpath(name).read_text()` (or `as_file()` when a real filesystem path is genuinely required). Do not build paths with `Path(__file__).parent / ...` or `os.path.dirname(__file__)`, and do not use the deprecated `pkg_resources`.


**Why:** files() gives stable, consistent semantics and works even when the package is loaded from a zip/wheel or a non-filesystem loader, where `__file__` math breaks. The docs state it provides pkg_resources-like access without that package's performance overhead. `as_file()` materializes a temp path only when needed and cleans up after, avoiding leaked temp files.


**Avoid:**
```python
from pathlib import Path

def load_vocab() -> str:
    p = Path(__file__).parent / "data" / "vocab.json"  # breaks inside a zip/wheel
    return p.read_text()
```

**Prefer:**
```python
from importlib.resources import files

def load_vocab() -> str:
    return files("minicoil_v2").joinpath("data", "vocab.json").read_text(encoding="utf-8")
```

Source: [Python 3.12 docs — importlib.resources](https://docs.python.org/3.12/library/importlib.resources.html)

## Use encoding='locale' when you genuinely want the OS locale encoding  ·  `low`

On the rare occasion you actually want the platform locale encoding (e.g. reading a file another locale-bound tool wrote), pass encoding='locale' (added in 3.10) rather than leaving encoding out. This documents intent and silences EncodingWarning when PYTHONWARNDEFAULTENCODING is on.


**Why:** An omitted encoding and a deliberate locale encoding look identical in code, so a reviewer cannot tell a bug from intent. PEP 597 added the explicit 'locale' sentinel precisely so the intentional case is distinguishable and does not trip the warning. Running the test suite with PYTHONWARNDEFAULTENCODING=1 then surfaces only the genuinely accidental omissions.


**Avoid:**
```python
with open(path) as f:        # did the author mean locale, or forget utf-8?
    data = f.read()
```

**Prefer:**
```python
with open(path, encoding="locale") as f:   # explicit: locale is intended
    data = f.read()
```

Source: [PEP 597 - Add optional EncodingWarning (peps.python.org)](https://peps.python.org/pep-0597/)

## Prefer Path.walk (3.12+) over os.walk, but mind the symlink default flip  ·  `low`

On the 3.12 target, use Path.walk() instead of os.walk(): it yields Path dirpaths and takes an on_error callback. Note the deliberate behavior change: Path.walk does NOT follow symlinks by default (lists them under filenames), whereas os.walk follows them. If you relied on os.walk descending symlinked dirs, pass follow_symlinks=True.


**Why:** Path.walk yields a Path for dirpath so children compose with the / operator directly (no os.path.join str-wrangling), and its on_error default of swallowing errors is replaceable with a callback. The symlink default is intentionally the safer one but differs from os.walk, so a silent behavior change can slip in during migration and a linter will not flag it.


**Avoid:**
```python
import os
for dirpath, _dirs, files in os.walk(root):
    for name in files:
        p = os.path.join(dirpath, name)   # str concatenation
        ...
```

**Prefer:**
```python
for dirpath, _dirs, files in root.walk(on_error=print):  # 3.12+
    for name in files:
        p = dirpath / name        # dirpath is a Path
        ...
```

Source: [pathlib - Object-oriented filesystem paths (docs.python.org)](https://docs.python.org/3/library/pathlib.html)

## Use json.load(fp) on the file object, not json.loads(fp.read())  ·  `low`

To read JSON from a file, call json.load(fp) directly on the open file object instead of json.loads(fp.read()). The latter reads the whole file into a Python str first, then re-scans it, doubling peak memory and adding a redundant decode step for no benefit.


**Why:** json.load and json.loads are documented as identical except for the input type, so json.loads(fp.read()) is strictly the same parse with an extra full-size intermediate string held in memory. On large corpus manifests that intermediate is wasteful, and the idiom signals the author did not know json.load exists. The symmetric write idiom is json.dump(obj, fp), not fp.write(json.dumps(obj)).


**Avoid:**
```python
import json
with open(path, encoding="utf-8") as f:
    data = json.loads(f.read())     # whole file buffered as a str first
```

**Prefer:**
```python
import json
with open(path, encoding="utf-8") as f:
    data = json.load(f)             # parses straight from the stream
```

Source: [json - JSON encoder and decoder (docs.python.org)](https://docs.python.org/3/library/json.html)

## On 3.12, set delete_on_close=False to reopen a NamedTemporaryFile by name  ·  `low`

When you need to write a NamedTemporaryFile and then reopen it by .name within the same with-block (common when handing a path to another library), set delete_on_close=False (added in 3.12). The default delete_on_close=True deletes the file the instant it is first closed, and on Windows you cannot reopen a still-open temp file at all without it.


**Why:** Before 3.12 this pattern was a portability minefield: on Windows the file could not be reopened while open, and naive workarounds left files undeleted. The 3.12 delete_on_close flag is the documented fix, recommended because it 'provides assistance in automatic cleaning of the temporary file upon the context manager exit' while still allowing a reopen. It is version-specific and the kind of detail a reviewer who has been bitten on Windows will flag.


**Avoid:**
```python
import tempfile
with tempfile.NamedTemporaryFile(suffix=".onnx") as tmp:
    tmp.write(model_bytes); tmp.flush()
    load_model(tmp.name)   # fails on Windows; risky on close elsewhere
```

**Prefer:**
```python
import tempfile
with tempfile.NamedTemporaryFile(suffix=".onnx", delete_on_close=False) as tmp:  # 3.12+
    tmp.write(model_bytes); tmp.close()
    load_model(tmp.name)   # safe to reopen; deleted on block exit
```

Source: [tempfile - Generate temporary files and directories (docs.python.org)](https://docs.python.org/3/library/tempfile.html)
