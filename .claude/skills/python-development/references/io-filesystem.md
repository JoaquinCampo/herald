# IO and filesystem

IO code should be explicit about paths, encodings, formats, and resource
lifetimes.

## Paths

Prefer `pathlib.Path` for path manipulation. Accept `str | PathLike[str]` at
boundaries when helpful, then normalize to `Path` internally.

Do not join paths with string concatenation. Be careful when combining a trusted
base path with user-provided or absolute child paths.

## Text IO

Specify encodings for text files. `utf-8` is the usual default for project data,
configuration, JSON, Markdown, and source-adjacent files. Use locale encoding
only when the OS locale is truly the contract.

## Resource management

Use context managers for files, sockets, locks, temporary directories,
subprocesses, and clients that own external resources.

Prefer `Path.read_text` and `Path.write_text` for small files. Stream large
files line by line or in chunks.

## Temp files

Use standard temp utilities or pytest's `tmp_path`. Do not hand-build paths in
shared temp directories with predictable names.

## JSON and CSV

- Use `json.load(fp)` and `json.dump(obj, fp)` with file objects.
- Use explicit encoding for JSON files.
- Consider `ensure_ascii=False` when preserving non-ASCII text matters.
- Open CSV files with `newline=""` and explicit encoding.

## Sources

- pathlib docs: https://docs.python.org/3/library/pathlib.html
- open docs: https://docs.python.org/3/library/functions.html#open
- tempfile docs: https://docs.python.org/3/library/tempfile.html
- json docs: https://docs.python.org/3/library/json.html
- csv docs: https://docs.python.org/3/library/csv.html
