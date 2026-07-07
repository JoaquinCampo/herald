# Naming, docstrings, comments

Code-quality rubric entries for **naming, docstrings, comments** (6 entries). See `README.md` for the full topic index.

## Never shadow a builtin with an argument or variable name  ·  `high`

Do not name parameters or locals after builtins (id, type, list, dict, input, str, filter, map, format, sum, min, max). Append a clarifying word or trailing underscore instead. This repo does NOT enable ruff's A (flake8-builtins) or N rules, so nothing catches it.


**Why:** Shadowing rebinds the builtin in that scope, so a later call like list(result) raises 'TypeError: object is not callable'. It is a genuine runtime bug, not just style, and the reader silently mistakes the argument for the builtin.


**Avoid:**
```python
def renumber_concepts(old_concepts: dict, id: int) -> dict:
    type = old_concepts[id]["kind"]   # rebinds builtin type()
    return {type(k): v for k, v in old_concepts.items()}  # TypeError
```

**Prefer:**
```python
def renumber_concepts(old_concepts: dict, concept_id: int) -> dict:
    kind = old_concepts[concept_id]["kind"]
    return {builtins_type(k): v for k, v in old_concepts.items()}
```

Source: [Ruff rule A002 builtin-argument-shadowing (flake8-builtins)](https://docs.astral.sh/ruff/rules/builtin-argument-shadowing/)

## Docstrings document semantics, not the types already in annotations  ·  `medium`

When parameters and the return are type-annotated, do not repeat the types in the docstring. Describe what the value means and how it behaves; only state a type if there is no annotation, or to convey something the annotation cannot (units, invariants, ownership).


**Why:** With type hints standard, restating ':type x: int' is redundant and goes stale when the signature changes. PEP 257 already says the return's nature (not its bare type) is what introspection can't give you; the Google guide makes the no-redundant-type rule explicit.


**Avoid:**
```python
def trim_around_word(sentence: str, window: int) -> str:
    """Trim sentence.

    Args:
        sentence (str): the sentence string.
        window (int): an int window.
    Returns:
        str: a string.
    """
```

**Prefer:**
```python
def trim_around_word(sentence: str, window: int) -> str:
    """Return the sentence cropped to `window` tokens on each side of the target.

    Args:
        sentence: Raw sentence; whitespace-tokenized internally.
        window: Tokens kept on each side; 0 keeps only the target word.
    Returns:
        The cropped slice, or the original if shorter than the window.
    """
```

Source: [Google Python Style Guide, Functions and Methods docstrings](https://google.github.io/styleguide/pyguide.html)

## Write the summary line in imperative mood, not descriptive  ·  `medium`

A docstring's first line is a command that prescribes the effect: 'Return the cropped slice', 'Build the vocabulary'. Do not write it as a description ('Returns the...', 'This function builds...'). Keep it to one line and follow it with a blank line in multi-line docstrings.


**Why:** PEP 257 mandates imperative mood so the one-line summary reads uniformly and is usable by indexing tools. Descriptive phrasing ('Returns the pathname...') is the most common PEP 257 violation and ruff does not check docstring mood.


**Avoid:**
```python
def parse_muse_dict(path: Path) -> list[tuple[str, str]]:
    """This function returns a list of the parsed (source, target) pairs."""
```

**Prefer:**
```python
def parse_muse_dict(path: Path) -> list[tuple[str, str]]:
    """Parse a MUSE dictionary file into (source, target) word pairs."""
```

Source: [PEP 257 Docstring Conventions](https://peps.python.org/pep-0257/)

## Comments explain WHY; a comment that restates code rots into a lie  ·  `medium`

Comment the reasoning a reader cannot infer (a workaround, a non-obvious constraint, why this branch exists), not what the line plainly does. A comment that paraphrases the code is worse than none, because it is not updated when the code changes and then actively misleads.


**Why:** PEP 8: 'Comments that contradict the code are worse than no comments.' Restating code adds a second source of truth that silently goes stale; the Google guide warns 'Never describe the code.' Reviewers must flag both the redundancy and the staleness risk; ruff cannot read intent.


**Avoid:**
```python
# add one to the offset for each sentence boundary
offset = offset + len(boundaries)  # later code changes to len(boundaries)-1, comment now wrong
```

**Prefer:**
```python
# Boundaries are inclusive on both ends, so the last index is double-counted upstream;
# compensate here rather than touching the shared tokenizer.
offset = offset + len(boundaries)
```

Source: [PEP 8, Comments](https://peps.python.org/pep-0008/)

## Use intention-revealing names; never encode the type in the name  ·  `medium`

Make names descriptive in proportion to their scope, and never bake the variable's type into its name (id_to_name_dict, words_list, count_int). Reserve single-character names (i, j, k, v) for short-lived counters/iterators, exception handles, and with-statement handles.


**Why:** The Google guide explicitly prohibits 'names that needlessly include the type of the variable (for example: id_to_name_dict)'. Type-in-name suffixes lie when the type changes and duplicate what the annotation already states, and short cryptic names in wide scopes hide intent. None of this is caught by ruff E/F/I/UP/B/SIM.


**Avoid:**
```python
concept_id_to_word_dict: dict[int, str] = {}
v = load_vocab(data_dir)  # 'v' lives for 80 lines
```

**Prefer:**
```python
word_by_concept_id: dict[int, str] = {}
vocab = load_vocab(data_dir)
```

Source: [Google Python Style Guide, Naming](https://google.github.io/styleguide/pyguide.html)

## TODO comments need a tracking link, not a person's name  ·  `low`

Format actionable debt as 'TODO: <issue-link> - <description>'. Anchor it to a tracked issue/bug reference, not a developer's name or a bare 'TODO: fix later'. Either resolve it or make it findable and ownable.


**Why:** The Google guide specifies 'a TODO comment begins with the word TODO in all caps, a following colon, and a link to a resource that contains the context, ideally a bug reference', and discourages naming individuals. A nameless or person-stamped TODO is invisible to the tracker and orphaned when that person leaves.


**Avoid:**
```python
# TODO(joaquin): make the pruning threshold configurable someday
```

**Prefer:**
```python
# TODO: github.com/pento/minicoil-v2/issues/142 - make pruning threshold configurable
```

Source: [Google Python Style Guide, TODO Comments](https://google.github.io/styleguide/pyguide.html)
