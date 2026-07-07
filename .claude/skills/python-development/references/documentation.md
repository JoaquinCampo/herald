# Documentation and comments

Good documentation explains contracts, intent, and surprising constraints. It
does not restate obvious code.

## Docstrings

Use docstrings for public modules, classes, functions, and methods. Also add
docstrings to private helpers when their behavior is not obvious from the name
and signature.

A useful docstring explains:

- what the object does
- important arguments when annotations are not enough
- return behavior when non-obvious
- raised exceptions when callers need to handle them
- side effects, external calls, or performance costs

## Comments

Comments should explain why, not what. Remove comments that merely repeat the
next line of code.

## TODOs

TODOs should be actionable and traceable. Prefer an issue link or clear owner
context. Avoid vague TODOs that become permanent clutter.

## Sources

- PEP 257 docstring conventions: https://peps.python.org/pep-0257/
- Google Python style guide, comments and docstrings: https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings
