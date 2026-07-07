# Style and naming

Style should make code easier to read and maintain. Prefer tool-enforced style
where possible, and spend human attention on names and structure that tools
cannot judge.

## Names

- Use names that reveal intent.
- Avoid shadowing builtins such as `id`, `type`, `list`, `dict`, and `file`.
- Avoid single-letter names except for tiny local scopes where the convention is
  obvious.
- Name booleans as predicates when possible, such as `is_ready` or `has_cache`.
- Name functions after what they do, not how they are implemented.

## Human review scope

Do not bikeshed formatting, import order, or quote style when tools own those
choices. Do comment when a name is actively misleading or a structure hides the
reader's path through the code.

## Sources

- PEP 8: https://peps.python.org/pep-0008/
- Google Python style guide: https://google.github.io/styleguide/pyguide.html
