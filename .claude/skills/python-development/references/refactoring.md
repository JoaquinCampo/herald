# Refactoring

Refactoring changes structure while preserving behavior. If behavior changes,
call it a behavior change and test it as one.

## Safe refactor workflow

1. Identify the behavior that must not change.
2. Run existing focused tests or add characterization tests.
3. Make one kind of change at a time.
4. Validate after each meaningful step.
5. Keep renames, moves, and semantic edits separate when possible.
6. Delete dead code after proving it has no current caller.

## Good refactor targets

- Long functions mixing abstraction levels.
- Deeply nested conditionals.
- Duplicated logic.
- Poor names that hide intent.
- Hidden side effects.
- Over-generalized APIs with one caller.
- Tests coupled to implementation details.

## Avoid

- Rewriting working code without a clear quality or behavior goal.
- Moving code and changing behavior in the same diff without tests.
- Adding an abstraction before there are multiple real use cases.
- Preserving compatibility with dead call sites.

## Review questions

- Is the new shape easier to explain?
- Did public behavior stay stable?
- Are tests still behavior-focused?
- Did the refactor remove complexity, or just move it?

## Sources

- Martin Fowler, refactoring catalogue: https://refactoring.com/catalog/
- Martin Fowler, YAGNI: https://martinfowler.com/bliki/Yagni.html
