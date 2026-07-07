# Typing

Type hints should make contracts clearer for humans and tools. They should not
turn simple Python into type-system performance art.

## Match syntax to supported Python versions

Before changing annotations, check the project's minimum Python version.

- Python 3.9+: built-in generics such as `list[str]` and `dict[str, int]`.
- Python 3.10+: union operator such as `str | None`.
- Python 3.11+: `typing.Self`, `typing.Never`, `typing.assert_never`.
- Python 3.12+: PEP 695 generic class and function syntax is available, but use
  it only when the project supports 3.12+ and the checker supports it well.

## Do not add future annotations by habit

`from __future__ import annotations` was introduced by PEP 563 and turns
annotations into strings through Python 3.14. That can help with forward
references or import cycles on older supported versions, but it can surprise
runtime introspection and libraries that inspect annotations. For Python 3.10+
syntax alone, it is not needed.

Python 3.14 changed annotation behavior again through PEP 649 and PEP 749:
annotations are deferred by default, evaluated only when needed, and can be
inspected through `annotationlib`. The future import behavior is unchanged in
3.14, but it is deprecated and expected to be removed after Python 3.13 reaches
end of life. If a project supports only 3.14+, consider removing quoted forward
references and the future import after checking libraries that inspect
annotations.

## Prefer modern aliases

For new code on supported versions, prefer:

- `list[T]`, not `typing.List[T]`
- `dict[K, V]`, not `typing.Dict[K, V]`
- `T | None`, not `typing.Optional[T]`
- `A | B`, not `typing.Union[A, B]`

## Accept broad, return useful

Accept the broadest interface the function needs, such as `Sequence[str]` when
it only iterates. Return concrete types when callers benefit from concrete
behavior, such as `list[str]` when the result is reusable.

## Use Protocol for dependency seams

Use `Protocol` when code depends on behavior rather than inheritance. Avoid
`@runtime_checkable` unless you explicitly understand that runtime checks only
verify member presence, not full signatures or return types.

## Use overloads sparingly

Use `@overload` when return type depends on input type and callers benefit from
precise inference. Do not use overloads to decorate simple functions whose return
is naturally one type.

## Exhaustiveness

For finite modes represented by `Literal` or enums, `assert_never` can make
missing branches visible to the type checker.

## Sources

- Python typing docs: https://docs.python.org/3/library/typing.html
- Python 3.14 annotation changes: https://docs.python.org/3.14/whatsnew/3.14.html#pep-649-pep-749-deferred-evaluation-of-annotations
- future annotations docs: https://docs.python.org/3/library/__future__.html
- annotationlib docs: https://docs.python.org/3/library/annotationlib.html
- PEP 585 built-in generics: https://peps.python.org/pep-0585/
- PEP 604 union operator: https://peps.python.org/pep-0604/
- PEP 673 Self type: https://peps.python.org/pep-0673/
- PEP 695 type parameter syntax: https://peps.python.org/pep-0695/
- Typing protocol spec: https://typing.python.org/en/latest/spec/protocol.html
