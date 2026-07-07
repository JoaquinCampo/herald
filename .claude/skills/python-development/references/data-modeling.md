# Data modeling

Choose the lightest data shape that makes invalid states difficult and valid
states easy to read.

## Plain collections

Use plain `dict`, `list`, `tuple`, and `set` for local, obvious, short-lived
shapes. Promote them when keys, positions, or invariants become important.

## dataclass

Use `@dataclass` for internal records with named fields and little or no runtime
validation.

Good defaults:

- `frozen=True` when instances should be hashable or immutable.
- `slots=True` for many instances or when accidental attributes would be bugs.
- `kw_only=True` when there are several fields or defaults make construction
  ambiguous.

Avoid putting complex business logic into dataclasses. Keep methods small and
close to the data's invariants.

## TypedDict

Use `TypedDict` for JSON-like dictionaries, request or response payloads, and
config fragments where runtime validation is not needed but static key checking
helps.

Use `NotRequired` for optional keys rather than making every value optional.

## Pydantic

Use Pydantic v2 when runtime validation, parsing, serialization, or settings
integration matters. Keep models flat and validators simple. Prefer explicit
fields and simple validators over clever metaprogramming.

Use `pydantic-settings` for environment-backed settings rather than hand-rolled
environment parsing when Pydantic is already an acceptable dependency.

## Enum and StrEnum

Use `Enum` or `StrEnum` for named finite modes. Use `IntEnum` only when real
integer interop is required. For serialized values, spell strings explicitly,
especially with `StrEnum`.

## NamedTuple

Use `NamedTuple` only when tuple behavior is part of the intended API. For new
opaque result records, prefer a dataclass.

## Sources

- dataclasses docs: https://docs.python.org/3/library/dataclasses.html
- typing TypedDict docs: https://docs.python.org/3/library/typing.html#typing.TypedDict
- enum docs: https://docs.python.org/3/library/enum.html
- Pydantic models docs: https://docs.pydantic.dev/latest/concepts/models/
- Pydantic settings docs: https://docs.pydantic.dev/latest/concepts/pydantic_settings/
