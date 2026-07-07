# Generated code and migrations

Generated code and migrations need different review discipline than ordinary
hand-written code.

## Generated code

Before editing generated files, find the generator and source of truth. Prefer
changing the generator input and regenerating output. If hand-editing generated
code is unavoidable, document why.

Check generated changes for:

- deterministic output
- minimal diff from generator changes
- no local paths, timestamps, secrets, or machine-specific data
- compatibility with formatting and linting policy

## Data and schema migrations

Migrations are often hard to reverse. Treat them as high-risk when they affect
persistent data, public APIs, or external systems.

Check:

- upgrade path
- downgrade or rollback path, when supported
- data preservation
- idempotency or clear non-idempotency
- compatibility with old and new code during deploys
- tests or dry-run evidence for representative data

## Sources

- Python Packaging User Guide: https://packaging.python.org/
- Alembic documentation: https://alembic.sqlalchemy.org/
- Django migrations documentation: https://docs.djangoproject.com/en/stable/topics/migrations/
