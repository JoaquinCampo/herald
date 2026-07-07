# Application boundaries

Boundary code connects your Python code to users, frameworks, databases,
services, queues, files, and external libraries. Keep boundary code thin and
explicit.

## General rule

Parse, validate, and adapt at the boundary. Keep business logic in plain,
testable functions that do not depend on framework objects when possible.

## Web handlers

Route handlers should usually:

- validate request shape
- call application logic
- map domain results to responses
- map domain errors to appropriate status codes

Avoid burying business rules, database queries, and external API calls directly
inside handlers when they can be separated.

## Databases and external services

- Keep SQL, ORM, or client-specific details behind small repositories or
  adapters when the rest of the code should not know them.
- Make transaction boundaries explicit.
- Handle timeouts, retries, partial failures, and idempotency deliberately.
- Test domain logic without requiring live services, and keep a smaller number
  of integration tests for the real boundary.

## Sources

- FastAPI docs: https://fastapi.tiangolo.com/
- Python sqlite3 docs: https://docs.python.org/3/library/sqlite3.html
- unittest.mock docs: https://docs.python.org/3/library/unittest.mock.html
