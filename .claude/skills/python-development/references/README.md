# Python development references

Open only the topic needed for the current task. `SKILL.md` is the operating
playbook; these files provide the deeper rules, caveats, and citations.

| Topic | File | Use when |
| --- | --- | --- |
| Project discovery | `project-discovery.md` | Starting work in any repo or choosing commands |
| Tooling | `tooling.md` | Formatters, linters, type checkers, pre-commit, CI |
| Development loop | `development-loop.md` | Implementing features or general code changes |
| Validation | `validation.md` | Choosing checks by work mode and risk |
| API design | `api-design.md` | Designing functions, signatures, returns, and boundaries |
| Typing | `typing.md` | Writing annotations or debugging type-checker issues |
| Data modeling | `data-modeling.md` | Choosing dict, dataclass, Pydantic, enum, or TypedDict |
| Configuration | `configuration.md` | Settings precedence, environment, files, secrets |
| CLI apps | `cli-apps.md` | Command shape, exit codes, stdout, stderr, CLI tests |
| Application boundaries | `boundaries.md` | Web, database, external service, and adapter boundaries |
| Errors and logging | `errors-logging.md` | Handling validation, exceptions, and logs |
| Observability | `observability.md` | Logs, metrics, tracing, health, diagnostics |
| Testing | `testing.md` | Writing or reviewing tests |
| Debugging | `debugging.md` | Investigating crashes, failures, or wrong results |
| Refactoring | `refactoring.md` | Changing structure while preserving behavior |
| Review schema | `review-schema.md` | Emitting structured code review findings |
| Anti-patterns | `anti-patterns.md` | Common process, design, testing, and safety smells |
| IO and filesystem | `io-filesystem.md` | Paths, files, encodings, temp files, JSON, CSV |
| Imports and packaging | `imports-packaging.md` | Package metadata, imports, source roots, build behavior |
| Dependency management | `dependency-management.md` | Lockfiles, dependency groups, extras, upgrades |
| CI and release | `ci-release.md` | CI commands, builds, entry points, package smoke tests |
| Generated code and migrations | `generated-code-and-migrations.md` | Generated outputs, schema and data migrations |
| Performance | `performance.md` | Hot paths, profiling, batching, caching, memory |
| Concurrency | `concurrency.md` | Async, threads, processes, cancellation, cleanup |
| Security basics | `security.md` | Inputs, secrets, subprocesses, unsafe deserialization |
| Documentation | `documentation.md` | Docstrings, comments, TODOs, and reader contracts |
| Style and naming | `style-naming.md` | Names, builtin shadowing, style review boundaries |
| Notebooks | `notebooks.md` | Notebook reproducibility, review, and test extraction |
| Maintenance | `maintenance.md` | Consistency checks when editing this skill |
| Scientific Python | `scientific-python.md` | NumPy, pandas, torch, and array-heavy code |
| Libraries | `libraries.md` | Optional Pydantic, Typer, Loguru, FastAPI, ecosystem notes |

## Entry format

When adding to this catalogue, prefer this shape:

- Rule
- When it applies
- Why it matters
- Avoid
- Prefer
- Source

Keep examples generic. Do not encode project names, internal paths, or one
repository's commands as universal guidance.
