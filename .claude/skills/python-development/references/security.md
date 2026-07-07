# Security basics

Use this for general Python security hygiene. For security-critical systems,
also use a dedicated security review process.

## Inputs and boundaries

- Treat file paths, URLs, headers, environment variables, CLI arguments, and JSON
  bodies as untrusted until validated.
- Validate shape and range at boundaries.
- Avoid stringly typed modes when a finite enum or explicit parser would be
  safer.

## Secrets

- Do not log secrets, tokens, passwords, private keys, cookies, or full
  authorization headers.
- Do not commit secrets or generated credentials.
- Prefer environment variables or secret stores according to the platform.

## Subprocesses

- Prefer argument lists over shell strings.
- Avoid `shell=True` unless shell features are required and inputs are trusted or
  safely quoted.
- Set timeouts for external commands that can hang.

## Archives and paths

- Treat archive member names as untrusted paths.
- Prevent path traversal when extracting archives.
- Prefer safe extraction filters when using `tarfile` on Python versions that
  support them.
- Validate resolved paths stay under the intended base directory when writing
  user-named files.

## Deserialization

- Avoid unsafe pickle, YAML, or dynamic import loading for untrusted input.
- Use safe loaders and explicit schemas.

## Sources

- Python subprocess security considerations: https://docs.python.org/3/library/subprocess.html#security-considerations
- Python pickle warning: https://docs.python.org/3/library/pickle.html
- Python tarfile extraction filters: https://docs.python.org/3/library/tarfile.html#extraction-filters
- OWASP Top Ten: https://owasp.org/www-project-top-ten/
