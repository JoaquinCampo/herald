# Concurrency

Concurrency code must be explicit about ownership, cancellation, cleanup, and
shared state.

## Choose the model deliberately

- `asyncio`: many concurrent IO-bound operations in one thread.
- threads: blocking IO or libraries that release the GIL.
- processes: CPU-bound Python work or isolation needs.
- external workers or queues: durable background work and retries.

Do not add concurrency before a sequential version is correct and measured.

## asyncio

- Await tasks or keep explicit ownership of background tasks.
- Use timeouts at IO boundaries.
- Propagate cancellation unless you have a cleanup reason to delay it.
- Use `TaskGroup` on Python 3.11+ for structured concurrency when available.
- Avoid blocking calls in the event loop.

## Threads and processes

- Protect shared mutable state.
- Keep worker inputs and outputs serializable for processes.
- Shut down executors and pools.
- Make randomness and global caches explicit when they affect results.
- Be explicit about multiprocessing start methods when behavior depends on fork,
  spawn, or forkserver. Python 3.14 changed the default start method on many
  Unix platforms from fork to forkserver, which can expose pickling and global
  state assumptions.

## Tests

Concurrency tests need timeouts and deterministic synchronization where possible.
Avoid sleeps as the only coordination mechanism. Check for leaked tasks,
unhandled exceptions, and cleanup of resources.

## Sources

- asyncio docs: https://docs.python.org/3/library/asyncio.html
- asyncio TaskGroup docs: https://docs.python.org/3/library/asyncio-task.html#task-groups
- concurrent.futures docs: https://docs.python.org/3/library/concurrent.futures.html
- multiprocessing docs: https://docs.python.org/3/library/multiprocessing.html
- Python 3.14 multiprocessing start-method change: https://docs.python.org/3.14/whatsnew/3.14.html#porting-to-python-3-14
