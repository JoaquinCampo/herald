# Performance

Do not optimize by instinct. Measure first, then simplify or optimize the hot
path that matters.

## Workflow

1. Define the workload and success metric.
2. Capture a baseline.
3. Profile or inspect enough to locate the bottleneck.
4. Make one change.
5. Re-measure.
6. Verify correctness did not regress.

## Common Python performance wins

- Avoid accidental O(n squared) loops.
- Use set or dict membership for repeated membership checks.
- Use comprehensions for straightforward collection building.
- Stream into reducers instead of materializing huge intermediate lists.
- Batch IO, database, network, or model calls.
- Bound caches in long-lived processes.
- Avoid repeated expensive validation or adapter construction inside hot loops.
- Avoid logging or formatting work on disabled log paths.

## Caching

Use caching only when keys are stable, memory growth is bounded, and stale data
is acceptable. Prefer bounded `lru_cache` for long-lived processes unless the
input space is naturally tiny.

## Profiling tools

Choose tools appropriate to the workload:

- `timeit` for small pure functions.
- `cProfile` or sampling profilers for application flows.
- memory profilers for memory pressure.
- framework-specific profilers for database, web, or ML workloads.

## Sources

- Python timeit docs: https://docs.python.org/3/library/timeit.html
- Python profile docs: https://docs.python.org/3/library/profile.html
- functools cache docs: https://docs.python.org/3/library/functools.html
- itertools docs: https://docs.python.org/3/library/itertools.html
