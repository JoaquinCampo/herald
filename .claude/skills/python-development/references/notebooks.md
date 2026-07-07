# Notebooks

Notebook work is still software work. Keep it reproducible, reviewable, and
clear about what is exploratory versus production.

## Reproducibility

- Restart and run all cells before treating results as evidence.
- Keep random seeds, data paths, and environment assumptions visible.
- Avoid hidden state from out-of-order execution.
- Move reusable logic into importable modules when it becomes production code.

## Review

Check notebooks for:

- clear narrative and purpose
- no embedded secrets
- stable outputs when outputs are committed
- reasonable output size
- no dependency on local absolute paths

## Testing

For notebooks that define important logic, extract the logic into modules and
cover it with ordinary tests. Notebook execution can be a smoke test, not the
only correctness signal.

## Sources

- Jupyter documentation: https://docs.jupyter.org/
- pytest docs: https://docs.pytest.org/
