# Predictor pilot

## Aim

HERALD v1 is a pre-compression controller.

At each candidate switch position `s`, predict the final quality damage
that would result if compression starts at `s` and remains active until
the generation ends.

```text
D(s) = q(reference) - q(hybrid with compression from s onward)
```

The prediction answers a serving question:

> Given what we have seen so far, is it safe to turn compression on now?

## Primary target

Train on continuous `dq`, including negative values.

```text
dq = q_ref - q_hybrid
```

Binary risk labels are downstream views, not the primary target:

```text
damaged = dq > 0
major_damage = dq >= 0.5
```

This keeps magnitude information and lets thresholds be calibrated later.

## Inputs

Use only information available before compression starts:

- causal `feat__*` logit statistics from the reference stream at `s`
- the numeric compression `ratio`

Do not use these as primary-model inputs:

- `compressor`
- `prompt_id`
- `q_ref` or `q_hybrid`
- `damaged` or `major_damage`
- raw `s`, `relative_s`, or `ref_len`

`feat__position` is allowed for the operational model because position is
known online, but it must be ablated to test whether signal remains beyond
position.

## First pilot

Use the switch-level dataset:

```text
results/predictor/switch_dataset.parquet
```

Start with completed real compressors:

- `streaming_llm`
- `expected_attention`
- `knorm`

Add `snapkv` with a completeness note. Keep `random` separate until its
IFEval rows finish; use it as a degradation-floor stress test, not the
main transfer claim.

## Evaluation

Main split:

```text
leave-one-compressor-out
```

Leakage control:

```text
group by (model, task, prompt_id)
```

Never use row-random splits, because switch positions from the same prompt
share context and labels.

The model must beat train-only deployable baselines, especially:

- mean `dq` by `ratio`
- mean `dq` by `task, ratio`
- mean `dq` by `task, ratio, position_bucket`

The key question is whether causal logit features add predictive power
beyond task, ratio, and position.

## First model

Use a dependency-clean baseline first:

- PyTorch linear regressor
- Huber loss
- train-fitted median imputation and standardization
- grouped validation for regularization

Only add a small MLP if the linear model shows signal.

## Non-goal for v1

HERALD v1 is not an already-compressed monitor.

That later question is different:

> If compression is already active and remains active, are we in trouble?

Answering it needs compressed-stream features and different
counterfactuals. The current pilot is only about deciding whether to turn
compression on at `s`.
