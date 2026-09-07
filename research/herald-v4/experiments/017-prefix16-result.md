# Fixed prefix16 observation result

The single fixed prefix-disagreement observation did not meet the exploratory
proceed rule. Adding it to the same seven-feature baseline reduced grouped OOF
MAE from 0.349474 to 0.329737, a 5.65% gain against the 10% requirement. Three folds
improved. Within-task Spearman was 0.2372 for CWE and -0.1248 for NIAH, so the
positive-in-both-tasks rule also failed. No predictor is validated.

Before fitting, 015 fixed a 16-token observation and 016 fixed the same fold-local
StandardScaler/Ridge(alpha=1) comparison. This used existing exposed development
outcomes, not new confirmation. No horizon or penalty sweep was performed.

The descriptive observation showed both signal and ambiguity:

| Prefixes | Degradations | Unchanged scores | Improvements |
|---|---:|---:|---:|
| Different, 29 actions |20|8|1|
| Same, 31 actions |10|17|4|

The signal caught 20/30 degradations and differed on 8/25 unchanged outcomes.
It cannot determine the direction of quality change by itself. Computation at
an action boundary would require up to 16 generated tokens per candidate arm
and a shared reference continuation. Wall time was not measured in this offline
extraction; this is not a free observation or a deployment-overhead claim.

Source: scripts/analyze_prefix16.py SHA256
206b7653d1730f55d61c4e99244ee5165356b7ebf6a7eb219fa03d26ee0a058f,
composing the previously audited fold-local analyzer. Raw prefixes and counts
are in results/prefix16-diagnostic/, models and predictions in prefix16-model/.
Independent arithmetic/model audit PASSED in results/query-window-v1-audit.json.

Stop feature variants and revisit baseline/evaluation adequacy. In particular,
the current protocol intentionally omitted task identity even though the quality
metric and task family may be known at decision time. A fair matched-information
baseline must be explicit about that availability before further prediction
claims. Preserve these results and do not retroactively relabel them successful.
