# Why these controller metrics, and why the MAE gate retired

## The problem with the first target

The first predictor target was relative MAE against a locked
grouped-mean baseline, plus pooled top-decile lift. Two defects
surfaced once models were run against it:

1. **MAE on a zero-inflated target rewards re-aggregation, not
   foresight.** With 62-89 percent of dq values exactly zero, grouped
   medians beat grouped means by 17-19 percent relative MAE using the
   same three inputs and no new information. A target that a
   statistical identity can improve that much measures loss-shape
   alignment, not predictive power (see
   `results/predictor/experiments/result_report.md`).
2. **Pooled lift is not the controller's decision.** The controller
   ranks candidate switch positions within one generation; pooled
   lift ranks rows across prompts, tasks, and ratios, so most of it
   can be earned from coarse structure the baseline already knows.
   Empirically, the median-mix predictor that jumped MAE by 17
   percent left every lift unchanged relative to the mean baseline:
   the two metric families measure different things, and neither is
   the deployment decision.

## What the deployment decision actually is

Pick the earliest switch position whose predicted damage is
acceptable. The natural currency is therefore: memory saved, at a
bounded realized quality cost, on a compressor never seen in
training, with the threshold chosen without access to that
compressor. The primary metric simulates exactly that and nothing
else. The dataset supports it directly because every decision group
samples the full switch grid (median 18 positions per group), and 95
percent of groups contain at least one zero-damage position, so an
informed policy has genuine savings to find.

## Why each choice

- **Savings = 1 - s/ref_len**: the fraction of the generation run
  under compression; monotone in the real KV-memory saving within a
  group, comparable across groups, and free of compressor-specific
  constants.
- **epsilon = 0.01**: at most one point of quality on the 0-1 task
  scale, the conventional "under 1 percent degradation" bar. Chosen
  by convention before any policy was scored, not tuned.
- **Train-only tau**: threshold transfer IS the deployment problem;
  letting tau see the held-out compressor would delete the hard part
  (the knorm shift documented in the experiment log).
- **Worst-case across held-out compressors**: deployment happens on
  one unknown compressor, so the mean across three of them overstates
  the guarantee. Budget-busting splits score zero savings in the
  headline to make overconfident thresholds visibly expensive rather
  than silently averaged away.
- **Catastrophe recall at 10 percent FPR**: the asymmetric failure is
  switching into major damage; a savings metric alone would tolerate
  a policy that is right on average and occasionally catastrophic.
- **Nothing else**: calibration bins, AUPRC, within-prompt precision
  and the MAE diagnostic stay available in reports but are not
  targets. Every additional locked metric is another surface to
  overfit and another reason to fudge later; the two locked metrics
  are jointly sufficient to say "it saves memory, respects the
  budget, and avoids catastrophes, on an unseen compressor".
