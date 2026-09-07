# Reuse inventory and historical boundary

Read `reuse-manifest.json` for exact absolute source paths, content SHA-256 values, checkout revisions and verification results. This draft does not copy data, weights, environments, or source modules into v3.

## Recovered work, not repeated audits

The final messages of audit tasks `01a0730c-933f-76f1-a318-4d4d430e031a` and `01a0730c-bda9-7ac0-91b9-6e51ec3aa70e` were recovered through `read_thread` on local. Their scoped findings inform this proposal. The present check is limited to source timing, artifact currency, and precise reuse hashes.

| Reuse candidate | Source relative to checkout | Allowed use and limitation |
|---|---|---|
| IFEval scorer | main and cero: `src/herald/ifeval.py` | Reuse after agreement with a pinned official scorer on actual pilot answers; preserve instruction vectors and strict/loose fractions |
| Cache-native paired continuation | main and cero: `src/herald/intervention_sweep.py`, `src/herald/live_controller.py` | Adapt pending-token and fork logic; verify full mutable-state isolation and uninterrupted-reference parity on the intended model |
| Compressor wrapper | main and cero: `src/herald/presses.py` | Verify fraction-removed convention and exact dependency version; record actual retained indices/bytes |
| Fresh historical collection harness | cero: `scripts/collect_hidden_state_development.py`, `src/herald/hidden_state_tap.py` | Reuse provenance and collection discipline after validating intervention timing; checkpoint names alone do not establish action timing |
| Paired scores and exposed manifests | cero: `results/recovered/quality-risk-v1/audit-2026-09-04/hidden_state_{development,moderate}/` and corresponding probe prediction CSVs | Exploratory reconstruction and implementation checks only; do not relabel as untouched acceptance data or automatically join to a token-32 switch |
| Prompt-group evaluation | cero: `scripts/probe_hidden_state_risk.py`, `scripts/bootstrap_hidden_state_probe.py` | Reuse grouping concepts and corrected bootstrap multiplicities; binary logistic loss/PCA specification is not the v3 regression target |
| Probe and attention history | cero: `docs/implementation/feature_extension.md` | Determines the narrow novelty claim; not proof that the proposed measurement predicts quality |

Main checkout is `/Users/joaquincamponario/Documents/INCO/RESEARCH/herald-v2`; cero is `/Users/joaquincamponario/orca/workspaces/herald-v2/cero`. The manifest distinguishes their content hashes; neither a Git HEAD nor a historical commit alone identifies dirty working-tree source.

## What prior evidence does and does not establish

The recovered audit reports 160 already-exposed prompts (80 GSM8K, 80 IFEval), five prompt folds, Qwen2.5-7B-Instruct, greedy decoding, and a 512-token cap. Severe and mild collections together represent 960 generations, with only 160 independent prompt clusters. Their prediction target was binary damage, not expected signed loss, although stored paired score columns allow signed exploratory reconstruction.

Direct source verification corrects a timing error in the recovered audit summary: the collector wraps the entire `model.generate` call in KnormPress, applying compression at prefill. `HiddenStateTap` index 32 is a later observation checkpoint on that trajectory, not a token-32 intervention. Thus these paired outcomes concern prefill compression and do not supply labels for the proposed full-cache token-32 switch. Their compressed state32 observations are also not free pre-decision observations for that switch.

The older main-checkout parity notes describe unresolved checks in that workflow. Newer integrity artifacts record equal prefill observations across arms and reference-token parity for the moderate collection. Direct file verification found all 960 listed artifact entries in each collection with matching SHA-256 values, totaling 1,920 checked entries. Those records must be credited within their actual scope; they do not establish every mutable-state fork check needed by a new switch experiment. No blanket assertion that all historical labels are invalid is justified.

Moderate collection provenance matches all 13 recorded current source files. Three of nine severe-collection source hashes differ from the current working tree; frozen source copies exist. Use the exact snapshot paths in the manifest, not today's scripts, when reproducing that historical collection.

The recovered audits report failure of tested scalar and state32/PCA predictors to improve their matched binary-loss comparators. The older signed-loss regressors also failed their recorded specification. These are reasons to avoid another undirected model/checkpoint sweep, not impossibility results about the agreed target. Earlier positive quality-risk gates were withdrawn after evaluation defects, so those positives cannot motivate a v3 success claim.

## What is new in this proposal

`feature_extension.md` already defines a candidate-specific one-token probe, compressed step-zero scalar features, scalar deltas, and greedy token match. Its revision explicitly says raw scalar deltas failed and a native probe would need distribution-level divergence. Generic attention reliance and press-eviction mass were also explored, with a negative aggregate cross-compressor result. Calling any of these broad ideas new would overstate the evidence.

The proposed incremental test is full-vocabulary JS added to matched scalar-probe inputs, evaluated for signed final compliance loss from an explicitly isolated token-32 full-cache decision state. Its novelty is relative to the verified local tested specification. No literature-wide novelty search was completed, and the history already anticipates distribution divergence. The causal timing and acquisition cost are part of the hypothesis: an observation after committed compression is not interchangeable with a sandbox candidate assay before commitment.

## Integration rule

Owner should port the smallest verified functions with their source hashes, not wholesale copy a dirty checkout or import its editable environment. Record the imported implementation and dependency versions before any generation. Historical artifact verification establishes bytes and declared checks; it does not replace running the new acceptance slice. No v3 model revision, unseen-data inventory, or real parity result is claimed by these drafts.
