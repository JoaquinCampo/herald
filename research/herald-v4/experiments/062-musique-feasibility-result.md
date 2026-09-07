# MuSiQue058 feasibility result

Owner rescored all80 raw branch answers from16 canonical records using the official max-alias tokenF1. Fixed feasibility FAIL. Independent final audit accepted: all80 scores and16 technical records verified; results/musique-pilot-independent-audit.json SHA2568fb795f636925046087ef59dc3aa676e279dad71484a54a0a0efcdd4ac49c1d7.

Canonical records use the original first row, original second row, corrected third-row reproduction, and13 final rows. Reproductions are not extra samples. All16 shared-boundary technical checks pass. Supplemental full-prefill parity holds15/16 and is not the reference.

Reference meanF1=.42152778; only5/16 reachF1>=.8, below the frozen12. EA.10 has identical per-row F1, hence16zero signed losses,0positive and0negative. Its five score levels satisfy the score-level gate but do not create loss variation. Knorm.10 is descriptive: meanF1=.3875, onepositive loss, onenegative loss,14zero. It cannot rescue EA's failed pilot.

This closes the fixed model/task/prompt/horizon/action slice without severity, prompt, horizon, feature or learner tuning on these outcomes. No predictor fit. It does not show that QA or EA cannot support risk prediction elsewhere. The16rows and all related full-dev component groups are exposed. Two task pilots, VT and MuSiQue, have not produced a viable new prediction setting; next return to assumptions and strategy before any new collection.

Reproduction: .venv-v4/bin/python scripts/analyze_musique_pilot.py
Summary: results/musique-pilot-summary.json, including per-row source hashes, scores, signed losses and exact frozen gates. Official scorer provenance and all original failures remain preserved. The earlier+0.126984 claim used an incorrect full-prefill reference and remains invalid, see061.
