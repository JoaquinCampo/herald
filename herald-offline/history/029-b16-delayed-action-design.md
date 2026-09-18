# Fixed B16 action, explicitly changed estimand

The frozen B0attention variants failed (028). Move the action to one fixed
later state, without changing the feature formula or probing wording again.
B16 means16 shared greedy output tokens have been emitted: original prompt
and generated tokens1..15 are cached; token16 is pending. Compress .10 of this
whole cache, including cached generated tokens. The action can affect token17
onward. Target d16 remains reference complete score minus action complete score.
This is a different estimand from B0 loss and cannot solve B0 by relabeling it.

Use all12exposedNIAH prompts, same model/tokenization/seed/original total128cap.
Construct ORIGINAL shared B0 with run_pair_pilot.build_last_prompt_boundary.
Clone its cache, restore its RNG, then use engine._continue_cache with TOTAL
max_new_tokens=16. Construct B16 from the resulting16IDs and in-place advanced
cache; logical position/cache length=P+15. Capture RNG and decoder fingerprint.
Do not switch to engine.build_boundary(fullpromptprefill), which changes the
reference path. Any EOS at/before16 or otherwise unavailable boundary is retained
as ineligible, never dropped or replaced. Stop interpretation if anycaseineligible.

Both reference and noop B16 continuations must reproduce the OLD uninterrupted
reference tokens, termination and score exactly. Pass ORIGINAL totalcap128 to
engine.continue_from_boundary, because its generated_ids already contain16;
do not pass112 and shorten totaloutput. Independently repeat prefix construction
and require exact IDs/cache. All branches preserve the shared16prefix and source
cache. Candidate Knorm retains native dtype mask semantics and logical positions.

Observe ONLY027unforcedevicted-mass scalar from actual pendingtoken16Q in a
full B16clone, same attention timesVnorm/GQAmax/normalization andmean removed
mass at .10. No cue, oracle spans, answerfeatures, generatedfuturelookahead or
new formula. Compute feature before paired outcomes, collecting both in onerun
avoids duplicate state construction. Persist rawprobabilities/Vnorm/c/masks for
audit. Probe instrumentation must preserve logits; source equality/disjointness,
maskcardinality, physicalbytes, logicalposition and no-op parity are required.
Record ordinary16token advancement separately from additional observationcost.

Staged interpretation: accept complete12case boundary/replay/scoring evidence
first. Require at least3positive and3zero d16 outcomes to evaluate prediction;
if variation is insufficient, close this fixed study without moving timing again.
Otherwise locked LOO single TRAINstandardizedz, Ridgealpha1/interceptTrue versus
LOOtrainingmean. Require10%lowerMSE,8/12promptwins,rawpositivezAUC>=.80. Preserve
all signed labels; referenceperfectscope means these NIAHlossescanonlybe0or1.
Report descriptive exact permutation and fixedOOFpairbootstrap as027. No timing,
action, feature, head or threshold search. Failure closes this specific B16
unforcedmass observation. All data remain exposed, no confirmation claim.

Execution freeze before first GPU outcomes: collector SHA256
c596a5cb10927f1e256ff901d3e29fede4706399a8efa67bfc1fb6949802ef9a;
analyzer 464e3638982e52dfa2d03918bb2734998ba044ba813fa6ab86a2859b63289657.
Owner CPU reproduction in results/b16-cpu-owner passed all controls.
