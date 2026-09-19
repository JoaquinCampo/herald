# 14. Conditional mediation, recipient context, and persistent state

**Research record: CPU reanalysis of the completed mediation follow-up.** No predictor, controller, compressor, or new model forward pass. Source is `HERALD_MEDIATION_TEXT_PACKET.md`, SHA256 `e57e53661e9e35196ca59cc669d2b9943959efddde2ac5e3d86e0d38831a7be7`. Executed v2 source hash: `f5ceb58db806f948b9ff73660452db0057074562307794effa85476cb686bc8b`. Layer indices are zero-based.

This versioned summary preserves the main reasoning, exact exceptions and regeneration entrypoint. The longer record `docs/14_mediation_context_state_and_quality.md`, complete numeric ledgers, independent tests and new staged experiment are in the accompanying `herald-mediation-reanalysis.zip`. Raw input arrays are not committed.

## Assumptions, units and independent verification

The task and checkpoint are unchanged: exact seven-digit NIAH recall, eight prompts in four base/planted families. The latest treatment is Knorm0.1, not a new dose sweep. Thirty cells comprise five first errors, thirteen earlier decisions in failed arms, and twelve decisions in preserved arms. Two post-error02-planted cells remain censored. There are390 patch outputs plus60 baselines,390 late-MLP projection rows, and75 complete continuations. Five errors belong to only three families: four last-digit truncations and one fifth-digit swap.

The new analysis independently checks all450 decision summaries,390 projection rows,75 number-run/score records,60 self-patch summary identities and10 native/self full-trajectory comparisons. All75 texts decode exactly with the supplied tokenizer. Floating-point fields are restored to their original binary32 representations before computing contrasts. Five reused old NPZs match the earlier manifest and their native margins match the new baselines. Full unsaved logits/caches and GPU execution gates remain producer evidence.

The user's prior analysis reproduction is preserved accurately: five of seven artifacts byte-identical; two have last-bit/serialization discrepancies up to7.105427357601002e-15. That is not seven-file byte identity. The user's75-test suite and this analysis's28 new CPU tests are separate validations. New main outputs regenerate byte-identically, and a second standalone implementation reproduces the main contrasts.

## M1. The freeze has a continuous effect in5/5, not merely a binary effect in3/5

Hold the correct token and the compressed baseline's strongest non-target fixed across every condition. Let C be the baseline margin, H the both-reader margin, and F the margin with those readers plus compressed late-MLP outputs frozen. All arm names below end in `/knorm:0.1`.

| Cell | C | H | F | F-C | H-F |
|---|---:|---:|---:|---:|---:|
|00-base/d6|-5.250|5.625|-0.125|5.125|5.750|
|00-planted/d6|-4.875|6.125|0.500|5.375|5.625|
|01-base/d6|-13.250|0.375|-10.000|3.250|10.375|
|01-planted/d6|-6.500|3.125|-4.000|2.500|7.125|
|02-planted/d4|-4.750|10.250|0.250|5.000|10.000|

The freeze removes correction in3/5, but reduces the margin benefit in5/5. Reader restoration still helps under the freeze in5/5. The surviving frozen corrections00-planted and02-planted have only+0.5 and+0.25 reserve;00-base is-0.125. These exact-runtime near-boundary outcomes must not be overgeneralized.

The identity `H-C=(F-C)+(H-F)` is controlled contrast arithmetic, not a unique fraction of natural damage. Releasing MLP outputs also changes their descendants, potentially including later attention, and can interact with normalization. It does not isolate the direct projection of three vectors. The result supports a causal dependence of this reader intervention's benefit on these late outputs; it does not establish an exclusive natural circuit.

## M2. Binary complementarity does not imply positive synergy

The interaction `H-L22-L23+C` is negative in all five first-error cells:
`-1.125,-1.375,-1.750,-0.625,-0.250`, in table order.

For01-base, L22 alone leaves-0.5 and L23 alone-10.625; both give+0.375. Their separate gains12.75 and2.625 would add to+2.125. The observed joint result is smaller but crosses zero. Positive continuous synergy is not needed to explain the apparent AND-like binary outcome.

This is specific to the fixed margin and intervention scale, not a denial of internal nonlinearity. L23 alone corrects0/5 but improves every error margin by0.5–2.625. L22 corrects4/5; its exception is01-base. The alternative L23G1 reference patch also corrects4/5, except01-base. It transfers genuine task-conditioned reference vectors, not norm-matched irrelevant noise. A successful alternative limits uniqueness without establishing that any perturbation would work.

## M3. Recipient context changes both gain and threshold

Reverse reader patches leave all five reference first answers correct. This is not inertness:02-planted's fixed truth/distractor margin falls21 to4. Across the four truncations, forward restoration adds9.625–13.625 to the compressed margin, while reverse replacement removes only0.1875–3.5625 from the reference margin. The swap is the exception: forward gain15, reverse loss17, but reference still wins.

Thus the realized finite response is recipient-dependent, not merely a single universal vector effect applied to different initial thresholds. These are hybrid contrasts, not local Jacobian estimates. Larger reserve, alternative evidence, downstream compensation, and bypass remain possible explanations; active self-repair is not uniquely established.

Reverse late-MLP patches impair01-base and01-planted first answers but leave00-base,00-planted and02-planted correct. The same selected compressed late vectors can be compatible with a wrong compressed answer and a correct reference-recipient answer. Calling them intrinsically wrong-answer vectors is insufficient.

## M4. Correct answers do not require restoring the reference computation

Project the late MLP25–27 outputs onto the fixed target/native-rival direction, omitting final RMS consistently. Reader patches recover only18.44%,17.58%,29.03%,23.75%,60.63% of the reference-minus-compressed projected support gap in the five-cell order. All five first answers become correct. These are fractions of one projection gap, not causal or full-vector recovery fractions.

MLP27 remains opposed to the correct token in two successful reader repairs:01-base and01-planted. In both00 cases, MLP26 becomes more opposed than its compressed baseline while the complete answer becomes correct.

Five first-error patch conditions have positive summed late support but remain wrong:01-base with L22G0,L23G1,MLP25;01-planted with MLP26;02-planted with MLP27. Five have negative summed late support but become correct:00-base reverse MLPs;00-planted freeze and reverse MLPs;02-planted freeze and reverse MLPs. The sign of one component sum is not the sign of the total decision margin.

## M5. Content and closure are distinct competitions

In all60 output conditions at the four truncation cells, the correct token remains the highest-logit digit. Native correct-versus-other-digit margins are3.125,3.250,3.375,1.000, while period outranks the target by5.25,4.875,13.25,6.5. At02-planted's swap, the target loses the digit contest by4.75 but still beats period by19.5625.

These are measured decision competitions, not proof that the model consciously knows the fact or that dedicated content/closure modules exist. They motivate distinguishing a source competitor from premature closure rather than treating both as one uncertainty magnitude.

## M6. Completed-answer repair is real, but not synonymous with cache repair

All75 continuations terminate at EOS. Forty-four have correct first numbers; all44 complete token sequences equal the corresponding same-prefix reference continuation. There are eleven distinct arm/text outputs. These endpoints remain a narrow horizon: four interventions occur at the last required digit; the swap has only two remaining digits, which agree across the source pair.

The continuation code uses the patched forward's returned cache after removing the hooks. Therefore a reader intervention can change both the selected token and K/V written for subsequent steps. Hook removal does not make all consequences transient.

The pinned Qwen2 control flow yields a stronger, source-grounded deduction. K/V is written inside attention before the MLP; the final MLP27 follows all cache writes. Replacing only its output cannot alter K/V in that forward. Consequently, the four final-MLP truncation repairs do not require cache repair. For the25 earlier/preserved MLP27 patches whose selected token remains unchanged, deterministic continuation must also remain unchanged because cache, token and position agree. Those25 continuations were not measured; exact equality is staged as a mandatory control.

A large confidence increase on an unchanged token can therefore be confined to a branch discarded after token selection. This does not mean confidence is statistically uninformative. It means confidence improvement is not itself proof of persistent repair.

Only for this final-MLP intervention, an ideal donor-ray contrast has numerator `d·h_C+alpha*d·(m27_R-m27_C)`. Positive RMS cannot move its exact zero. Five matching old NPZs and new donor projections predict crossings0.240,0.219,0.569,0.328,1.276. Intermediate alpha behavior is unmeasured; BF16 mixing/addition and other competitors can change observed boundaries. The staged ray checks are not a robustness theorem or a natural-compression model.

## M7. Original utility and first-answer correctness differ

The sole discrepancy is01-planted with compressed MLP outputs in reference. Its first number is486015; it later mentions4860455 and4860155 in a longer explanation. First-number correctness is0, original any-exact-number score1. Reverse MLP patches therefore preserve3/5 first answers but4/5 original scores.

Preserve both metrics. Predicting a wrong token is not automatically predicting the loss defined by a scorer. The packet does not establish a continuous human-grounded quality scale; margin, length and edit distance cannot substitute for that definition.

## M8. A new reference-free signal and its limits

Let w be the current selected token and r its runner-up. Define `C_late=sum_l (W_w-W_r)^T diag(gamma) m_l`, for l25–27. The current compressed MLP outputs and readout weights suffice. The gold pivot used to serialize projections cancels by subtracting two columns. No reference donor, gold digit or extra forward is needed; overhead is unbenchmarked.

Negative support means the late updates oppose the current winner relative to its runner-up. On30 native snapshots, it flags3/5 failed arms early and0/3 preserved arms. First hits:00-planted/d4, lead2;01-base/d5 and01-planted/d5, lead1. Misses:00-base and02-planted. At all five actual errors it is positive for the newly winning wrong token. It is a transient conflict marker, not a persistent wrong-answer detector.

All early conflict hits already have a logit gap belowln(10000). There is no observed additional token lead. This statistic is not record11's oracle sum across all28 stored MLP increments.

| Candidate on the eight-arm sampled scope | Failed arms flagged | Preserved negatives | Earlier failures | At least2 decisions early |
|---|---:|---:|---:|---:|
|Entropy>0.02|5/5|1/3|4/5|2/5|
|Winner gap<ln(10000)|5/5|0/3|4/5|3/5|
|Negative last-three winner support|3/5|3/3|3/5|1/5|
|Negative last-MLP winner support|4/5|2/3|3/5|0/5|
|Digit winner; period runner before digit index6|4/5|3/3|4/5|0/5|

The last-MLP-only false positive is03-base; its extra failure hit is02-planted at the error step. The period rule misses02-planted. The entropy false positives are03-base and03-planted; gap false positives are all three preserved arms. Full hit/miss/lead ledgers are in the bundle.

On the earlier72-arm non-excision population, the period rule has only9/22 total detections,7/22 early, and49/50 preserved negatives. Its false positive is02-base/Knorm0.5. Two detections are post-error. This is retrospective reuse of the same families, not independent validation. The rule needs the required seven-digit length, even without knowing the digits.

## Computational target for the next stage

Represent a step as `(z,K_written)=F(K,prefix,position)`; choose `a=argmax(z)` and continue from `(K_written,prefix+a,next_position)`. For a declared utility U, final quality is `V_U(K_written,prefix,a)`, not entropy or margin itself.

A warning must distinguish changed computation, the upcoming decision, and the consequence under U. Current output support can be altered without changing persistent state. Earlier reader changes can potentially alter persistent state while leaving the current token unchanged. Successful oracle patching does not tell us which reference-free information anticipated the failure.

## Focused staged handoff

The new `scripts/staged_token_state_factorial.py` in the companion bundle stages:

- `errors`: five first-error cells, four existing patch types, baseline/patched returned cache crossed with native/reference current token:80 labeled continuations.
- `earlier`: all25 earlier/preserved cells, both-reader and final-MLP patches, same current token with baseline/patched cache:100 labeled continuations, with repeated baselines explicit.
- `ray`: optional25 one-step checks around predicted final-MLP crossings, including exact0/1 endpoints.

Correct token with native cache tests output redirection alone. Native token with patched cache tests persistent-state consequences. Both changed reproduces the existing patch; neither changed reproduces native. Earlier token-preserving patches directly test future quality effects; finalMLP27 must have exact cache and continuation identity.

The runner reuses the executed helpers and original runtime, checks source hashes and native calibration, verifies all unchanged old cache columns, records changed final-column layers, clones every crossed state, reproduces existing diagonal continuations, and preserves failed gates. It has been compiled and CPU-tested, not GPU-executed. No new compressor, ratio, model or training is introduced.

## Regeneration and external context

`uv run python herald-offline/scripts/mediation_core.py /path/to/HERALD_MEDIATION_TEXT_PACKET.md` is a standalone stdlib cross-check. The full companion bundle includes the separate main parser, all effect/exception/source-line ledgers,28 targeted tests, optional old-archive checks and exact GPU handoff.

Pinned implementation: https://raw.githubusercontent.com/huggingface/transformers/v4.57.6/src/transformers/models/qwen2/modeling_qwen2.py . Activation-patching interpretation: Heimersheim and Nanda, arXiv2404.15255. Compensation context: McGrath et al., arXiv2307.15771. These external sources do not establish self-repair or a unique HERALD circuit.

No GPU work was done in this reanalysis; the original source packet and prior arrays are unchanged. Full repository testing is not certified. The current independent checks concern exported evidence, not a recreation of all original unsaved internal states.
