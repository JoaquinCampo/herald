# HERALD OpenAI Showcase Video Design

Date: 2026-07-10

## Objective

Create an 85-second, 1920 by 1080 product film that presents HERALD to OpenAI's inference team as a credible safety layer for adaptive KV-cache compression.

The film must feel suitable for an internal product launch while preserving research precision. It must not imply that OpenAI has integrated, endorsed, or deployed HERALD.

## Audience and message

Primary audience: OpenAI inference, systems, and research leaders evaluating whether HERALD deserves technical diligence.

Core message: compression damage can emerge silently during generation. HERALD uses the model's own causal logit signals and a private two-token grace window to reject unsafe compression attempts before rejected tokens reach the user.

Closing line: **HERALD. Compression, with an undo button.**

## Storyboard

### 0 to 7 seconds: The hidden cost

Open on a clean token stream beside a shrinking cache representation. The interface initially communicates efficiency, then one token path subtly destabilizes.

Copy:

> More context. Less memory. One hidden cost.

### 7 to 18 seconds: The failure

Show the real GSM8K example for `gsm8k-0`. The uncompressed path reaches the correct answer, 18. The StreamingLLM path with ratio 0.75 at token 128 loses the problem state, invents an unrelated boxes problem, and answers 3.

The visual must distinguish verbatim artifact excerpts from explanatory labels. No synthetic model output may be presented as experimental evidence.

### 18 to 33 seconds: The signal

Transition from generated tokens into live signal traces. Reveal entropy, confidence margin, KL change, and rolling dynamics as causal statistics computed from logits already produced during decoding.

Copy:

> The warning is already inside the model.

Supporting copy:

> No additional forward pass.

### 33 to 52 seconds: The two-token undo button

Animate the runtime mechanism as one continuous causal sequence:

1. Hold the uncompressed cache in reserve.
2. Attempt compression.
3. Generate two private probe tokens.
4. If risk rises, discard the probe and resume from the held cache.
5. If the probe is safe, commit compression and release the reserve.

The first attempt should fail and visibly rewind. A later attempt should pass and settle into a calm compressed state. This is the film's main visual sequence.

### 52 to 72 seconds: Live proof

Present the Orion campaign as a restrained evidence board:

- 552 live held-out IFEval episodes
- 46 prompts, four ratios, three compressors
- Compressed-generation fraction: ExpectedAttention 79.37%, Knorm 12.08%, StreamingLLM 36.20%
- Live-internal quality cost: 0.0027, 0.0118, 0.0208
- Probe and rollback wall overhead: 2.0%, 8.4%, 5.1%

The three compressor lanes should make adaptation visible. HERALD is aggressive where evidence supports it and cautious where it does not.

Footnote the metrics as results from the live Llama IFEval campaign. Do not present compressed-generation fraction as measured memory savings.

### 72 to 85 seconds: OpenAI framing

Pull back from the evidence into a simplified inference-stack view. Position HERALD between token generation and the compression controller, observing signals and approving or rejecting compression attempts.

Copy:

> A safety layer for adaptive KV-cache compression.

End card:

> HERALD
>
> Compression, with an undo button.

## Visual system

The treatment uses warm alabaster, deep graphite, muted signal blue, and restrained coral for unsafe states. It pairs an editorial serif for headline statements with a neutral grotesk for explanation and a monospaced face for tokens and metrics.

Each shot has one focal idea. Product evidence remains large enough to read at 1080p. Motion uses measured pushes, pans, crops, masks, and spatial handoffs. Decorative bouncing, generic AI imagery, glowing brains, excessive particles, and fake terminal activity are excluded.

The OpenAI audience affects the framing and technical specificity, not the branding. The film remains visibly HERALD and does not imitate OpenAI's identity or claim customer status.

## Sound

Use calm, technically confident narration. Every important statement also appears visually, so the film remains understandable when muted.

Create an understated original ambient score with restrained low percussion, soft tonal movement, subtle token ticks, a short rollback cue, and a resolved commit cue. Leave a brief pocket of near-silence before the first unsafe probe is revealed.

## Production architecture

Build the film as a dedicated Remotion project under `showcase/herald-video/`.

Keep the composition modular:

- `scenes/`: one component per storyboard chapter
- `components/`: reusable token stream, cache, signal trace, metric lane, and title primitives
- `data/`: checked-in, minimal JSON extracts derived from current HERALD artifacts
- `public/`: narration, score, sound effects, and any rendered source assets
- `scripts/`: deterministic extraction and validation helpers

The visual layer must consume curated data extracts, not parse multi-gigabyte experiment files during render. Extraction scripts must preserve source paths and metric definitions.

## Error handling and claim safety

The build must fail clearly when a required audio or data asset is missing. Metric validation must compare the curated extracts with their source artifacts before the final render.

Do not claim:

- OpenAI adoption, endorsement, or integration
- Compressor-agnostic transfer
- Zero quality loss
- Measured memory savings from compressed-generation fraction
- Generalization beyond the current Llama evidence
- Total end-to-end wall overhead below 15%

The phrase "before it reaches the user" refers specifically to the private two-token probe and rollback mechanism.

## Verification

Before delivery:

1. Validate every numeric claim against current repository artifacts.
2. Run type checking and the Remotion project's tests.
3. Render representative stills from every scene and inspect them at full resolution.
4. Render a draft video and inspect pacing, text readability, transitions, audio balance, and frame boundaries.
5. Check the final MP4 with `ffprobe` for resolution, duration, frame rate, codec, and audio stream.
6. Watch the final render from start to finish before reporting completion.

## Deliverables

- Final 1920 by 1080 H.264 MP4
- Remotion source project
- Narration script
- Artifact-backed metric extract and validation script
- Contact sheet of representative frames

