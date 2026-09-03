# Which intervention defines compression damage, and why

> **Status: current protocol.** This hierarchy is frozen before model
> training. The parity experiments below are outstanding and decide whether
> the historical switch dataset can be reused.

## The decision boundary

A switch position $s$ means that exactly $s$ generated tokens are fixed and
compression is considered immediately before computing the next token. We use
$s$, rather than an ambiguous token index $t$, throughout the dataset and
paper.

The state at that boundary is the complete decoder state $S_s$, not only its
KV tensors. It includes the KV cache, any pending input token, the attention
mask, cache and position indices, compressor auxiliary state, remaining token
budget, stopping state, decoding configuration, and numerical configuration.
This matters because an emitted token may not yet have been incorporated into
the cache at the instant a generation loop exposes the decision boundary.

## Preferred target: fork the live decoder state

For compressor $c$ and removal ratio $r$, the preferred paired intervention is

$$
d^{\mathrm{fork}}_{c,r}(s)
=
q\!\left(\operatorname{continue}(S_s)\right)
-
q\!\left(\operatorname{continue}(\mathcal C_{c,r}(S_s))\right),
$$

where $\mathcal C_{c,r}$ modifies one independent fork of $S_s$ and the other
fork remains uncompressed. Positive values are degradation, zero is no
measured effect, and negative values are compression-associated lift.

This target most directly identifies the final task-quality effect of
activating a known compressor on the live generation state at that moment.
The two branches must preserve the same prompt, emitted prefix, length budget,
EOS rules, decoding configuration, attention backend, numerical precision,
and initial decoder state. Compressing one fork must not mutate the other
through shared storage or model hooks.

## Compressor semantics come before a universal mechanism

Each compressor must be classified before its labels are interpreted:

1. **Cache-native.** The compressor is a well-defined transformation of the
   state available at the decision boundary.
2. **Auxiliary-state.** A live intervention is valid, but it also requires
   query, attention, statistics, or other state beyond the KV tensors.
3. **Re-prefill-defined.** The faithful compressor operation exists only as
   part of a prefill over the prompt and emitted prefix.

The fork target is preferred for the first two classes. A matched re-prefill
pair is the faithful target for the third class. We do not force compressors
with different execution semantics into one intervention merely to simplify
the dataset.

## Matched re-prefill target

When re-prefill is the faithful or only available operation, both branches are
reconstructed from the same token IDs:

$$
d^{\mathrm{refill}}_{c,r}(s)
=
q\!\left(\operatorname{continue}(R(S_s))\right)
-
q\!\left(\operatorname{continue}(\mathcal C_{c,r}(R(S_s)))\right),
$$

where $R$ rebuilds the state by prefilling the prompt plus the first $s$
reference tokens. Both arms use the same re-prefill path, batch shape,
attention backend, numerical configuration, decoding rules, and remaining
budget.

The historical hybrid sweep already generated the compressed arm this way.
Its uncompressed label currently uses the stored continuously decoded
reference, not a matched no-press re-prefill. Reusing that shortcut therefore
requires evidence, not an assumption.

## Required validation ladder

Run the following checks for every intended compressor and ratio on the real
model configuration, with enough prompts and switch positions to exercise the
supported range.

### 1. Fork-clone parity and isolation

Fork the live state twice without compression and continue both branches.
Require identical continuation token IDs. Separately verify that the forks do
not share writable cache storage, that compressing one fork leaves every tensor
and state field in the other unchanged, that branch execution order does not
change outputs, and that compressor hooks detach completely.

### 2. Sham re-prefill parity

Compare the live full-cache continuation with a no-press re-prefill from the
same token prefix. Record exact continuation match rate, first divergence
position, and the final quality-delta distribution. Token identity is the
strongest criterion; quality equality alone does not establish numerical
parity.

### 3. Intervention parity

Where a live-cache operator exists, compare the compressed live-state fork
with pressed re-prefill from the same prefix. Record the same token- and
quality-level diagnostics.

## Consequences for historical data

- If sham and intervention parity pass, the stored reference quality and
  historical pressed-reprefill hybrids are justified shortcuts for the live
  fork target.
- If sham parity fails, generate a matched no-press re-prefill for every
  labelled switch point before using the re-prefill target.
- If intervention parity fails, the historical hybrids measure pressed
  re-prefill only and must not be described as equivalent to live-cache
  activation.
- If clone isolation fails, repair the fork implementation before producing
  any live-fork labels.

The parity report is the gate for dataset recovery, model training, and every
causal wording in the paper.
