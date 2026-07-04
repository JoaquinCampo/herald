# What the predictor predicts, and why

The predictor is an online estimator that runs alongside generation.
At every generated token $t$ it emits a scalar $\hat{D}(t)$ estimating
the damage that would result if the compressor activated at $t$ and
ran to the end of the generation. Its target is the damage curve of
the measurement section, read at position $t$.

## Why this target

### Why a per-position target rather than a run-level one

A run-level target (predict the whole-run damage label at every token)
gives the predictor the same answer to produce at every position of a
run. It cannot localize anything: the prediction at token 10 and at
token 400 are estimates of the same scalar, and the only thing that
changes along the run is the predictor's confidence. The damage curve
already tells us that damage depends strongly on *when* compression
activates, and a run-level target throws that structure away.

The per-position target is the question an online estimate is for:
given everything observed so far, how harmful is compression *here*.
It varies along the run, it is grounded in a measured quantity at
every supervised position, and it is the value of the damage curve at
$t$, so the predictor and the measurement share one definition of
damage rather than two.

### Why not a fixed-horizon target

A target of the form "damage if compression activates $H$ tokens from
now" introduces a horizon hyperparameter with no natural value, and it
answers an indirect question: the decision a serving system faces at
token $t$ concerns activating compression at $t$, not at $t + H$. The
activation-at-$t$ target is the direct form of the question and has
labels measured at exactly that operationalisation.

## Why the inputs are logit statistics plus ratio, and not compressor identity

At prediction time, compression has not yet activated, so the
predictor's input comes entirely from the uncompressed stream. That
input is identical no matter which compressor is about to switch on.
Whatever the input carries, therefore, is information about the
*position*: how vulnerable this point of the generation is to losing
cache content, not which compressor will exploit that vulnerability.

Conditioning on compressor identity could only add a learned
per-compressor offset on top of that shared signal. Such an offset is
memorized from the training compressors and is undefined for a
compressor never seen in training, so it is precisely the component of
the prediction that cannot transfer. Leaving identity out makes the
compressor-agnostic claim literal: the predictor learns the shared
vulnerability signal, and cross-compressor evaluation (training with
entire compressors held out) tests exactly that. An identity-
conditioned variant remains available as an ablation to quantify how
much compressor-specific signal exists; a small gap supports the
thesis that compression damage has a shared signature.

Ratio is different in kind. It is a numeric quantity, always known to
the system applying compression, and damage at light and heavy ratios
are genuinely different questions; a ratio-blind predictor could only
output an average over the sweep, which answers none of them.
Conditioning on a known numeric knob specifies the question being
asked; it memorizes nothing.

## Why scalar regression rather than binary classification

The label is continuous (a graded quality delta), and a scalar
prediction preserves magnitude: mild degradation and severe derailment
get different values rather than the same bit. Any binary decision is
recoverable downstream by thresholding the scalar, and the threshold
can be chosen, changed, or calibrated after training without touching
the predictor. Training a classifier instead would fix a threshold
before we know what consumers of the prediction need, and would
discard the magnitude information the damage measure was designed to
provide. If the cosine label proves too noisy to regress against, that
is an empirical finding to report, not a reason to pre-commit to a
coarser target.

## Why the streaming constraint is stated as "no additional forward pass"

The predictor's value rests on being effectively free relative to the
generation it monitors. The constraint that guarantees this is
architectural, not a measured latency figure: the predictor may use
only quantities that fall out of the forward pass the model already
performs to generate, may never look at future tokens, and must do a
constant amount of work per token. Causal rolling statistics over past
tokens respect all three. A second forward pass, even a partial one,
would put the monitor's cost in the same class as the thing it
monitors, and would undercut the premise that the signal is already
present in the model's own next-token distribution.
