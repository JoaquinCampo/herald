"""Horizon-based hazard labels for catastrophe prediction.

Only looping and non_termination are trainable catastrophe types.
wrong_answer is normal model behavior — not a training signal.
"""

TRAINABLE_CATASTROPHES = {"looping", "non_termination"}

# For non_termination, the true onset (last token) is useless for
# prediction — by that point generation is already over.  Use a proxy
# onset at this fraction of max_new_tokens instead (kvguard lesson).
DEFAULT_NT_ONSET_FRAC = 0.75


def earliest_onset(
    catastrophe_onsets: dict[str, int],
    catastrophes: list[str],
    max_new_tokens: int = 512,
    nt_onset_frac: float = DEFAULT_NT_ONSET_FRAC,
    n_tokens: int | None = None,
) -> int | None:
    """Return the earliest onset among trainable catastrophe types.

    For looping, uses the detected onset position directly.
    For non_termination, uses a proxy onset at nt_onset_frac *
    max_new_tokens (the true last-token onset is useless for
    prediction).

    Filters out wrong_answer (not a training signal).
    """
    onsets: list[int] = []

    if "looping" in catastrophe_onsets:
        onsets.append(catastrophe_onsets["looping"])

    if "non_termination" in catastrophes:
        proxy = int(nt_onset_frac * max_new_tokens)
        if n_tokens is not None:
            proxy = min(proxy, n_tokens - 1)
        onsets.append(proxy)

    return min(onsets) if onsets else None


def create_horizon_labels(
    n_tokens: int,
    catastrophe_onset: int | None,
    horizon: int,
) -> list[int]:
    """Create per-token binary labels for horizon-based prediction.

    label[t] = 1 if a catastrophe will begin within the next H
    tokens (i.e. onset is in [t+1, t+horizon]), OR if the
    catastrophe has already started (t >= onset).

    Censored sequences (onset=None) get all-zero labels.
    """
    if n_tokens == 0:
        return []

    labels = [0] * n_tokens

    if catastrophe_onset is None:
        return labels

    # Tokens at and after onset: catastrophe already happening
    for t in range(catastrophe_onset, n_tokens):
        labels[t] = 1

    # Tokens within H steps before onset: warning window
    warn_start = max(0, catastrophe_onset - horizon)
    for t in range(warn_start, catastrophe_onset):
        labels[t] = 1

    return labels
