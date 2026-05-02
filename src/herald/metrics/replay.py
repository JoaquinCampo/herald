"""Matched-prefix replay: forward pass + per-position metric extraction.

`replay_forward` (added in Task 5) runs a single teacher-forced
forward pass on the uncompressed model. `compute_replay_position`
turns paired (compressed, uncompressed) per-position logits into the
union-top-K row plus full-vocab exact scalars (KL both directions, JS,
top-1 ranks). The full-vocab tensors are not retained after this call.
"""

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from herald.config import GenerationArtifact
from herald.metrics.io import PerRunPaths, write_replay_rows


@torch.no_grad()
def replay_forward(
    model: Any,
    input_ids: torch.Tensor,
    generated_token_ids: list[int],
) -> torch.Tensor:
    """Single teacher-forced uncompressed forward.

    Returns logits at every generated position with shape
    (gen_len, vocab_size). Caller is responsible for being outside
    any kvpress context.

    Index alignment: position prompt_len-1+i predicts gen[i], so we
    slice [:, prompt_len-1 : prompt_len-1+N, :].
    """
    if input_ids.shape[0] != 1:
        raise ValueError("replay_forward expects batch size 1")
    if not generated_token_ids:
        return torch.empty(
            (0, model.config.vocab_size), device=input_ids.device
        )

    device = input_ids.device
    gen_t = torch.tensor(
        generated_token_ids, dtype=input_ids.dtype, device=device
    ).unsqueeze(0)
    full = torch.cat([input_ids, gen_t], dim=1)
    out = model(input_ids=full, use_cache=False)
    prompt_len = input_ids.shape[1]
    n = len(generated_token_ids)
    sliced = out.logits[0, prompt_len - 1 : prompt_len - 1 + n, :]
    result: torch.Tensor = sliced.contiguous()
    return result


def _kl(log_p: torch.Tensor, log_q: torch.Tensor, p: torch.Tensor) -> float:
    """KL(p || q) given log_p, log_q, and p on the full vocab."""
    return (p * (log_p - log_q)).sum().item()


def compute_replay_position(
    logits_compressed: torch.Tensor,
    logits_uncompressed: torch.Tensor,
    realized_token_id: int,
    top_k: int = 128,
) -> dict[str, object]:
    """Compute exact full-vocab scalars + union top-K row.

    All inputs are 1-D vocab-size tensors on the same device. The
    full-vocab tensors are not retained after this call.
    """
    log_p_c = F.log_softmax(logits_compressed.float(), dim=-1)
    log_p_u = F.log_softmax(logits_uncompressed.float(), dim=-1)
    p_c = log_p_c.exp()
    p_u = log_p_u.exp()

    log_m = torch.logaddexp(log_p_c, log_p_u) - math.log(2.0)
    js_full = 0.5 * _kl(log_p_c, log_m, p_c) + 0.5 * _kl(log_p_u, log_m, p_u)

    kl_unc_comp = _kl(log_p_u, log_p_c, p_u)
    kl_comp_unc = _kl(log_p_c, log_p_u, p_c)

    argmax_c = int(p_c.argmax().item())
    argmax_u = int(p_u.argmax().item())
    top1_match = argmax_c == argmax_u

    rank_c_under_u = int((p_u > p_u[argmax_c]).sum().item())
    rank_u_under_c = int((p_c > p_c[argmax_u]).sum().item())

    k = min(top_k, log_p_c.shape[-1])
    top_c = torch.topk(log_p_c, k=k).indices
    top_u = torch.topk(log_p_u, k=k).indices
    union = torch.unique(torch.cat([top_c, top_u]))
    lp_c_union = log_p_c[union]
    lp_u_union = log_p_u[union]
    tail_c = float(1.0 - p_c[union].sum().item())
    tail_u = float(1.0 - p_u[union].sum().item())

    realized_lp_c = float(log_p_c[realized_token_id].item())
    realized_lp_u = float(log_p_u[realized_token_id].item())

    return {
        "realized_token_id": int(realized_token_id),
        "union_top_k_token_ids": union.tolist(),
        "logprobs_compressed": lp_c_union.tolist(),
        "logprobs_uncompressed": lp_u_union.tolist(),
        "tail_mass_compressed": max(tail_c, 0.0),
        "tail_mass_uncompressed": max(tail_u, 0.0),
        "realized_logprob_compressed": realized_lp_c,
        "realized_logprob_uncompressed": realized_lp_u,
        "js_full": float(max(js_full, 0.0)),
        "kl_unc_comp_full": float(max(kl_unc_comp, 0.0)),
        "kl_comp_unc_full": float(max(kl_comp_unc, 0.0)),
        "top1_match": top1_match,
        "top1_rank_comp_under_unc": rank_c_under_u,
        "top1_rank_unc_under_comp": rank_u_under_c,
    }


@dataclass(frozen=True)
class ReplayMetrics:
    run_id: str
    num_positions: int
    js_full_max: float
    js_full_mean: float


def replay_run(
    model: Any,
    tokenizer: Any,
    artifact: GenerationArtifact,
    output_root: Path,
    top_k: int = 128,
) -> ReplayMetrics:
    """Run uncompressed replay forward and emit per-run replay parquet.

    Must be called outside any kvpress context. Consumes
    `artifact.compressed_scores` while they are still on GPU.
    """
    paths = PerRunPaths(root=output_root, run_id=artifact.run_id)
    if not artifact.generated_token_ids:
        write_replay_rows([], paths.replay)
        return ReplayMetrics(
            run_id=artifact.run_id,
            num_positions=0,
            js_full_max=0.0,
            js_full_mean=0.0,
        )

    uncompressed_logits = replay_forward(
        model, artifact.input_ids, artifact.generated_token_ids
    )
    rows: list[dict[str, Any]] = []
    js_values: list[float] = []
    for t, gen_id in enumerate(artifact.generated_token_ids):
        comp = artifact.compressed_scores[t]
        unc = uncompressed_logits[t]
        row = compute_replay_position(
            logits_compressed=comp,
            logits_uncompressed=unc,
            realized_token_id=gen_id,
            top_k=top_k,
        )
        row["run_id"] = artifact.run_id
        row["token_pos"] = t
        rows.append(row)
        js_values.append(float(row["js_full"]))  # type: ignore[arg-type]

    write_replay_rows(rows, paths.replay)
    return ReplayMetrics(
        run_id=artifact.run_id,
        num_positions=len(rows),
        js_full_max=max(js_values) if js_values else 0.0,
        js_full_mean=(sum(js_values) / len(js_values) if js_values else 0.0),
    )
