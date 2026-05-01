"""Matched-prefix replay: forward pass + per-position metric extraction.

`replay_forward` (added in Task 5) runs a single teacher-forced
forward pass on the uncompressed model. `compute_replay_position`
turns paired (compressed, uncompressed) per-position logits into the
union-top-K row plus full-vocab exact scalars (KL both directions, JS,
top-1 ranks). The full-vocab tensors are not retained after this call.
"""

import math

import torch
import torch.nn.functional as F


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
