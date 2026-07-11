"""Exact attention over physically separate KV segments."""

import torch


def combine_segment_attention(
    segment_outputs: list[torch.Tensor],
    segment_logsumexp: list[torch.Tensor],
) -> torch.Tensor:
    """Combine segment outputs using their log partition functions."""
    if not segment_outputs or len(segment_outputs) != len(segment_logsumexp):
        raise ValueError(
            "segment outputs and logsumexp must have the same non-zero length"
        )

    stacked_logsumexp = torch.stack(segment_logsumexp, dim=0)
    global_logsumexp = torch.logsumexp(stacked_logsumexp, dim=0)
    weights = torch.exp(stacked_logsumexp - global_logsumexp.unsqueeze(0))
    stacked_outputs = torch.stack(segment_outputs, dim=0)
    return torch.sum(stacked_outputs * weights.unsqueeze(-1), dim=0)


def segmented_flash_attention(
    query: torch.Tensor,
    key_segments: list[torch.Tensor],
    value_segments: list[torch.Tensor],
    *,
    scale: float,
) -> torch.Tensor:
    """Run native CUDA flash attention per KV segment and combine exactly."""
    if not key_segments or len(key_segments) != len(value_segments):
        raise ValueError(
            "key and value segments must have the same non-zero length"
        )
    if query.device.type != "cuda":
        raise ValueError("segmented flash attention requires CUDA tensors")

    outputs: list[torch.Tensor] = []
    logsumexp: list[torch.Tensor] = []
    for keys, values in zip(key_segments, value_segments, strict=True):
        result = torch.ops.aten._scaled_dot_product_flash_attention(
            query,
            keys,
            values,
            0.0,
            False,
            False,
            scale=scale,
        )
        outputs.append(result[0])
        logsumexp.append(result[1])
    return combine_segment_attention(outputs, logsumexp)
