import pytest
import torch

from herald.segmented_attention import combine_segment_attention


def test_combine_segment_attention_matches_concatenated_softmax() -> None:
    query = torch.tensor([[[[0.3, -0.2]]]], dtype=torch.float32)
    key_segments = [
        torch.tensor([[[[0.1, 0.4], [-0.3, 0.2]]]], dtype=torch.float32),
        torch.tensor([[[[0.8, -0.1]]]], dtype=torch.float32),
    ]
    value_segments = [
        torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]], dtype=torch.float32),
        torch.tensor([[[[-2.0, 5.0]]]], dtype=torch.float32),
    ]
    scale = query.shape[-1] ** -0.5
    segment_outputs: list[torch.Tensor] = []
    segment_logsumexp: list[torch.Tensor] = []
    for keys, values in zip(key_segments, value_segments, strict=True):
        logits = torch.matmul(query, keys.transpose(-2, -1)) * scale
        segment_outputs.append(torch.softmax(logits, dim=-1) @ values)
        segment_logsumexp.append(torch.logsumexp(logits, dim=-1))

    actual = combine_segment_attention(segment_outputs, segment_logsumexp)

    keys = torch.cat(key_segments, dim=-2)
    values = torch.cat(value_segments, dim=-2)
    expected = (
        torch.softmax(
            torch.matmul(query, keys.transpose(-2, -1)) * scale,
            dim=-1,
        )
        @ values
    )
    torch.testing.assert_close(actual, expected)


def test_combine_segment_attention_rejects_mismatched_inputs() -> None:
    with pytest.raises(ValueError, match="same non-zero length"):
        combine_segment_attention([torch.zeros(1, 1, 1, 1)], [])
