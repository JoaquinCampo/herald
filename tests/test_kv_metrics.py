from dataclasses import dataclass

import torch

from herald.kv_metrics import kv_cache_nbytes


@dataclass
class FakeLayer:
    keys: torch.Tensor
    values: torch.Tensor


@dataclass
class FakeCache:
    layers: list[FakeLayer]


def test_counts_key_and_value_tensor_bytes_across_layers() -> None:
    cache = FakeCache(
        layers=[
            FakeLayer(
                keys=torch.zeros((1, 2, 3, 4), dtype=torch.bfloat16),
                values=torch.zeros((1, 2, 3, 4), dtype=torch.bfloat16),
            ),
            FakeLayer(
                keys=torch.zeros((1, 2, 3, 4), dtype=torch.float32),
                values=torch.zeros((1, 2, 3, 4), dtype=torch.float32),
            ),
        ]
    )

    assert kv_cache_nbytes(cache) == 24 * 2 * 2 + 24 * 4 * 2


def test_counts_legacy_cache_tuples() -> None:
    cache = (
        (
            torch.zeros((1, 1, 5, 2), dtype=torch.float16),
            torch.zeros((1, 1, 5, 2), dtype=torch.float16),
        ),
    )

    assert kv_cache_nbytes(cache) == 40


def test_none_cache_has_zero_bytes() -> None:
    assert kv_cache_nbytes(None) == 0


def test_rejects_unknown_cache_shape() -> None:
    try:
        kv_cache_nbytes(object())
    except TypeError as exc:
        assert "unsupported cache" in str(exc)
    else:
        raise AssertionError(
            "unknown cache must not be silently counted as zero"
        )
