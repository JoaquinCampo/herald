import torch

from herald.presses import get_press


def test_snapkv_skips_context_not_longer_than_window() -> None:
    press = get_press("snapkv", 0.25)
    keys = torch.randn(1, 2, 64, 8)
    values = torch.randn(1, 2, 64, 8)

    compressed_keys, compressed_values = press.compress(
        None,
        torch.empty(0),
        keys,
        values,
        torch.empty(0),
        {},
    )

    assert compressed_keys is keys
    assert compressed_values is values
