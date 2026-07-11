"""Exact isolated byte accounting for retained KV-cache tensors."""

from typing import Any

import torch


def kv_cache_nbytes(cache: Any) -> int:
    """Return bytes retained by key/value tensor storages in ``cache``.

    Storage bytes, rather than CUDA allocator totals or logical tensor
    sizes, prevent a sliced cache view from claiming savings while its
    original backing allocation is still retained.
    """
    if cache is None:
        return 0

    tensors: list[torch.Tensor] = []
    layers = getattr(cache, "layers", None)
    if layers is not None:
        for layer in layers:
            retained_tensors = getattr(layer, "retained_tensors", None)
            if callable(retained_tensors):
                layer_tensors = retained_tensors()
                if not all(
                    isinstance(tensor, torch.Tensor)
                    for tensor in layer_tensors
                ):
                    raise TypeError(
                        "cache layer retained_tensors must return tensors"
                    )
                tensors.extend(layer_tensors)
                continue
            keys = getattr(layer, "keys", None)
            values = getattr(layer, "values", None)
            if not isinstance(keys, torch.Tensor) or not isinstance(
                values, torch.Tensor
            ):
                raise TypeError(
                    "unsupported cache layer: keys and values must be tensors"
                )
            tensors.extend((keys, values))
    elif isinstance(cache, (tuple, list)):
        for layer in cache:
            if (
                not isinstance(layer, (tuple, list))
                or len(layer) < 2
                or not isinstance(layer[0], torch.Tensor)
                or not isinstance(layer[1], torch.Tensor)
            ):
                raise TypeError("unsupported legacy cache layer")
            tensors.extend((layer[0], layer[1]))
    else:
        raise TypeError(f"unsupported cache type: {type(cache).__name__}")

    seen: set[tuple[str, int, int]] = set()
    total = 0
    for tensor in tensors:
        storage = tensor.untyped_storage()
        identity = (str(tensor.device), storage.data_ptr(), storage.nbytes())
        if identity in seen:
            continue
        seen.add(identity)
        total += storage.nbytes()
    return total
