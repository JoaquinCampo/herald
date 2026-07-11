"""Benchmark exact two-segment native flash attention on CUDA."""

import argparse
import json
import statistics
from collections.abc import Callable
from functools import partial
from typing import cast

import torch

from herald.segmented_attention import segmented_flash_attention


def _measure(
    callable_: Callable[[], object], repetitions: int
) -> list[float]:
    timings: list[float] = []
    for _ in range(repetitions):
        start = torch.cuda.Event(enable_timing=True)  # type: ignore[no-untyped-call]
        end = torch.cuda.Event(enable_timing=True)  # type: ignore[no-untyped-call]
        start.record()  # type: ignore[no-untyped-call]
        callable_()
        end.record()  # type: ignore[no-untyped-call]
        end.synchronize()
        timings.append(float(start.elapsed_time(end)))  # type: ignore[no-untyped-call]
    return timings


def _contiguous_flash_attention(
    query: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    *,
    scale: float,
) -> torch.Tensor:
    return cast(
        torch.Tensor,
        torch.ops.aten._scaled_dot_product_flash_attention(
            query,
            keys,
            values,
            0.0,
            False,
            False,
            scale=scale,
        )[0],
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lengths", default="256,1024,4096")
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--repetitions", type=int, default=200)
    args = parser.parse_args()

    torch.manual_seed(0)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    scale = 128**-0.5
    rows: list[dict[str, float | int]] = []

    for total_length in [int(value) for value in args.lengths.split(",")]:
        sink_length = max(4, total_length // 8)
        recent_length = total_length - sink_length
        query = torch.randn(1, 32, 1, 128, device=device, dtype=dtype)
        key_segments = [
            torch.randn(1, 8, sink_length, 128, device=device, dtype=dtype),
            torch.randn(1, 8, recent_length, 128, device=device, dtype=dtype),
        ]
        value_segments = [torch.randn_like(keys) for keys in key_segments]
        contiguous_keys = torch.cat(key_segments, dim=-2)
        contiguous_values = torch.cat(value_segments, dim=-2)

        contiguous = partial(
            _contiguous_flash_attention,
            query,
            contiguous_keys,
            contiguous_values,
            scale=scale,
        )
        segmented = partial(
            segmented_flash_attention,
            query,
            key_segments,
            value_segments,
            scale=scale,
        )

        for _ in range(args.warmup):
            contiguous()
            segmented()
        torch.cuda.synchronize()

        expected = contiguous()
        actual = segmented()
        absolute_error = (actual - expected).abs().float()
        first_contiguous = _measure(contiguous, args.repetitions // 2)
        first_segmented = _measure(segmented, args.repetitions // 2)
        second_segmented = _measure(segmented, args.repetitions // 2)
        second_contiguous = _measure(contiguous, args.repetitions // 2)
        contiguous_ms = statistics.median(
            first_contiguous + second_contiguous
        )
        segmented_ms = statistics.median(first_segmented + second_segmented)
        retained_bytes = sum(
            tensor.numel() * tensor.element_size()
            for tensor in [*key_segments, *value_segments]
        )
        rows.append(
            {
                "total_length": total_length,
                "sink_length": sink_length,
                "contiguous_ms": contiguous_ms,
                "segmented_ms": segmented_ms,
                "slowdown": segmented_ms / contiguous_ms - 1.0,
                "max_abs_error": float(absolute_error.max().item()),
                "mean_abs_error": float(absolute_error.mean().item()),
                "retained_kv_bytes": retained_bytes,
                "contiguous_candidate_bytes_avoided": retained_bytes,
            }
        )

    print(json.dumps({"rows": rows}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
