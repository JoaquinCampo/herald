"""Fixed causal layer-band features for the locked M3 protocol.

M3 stores four contiguous attention-layer means for each of the nine immutable
per-layer compressor quantities.  The underlying quantities and eviction
semantics are imported from the frozen M2 implementation; this module owns the
layer-band transform and its schema.
"""

from collections.abc import Mapping, Sequence
from math import floor

import numpy as np

from herald.press_sensors import SENSOR_NAMES, SensorCaptureError

QUARTILES = 4
LAYER_COUNT = 32
LAYER_BAND_PROTOCOL = "herald.m3.layer_band_replay.v1"


def layer_band_feature_names() -> list[str]:
    """Return the locked sensor-major 36-column layer-band schema."""
    return [
        f"{name}_layer_q{quartile}_mean"
        for name in SENSOR_NAMES
        for quartile in range(1, QUARTILES + 1)
    ]


LAYER_BAND_FEATURE_NAMES: tuple[str, ...] = tuple(layer_band_feature_names())


def _layer_matrix(layer_values: Sequence[Mapping[str, float]]) -> np.ndarray:
    if not layer_values:
        raise SensorCaptureError("no layers captured for layer bands")
    try:
        matrix = np.asarray(
            [
                [float(row[name]) for name in SENSOR_NAMES]
                for row in layer_values
            ],
            dtype=np.float64,
        )
    except (KeyError, TypeError, ValueError) as error:
        raise SensorCaptureError("missing or invalid layer sensor") from error
    if matrix.ndim != 2 or matrix.shape != (
        len(layer_values),
        len(SENSOR_NAMES),
    ):
        raise SensorCaptureError("layer sensor matrix has an invalid shape")
    if not np.isfinite(matrix).all():
        raise SensorCaptureError(
            "layer sensor matrix contains nonfinite values"
        )
    return matrix


def layer_band_values(
    layer_values: Sequence[Mapping[str, float]],
) -> dict[str, float]:
    """Compute the 36 contiguous quartile means in locked feature order.

    For ``L`` ordered attention layers, quartile ``q`` is exactly
    ``[floor((q-1)L/4), floor(qL/4))``.  Empty bands are rejected rather than
    silently manufacturing a value; the locked Llama model has 32 layers.
    """
    matrix = _layer_matrix(layer_values)
    layer_count = matrix.shape[0]
    result: dict[str, float] = {}
    for sensor_index, name in enumerate(SENSOR_NAMES):
        for quartile in range(1, QUARTILES + 1):
            start = floor((quartile - 1) * layer_count / QUARTILES)
            end = floor(quartile * layer_count / QUARTILES)
            if end <= start:
                raise SensorCaptureError(
                    f"quartile q{quartile} is empty for {layer_count} layers"
                )
            value = float(matrix[start:end, sensor_index].mean())
            feature = f"{name}_layer_q{quartile}_mean"
            if not np.isfinite(value):
                raise SensorCaptureError(f"nonfinite layer-band: {feature}")
            result[feature] = value
    if tuple(result) != LAYER_BAND_FEATURE_NAMES:
        raise SensorCaptureError("layer-band feature order is not locked")
    return result


def aggregate_layer_bands(
    layer_values: Sequence[Mapping[str, float]],
) -> dict[str, float]:
    """Alias with an explicit aggregation name for replay callers."""
    return layer_band_values(layer_values)


# Conventional names mirror the M2 sensor API while keeping the M3 schema
# separate from the frozen global-sensor columns.
sensor_feature_names = layer_band_feature_names


def layer_band_schema() -> dict[str, object]:
    """Describe the immutable transform for lock validation and manifests."""
    return {
        "names": list(SENSOR_NAMES),
        "quartiles": QUARTILES,
        "feature_names": list(LAYER_BAND_FEATURE_NAMES),
        "boundary_rule": "[floor((q-1)L/4), floor(qL/4))",
        "layer_order": "ascending attention module layer_idx",
        "layer_count": LAYER_COUNT,
        "aggregation": "arithmetic mean within each contiguous quartile",
    }
