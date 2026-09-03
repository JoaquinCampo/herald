"""Focused contracts for the locked M3 layer-band transform."""

import numpy as np
import pytest

from herald.press_sensors import SENSOR_NAMES, SensorCaptureError
from herald.press_sensors_m3 import (
    LAYER_BAND_FEATURE_NAMES,
    layer_band_feature_names,
    layer_band_schema,
    layer_band_values,
)


def test_layer_bands_use_sensor_major_order_and_floor_boundaries() -> None:
    rows = [
        {
            name: float(index * 10 + sensor)
            for sensor, name in enumerate(SENSOR_NAMES)
        }
        for index in range(8)
    ]
    result = layer_band_values(rows)
    assert layer_band_feature_names() == list(LAYER_BAND_FEATURE_NAMES)
    assert list(result) == list(LAYER_BAND_FEATURE_NAMES)
    assert result[
        "removed_k_norm_mass_fraction_layer_q1_mean"
    ] == pytest.approx(5.0)
    assert result[
        "removed_k_norm_mass_fraction_layer_q2_mean"
    ] == pytest.approx(25.0)
    assert result[
        "removed_k_norm_mass_fraction_layer_q3_mean"
    ] == pytest.approx(45.0)
    assert result[
        "removed_k_norm_mass_fraction_layer_q4_mean"
    ] == pytest.approx(65.0)
    assert all(np.isfinite(value) for value in result.values())


def test_layer_bands_fail_closed_on_empty_or_nonfinite_quartile_inputs() -> (
    None
):
    with pytest.raises(SensorCaptureError):
        layer_band_values([])
    rows = [{name: 1.0 for name in SENSOR_NAMES} for _ in range(4)]
    rows[2][SENSOR_NAMES[0]] = float("nan")
    with pytest.raises(SensorCaptureError):
        layer_band_values(rows)
    with pytest.raises(SensorCaptureError):
        layer_band_values([{name: 1.0 for name in SENSOR_NAMES}])


def test_layer_band_schema_binds_locked_llama_geometry() -> None:
    schema = layer_band_schema()
    assert schema["quartiles"] == 4
    assert schema["layer_count"] == 32
    assert schema["boundary_rule"] == "[floor((q-1)L/4), floor(qL/4))"
    assert schema["feature_names"] == list(LAYER_BAND_FEATURE_NAMES)
