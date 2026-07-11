import importlib.util
from pathlib import Path
from typing import Any

import numpy as np

from herald.features import FEATURE_NAMES


def _module() -> Any:
    script = Path(__file__).parents[1] / "scripts" / "export_alarm_bundle.py"
    spec = importlib.util.spec_from_file_location(
        "export_alarm_bundle", script
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load export_alarm_bundle.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_preserves_design_nan_as_missing_feature() -> None:
    module = _module()
    width = len(FEATURE_NAMES)
    blocks = np.zeros((1, 2, width), dtype=np.float16)
    blocks[0, 0, FEATURE_NAMES.index("kl_prev")] = np.nan
    matrix, names = module.hyb_summaries(
        blocks,
        np.asarray([2]),
        np.zeros((1, width), dtype=np.float32),
        2,
    )

    feature = matrix[0, names.index("hyb__step0_kl_prev_k2")]
    assert np.isnan(feature)
    assert module._optional_finite_float(feature, source="feature") is None
    assert module._optional_finite_float(1.5, source="feature") == 1.5
    try:
        module._optional_finite_float(float("inf"), source="feature")
    except ValueError as error:
        assert "finite" in str(error)
    else:
        raise AssertionError("infinite feature must be rejected")
