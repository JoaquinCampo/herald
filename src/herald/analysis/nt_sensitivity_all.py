"""NT-sensitivity across every prediction horizon.

`evaluate.nt_sensitivity` retrains and 5-fold evaluates at
each NT onset fraction for ONE horizon. This module loops
that over the full horizon set so the paper can show the
predictor's robustness to the NT labeling choice is a
property of the approach, not an artifact of H=10.
"""

from pathlib import Path

from loguru import logger

from herald.analysis.common import DEFAULT_HORIZONS, write_json
from herald.evaluate import DEFAULT_NT_FRACS, nt_sensitivity


def run_nt_sensitivity_all(
    results_dir: Path,
    output_path: Path,
    horizons: list[int] | None = None,
    fracs: list[float] | None = None,
) -> dict[str, object]:
    horizons = horizons or DEFAULT_HORIZONS
    fracs = fracs or DEFAULT_NT_FRACS

    by_horizon: dict[str, dict[str, object]] = {}
    out: dict[str, object] = {
        "horizons": horizons,
        "fracs": fracs,
        "by_horizon": by_horizon,
    }

    for h in horizons:
        logger.info(f"NT-sensitivity: H={h}")
        by_horizon[f"H{h}"] = nt_sensitivity(
            results_dir, horizon=h, fracs=fracs
        )

    write_json(output_path, out)
    return out
