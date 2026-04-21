"""Feature importance extraction for each trained booster.

XGBoost exposes three importance types, none of which is
strictly correct on its own:

  - gain: avg loss reduction when the feature is used in a
    split (most common for "which features matter").
  - weight: raw split count.
  - cover: avg Hessian coverage on splits using the feature.

We dump all three per horizon so the paper can pick whichever
tells the cleanest story and show the rank agreement across
metrics as a robustness check.
"""

from pathlib import Path

from herald.analysis.common import (
    DEFAULT_HORIZONS,
    load_booster,
    model_path_for,
    write_json,
)

IMPORTANCE_TYPES = ("gain", "weight", "cover")


def run_importances(
    model_dir: Path,
    output_path: Path,
    horizons: list[int] | None = None,
) -> dict[str, object]:
    horizons = horizons or DEFAULT_HORIZONS
    by_horizon: dict[str, dict[str, object]] = {}
    out: dict[str, object] = {
        "horizons": horizons,
        "importance_types": list(IMPORTANCE_TYPES),
        "by_horizon": by_horizon,
    }

    for h in horizons:
        mp = model_path_for(model_dir, h)
        if not mp.exists():
            continue
        booster = load_booster(mp)

        entry: dict[str, object] = {}
        for imp in IMPORTANCE_TYPES:
            raw = booster.get_score(importance_type=imp)
            scalar_items: list[tuple[str, float]] = [
                (f, float(v))
                for f, v in raw.items()
                if not isinstance(v, list)
            ]
            scalar_items.sort(key=lambda kv: kv[1], reverse=True)
            entry[imp] = {
                "ranked": [
                    {"feature": f, "value": round(v, 6)}
                    for f, v in scalar_items
                ],
            }
        by_horizon[f"H{h}"] = entry

    write_json(output_path, out)
    return out
