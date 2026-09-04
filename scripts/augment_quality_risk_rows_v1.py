"""Join frozen v1 divergence OOF predictions onto v2 development rows.

Missing bands (frozen model abstains near EOS) are filled with 0.0 and no
missingness indicator is stored, so no final-length signal enters.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from herald.quality_risk_features import (  # noqa: E402
    V1_DIV_COLUMNS as V1_BAND_COLUMNS,
)
from herald.quality_risk_features import (  # noqa: E402
    V1_FILL_VALUE as FILL_VALUE,
)
from herald.quality_risk_features import (  # noqa: E402
    V1_PRED_COLUMNS,
)

SCHEMA_VERSION = "herald.quality_risk_development_data_v3.v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows-root", type=Path, required=True)
    parser.add_argument("--v1-oof", type=Path, required=True)
    parser.add_argument("--v1-tabular-report", type=Path, required=True)
    parser.add_argument("--protocol-lock", type=Path, required=True)
    parser.add_argument("--label-audit", type=Path, required=True)
    parser.add_argument("--v2-manifest", type=Path, required=True)
    parser.add_argument("--v1-expected-sha256", type=str, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def join_v1_divergence(
    rows: pd.DataFrame, v1: pd.DataFrame, scales: list[float]
) -> pd.DataFrame:
    """Left-join calibrated v1 band rates; fill abstentions with zero."""
    if len(scales) != len(V1_PRED_COLUMNS):
        raise ValueError("v1 calibration scale roster changed")
    merged = rows.merge(
        v1[["run_id", "token_pos", *V1_PRED_COLUMNS]],
        on=["run_id", "token_pos"],
        how="left",
        validate="many_to_one",
    )
    for name, source, scale in zip(
        V1_BAND_COLUMNS, V1_PRED_COLUMNS, scales, strict=True
    ):
        merged[name] = merged[source].to_numpy(dtype=np.float64) * float(
            scale
        )
        merged.loc[merged[name].isna(), name] = FILL_VALUE
        merged[name] = merged[name].astype(np.float32)
    return merged.drop(columns=list(V1_PRED_COLUMNS))


def main() -> None:
    args = parse_args()
    if args.output_root.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_root}")
    lock = load_json(args.protocol_lock)
    if lock.get("schema_version") != "herald.quality_risk.v1":
        raise ValueError("unexpected protocol schema")
    if sha256_file(args.v1_oof) != args.v1_expected_sha256:
        raise ValueError("v1 OOF hash mismatch")
    manifest = load_json(args.v2_manifest)
    if manifest.get("protocol_lock_sha256") != sha256_file(
        args.protocol_lock
    ):
        raise ValueError("v2 rows belong to another protocol")
    label_audit = load_json(args.label_audit)
    if label_audit.get("pass") is not True:
        raise ValueError("label audit did not pass")
    scales = [
        float(value)
        for value in load_json(args.v1_tabular_report)["calibration_scales"]
    ]
    v1_table = (
        ds.dataset(args.v1_oof, format="parquet")  # type: ignore[no-untyped-call]
        .to_table(columns=["run_id", "token_pos", *V1_PRED_COLUMNS])
        .to_pandas()
    )
    args.output_root.mkdir(parents=True)
    files: dict[str, Any] = {}
    for source in sorted(args.rows_root.glob("*.parquet")):
        frame = (
            ds.dataset(source, format="parquet")  # type: ignore[no-untyped-call]
            .to_table()
            .to_pandas()
        )
        merged = join_v1_divergence(frame, v1_table, scales)
        if len(merged) != len(frame) or set(merged["run_id"].unique()) != set(
            frame["run_id"].unique()
        ):
            raise ValueError(f"lossy v1 join in {source.name}")
        destination = args.output_root / source.name
        pq.write_table(  # type: ignore[no-untyped-call]
            pa.Table.from_pandas(merged, preserve_index=False),
            destination,
            compression="zstd",
            use_dictionary=("run_id", "prompt_id", "task", "press"),
        )
        files[source.stem] = {
            "rows": len(merged),
            "runs": int(merged["run_id"].nunique()),
            "filled_zero_bands": int(
                (merged[list(V1_BAND_COLUMNS)].to_numpy() == FILL_VALUE)
                .all(axis=1)
                .sum()
            ),
            "sha256": sha256_file(destination),
            "columns": list(merged.columns),
        }
    out_manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "development_v3_with_v1_divergence_confirmation_unread",
        "protocol_lock_sha256": sha256_file(args.protocol_lock),
        "label_audit_sha256": sha256_file(args.label_audit),
        "v2_manifest_sha256": sha256_file(args.v2_manifest),
        "v1_oof_sha256": sha256_file(args.v1_oof),
        "v1_calibration_scales": scales,
        "fill_rule": (
            "missing v1 bands filled with 0.0, no missingness stored"
        ),
        "confirmation_prompts_projected": 0,
        "files": files,
    }
    (args.output_root / "manifest.json").write_text(
        json.dumps(out_manifest, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(out_manifest, sort_keys=True))


if __name__ == "__main__":
    main()
