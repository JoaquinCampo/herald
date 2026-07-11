# pyright: reportMissingImports=false

import pytest

from herald.alarm_bundle_provenance import (
    bundle_source_provenance,
    validate_bundle_target_binding,
)


def _source() -> dict[str, str]:
    return bundle_source_provenance(
        parquet_sha256="a" * 64,
        streams_sha256="b" * 64,
        sweep_config_sha256="c" * 64,
    )


def test_accepts_identical_bundle_and_target_source_provenance() -> None:
    source = _source()
    bundles = {"expected_attention_stats": {"source_provenance": source}}
    targets = {
        "source_provenance": source,
        "compressors": {
            "expected_attention_stats": {"source_provenance": source},
            "knorm": {"source_provenance": source},
        },
    }

    validate_bundle_target_binding(bundles, targets)


def test_rejects_mixed_bundle_or_target_source_provenance() -> None:
    source = _source()
    stale = {**source, "streams_sha256": "d" * 64}
    bundles = {"expected_attention_stats": {"source_provenance": source}}
    targets = {
        "source_provenance": source,
        "compressors": {
            "expected_attention_stats": {"source_provenance": stale},
        },
    }

    with pytest.raises(ValueError, match="source provenance"):
        validate_bundle_target_binding(bundles, targets)
