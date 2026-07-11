"""Immutable source bindings for frozen alarm bundles."""

from collections.abc import Mapping
from typing import Any

SOURCE_FIELDS = (
    "parquet_sha256",
    "streams_sha256",
    "sweep_config_sha256",
)


def bundle_source_provenance(
    *,
    parquet_sha256: str,
    streams_sha256: str,
    sweep_config_sha256: str,
) -> dict[str, str]:
    """Validate and name the exact artifacts used to train one bundle."""
    source = {
        "parquet_sha256": parquet_sha256,
        "streams_sha256": streams_sha256,
        "sweep_config_sha256": sweep_config_sha256,
    }
    _validate_source_provenance(source, source="bundle source provenance")
    return source


def validate_bundle_target_binding(
    bundle_meta: Mapping[str, Mapping[str, Any]],
    targets: Mapping[str, Any],
) -> None:
    """Reject a target file not produced with the loaded frozen bundles."""
    target_source = _source_from(targets, source="fidelity targets")
    compressors = targets.get("compressors")
    if not isinstance(compressors, Mapping):
        raise ValueError("fidelity targets need a compressors object")
    missing_compressors = set(bundle_meta) - set(compressors)
    if missing_compressors:
        raise ValueError(
            "fidelity targets lack loaded bundle compressors: "
            f"{sorted(missing_compressors)}"
        )
    for compressor, meta in bundle_meta.items():
        bundle_source = _source_from(meta, source=f"bundle {compressor}")
        target = compressors[compressor]
        if not isinstance(target, Mapping):
            raise ValueError(
                f"fidelity target {compressor} must be an object"
            )
        compressor_source = _source_from(
            target,
            source=f"fidelity target {compressor}",
        )
        source_matches = (
            bundle_source == target_source
            and bundle_source == compressor_source
        )
        if not source_matches:
            raise ValueError(
                "source provenance mismatch for "
                f"{compressor} bundle and target"
            )


def _source_from(value: Mapping[str, Any], *, source: str) -> dict[str, str]:
    provenance = value.get("source_provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError(f"{source} needs source_provenance")
    result: dict[str, str] = {}
    for name in SOURCE_FIELDS:
        result[name] = _required_digest(
            provenance.get(name),
            source=source,
            name=name,
        )
    return result


def _validate_source_provenance(
    value: Mapping[str, Any],
    *,
    source: str,
) -> None:
    for name in SOURCE_FIELDS:
        _required_digest(value.get(name), source=source, name=name)


def _required_digest(value: Any, *, source: str, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(char not in "0123456789abcdef" for char in value)
    ):
        raise ValueError(f"{source} has invalid {name}")
    return value
