# pyright: reportMissingImports=false

"""Bind switch data to the frozen generation-sweep configuration."""

from hashlib import sha256
from pathlib import Path
from typing import Any

from herald.config import Config

SWEEP_CONFIG_SHA256_METADATA_KEY = b"herald.sweep_config_sha256"


def sha256_file(path: Path) -> str:
    """Return the SHA-256 of one regular artifact file."""
    if not path.is_file():
        raise ValueError(f"artifact file does not exist: {path}")
    digest = sha256()
    with path.open("rb") as file:
        while chunk := file.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def bind_table_to_sweep_config(table: Any, config_path: Path) -> Any:
    """Attach the exact sweep config digest to a parquet table schema."""
    config_sha256 = sha256_file(config_path)
    metadata = dict(table.schema.metadata or {})
    metadata[SWEEP_CONFIG_SHA256_METADATA_KEY] = config_sha256.encode()
    return table.replace_schema_metadata(metadata)


def validate_parquet_sweep_config(
    parquet_path: Path,
    config_path: Path,
    *,
    required_compressor: str | None = None,
    expected_statistics_sha256: str | None = None,
) -> Config:
    """Require a parquet table to name this exact compatible sweep config."""
    config_sha256 = sha256_file(config_path)
    config = _load_config(config_path)
    metadata = _parquet_metadata(parquet_path)
    bound_sha256 = metadata.get(SWEEP_CONFIG_SHA256_METADATA_KEY)
    if bound_sha256 is None:
        raise ValueError("switch parquet lacks sweep config sha256 metadata")
    if bound_sha256.decode() != config_sha256:
        raise ValueError("switch parquet sweep config sha256 does not match")
    if (
        required_compressor is not None
        and required_compressor not in config.compressors
    ):
        raise ValueError(
            f"sweep config does not include {required_compressor!r}"
        )
    statistics_match = (
        config.expected_attention_stats_sha256 == expected_statistics_sha256
    )
    if expected_statistics_sha256 is not None and not statistics_match:
        raise ValueError("sweep config statistics digest does not match")
    return config


def _load_config(path: Path) -> Config:
    try:
        return Config.model_validate_json(path.read_text())
    except (OSError, ValueError) as error:
        raise ValueError(f"could not read sweep config: {path}") from error


def _parquet_metadata(path: Path) -> dict[bytes, bytes]:
    try:
        import pyarrow.parquet as pq

        metadata = pq.read_schema(path).metadata  # type: ignore[no-untyped-call]
    except (ImportError, OSError, ValueError) as error:
        raise ValueError(
            f"could not read switch parquet schema: {path}"
        ) from error
    return dict(metadata or {})
