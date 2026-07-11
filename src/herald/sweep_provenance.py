# pyright: reportMissingImports=false

"""Bind switch data to the frozen generation-sweep configuration."""

from collections.abc import Sequence
from hashlib import sha256
from pathlib import Path
from typing import Any

from herald.config import Config
from herald.storage import hybrid_done, load_reference, reference_done

SWEEP_CONFIG_SHA256_METADATA_KEY = b"herald.sweep_config_sha256"


def initialize_sweep_config(results_dir: Path, config: Config) -> Path:
    """Write a sweep config once, refusing to mix it with prior results."""
    config_path = results_dir / "config.json"
    expected = config.model_dump_json(indent=2)
    if config_path.is_file():
        try:
            actual = config_path.read_text()
        except OSError as error:
            raise ValueError(
                f"could not read existing sweep config: {config_path}"
            ) from error
        if actual != expected:
            raise ValueError(
                "existing sweep config does not match invocation"
            )
        return config_path
    if results_dir.exists() and any(results_dir.iterdir()):
        raise ValueError(
            "sweep results directory contains artifacts without config.json"
        )
    results_dir.mkdir(parents=True, exist_ok=True)
    config_path.write_text(expected)
    return config_path


def validate_sweep_completeness(
    results_dir: Path,
    config: Config,
    *,
    models: Sequence[str] | None = None,
    tasks: Sequence[str] | None = None,
) -> None:
    """Reject missing references or expected hybrid cells before training."""
    selected_models = list(models) if models is not None else config.models
    selected_tasks = list(tasks) if tasks is not None else config.tasks
    unknown_models = set(selected_models) - set(config.models)
    if unknown_models:
        raise ValueError(
            "models not in sweep config: " + ", ".join(sorted(unknown_models))
        )
    unknown_tasks = set(selected_tasks) - set(config.tasks)
    if unknown_tasks:
        raise ValueError(
            "tasks not in sweep config: " + ", ".join(sorted(unknown_tasks))
        )
    errors: list[str] = []
    for model in selected_models:
        for task in selected_tasks:
            prompt_ids = reference_done(results_dir, model, task)
            if len(prompt_ids) != config.prompts_per_task:
                errors.append(
                    f"{model}/{task}: expected {config.prompts_per_task} "
                    f"references, found {len(prompt_ids)}"
                )
                continue
            expected_by_prompt: dict[str, set[int]] = {}
            for prompt_id in prompt_ids:
                try:
                    reference = load_reference(
                        results_dir,
                        model,
                        task,
                        prompt_id,
                    )
                    gen_ids = reference["gen_ids"]
                except (FileNotFoundError, KeyError) as error:
                    errors.append(f"{model}/{task}/{prompt_id}: {error}")
                    continue
                if not isinstance(gen_ids, list) or not gen_ids:
                    errors.append(
                        f"{model}/{task}/{prompt_id}: "
                        "invalid reference gen_ids"
                    )
                    continue
                expected_by_prompt[prompt_id] = set(
                    range(0, len(gen_ids), config.switch_stride)
                )
            for compressor in config.compressors:
                for ratio in config.ratios:
                    actual = hybrid_done(
                        results_dir,
                        model,
                        task,
                        compressor,
                        ratio,
                        require_features=True,
                    )
                    expected = {
                        (prompt_id, switch_s)
                        for prompt_id, switch_positions in (
                            expected_by_prompt.items()
                        )
                        for switch_s in switch_positions
                    }
                    missing = expected - actual
                    if missing:
                        errors.append(
                            f"{model}/{task}/{compressor}/{ratio}: "
                            f"missing {len(missing)} hybrid cells"
                        )
    if errors:
        raise ValueError("incomplete hybrid sweep: " + "; ".join(errors))


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


def load_sweep_config(path: Path) -> Config:
    """Load and validate a serialized sweep configuration."""
    return _load_config(path)


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
