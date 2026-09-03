"""Train fixed per-compressor IFEval damage-magnitude evidence.

Example::

    uv run python scripts/train_switch_magnitude.py \
        results/predictor/switch_dataset_validated.parquet \
        --sweep-config results/ifeval-intervention-v1/config.json \
        --source-manifest \
        results/ifeval-intervention-v1/llama/ifeval/manifest.json \
        --out results/predictor/magnitude.json
"""

import argparse
from pathlib import Path

from herald.magnitude import (
    DEFAULT_BOOTSTRAP_RESAMPLES,
    DEFAULT_BOOTSTRAP_SEED,
    DEFAULT_TEST_FRACTION,
    fit_evidence,
    load_validated_parquet,
    write_model_bundle,
    write_report,
)
from herald.sweep_provenance import sha256_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("parquet", type=Path)
    parser.add_argument("--sweep-config", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--compressors", nargs="+", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--test-fraction", type=float, default=DEFAULT_TEST_FRACTION
    )
    parser.add_argument(
        "--bootstrap-resamples", type=int, default=DEFAULT_BOOTSTRAP_RESAMPLES
    )
    parser.add_argument(
        "--bootstrap-seed", type=int, default=DEFAULT_BOOTSTRAP_SEED
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows, data_hash = load_validated_parquet(
        args.parquet,
        compressors=args.compressors,
        sweep_config=args.sweep_config,
        source_manifest=args.source_manifest,
    )
    evidence = fit_evidence(
        rows,
        compressors=args.compressors,
        seed=args.seed,
        test_fraction=args.test_fraction,
        bootstrap_resamples=args.bootstrap_resamples,
        bootstrap_seed=args.bootstrap_seed,
    )
    evidence.report["input"] = {
        "path": str(args.parquet),
        "sha256": data_hash,
    }
    evidence.report["sweep_config"] = {
        "path": str(args.sweep_config),
        "sha256": sha256_file(args.sweep_config),
    }
    evidence.report["source_manifest"] = {
        "path": str(args.source_manifest),
        "sha256": sha256_file(args.source_manifest),
    }
    evidence.report["config"] = {
        "seed": args.seed,
        "test_fraction": args.test_fraction,
        "bootstrap_resamples": args.bootstrap_resamples,
        "bootstrap_seed": args.bootstrap_seed,
        "compressors": sorted(evidence.fits),
    }
    write_report(evidence.report, args.out)
    model_dir = args.model_dir or args.out.with_name(
        f"{args.out.stem}_models"
    )
    write_model_bundle(evidence, model_dir)
    print(
        "wrote magnitude evidence and native models for "
        f"{len(evidence.fits)} compressors -> {args.out}, {model_dir}"
    )


if __name__ == "__main__":
    main()
