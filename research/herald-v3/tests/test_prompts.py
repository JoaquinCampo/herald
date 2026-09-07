"""Prompt manifest tests using an injected Arrow-shaped dataset."""

import json
from pathlib import Path

import pytest

from herald_v3.engineering.prompts import (
    load_prompt_manifest,
    write_prompt_manifest,
)


def _fixture_files(
    tmp_path: Path,
) -> tuple[Path, Path, list[dict[str, object]]]:
    source_rows: list[dict[str, object]] = []
    official_rows: list[dict[str, object]] = []
    for index in range(10):
        key = 100 + index
        user = f"Give a short answer for fixture {index}."
        prompt = (
            "<|im_start|>system\nFollow exactly.<|im_end|>\n"
            f"<|im_start|>user\n{user}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        source_rows.append(
            {
                "fold": index % 5,
                "fold_hash": f"fold-{index % 5}",
                "split_hash": f"split-{index}",
                "prompt_id": f"ifeval_{key}",
                "task": "ifeval",
                "prompt_text": prompt,
            }
        )
        ids = ["punctuation:no_comma"]
        kwargs: list[dict[str, object]] = [{}]
        if index < 5:
            ids.append("detectable_format:title")
            kwargs.append({})
        official_rows.append(
            {
                "key": key,
                "prompt": user,
                "instruction_id_list": ids,
                "kwargs": kwargs,
            }
        )
    source = tmp_path / "inputs.json"
    source.write_text(json.dumps({"prompts": source_rows}), encoding="utf-8")
    arrow = tmp_path / "if_eval-train.arrow"
    arrow.write_bytes(b"injected Arrow provenance")
    return source, arrow, official_rows


def test_selection_is_deterministic_and_covers_folds(tmp_path: Path) -> None:
    source, arrow, dataset = _fixture_files(tmp_path)
    first = load_prompt_manifest(source, arrow_path=arrow, dataset=dataset)
    second = load_prompt_manifest(source, arrow_path=arrow, dataset=dataset)

    assert len(first.prompts) == 8
    assert first.fingerprint == second.fingerprint
    assert {prompt.fold for prompt in first.prompts} == {0, 1, 2, 3, 4}
    assert all(prompt.multi_instruction for prompt in first.prompts[:5])
    assert first.prompts[0].prompt_text.endswith("<|im_start|>assistant\n")
    assert first.prompts[0].prompt_text_sha256


def test_manifest_roundtrip_can_be_sliced_for_smoke(tmp_path: Path) -> None:
    source, arrow, dataset = _fixture_files(tmp_path)
    manifest = load_prompt_manifest(source, arrow_path=arrow, dataset=dataset)
    saved = tmp_path / "engineering-prompts.json"
    write_prompt_manifest(manifest, saved)

    loaded = load_prompt_manifest(saved, limit=1)

    assert len(loaded.prompts) == 1
    assert loaded.prompts[0].prompt_id == manifest.prompts[0].prompt_id
    assert loaded.prompts[0].kwargs == manifest.prompts[0].kwargs


def test_missing_official_row_fails_instead_of_dropping_prompt(
    tmp_path: Path,
) -> None:
    source, arrow, dataset = _fixture_files(tmp_path)
    dataset[0]["prompt"] = "wrong prompt"

    with pytest.raises(ValueError, match="official IFEval row is missing"):
        load_prompt_manifest(source, arrow_path=arrow, dataset=dataset)
