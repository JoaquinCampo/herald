import json
import shutil
from pathlib import Path

import pytest
from build_evidence import build_evidence, check_generated_evidence

ROOT = Path(__file__).resolve().parents[3]

ARTIFACT_PATHS = (
    Path("results/live_controller_v3/episodes.jsonl"),
    Path("results/live_controller_v3/baseline.jsonl"),
    Path("results/sweep/llama/gsm8k/references/gsm8k-0.json"),
    Path("results/sweep/llama/gsm8k/hybrids/streaming_llm__0.7500.jsonl"),
)


@pytest.fixture
def artifact_root(tmp_path: Path) -> Path:
    for relative_path in ARTIFACT_PATHS:
        target = tmp_path / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative_path, target)
    return tmp_path


def read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text("".join(f"{json.dumps(row)}\n" for row in rows))


def test_live_campaign_metrics_match_current_artifacts() -> None:
    evidence = build_evidence(ROOT)
    assert evidence["campaign"]["episode_count"] == 552
    assert evidence["campaign"]["prompt_count"] == 46
    assert evidence["campaign"]["ratio_count"] == 4
    expected = {
        "expected_attention": (0.8092, 0.0027, 0.020),
        "knorm": (0.1271, 0.0118, 0.084),
        "streaming_llm": (0.3648, 0.0208, 0.051),
    }
    for compressor, values in expected.items():
        row = evidence["campaign"]["compressors"][compressor]
        assert round(row["compressed_generation_fraction"], 4) == values[0]
        assert round(row["quality_cost"], 4) == values[1]
        assert round(row["revert_wall_overhead"], 3) == values[2]


def test_gsm8k_example_is_the_real_s128_failure() -> None:
    example = build_evidence(ROOT)["gsm8k_example"]
    assert example["prompt_id"] == "gsm8k-0"
    assert example["switch_position"] == 128
    assert example["ratio"] == 0.75
    assert example["reference_answer"] == "18"
    assert example["compressed_answer"] == "3"
    assert example["compressed_quality"] == 0.0
    assert example["reference_excerpt"] == (
        "Money made = 9 (eggs left) * 2 (price per egg) = $18"
    )
    assert example["compressed_excerpt"] == (
        "Number of boxes = Eggs left / Eggs per box = 9 / 3 = 3 boxes"
    )


def test_check_generated_evidence_requires_full_json_equality(
    tmp_path: Path,
) -> None:
    output = tmp_path / "evidence.generated.json"
    output.write_text(json.dumps(build_evidence(ROOT), indent=2) + "\n")
    check_generated_evidence(ROOT, output)

    stale = json.loads(output.read_text())
    stale["campaign"]["episode_count"] = 551
    output.write_text(json.dumps(stale, indent=2) + "\n")

    with pytest.raises(ValueError, match="out of date"):
        check_generated_evidence(ROOT, output)


@pytest.mark.parametrize(
    ("relative_path", "excerpt", "message"),
    (
        (
            Path("results/sweep/llama/gsm8k/references/gsm8k-0.json"),
            "Money made = 9 (eggs left) * 2 (price per egg) = $18",
            "reference excerpt",
        ),
        (
            Path(
                "results/sweep/llama/gsm8k/hybrids/"
                "streaming_llm__0.7500.jsonl"
            ),
            "Number of boxes = Eggs left / Eggs per box = 9 / 3 = 3 boxes",
            "compressed excerpt",
        ),
    ),
)
def test_excerpt_must_be_an_exact_artifact_line(
    artifact_root: Path,
    relative_path: Path,
    excerpt: str,
    message: str,
) -> None:
    path = artifact_root / relative_path
    path.write_text(path.read_text().replace(excerpt, f"prefix {excerpt}"))

    with pytest.raises(ValueError, match=message):
        build_evidence(artifact_root)


def test_selected_hybrid_row_must_exist(artifact_root: Path) -> None:
    path = artifact_root / ARTIFACT_PATHS[-1]
    rows = read_jsonl(path)
    rows = [
        row
        for row in rows
        if not (row["prompt_id"] == "gsm8k-0" and row["s"] == 128)
    ]
    write_jsonl(path, rows)

    with pytest.raises(ValueError, match="exactly one.*found 0"):
        build_evidence(artifact_root)


def test_selected_hybrid_row_must_be_unique(artifact_root: Path) -> None:
    path = artifact_root / ARTIFACT_PATHS[-1]
    rows = read_jsonl(path)
    selected = next(
        row
        for row in rows
        if row["prompt_id"] == "gsm8k-0" and row["s"] == 128
    )
    rows.append(selected)
    write_jsonl(path, rows)

    with pytest.raises(ValueError, match="exactly one.*found 2"):
        build_evidence(artifact_root)


def test_reference_requires_gsm8k_answer_delimiter(
    artifact_root: Path,
) -> None:
    path = artifact_root / ARTIFACT_PATHS[2]
    reference = json.loads(path.read_text())
    reference["text"] = reference["text"].replace("#### 18", "18")
    path.write_text(json.dumps(reference))

    with pytest.raises(ValueError, match="####.*delimiter"):
        build_evidence(artifact_root)


def test_reference_requires_numeric_gsm8k_answer(
    artifact_root: Path,
) -> None:
    path = artifact_root / ARTIFACT_PATHS[2]
    reference = json.loads(path.read_text())
    reference["text"] = reference["text"].replace("#### 18", "#### eighteen")
    path.write_text(json.dumps(reference))

    with pytest.raises(ValueError, match="answer.*numeric"):
        build_evidence(artifact_root)


def test_baseline_prompt_ids_must_be_unique(artifact_root: Path) -> None:
    path = artifact_root / ARTIFACT_PATHS[1]
    rows = read_jsonl(path)
    rows[-1]["prompt_id"] = rows[0]["prompt_id"]
    write_jsonl(path, rows)

    with pytest.raises(ValueError, match="duplicate baseline prompt_id"):
        build_evidence(artifact_root)


def test_campaign_requires_exact_compressor_membership(
    artifact_root: Path,
) -> None:
    path = artifact_root / ARTIFACT_PATHS[0]
    rows = read_jsonl(path)
    rows[0]["compressor"] = "unexpected"
    write_jsonl(path, rows)

    with pytest.raises(ValueError, match="compressors.*unexpected"):
        build_evidence(artifact_root)


def test_campaign_requires_184_episodes_per_compressor(
    artifact_root: Path,
) -> None:
    path = artifact_root / ARTIFACT_PATHS[0]
    rows = read_jsonl(path)
    rows.pop(0)
    write_jsonl(path, rows)

    with pytest.raises(ValueError, match="expected_attention.*184.*183"):
        build_evidence(artifact_root)


def test_campaign_requires_exact_ratio_set(artifact_root: Path) -> None:
    path = artifact_root / ARTIFACT_PATHS[0]
    rows = read_jsonl(path)
    rows[0]["ratio"] = 0.33
    write_jsonl(path, rows)

    with pytest.raises(ValueError, match="ratios"):
        build_evidence(artifact_root)
