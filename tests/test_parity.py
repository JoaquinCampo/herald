"""Contract tests for intervention parity evidence."""

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import pytest
import torch

import herald.generate as G
import herald.live_controller as LC
from herald.generate import LoadedModel, load_model
from herald.parity import run_parity_case
from herald.tasks import PromptRecord

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

import run_parity  # type: ignore[import-not-found, unused-ignore]  # noqa: E402

TINY = "hf-internal-testing/tiny-random-LlamaForCausalLM"
LONG = " ".join(["alpha beta gamma delta epsilon zeta"] * 16)


@dataclass
class _Layer:
    keys: torch.Tensor
    values: torch.Tensor


@dataclass
class _Cache:
    layers: list[_Layer]


def _rec(content: str = LONG, pid: str = "p0") -> PromptRecord:
    return PromptRecord(
        task="gsm8k",
        prompt_id=pid,
        messages=[{"role": "user", "content": content}],
        gold={"answer": "0"},
    )


def test_boundary_uses_pending_first_post_boundary_token() -> None:
    assert LC.split_switch_boundary([10, 11, 12], 2) == ([10, 11], 12)
    with pytest.raises(ValueError):
        LC.split_switch_boundary([10], 1)


def test_cache_clone_has_independent_storage_and_mutation_isolation() -> None:
    source = _Cache([_Layer(torch.ones(1, 1, 3), torch.zeros(1, 1, 3))])
    cloned = LC.clone_cache(source)
    assert LC.cache_storage_independent(source, cloned)
    assert LC.check_cache_mutation_isolation(source, cloned)
    assert torch.equal(source.layers[0].keys, torch.ones(1, 1, 3))


@pytest.fixture(scope="module")
def lm() -> LoadedModel:
    return load_model(
        "llama",
        dtype="float32",
        device="cpu",
        attn_implementation="sdpa",
        model_id=TINY,
    )


@pytest.fixture(autouse=True)
def _direct_tokenize(monkeypatch: pytest.MonkeyPatch) -> None:
    def build(lm: LoadedModel, record: PromptRecord) -> torch.Tensor:
        ids = lm.tokenizer(
            record.messages[-1]["content"], return_tensors="pt"
        ).input_ids
        return cast(torch.Tensor, ids)[0]

    monkeypatch.setattr(G, "build_input_ids", build)
    monkeypatch.setattr(LC, "build_input_ids", build)


@pytest.mark.parametrize("compressor", ["streaming_llm", "knorm"])
@pytest.mark.parametrize("s", [0, 4])
def test_cache_native_direct_and_reprefill_parity(
    lm: LoadedModel, compressor: str, s: int
) -> None:
    result = run_parity_case(
        lm,
        _rec(),
        compressor=compressor,
        s=s,
        ratio=0.5,
        continuation_budget=8,
    )
    assert result.complete and result.passed
    assert result.checks["sham_no_press"].exact_match
    assert result.checks["compressed_live_fork"].exact_match


@pytest.mark.parametrize("s", [0, 4])
def test_expected_attention_live_parity_is_not_applicable(
    lm: LoadedModel, s: int
) -> None:
    result = run_parity_case(
        lm,
        _rec(pid="expected"),
        compressor="expected_attention",
        s=s,
        ratio=0.5,
        continuation_budget=8,
    )
    assert result.complete
    check = result.checks["compressed_live_fork"]
    assert check.status == "not_applicable"
    assert check.exact_match is None
    assert "not applicable" in (check.reason or "")


def test_report_json_shape_is_stable() -> None:
    from herald.parity import CacheIsolation, ParityCheck, ParityResult

    result = ParityResult(
        schema_version="herald.parity.v1",
        model_key="llama",
        model_id="tiny",
        task="gsm8k",
        prompt_id="p0",
        compressor="streaming_llm",
        intervention_classification="cache_native_live_fork",
        ratio=0.5,
        s=0,
        continuation_budget=1,
        reference_token_ids=[1],
        reference_token_sha256="a",
        checks={"x": ParityCheck("passed", True, None, [1], [1], "a", "a")},
        quality_scores=None,
        quality_deltas=None,
        cache_isolation=CacheIsolation(
            source_cache_available=True,
            clone_storage_independent=True,
            source_unchanged_after_probe_mutation=True,
            compressed_fork_storage_independent=True,
            source_unchanged_after_compression=True,
            mutation_checked=True,
        ),
        complete=True,
        passed=True,
        failure_reasons=[],
        provenance={},
    )
    assert result.to_dict() == result.to_dict()
    assert result.to_json() == result.to_json()


def test_cli_argument_parsing_without_loading_model() -> None:
    args = run_parity.build_parser().parse_args(
        [
            "--model",
            "llama",
            "--task",
            "gsm8k",
            "--prompt-ids",
            "gsm8k-3",
            "--compressors",
            "streaming_llm,knorm",
            "--ratios",
            "0.25,0.5",
            "--switch-positions",
            "0,4",
            "--continuation-budget",
            "8",
        ]
    )
    assert args.prompt_ids == ["gsm8k-3"]
    assert args.compressors == ["streaming_llm", "knorm"]
    assert args.ratios == [0.25, 0.5]
    assert args.switch_positions == [0, 4]


@pytest.mark.parametrize(
    ("flag", "value"),
    [("--ratios", "nan"), ("--ratios", "0"), ("--switch-positions", "-1")],
)
def test_cli_rejects_invalid_numeric_inputs(flag: str, value: str) -> None:
    with pytest.raises(SystemExit):
        run_parity.build_parser().parse_args([flag, value])


def test_cli_rejects_switch_outside_budget_before_loading_model() -> None:
    args = run_parity.build_parser().parse_args(
        ["--switch-positions", "8", "--continuation-budget", "8"]
    )
    with pytest.raises(ValueError, match="switch positions"):
        run_parity.run_cli(args)
