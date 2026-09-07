from pathlib import Path
from typing import cast

import pytest
import torch
from torch import nn

import herald_v3.retrieval_discovery as discovery
from herald_v3.retrieval_discovery import (
    DecodeResult,
    DiscoveryRuntime,
    RetrievalCase,
    RetrievalDiscoveryError,
    _decode,
    _select_cells,
    attention_copy_hits,
    build_cases,
    exact_answer_match,
    load_triples,
    query_to_kv_mapping,
    run_discovery,
    set_attention_backend,
    validate_sentinel_ids,
    verify_model_gqa_mapping,
    verify_source_manifest,
)


class CharacterTokenizer:
    """Small reversible tokenizer for the deterministic CPU assay tests."""

    def __call__(
        self,
        text: str,
        *,
        add_special_tokens: bool = False,
        return_offsets_mapping: bool = False,
        **_: object,
    ) -> dict[str, object]:
        del add_special_tokens
        result: dict[str, object] = {
            "input_ids": [ord(character) + 1 for character in text]
        }
        if return_offsets_mapping:
            result["offset_mapping"] = [
                (index, index + 1) for index in range(len(text))
            ]
        return result

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool = False,
        add_generation_prompt: bool = False,
        return_tensors: str | None = None,
    ) -> object:
        content = messages[0]["content"]
        rendered = f"<user>\n{content}\n</user>"
        if add_generation_prompt:
            rendered += "\n<assistant>\n"
        if not tokenize:
            return rendered
        encoded = self(rendered, add_special_tokens=False)
        ids = encoded["input_ids"]
        if return_tensors == "pt":
            return {"input_ids": torch.tensor([ids])}
        return {"input_ids": ids}

    def decode(
        self,
        token_ids: list[int],
        *,
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: bool = False,
    ) -> str:
        del skip_special_tokens, clean_up_tokenization_spaces
        return "".join(chr((token_id - 1) % 128) for token_id in token_ids)


class ExactGQAModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(1))
        self.config = type(
            "Config",
            (),
            {
                "model_type": "qwen2",
                "hidden_size": 3584,
                "num_hidden_layers": 28,
                "num_attention_heads": 28,
                "num_key_value_heads": 4,
                "_attn_implementation": "eager",
            },
        )()
        self.model = type("Backbone", (), {})()
        self.model.layers = [
            type(
                "Layer",
                (),
                {
                    "self_attn": type(
                        "Attention",
                        (),
                        {
                            "num_heads": 28,
                            "num_key_value_heads": 4,
                            "num_key_value_groups": 7,
                        },
                    )()
                },
            )()
            for _ in range(28)
        ]
        self.backend_calls: list[str] = []

    def set_attn_implementation(self, backend: str) -> None:
        self.config._attn_implementation = backend
        self.backend_calls.append(backend)


def _stub_decode_result() -> DecodeResult:
    return DecodeResult(
        generated_token_ids=(1,),
        stop_reason="eos",
        success=True,
        answer_match_count=1,
        head_copy_fraction={"0:0": 1.0},
        elapsed_seconds=0.01,
        prefill_seconds=0.002,
        decode_forward_seconds=0.003,
        peak_allocated_bytes=None,
        peak_reserved_bytes=None,
    )


@pytest.fixture
def source_root() -> Path:
    return (
        Path(__file__).resolve().parents[1]
        / "data"
        / "retrieval-discovery-v1"
        / "source"
    )


def test_source_manifest_and_case_grid_are_frozen(source_root: Path) -> None:
    manifest = verify_source_manifest(source_root)
    triples = load_triples(source_root)
    cases = build_cases(CharacterTokenizer(), source_root)

    assert manifest["commit"] == "3ac171a6f71ce7ef1cda57d4215c390fb6ab51f2"
    assert len(triples) == 3
    assert len(cases) == 24
    assert {case.panel for case in cases} == {"discovery", "validation"}
    assert all(len(case.prompt_ids) > case.context_length for case in cases)
    assert all(
        _decode(CharacterTokenizer(), case.answer_token_ids)
        == triples[case.triple_index].answer
        for case in cases
    )
    tokenizer = CharacterTokenizer()
    for case in cases:
        rendered = _decode(tokenizer, case.prompt_ids)
        answer = triples[case.triple_index].answer
        answer_at = rendered.find(answer)
        assert answer_at >= 1
        assert rendered[answer_at - 1] == "\n"


def test_attention_copy_uses_prompt_token_at_generation_step() -> None:
    prompt_ids = (10, 20, 30, 40)
    attention = torch.zeros((1, 2, 1, 5))
    attention[0, 0, 0, 1] = 1.0
    attention[0, 1, 0, 3] = 1.0

    hits = attention_copy_hits(
        (attention,), prompt_ids, 1, 3, generated_token_id=20
    )

    assert hits == {"0:0": True, "0:1": False}


def test_gqa_mapping_verifies_contiguous_repeat_groups() -> None:
    assert query_to_kv_mapping(28, 4) == (
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        3,
        3,
        3,
        3,
        3,
        3,
        3,
    )


def test_selection_includes_threshold_boundary_and_deduplicates_kv_head() -> (
    None
):
    mapping = query_to_kv_mapping(28, 4)
    discovery_means = {f"0:{query}": 0.1 for query in range(7)}
    validation_means = {f"0:{query}": 0.1 for query in range(7)}

    selected = _select_cells(
        {"discovery": discovery_means, "validation": validation_means},
        mapping,
    )

    assert selected == [{"layer": 0, "kv_head": 0, "stability_score": 0.1}]


def test_noncopy_does_not_count_as_success() -> None:
    assert exact_answer_match((9, 8, 7), (1, 2)) == (False, 0)
    with pytest.raises(RetrievalDiscoveryError, match="empty"):
        exact_answer_match((1,), ())


def test_sentinel_mismatch_stops_assay() -> None:
    with pytest.raises(RetrievalDiscoveryError, match="sentinel"):
        validate_sentinel_ids((1, 2), (1, 3))


def test_run_reuses_eager_sentinels_and_caps_call_count(
    source_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cases = build_cases(CharacterTokenizer(), source_root)
    calls: list[bool] = []

    def fake_decode(
        model: object,
        tokenizer: object,
        case: object,
        *,
        collect_attention: bool,
        runtime: DiscoveryRuntime,
    ) -> DecodeResult:
        del model, tokenizer, case, runtime
        calls.append(collect_attention)
        return _stub_decode_result()

    monkeypatch.setattr(discovery, "decode_case", fake_decode)
    result = run_discovery(
        ExactGQAModel(),
        CharacterTokenizer(),
        cases,
        runtime=DiscoveryRuntime(require_cuda=False),
    )

    assert result["status"] == "completed"
    assert calls.count(True) == 24
    assert calls.count(False) == 4
    timing = cast(dict[str, object], result["timing"])
    assert timing["completed_continuations"] == 28
    assert len(cast(list[object], result["sentinels"])) == 4


def test_deadline_returns_partial_operational_record(
    source_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cases = build_cases(CharacterTokenizer(), source_root)
    calls = 0

    def fake_decode(
        model: object,
        tokenizer: object,
        case: object,
        *,
        collect_attention: bool,
        runtime: DiscoveryRuntime,
    ) -> DecodeResult:
        del model, tokenizer, case, collect_attention
        nonlocal calls
        calls += 1
        runtime._deadline = runtime.clock() - 1
        return _stub_decode_result()

    monkeypatch.setattr(discovery, "decode_case", fake_decode)
    result = run_discovery(
        ExactGQAModel(),
        CharacterTokenizer(),
        cases,
        runtime=DiscoveryRuntime(require_cuda=False),
    )

    assert result["status"] == "operational_stop"
    assert calls == 1
    timing = cast(dict[str, object], result["timing"])
    assert timing["completed_continuations"] == 1
    assert result["completed_eager_sentinel_ids"]


def test_tiny_qwen2_cpu_loop_matches_eager_and_sdpa() -> None:
    transformers = pytest.importorskip("transformers")
    config = transformers.Qwen2Config(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=128,
    )
    model = transformers.Qwen2ForCausalLM(config).eval()
    tokenizer = CharacterTokenizer()
    case = RetrievalCase(
        case_id="tiny",
        panel="discovery",
        triple_index=0,
        source_part="part1",
        original_needle="needle",
        transformed_needle="needle",
        answer="answer",
        context_length=4,
        depth_percent=20.0,
        prompt_ids=(1, 2, 3, 4),
        answer_token_ids=(1,),
        answer_start=0,
        answer_end=1,
        rendered_prompt_sha256="0" * 64,
    )
    runtime = DiscoveryRuntime(require_cuda=False, deadline_seconds=30.0)
    set_attention_backend(model, "eager")
    eager = discovery.decode_case(
        model, tokenizer, case, collect_attention=True, runtime=runtime
    )
    set_attention_backend(model, "sdpa")
    sdpa = discovery.decode_case(
        model, tokenizer, case, collect_attention=False, runtime=runtime
    )

    assert eager.generated_token_ids == sdpa.generated_token_ids
    assert len(eager.generated_token_ids) <= 50


def test_real_qwen2_gqa_contract_and_rejects_mismatches() -> None:
    transformers = pytest.importorskip("transformers")
    config = transformers.Qwen2Config(
        vocab_size=64,
        hidden_size=3584,
        intermediate_size=64,
        num_hidden_layers=28,
        num_attention_heads=28,
        num_key_value_heads=4,
        max_position_embeddings=128,
    )
    with torch.device("meta"):
        attention = transformers.models.qwen2.modeling_qwen2.Qwen2Attention(
            config, layer_idx=0
        )
    model = type("Model", (), {})()
    model.config = config
    model.model = type("Backbone", (), {})()
    model.model.layers = [
        type("Layer", (), {"self_attn": attention})() for _ in range(28)
    ]

    assert not hasattr(attention, "num_heads")
    assert not hasattr(attention, "num_key_value_heads")
    assert verify_model_gqa_mapping(model) == tuple(
        query // 7 for query in range(28)
    )

    attention.num_key_value_groups = 8
    with pytest.raises(RetrievalDiscoveryError, match="28/4/7"):
        verify_model_gqa_mapping(model)
    attention.num_key_value_groups = 7

    config.num_attention_heads = 27
    with pytest.raises(RetrievalDiscoveryError, match="28 query heads"):
        verify_model_gqa_mapping(model)


def test_real_pinned_tokenizer_case_builder_when_available() -> None:
    tokenizer_path = __import__("os").environ.get("HERALD_QWEN_TOKENIZER")
    if tokenizer_path is None:
        pytest.skip("set HERALD_QWEN_TOKENIZER to run the pinned tokenizer")
    transformers = pytest.importorskip("transformers")
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        tokenizer_path, local_files_only=True
    )
    cases = build_cases(tokenizer)
    assert len(cases) == 24
    assert all(case.answer_start < case.answer_end for case in cases)
