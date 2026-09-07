import json
import sys
import types
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import check_shadow_attention as shadow_cli  # noqa: E402

from herald_v3.shadow_attention import (
    ShadowAttentionCollector,
    ShadowAttentionError,
    ShadowRuntime,
    first_divergence,
    run_paired_continuations,
)


class FakeCache:
    def __init__(self, key: torch.Tensor, value: torch.Tensor) -> None:
        self.layers = [SimpleNamespace(keys=key, values=value)]

    def get_seq_length(self) -> int:
        return int(self.layers[0].keys.shape[-2])


class FakeAttention:
    num_key_value_groups = 2
    layer_idx = 0
    training = False


class TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(1))
        self.attention = FakeAttention()
        self.config = SimpleNamespace(_attn_implementation="sdpa")

    def forward(self, input_ids: torch.Tensor, **kwargs: object) -> object:
        query_length = int(input_ids.shape[1])
        previous = kwargs.get("past_key_values")
        if isinstance(previous, FakeCache):
            key_length = previous.get_seq_length() + 1
        else:
            key_length = query_length
        query = torch.ones((1, 4, query_length, 2))
        key = torch.ones((1, 2, key_length, 2))
        value = torch.ones_like(key)
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

        attention = ALL_ATTENTION_FUNCTIONS["sdpa"]
        attention(
            self.attention,
            query,
            key,
            value,
            None,
            dropout=0.0,
            scaling=2.0**-0.5,
            position_ids=torch.tensor([[key_length - 1]]),
        )
        logits = torch.zeros((1, query_length, 10))
        logits[:, :, 5] = 1.0
        return SimpleNamespace(
            logits=logits,
            past_key_values=FakeCache(key, value),
        )


class TinyTokenizer:
    eos_token_id = None

    def decode(
        self,
        token_ids: list[int] | tuple[int, ...],
        *,
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: bool = False,
    ) -> str:
        del skip_special_tokens, clean_up_tokenization_spaces
        return "".join("x" for _ in token_ids)


def _install_registry(monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    registry: dict[str, object] = {}

    def native(
        module: object,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: torch.Tensor | None,
        *args: object,
        **kwargs: object,
    ) -> object:
        del module, attention_mask, args, kwargs
        if key.shape[1] != query.shape[1]:
            groups = query.shape[1] // key.shape[1]
            key = torch.repeat_interleave(key, groups, dim=1)
            value = torch.repeat_interleave(value, groups, dim=1)
        return (torch.matmul(query, key.transpose(-2, -1)), value)

    registry["sdpa"] = native

    class AttentionInterface:
        @classmethod
        def register(cls, key: str, value: object) -> None:
            del cls
            registry[key] = value

    transformers = cast(Any, types.ModuleType("transformers"))
    transformers.AttentionInterface = AttentionInterface
    modeling_utils = cast(
        Any, types.ModuleType("transformers.modeling_utils")
    )
    modeling_utils.ALL_ATTENTION_FUNCTIONS = registry
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    monkeypatch.setitem(
        sys.modules, "transformers.modeling_utils", modeling_utils
    )
    return registry


def test_wrapper_preserves_native_result_and_matches_hf_mask_crop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = _install_registry(monkeypatch)
    native = registry["sdpa"]
    native_result = object()

    def native_with_sentinel(*args: object, **kwargs: object) -> object:
        del args, kwargs
        return native_result

    registry["sdpa"] = native_with_sentinel
    native = native_with_sentinel
    collector = ShadowAttentionCollector(
        witness_limit=1,
    )
    module = FakeAttention()
    query = torch.tensor(
        [[[[1.0, 0.0]], [[0.0, 1.0]], [[1.0, 0.0]], [[0.0, 1.0]]]]
    )
    key = torch.tensor(
        [
            [
                [[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]],
                [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
            ]
        ]
    )
    value = torch.zeros_like(key)
    mask = torch.tensor([[False, True, False, True, True]])
    with collector:
        collector.begin_step(0)
        wrapped = cast(Callable[..., object], registry["sdpa"])
        result = wrapped(module, query, key, value, mask, scaling=1.0)
        collector.finish_step(9)
    assert result is native_result
    assert registry["sdpa"] is native
    evidence = collector.to_dict()
    assert evidence["native_call_count"] == 1
    assert evidence["record_count"] == 1
    records = cast(list[object], evidence["records"])
    record = cast(dict[str, object], records[0])
    assert record["mask_kind"] == "boolean_allowed"
    assert record["key_length"] == 4
    assert record["top_indices"] == [1, 1, 1, 1]
    assert "hit" not in record
    assert record["index_checksum"]
    assert record["normalized_row_checksum"]
    assert evidence["attention_registry_restored"] is True


def test_additive_mask_is_added_to_scores_and_rows_are_normalized(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = _install_registry(monkeypatch)
    collector = ShadowAttentionCollector(witness_limit=0)
    query = torch.ones((1, 2, 1, 2))
    key = torch.ones((1, 1, 3, 2))
    value = torch.ones_like(key)
    additive = torch.tensor([[[[0.0, float("-inf"), 0.0]]]])
    with collector:
        collector.begin_step(0)
        native = cast(Callable[..., object], registry["sdpa"])
        native(FakeAttention(), query, key, value, additive)
        collector.finish_step(5)
    records = cast(list[object], collector.to_dict()["records"])
    record = cast(dict[str, object], records[0])
    assert record["mask_kind"] == "additive"
    assert record["normalized_attention_row_sums"] == pytest.approx(
        [1.0, 1.0]
    )
    assert record["top_indices"] == [0, 0]


def test_first_divergence_preserves_length_mismatch() -> None:
    assert first_divergence((1, 2, 3), (1, 9, 3)) == {
        "generated_index": 1,
        "unwrapped_token_id": 2,
        "wrapped_token_id": 9,
    }
    assert first_divergence((1,), (1, 2)) == {
        "generated_index": 1,
        "unwrapped_token_id": None,
        "wrapped_token_id": 2,
    }
    assert first_divergence((1,), (1,)) is None


def _shadow_contract_fixture() -> dict[str, object]:
    return {
        "seed": shadow_cli.SEED,
        "attention_backend": "sdpa",
        "greedy": {
            "do_sample": False,
            "max_new_tokens": shadow_cli.MAX_NEW_TOKENS,
        },
        "runtime": shadow_cli._shadow_runtime_contract(),
        "model": {
            "architecture": dict(shadow_cli.EXPECTED_ARCHITECTURE),
        },
        "protocol": shadow_cli._shadow_protocol(),
    }


def test_tampered_backend_and_protocol_metadata_are_rejected() -> None:
    seal = _shadow_contract_fixture()
    seal["attention_backend"] = "eager"
    with pytest.raises(shadow_cli.CliIntegrityError):
        shadow_cli._verify_shadow_contract_metadata(seal)

    seal = _shadow_contract_fixture()
    runtime = cast(dict[str, object], seal["runtime"])
    runtime["attention_backends"] = ["eager", "sdpa"]
    with pytest.raises(shadow_cli.CliIntegrityError):
        shadow_cli._verify_shadow_contract_metadata(seal)

    seal = _shadow_contract_fixture()
    protocol = cast(dict[str, object], seal["protocol"])
    protocol["head_selection"] = True
    with pytest.raises(shadow_cli.CliIntegrityError):
        shadow_cli._verify_shadow_contract_metadata(seal)


def test_failure_payload_retains_bounded_diagnostics(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    output = run_dir / "result.json"
    diagnostics = {
        "native_call_count": 2,
        "attention_registry_restored": True,
        "witnesses": [{"step_index": 0, "top_index": 3}],
    }
    failure = shadow_cli._preserve_failure(
        output,
        run_dir,
        {
            "case_id": "fixture",
            "case_prompt_token_ids": [1, 2],
            "unwrapped_token_ids": [5],
            "wrapped_token_ids": [],
            "first_divergence": {
                "generated_index": 0,
                "unwrapped_token_id": 5,
                "wrapped_token_id": None,
            },
            "diagnostics": diagnostics,
        },
        ShadowAttentionError("fixture failure"),
    )
    saved = json.loads(output.read_text(encoding="utf-8"))
    assert failure["diagnostics"] == diagnostics
    assert saved["diagnostics"] == diagnostics
    assert saved["first_divergence"]["generated_index"] == 0
    assert (run_dir / "shadow-integrity.json").exists()


def test_paired_continuations_keep_exact_tokens_cache_model_and_rng_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_registry(monkeypatch)
    case = SimpleNamespace(
        case_id="fixture",
        prompt_ids=(1, 2, 3, 4),
    )
    result = run_paired_continuations(
        TinyModel(),
        TinyTokenizer(),
        case,
        runtime=ShadowRuntime(require_cuda=False, deadline_seconds=10.0),
        witness_limit=1,
    )
    assert result.passed is True
    assert result.parity["tokens_exact"] is True
    assert result.parity["cache_fingerprint_equal"] is True
    assert result.parity["model_state_equal"] is True
    assert result.parity["rng_state_equal"] is True
    assert result.diagnostics["native_call_count"] == 51
    assert result.diagnostics["prefill_call_count"] == 1
    assert result.diagnostics["decode_call_count"] == 50
    assert result.diagnostics["attention_registry_restored"] is True
    serialized = result.to_dict()
    for branch in ("unwrapped", "wrapped"):
        branch_payload = cast(dict[str, object], serialized[branch])
        assert "success" not in branch_payload
        assert "answer_match_count" not in branch_payload


def test_failed_wrapped_branch_preserves_partial_tokens_and_restores_registry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = _install_registry(monkeypatch)
    native = registry["sdpa"]
    case = SimpleNamespace(
        case_id="fixture",
        prompt_ids=(1, 2, 3, 4),
    )
    state: dict[str, object] = {}
    with pytest.raises(ShadowAttentionError, match="GQA dimensions"):
        run_paired_continuations(
            TinyModel(),
            TinyTokenizer(),
            case,
            runtime=ShadowRuntime(
                require_cuda=False,
                deadline_seconds=10.0,
            ),
            strict_contract=True,
            state=state,
        )
    assert registry["sdpa"] is native
    assert state["unwrapped_token_ids"] == [5] * 50
    assert state["wrapped_token_ids"] == []
    assert state["first_divergence"] == {
        "generated_index": 0,
        "unwrapped_token_id": 5,
        "wrapped_token_id": None,
    }


def test_tiny_real_qwen_sdpa_prefill_passthrough_and_decode_shadow() -> None:
    transformers = pytest.importorskip("transformers")
    from transformers import Qwen2Config, Qwen2ForCausalLM

    config_type = cast(Any, Qwen2Config)
    model_type = cast(Any, Qwen2ForCausalLM)
    config = config_type(
        vocab_size=32,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=32,
        _attn_implementation="sdpa",
    )
    model = model_type(config).eval()
    prompt = torch.tensor([[1, 2, 3, 4]])
    mask = torch.ones_like(prompt)
    with torch.no_grad():
        native_prefill = model(
            input_ids=prompt,
            attention_mask=mask,
            use_cache=True,
            return_dict=True,
        )
    native_cache = native_prefill.past_key_values
    collector = ShadowAttentionCollector(
        witness_limit=1,
    )
    with collector, torch.no_grad():
        wrapped_prefill = model(
            input_ids=prompt,
            attention_mask=mask,
            use_cache=True,
            return_dict=True,
        )
        wrapped_prefill_key = wrapped_prefill.past_key_values.layers[
            0
        ].keys.clone()
        collector.begin_step(0)
        wrapped_decode = model(
            input_ids=torch.tensor([[4]]),
            attention_mask=torch.ones((1, 5), dtype=torch.long),
            position_ids=torch.tensor([[4]]),
            cache_position=torch.tensor([4]),
            past_key_values=wrapped_prefill.past_key_values,
            use_cache=True,
            return_dict=True,
        )
        collector.finish_step(
            int(wrapped_decode.logits[:, -1, :].argmax(dim=-1).item())
        )
    assert torch.equal(native_prefill.logits, wrapped_prefill.logits)
    assert torch.equal(
        native_cache.layers[0].keys,
        wrapped_prefill_key,
    )
    assert collector.mapping_restored is True
    assert collector.prefill_call_count == 2
    assert collector.decode_call_count == 2
    assert collector.to_dict()["record_count"] == 2
    del transformers
