import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

runner = pytest.importorskip("run_pair_pilot")
torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
tokenizers = pytest.importorskip("tokenizers")


def make_tiny_qwen(path):
    vocab = {
        "<unk>": 0,
        "<pad>": 1,
        "<eos>": 2,
        "alpha": 3,
        "beta": 4,
        "gamma": 5,
        "delta": 6,
    }
    tokenizer_backend = tokenizers.Tokenizer(
        tokenizers.models.WordLevel(vocab=vocab, unk_token="<unk>")
    )
    tokenizer_backend.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
    tokenizer = transformers.PreTrainedTokenizerFast(
        tokenizer_object=tokenizer_backend,
        unk_token="<unk>",
        pad_token="<pad>",
        eos_token="<eos>",
    )
    tokenizer.chat_template = (
        "{% for message in messages %}{{ message['content'] }}{% endfor %}"
        "{% if add_generation_prompt %}{{ '' }}{% endif %}"
    )
    tokenizer.save_pretrained(path)
    config = transformers.Qwen2Config(
        vocab_size=len(vocab),
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
        bos_token_id=0,
        eos_token_id=2,
        pad_token_id=1,
    )
    model = transformers.Qwen2ForCausalLM(config)
    model.save_pretrained(path, safe_serialization=False)


def test_tiny_qwen_cpu_last_prompt_pair(tmp_path):
    model_path = tmp_path / "tiny-qwen"
    model_path.mkdir()
    make_tiny_qwen(model_path)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            [
                {
                    "id": "tiny-1",
                    "prompt": "alpha beta gamma delta",
                    "answers": ["delta"],
                    "task": "tiny",
                    "max_new_tokens": 4,
                    "token_counts": {"qwen_chat_prompt_tokens": 4},
                }
            ]
        ),
        encoding="utf-8",
    )
    output = tmp_path / "output"
    result = runner.main(
        [
            "--manifest",
            str(manifest),
            "--model",
            str(model_path),
            "--engine-root",
            str(ROOT.parent / "herald-v3" / "src"),
            "--output-dir",
            str(output),
            "--device",
            "cpu",
            "--dtype",
            "float32",
            "--actions",
            "0",
            ".5",
        ]
    )
    assert result == 0
    run = json.loads((output / "run.json").read_text(encoding="utf-8"))
    assert run["status"] == "completed"
    record = json.loads((output / "0000-tiny-1.json").read_text(encoding="utf-8"))
    assert record["status"] == "completed"
    assert record["checks"]["noop_matches_reference"] is True
    assert record["checks"]["noop_source_unchanged"] is True
    assert record["boundary"]["clone_controls"]["equal_to_source"] is True
    assert record["boundary"]["clone_controls"]["source_a_disjoint"] is True
    assert record["boundary"]["prefill_clone_controls"] == {
        "equal": True,
        "disjoint": True,
    }
    assert record["arms"]["knorm:0"]["compression"]["physical_effect_exact"] is True
    assert record["arms"]["knorm:0.5"]["compression"]["physical_effect_exact"] is True
