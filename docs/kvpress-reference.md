# kvpress Library Reference

Comprehensive reference for the `kvpress` Python library (NVIDIA) for KV-cache compression in LLMs.

- **Repository**: https://github.com/NVIDIA/kvpress
- **PyPI**: https://pypi.org/project/kvpress/
- **Latest version**: 0.5.1 (Feb 16, 2026)
- **License**: Apache 2.0
- **Authors**: Simon Jegou, Maximilian Jeblick, Alessio Devoto, David Austin
- **Paper**: https://arxiv.org/abs/2510.00636v1
- **Python**: >=3.10
- **Transformers**: v5+ (since kvpress 0.5.0)

---

## 1. Architecture Overview

kvpress is a flat Python package (`kvpress/`) with a `presses/` subdirectory containing all compression methods. Key modules:

```
kvpress/
  __init__.py              # Exports all presses, calls patch_attention_functions()
  attention_patch.py       # Patches ALL_ATTENTION_FUNCTIONS for head-wise compression
  pipeline.py              # KVPressTextGenerationPipeline (HF pipeline integration)
  utils.py                 # Helpers: extract_keys_and_values, get_prerope_query_states, etc.
  presses/
    base_press.py          # BasePress (dataclass, context manager, hook registration)
    scorer_press.py        # ScorerPress (score-based pruning base class)
    ... (30+ press implementations)
```

---

## 2. Class Hierarchy

```
BasePress (dataclass)
  |
  +-- ScorerPress (score-based, has compression_ratio)
  |     +-- RandomPress
  |     +-- KnormPress
  |     +-- SnapKVPress
  |     |     +-- PyramidKVPress
  |     +-- ExpectedAttentionPress
  |     |     +-- ExpectedAttentionStatsPress
  |     +-- StreamingLLMPress
  |     +-- TOVAPress
  |     +-- ObservedAttentionPress
  |     +-- QFilterPress
  |     +-- KeyDiffPress
  |     +-- LagKVPress
  |     +-- CURPress
  |     +-- KVzapPress
  |     +-- LeverageScorePress
  |     +-- NonCausalAttnPress
  |     +-- CompactorPress
  |     +-- CriticalKVPress (wraps a ScorerPress)
  |
  +-- ThinKPress (dimension compression, not sequence)
  +-- SimLayerKVPress (lazy layer identification)
  +-- DuoAttentionPress (retrieval vs streaming heads)
  +-- FinchPress (prompt-guided, delimiter-based)
  +-- KVzipPress (context reconstruction, own __call__)
  +-- FastKVzipPress (learned gates, own __call__)
  +-- ComposedPress (chains multiple presses)
  +-- AdaKVPress (head-wise adaptive, wraps ScorerPress)
  +-- ChunkPress (chunk-wise uniform compression wrapper)
  +-- ChunkKVPress (semantic-preserving chunk selection)
  +-- BlockPress (block-wise iterative compression)
  +-- PerLayerCompressionPress (per-layer ratios)
  +-- KeyRerotationPress (RoPE-aware wrapper)
  +-- CriticalAdaKVPress (CriticalKV + AdaKV combined)
  +-- DecodingPress (compression during decoding phase)
  +-- PrefillDecodingPress (separate prefill + decoding strategies)
  +-- DMSPress (threshold-based adaptive compression)
```

---

## 3. compression_ratio Semantics

```python
compression_ratio: float = 0.0  # Range: [0.0, 1.0)
```

- **Meaning**: Fraction of KV pairs to **REMOVE** (not keep).
- `0.0` = no compression (keep everything)
- `0.5` = remove 50% of KV pairs (keep 50%)
- `0.9` = remove 90% of KV pairs (keep 10%)
- `1.0` = invalid (assertion fails)

**Internal calculation** (from `ScorerPress.compress`):
```python
n_kept = int(k_len * (1 - self.compression_ratio))
# Then topk(n_kept) on scores to select which to keep
```

**Herald mapping**: If Herald uses "fraction to keep" semantics, the conversion is:
```python
kvpress_ratio = 1.0 - herald_keep_fraction
# e.g., keep 12.5% of cache -> compression_ratio = 0.875
```

---

## 4. Context Manager API (`press(model)`)

### How `BasePress.__call__` works:

```python
@contextmanager
def __call__(self, model: PreTrainedModel) -> Generator:
    # 1. Validate model architecture
    if not isinstance(model, SUPPORTED_MODELS):
        logger.warning(f"Model {type(model)} not tested")

    # 2. Find all attention layers
    attention_layers = []
    for layer in model.model.layers:
        attention_layers.append(layer.self_attn)

    # 3. Call post_init_from_model (for presses that need model info)
    self.post_init_from_model(model)

    # 4. Register forward hooks on every attention layer
    hooks = []
    for layer in attention_layers:
        hooks.append(
            layer.register_forward_hook(self.forward_hook, with_kwargs=True)
        )

    # 5. Yield control to user code
    try:
        yield
    finally:
        # 6. Remove all hooks on exit
        for hook in hooks:
            hook.remove()
```

### What the hooks do (`forward_hook`):

Called **after** each attention layer's forward pass during **prefilling only** (when `q_len == k_len`). The hook:

1. Extracts `hidden_states`, `past_key_values` (cache), and `attentions` from kwargs/output
2. Calls `extract_keys_and_values(cache, layer_idx)` to get current K, V tensors
3. Calls `self.compress(module, hidden_states, keys, values, attentions, kwargs)` which returns pruned keys, values
4. Updates the cache in-place with compressed K, V
5. During **generation** (decoding), hooks are still registered but skip compression (`q_len != k_len`)

### Usage pattern:

```python
with torch.no_grad(), press(model):
    outputs = model.generate(
        input_ids,
        max_new_tokens=256,
        output_scores=True,           # Safe to use with kvpress
        return_dict_in_generate=True,  # Safe to use with kvpress
    )
```

### Special cases:
- **KVzipPress** and **FastKVzipPress** override `__call__` entirely with their own multi-pass logic
- **FinchPress** adds an additional hook on `model.model.embed_tokens` (for delimiter detection)
- **DecodingPress** adds a `reset()` call in the finally block

---

## 5. All Available Presses (Detailed)

### 5.1 Score-Based Presses (inherit from ScorerPress)

These all implement a `score()` method returning `(batch, num_kv_heads, seq_len)` tensor. Higher scores = more important = kept. Lowest-scored tokens are pruned.

#### RandomPress
- **Paper**: Baseline
- **Scoring**: Random scores (uniform)
- **Parameters**: `compression_ratio`, `seed: int | None`
- **Attention needed**: No
- **Use case**: Baseline comparison

#### KnormPress
- **Paper**: https://arxiv.org/abs/2406.11430
- **Scoring**: Negative L2 norm of key vectors (`-keys.norm(dim=-1)`)
- **Parameters**: `compression_ratio`
- **Attention needed**: No
- **Insight**: Keys with smaller norms are MORE important (negative scoring means high-norm keys are pruned)

#### SnapKVPress
- **Paper**: https://arxiv.org/abs/2404.14469
- **Scoring**: Attention patterns of last `window_size` tokens, with pooling kernel smoothing
- **Parameters**: `compression_ratio`, `window_size=64`, `kernel_size=5`
- **Attention needed**: Computes its own window attention (does NOT require `attn_implementation="eager"`)
- **Insight**: Recent tokens' attention patterns indicate which historical tokens are important

#### ExpectedAttentionPress
- **Paper**: https://arxiv.org/abs/2510.00636v1 (kvpress paper itself)
- **Scoring**: Statistical modeling of expected future attention using query mean/covariance + RoPE rotation
- **Parameters**: `compression_ratio`, `n_future_positions=512`, `n_sink=4`, `use_covariance=True`, `use_vnorm=True`, `epsilon=0.0`
- **Attention needed**: No (uses query statistics)
- **Insight**: Predicts what future queries will attend to, rescales by value norms

#### ExpectedAttentionStatsPress
- **Paper**: Same as ExpectedAttentionPress
- **Scoring**: Same as parent, but uses pre-computed query statistics (loaded from HuggingFace Hub)
- **Parameters**: Same as parent + `dataset_name="kmfoda/booksum"`, `num_samples=100`, `sample_seq_len=1000`
- **Requires**: Pre-computed stats available on HF Hub for the specific model

#### StreamingLLMPress
- **Paper**: https://arxiv.org/abs/2309.17453
- **Scoring**: Position-based: score=1 for sink tokens and recent tokens, score=0 for middle tokens
- **Parameters**: `compression_ratio`, `n_sink=4`
- **Attention needed**: No
- **Insight**: First few tokens are "attention sinks" + recent tokens matter; middle tokens discarded
- **Note**: For full paper match, wrap with `KeyRerotationPress`

#### TOVAPress
- **Paper**: https://arxiv.org/abs/2401.06104
- **Scoring**: Attention weight of the last token, averaged across heads
- **Parameters**: `compression_ratio`
- **Attention needed**: Uses `attentions` if available, otherwise computes window attention (window=1)
- **Insight**: Last token's attention pattern is a good global importance indicator

#### ObservedAttentionPress
- **Paper**: Related to H2O (https://arxiv.org/abs/2306.14048)
- **Scoring**: Average attention weight across all query positions from the prefill forward pass
- **Parameters**: `compression_ratio`
- **Attention needed**: YES, requires `attn_implementation="eager"` (will assert if not set)
- **Insight**: Uses the actual observed attention during prefilling as importance

#### QFilterPress
- **Paper**: https://arxiv.org/abs/2503.02812
- **Scoring**: Dot product between keys and learned filter vectors (loaded from HF Hub)
- **Parameters**: `compression_ratio`
- **Requires**: Pre-trained Q-filter parameters for the specific model (from `nthngdy/` on HF Hub)
- **Note**: Not all models have Q-filters available

#### KeyDiffPress
- **Paper**: https://arxiv.org/abs/2504.15364
- **Scoring**: Negative cosine similarity between key vectors and average key pattern
- **Parameters**: `compression_ratio`
- **Attention needed**: No
- **Insight**: Keeps tokens with distinctive key patterns, prunes those similar to average
- **Note**: For paper replication use `BlockPress(press=KeyDiffPress(...), block_size=N)`

#### LagKVPress
- **Paper**: https://arxiv.org/abs/2504.04704
- **Scoring**: Lag-relative information between sequence partitions (key + value based)
- **Parameters**: `compression_ratio`, `n_sink=4`, `lag_size=128`, `cross_scoring=False`
- **Attention needed**: No

#### CURPress
- **Paper**: https://arxiv.org/abs/2509.15038
- **Scoring**: Approximate leverage scores for keys (k2) and values (v2), combined as `k2 * v2` by default
- **Parameters**: `compression_ratio`, `num_sinks=4`, `leverage_type="kv_product"`, `use_random_leverage=False`, `use_local_approximation=True`, `local_window_size=16`
- **Attention needed**: No

#### KVzapPress
- **Paper**: https://arxiv.org/abs/2601.07891
- **Scoring**: Learned lightweight surrogate model (linear or MLP) applied to hidden states
- **Parameters**: `compression_ratio`, `model_type="mlp"` (or `"linear"`)
- **Requires**: Pre-trained KVzap model from `nvidia/KVzap-{type}-{model_name}` on HF Hub
- **Designed for**: Use with `DMSPress` for decoding-time compression

#### LeverageScorePress
- **Paper**: https://arxiv.org/pdf/2507.08143v1 (Compactor)
- **Scoring**: Approximate statistical leverage scores on pre-RoPE key embeddings via Gaussian sketch + Cholesky decomposition
- **Parameters**: `compression_ratio`, `sketch_dimension=48`
- **Attention needed**: No

#### NonCausalAttnPress
- **Paper**: https://arxiv.org/pdf/2507.08143v1 (Compactor)
- **Scoring**: Non-causal chunked attention column-sums, z-normalized
- **Parameters**: `compression_ratio`, `chunk_size=256`
- **Attention needed**: Computes its own non-causal attention

#### CompactorPress
- **Paper**: https://arxiv.org/pdf/2507.08143v1
- **Scoring**: Blends LeverageScorePress and NonCausalAttnPress scores
- **Parameters**: `compression_ratio`, `sink_size_start=8`, `sink_size_end=4`, `chunk_size=256`, `sketch_dimension=48`, `blending=None` (defaults to compression_ratio)
- **Attention needed**: Internally computes its own

### 5.2 Non-Score-Based Presses (inherit from BasePress directly)

#### ThinKPress
- **Paper**: https://arxiv.org/abs/2407.21018
- **Type**: **Dimension compression** (not sequence length)
- **Mechanism**: Zeros out least important key channels based on query attention scores
- **Parameters**: `key_channel_compression_ratio=0.0`, `window_size=32`
- **Note**: No memory savings currently (zeroed dims, same shape). Combinable with sequence presses via ComposedPress.

#### SimLayerKVPress
- **Paper**: https://arxiv.org/abs/2410.13846
- **Type**: Layer-adaptive
- **Mechanism**: Identifies "lazy" layers (attention concentrated on initial+recent tokens), applies StreamingLLM-style compression only to those layers
- **Parameters**: `lazy_threshold=1.0`, `n_last=1`, `n_recent=1024`, `n_initial=4`
- **Recommended thresholds**: Llama3=0.9, Llama2=0.65, Mistral=0.8, Qwen=0.85
- **Note**: `compression_ratio` is computed dynamically (read-only property)

#### DuoAttentionPress
- **Paper**: https://arxiv.org/abs/2410.10819
- **Type**: Head-adaptive
- **Mechanism**: Splits heads into retrieval (full cache) and streaming (sink+recent only)
- **Parameters**: `head_compression_ratio=0.0`, `on_the_fly_scoring=False`
- **Note**: Uses pre-computed attention patterns for supported models

#### FinchPress
- **Paper**: https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00716/125280
- **Type**: Prompt-guided
- **Mechanism**: SnapKV-style but with dynamic window size from delimiter token position
- **Parameters**: `compression_ratio`, `chunk_length=None`, `normalize_scores=True`, `rerotate_keys=True`
- **Requires**: Call `update_model_and_tokenizer()` to set delimiter token before use

#### KVzipPress
- **Paper**: https://arxiv.org/abs/2505.23416
- **Type**: Context reconstruction
- **Mechanism**: Multiple forward passes to identify redundant KV pairs
- **Parameters**: `compression_ratio`, `layerwise=False`, `n_sink=4`, `kvzip_plus_normalization=False`
- **Warning**: 2-3x computational overhead vs single prefill
- **Note**: Own `__call__` context manager, NOT compatible with ComposedPress

#### FastKVzipPress
- **Paper**: https://arxiv.org/abs/2601.17668
- **Type**: Learned gates
- **Mechanism**: Lightweight learned gate architecture predicts token importance
- **Parameters**: Loaded from HF Hub per model
- **Note**: Own `__call__` context manager. Added in v0.5.1 (latest).

### 5.3 Wrapper Presses

#### ComposedPress
- **Mechanism**: Chains multiple presses sequentially. Each operates on the output of the previous one.
- **Parameters**: `presses: list[BasePress]`
- **Final compression ratio**: Product of (1 - ratio) across all presses
- **Limitations**: Cannot contain AdaKVPress or KVzipPress. May fail if a press depends on features (hidden states, attention weights) that get invalidated by a prior press.

```python
press = ComposedPress([
    SnapKVPress(compression_ratio=0.3),
    ThinKPress(key_channel_compression_ratio=0.2)
])
```

#### AdaKVPress
- **Paper**: https://arxiv.org/abs/2407.11550
- **Mechanism**: Head-wise adaptive compression. Prunes bottom scores across ALL heads jointly, with safeguard minimum per head.
- **Parameters**: `press: ScorerPress`, `alpha_safeguard=0.20`
- **Note**: Uses attention_patch mechanism (fake keys where exp(<q,k>)=0) instead of actual pruning. Does NOT reduce peak memory.
- **Requires**: NOT `attn_implementation="eager"`

#### ChunkPress
- **Paper**: FINCH
- **Mechanism**: Applies ScorerPress independently to fixed-size chunks for uniform compression
- **Parameters**: `press: ScorerPress`, `chunk_length=1024`

#### ChunkKVPress
- **Paper**: https://arxiv.org/abs/2502.00299
- **Mechanism**: Computes global importance scores, then selects proportionally from each chunk
- **Parameters**: `press: ScorerPress`, `chunk_length=20`

#### BlockPress
- **Paper**: https://arxiv.org/abs/2504.15364 (KeyDiff)
- **Mechanism**: Block-wise iterative compression. Processes sequence in non-overlapping blocks.
- **Parameters**: `press: ScorerPress`, `block_size=128`

#### PerLayerCompressionPress
- **Mechanism**: Different compression ratio per layer
- **Parameters**: `press: ScorerPress`, `compression_ratios: list[float]`
- **Warning**: Experimental, only works with flash attention

#### KeyRerotationPress
- **Mechanism**: Re-applies RoPE after compression to maintain proper positional encoding for remaining tokens
- **Parameters**: `press: ScorerPress`
- **Use case**: Wrap StreamingLLMPress to match original paper implementation

#### CriticalKVPress
- **Paper**: https://arxiv.org/abs/2502.03805
- **Mechanism**: Two-stage: first stage uses base scorer, second stage rescales by L1 norm of Wo @ values
- **Parameters**: `press: ScorerPress`, `epsilon=1e-4`, `first_stage_ratio=0.5`

#### CriticalAdaKVPress
- **Mechanism**: CriticalKV + AdaKV combined

#### DecodingPress
- **Type**: Decoding-time compression (experimental)
- **Mechanism**: Accumulates hidden states during decoding, compresses every N steps
- **Parameters**: `base_press: ScorerPress | AdaKVPress`, `compression_interval=512`, `target_size=2048`, `hidden_states_buffer_size=256`

#### PrefillDecodingPress
- **Mechanism**: Combines separate prefill and decoding compression strategies
- **Parameters**: `prefilling_press: BasePress | None`, `decoding_press: DecodingPress | None`

#### DMSPress
- **Paper**: https://arxiv.org/abs/2506.05345
- **Mechanism**: Threshold-based adaptive compression (not fixed ratio). Evicts tokens with scores below threshold.
- **Parameters**: `press: ScorerPress`, `threshold: float | None`, `sliding_window_size=128`, `decoding=False`
- **Note**: `compression_ratio` is dynamically computed (read-only). Works during both prefill and decoding.

---

## 6. Supported Model Architectures

```python
SUPPORTED_MODELS = (
    LlamaForCausalLM,       # Llama 2, 3, 3.1, 3.2, etc.
    MistralForCausalLM,     # Mistral 7B, etc.
    Phi3ForCausalLM,        # Phi-3
    Qwen2ForCausalLM,       # Qwen2, Qwen2.5
    Qwen3ForCausalLM,       # Qwen3
    Gemma3ForConditionalGeneration,  # Gemma 3
)
```

**Important**: The check is a warning, not a hard block. Other architectures with similar structure may work but are untested.

### Architecture requirements:
- Model must have `model.model.layers[].self_attn` structure
- Each `self_attn` must expose key/value cache through HF `Cache` API
- For QKV projection, supports: `q_proj`/`k_proj` (Llama-like), `qkv_proj` (Phi3), and Qwen3/Gemma3 with QK norm

---

## 7. Score-Based vs Non-Score-Based Classification

### Score-based (ScorerPress subclasses):
**Use attention scores**: ObservedAttentionPress (requires eager), TOVAPress, SnapKVPress, NonCausalAttnPress, CompactorPress (internally)
**Use key geometry**: KnormPress, KeyDiffPress, LeverageScorePress, CURPress
**Use statistical modeling**: ExpectedAttentionPress, ExpectedAttentionStatsPress
**Use learned parameters**: QFilterPress, KVzapPress
**Position-based**: StreamingLLMPress (keeps sink + recent, prunes middle)
**Random**: RandomPress

### Non-score-based:
**Dimension compression**: ThinKPress (channel pruning)
**Layer-adaptive**: SimLayerKVPress (lazy layer detection)
**Head-adaptive**: DuoAttentionPress (retrieval vs streaming heads)
**Context reconstruction**: KVzipPress (multiple forward passes)
**Learned gates**: FastKVzipPress
**Prompt-guided**: FinchPress (delimiter-based window)

---

## 8. Quantization Support

```python
from transformers import QuantizedCache

cache = QuantizedCache(backend="quanto", nbits=4)
pipe(..., cache=cache)
```

kvpress handles `QuantizedCache` transparently via `extract_keys_and_values()` which dequantizes before scoring/compression, then re-quantizes after compression.

---

## 9. Key Implementation Details

### Hook mechanism
- Hooks registered via `register_forward_hook(self.forward_hook, with_kwargs=True)` on each `self_attn` layer
- `with_kwargs=True` is essential — provides access to `hidden_states`, `past_key_values`, `cache_position`, `position_embeddings`
- Hooks fire **after** the attention layer forward pass
- Compression only runs during **prefilling** (when `q_len == k_len`), skipped during generation

### Cache manipulation
- Keys/values extracted from `cache.layers[layer_idx].keys` / `.values`
- After compression, directly assigned back: `cache_layer.keys = compressed_keys`
- For QuantizedCache: dequantize -> compress -> re-quantize

### Attention patch (for AdaKV)
- On `import kvpress`, `patch_attention_functions()` is called
- Wraps every function in `transformers.modeling_utils.ALL_ATTENTION_FUNCTIONS`
- During decoding, if `module.masked_key_indices` is set, replaces those key positions with "fake keys" k such that `exp(<q, k>) ~ 0`
- Uses hyperplane search algorithm to find k vectors that zero out attention for all queries

### Gotchas with output_scores / return_dict_in_generate
- `output_scores=True` is **safe** to use with kvpress — the hooks operate on the KV cache, not on the logits/scores output
- `return_dict_in_generate=True` is **safe** — no interference
- The KV cache compression happens during the **prefill** forward pass, before generation begins
- Subsequent generation tokens see the already-compressed cache

### Batch size
- All presses support `batch_size >= 1`
- Score tensors are shaped `(batch_size, num_kv_heads, seq_len)`

### Multi-GPU
- kvpress supports multi-GPU via `accelerate` device_map
- Hooks are registered on the actual model layers regardless of device placement

---

## 10. Release History (Key Milestones)

| Version | Date | Key additions |
|---------|------|---------------|
| 0.5.1 | Feb 16, 2026 | FastKVzipPress |
| 0.5.0 | Jan 28, 2026 | Transformers v5 upgrade, KVzapPress, DMSPress (ThresholdPress) |
| 0.4.0 | Dec 5, 2025 | CURPress, CompactorPress, DecodingPress, post_init_from_model hook |
| 0.3.0 | Sep 4, 2025 | (Major release) |
| 0.2.10 | Aug 6, 2025 | Migration to uv |
| 0.2.x | Jun-Aug 2025 | Various additions (KVzipPress, FinchPress, BlockPress, etc.) |

---

## 11. Herald-Specific Notes

### Press instantiation for Herald experiments:
```python
from kvpress import StreamingLLMPress, SnapKVPress, ExpectedAttentionPress

# Herald uses "compression_ratio" as fraction to REMOVE
# So compression_ratio=0.875 means keep 12.5% of cache
press = StreamingLLMPress(compression_ratio=0.875)
```

### Using with model.generate():
```python
with torch.no_grad(), press(model):
    outputs = model.generate(
        input_ids,
        max_new_tokens=512,
        output_scores=True,            # SAFE: needed for Herald signal extraction
        return_dict_in_generate=True,   # SAFE: needed for Herald signal extraction
    )
# After exiting context manager, hooks are removed
# outputs.scores contains per-token logit distributions
# outputs.past_key_values contains the compressed cache
```

### Compression happens at prefill time only (by default):
1. Input tokens are processed in one forward pass (prefill)
2. After each attention layer, the hook fires and compresses the KV cache
3. Generation proceeds with the compressed cache — no further compression
4. This means catastrophic failures from compression manifest during generation, not prefill

### Methods most relevant to Herald:
- **StreamingLLMPress**: Position-based, simple, predictable failure modes (loses all middle context)
- **SnapKVPress**: Attention-based, more nuanced — may lose rare but important context
- **KnormPress**: Key-geometry-based, cheap to compute, no attention needed
- **ExpectedAttentionPress**: Most sophisticated, best quality, NVIDIA's own method
- **RandomPress**: Useful as degradation baseline
- **TOVAPress**: Single-token attention signal, interesting failure characteristics

### Presses that need special setup:
- `ObservedAttentionPress`: Must load model with `attn_implementation="eager"` (slower)
- `QFilterPress`: Needs pre-trained filters (not all models supported)
- `KVzipPress`: 2-3x overhead, own context manager
- `FastKVzipPress`: Needs pre-trained gates, own context manager
- `KVzapPress`: Needs pre-trained surrogate model
- `ExpectedAttentionStatsPress`: Needs pre-computed stats (or computes them on first run)
- `FinchPress`: Needs delimiter token configuration
- `DuoAttentionPress`: Pre-computed patterns for limited set of models
