import hashlib
import sys
from pathlib import Path

ROOT = Path('/Users/joaquincamponario/Documents/INCO/RESEARCH/herald-v4')
MODEL_PATH = Path('/private/tmp/herald-v4-ea-proof/model')
MANIFEST_PATH = Path('/private/tmp/herald-v4-ea-proof/manifest.json')
ENGINE_ROOT = Path('/Users/joaquincamponario/Documents/INCO/RESEARCH/herald-v3/src')
REMOVAL = 0.10
N_SINK = 4
N_FUTURE = 512
MAX_NEW = 4
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ENGINE_ROOT))
import run_pair_pilot as runner
from herald_v3.engineering import engine

import torch
import transformers
from kvpress import ExpectedAttentionPress


def install_rotary(model):
    rotary = model.model.rotary_emb
    previous = []
    for layer in model.model.layers:
        module = layer.self_attn
        had = hasattr(module, 'rotary_emb')
        old = getattr(module, 'rotary_emb', None)
        previous.append((module, had, old))
        if not had or old is not rotary:
            module.rotary_emb = rotary
    return previous


def restore_rotary(previous):
    for module, had, old in previous:
        if had:
            module.rotary_emb = old
        else:
            delattr(module, 'rotary_emb')


def cache_summary(cache):
    return {'lengths': list(engine.cache_lengths(cache)), 'bytes': engine.cache_nbytes(cache), 'fingerprint': engine.cache_fingerprint(cache)}


def mask_digest(indices):
    digest = hashlib.sha256()
    for layer in indices:
        digest.update(len(layer).to_bytes(4, 'little'))
        for head in layer:
            digest.update(len(head).to_bytes(8, 'little'))
            for pos in head:
                digest.update(int(pos).to_bytes(8, 'little'))
    return digest.hexdigest()


def collect(model, prompt_ids, press):
    scores = []
    hooks = []
    press.post_init_from_model(model)
    previous = install_rotary(model)
    def make_hook(index):
        def hook(module, args, kwargs, output):
            layer_cache = kwargs['past_key_values'].layers[module.layer_idx]
            values = press.score(module, kwargs['hidden_states'], layer_cache.keys, layer_cache.values, None, kwargs)
            scores.append((index, values.detach().clone()))
        return hook
    for index, layer in enumerate(model.model.layers):
        hooks.append(layer.self_attn.register_forward_hook(make_hook(index), with_kwargs=True))
    try:
        result = runner.build_last_prompt_boundary(engine, model, prompt_ids)
    finally:
        for h in hooks:
            h.remove()
        restore_rotary(previous)
    scores.sort(key=lambda x: x[0])
    return result, scores


def direct_indices(score):
    keep = int(score.shape[-1] * (1.0 - REMOVAL))
    return score.topk(keep, dim=-1).indices


def make_indices(scores):
    return [[[int(x) for x in row] for row in direct_indices(score)[0].detach().cpu().tolist()] for _, score in scores]


def continuation(boundary, cache, eos):
    return engine._continue_cache(model, boundary, cache, max_new_tokens=MAX_NEW, eos_ids=eos, first_logits_observer=None)

row = __import__('json').loads(MANIFEST_PATH.read_text())[0]
tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True, use_fast=True)
model = transformers.AutoModelForCausalLM.from_pretrained(MODEL_PATH, local_files_only=True, dtype=torch.float32, attn_implementation='sdpa')
model.eval()
prompt_ids = tokenizer.apply_chat_template([{'role': 'user', 'content': row['prompt']}], tokenize=True, return_tensors='pt', add_generation_prompt=True)
if prompt_ids.ndim == 1:
    prompt_ids = prompt_ids.unsqueeze(0)
press = ExpectedAttentionPress(compression_ratio=0.0, n_future_positions=N_FUTURE, n_sink=N_SINK, use_covariance=True, use_vnorm=True, epsilon=0.0)
plain_pair = runner.build_last_prompt_boundary(engine, model, prompt_ids)
plain_boundary, plain_source = plain_pair
inst_pair, scores = collect(model, prompt_ids, press)
inst_boundary, inst_source = inst_pair
det_pair, det_scores = collect(model, prompt_ids, press)
det_boundary, det_source = det_pair

score_shapes = [list(score.shape) for _, score in scores]
score_finite = all(bool(score.isfinite().all().item()) for _, score in scores)
indices = make_indices(scores)
det_indices = make_indices(det_scores)
mask_hash = mask_digest(indices)

action_cache = engine.clone_cache(plain_boundary.cache)
gather_checks = []
for layer_index, score in scores:
    base = plain_boundary.cache.layers[layer_index]
    act = action_cache.layers[layer_index]
    idx = direct_indices(score)
    expanded = idx.unsqueeze(-1).expand(-1, -1, -1, base.keys.shape[-1])
    expected_k = base.keys.gather(2, expanded).contiguous()
    expected_v = base.values.gather(2, expanded).contiguous()
    act.keys = expected_k
    act.values = expected_v
    gather_checks.append({'layer': layer_index, 'keys_equal': bool(act.keys.equal(expected_k)), 'values_equal': bool(act.values.equal(expected_v))})

action_pre_summary = cache_summary(action_cache)
eos = {tokenizer.eos_token_id}
uninterrupted = engine._greedy_from_prompt(model, prompt_ids, max_new_tokens=MAX_NEW, eos_ids=eos)
plain_noop = continuation(plain_boundary, engine.clone_cache(plain_boundary.cache), eos)
inst_noop = continuation(inst_boundary, engine.clone_cache(inst_boundary.cache), eos)
act_cont = continuation(plain_boundary, action_cache, eos)

plain_before = engine.cache_fingerprint(plain_boundary.cache)
inst_before = engine.cache_fingerprint(inst_boundary.cache)
plain_src_before = engine.cache_fingerprint(plain_source)
inst_src_before = engine.cache_fingerprint(inst_source)
source_unchanged = (engine.cache_fingerprint(plain_boundary.cache) == plain_before and engine.cache_fingerprint(inst_boundary.cache) == inst_before and engine.cache_fingerprint(plain_source) == plain_src_before and engine.cache_fingerprint(inst_source) == inst_src_before)
plain_summary = cache_summary(plain_boundary.cache)
action_summary = cache_summary(action_cache)
expected_keep = int(plain_summary['lengths'][0] * (1.0 - REMOVAL))
expected_bytes = sum((tensor.numel() // int(tensor.shape[-2])) * expected_keep * tensor.element_size() for tensor in engine._cache_tensors(plain_boundary.cache))
result = {
    'prompt_length': int(prompt_ids.shape[1]),
    'prompt_token_ids': [int(x) for x in prompt_ids[0].tolist()],
    'boundary': {'logical_position': int(plain_boundary.logical_position), 'pending_token_id': int(plain_boundary.pending_token_id), 'cache': plain_summary, 'source': cache_summary(plain_source)},
    'instrumented': {'cache_equal': engine.cache_tensors_equal(plain_boundary.cache, inst_boundary.cache), 'source_equal': engine.cache_tensors_equal(plain_source, inst_source), 'boundary_storage_disjoint': engine.cache_storage_independent(plain_boundary.cache, inst_boundary.cache), 'source_storage_disjoint': engine.cache_storage_independent(plain_source, plain_boundary.cache) and engine.cache_storage_independent(inst_source, inst_boundary.cache), 'det_boundary_equal': engine.cache_tensors_equal(inst_boundary.cache, det_boundary.cache), 'det_source_equal': engine.cache_tensors_equal(inst_source, det_source)},
    'scores': {'layer_count': len(scores), 'shapes': score_shapes, 'finite': score_finite, 'expected_kv_heads': int(model.config.num_key_value_heads)},
    'action': {'mask_hash': mask_hash, 'repeat_mask_hash': mask_digest(det_indices), 'indices_equal': indices == det_indices, 'kept_count_per_layer_head': [[len(head) for head in layer] for layer in indices], 'expected_keep': expected_keep, 'cache': action_pre_summary, 'expected_bytes': expected_bytes, 'sink_retained': all(set(range(N_SINK)).issubset(set(head)) for layer in indices for head in layer), 'gather_checks': gather_checks, 'action_storage_disjoint': engine.cache_storage_independent(plain_boundary.cache, action_cache)},
    'continuations': {'uninterrupted': {'ids': list(uninterrupted.token_ids), 'termination': uninterrupted.termination_reason}, 'plain_noop': {'ids': list(plain_noop.token_ids), 'termination': plain_noop.termination_reason}, 'instrumented_noop': {'ids': list(inst_noop.token_ids), 'termination': inst_noop.termination_reason}, 'action': {'ids': list(act_cont.token_ids), 'termination': act_cont.termination_reason}, 'no_op_exact': uninterrupted.token_ids == plain_noop.token_ids == inst_noop.token_ids and uninterrupted.termination_reason == plain_noop.termination_reason == inst_noop.termination_reason},
    'source_unchanged_after_continuations': source_unchanged,
    'native_ratio_zero_passthrough': None,
}
# Directly verify native ratio-zero short circuit without allowing score() to run.
keys = torch.zeros((1, 1, 3, 8), dtype=torch.float32)
values = torch.ones_like(keys)
ratio_zero = ExpectedAttentionPress(compression_ratio=0.0, n_future_positions=N_FUTURE, n_sink=N_SINK, use_covariance=True, use_vnorm=True, epsilon=0.0)
original = ratio_zero.score
def should_not_run(*args, **kwargs):
    raise AssertionError('ratio-zero native compress scored')
ratio_zero.score = should_not_run
try:
    out_k, out_v = ratio_zero.compress(None, None, keys, values, None, {})
finally:
    ratio_zero.score = original
result['native_ratio_zero_passthrough'] = out_k is keys and out_v is values
print(__import__('json').dumps(result, sort_keys=True))
