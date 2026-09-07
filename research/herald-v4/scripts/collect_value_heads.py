"""Fixed B0 numeric-value retention features and paired signed outcomes."""
import argparse
import hashlib
import json
import math
import sys
import time
import traceback
from pathlib import Path

import joblib
import numpy as np

import diagnose_needle_rescue as rescue
import run_pair_pilot as runner
from score_ruler_pilot import score_prediction
from value_span_adapter import locate_value

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    for name in ('manifest', 'model', 'engine-root', 'output'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--ids', nargs='+')
    parser.add_argument('--models', type=Path)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--dtype', choices=('float32', 'bfloat16'), default='float32')
    args = parser.parse_args()
    import torch
    import transformers
    torch.manual_seed(0)
    engine, _, engine_path = runner.load_engine(args.engine_root)
    manifest_path, rows = runner.load_manifest(args.manifest)
    if args.ids:
        assert set(args.ids) <= {r['id'] for r in rows}
        rows = [r for r in rows if r['id'] in args.ids]
    assert rows
    output = runner.ensure_output_dir(args.output)
    device = runner.choose_device(args.device, torch)
    dtype = runner.choose_dtype(args.dtype, device, torch)
    model, tokenizer = runner.load_model_and_tokenizer(args.model, device, dtype, transformers)
    bundle = joblib.load(args.models) if args.models else None
    if bundle:
        import sklearn
        assert bundle['sklearn_version'] == sklearn.__version__
    eos = runner.eos_ids(model, tokenizer)
    sources = [Path(__file__), ROOT/'scripts/value_span_adapter.py', ROOT/'scripts/run_pair_pilot.py', ROOT/'scripts/diagnose_needle_rescue.py', ROOT/'scripts/score_ruler_pilot.py', engine_path]
    run = {'status': 'running', 'manifest_sha256': sha(manifest_path), 'source_hashes': {str(p.resolve()): sha(p) for p in sources}, 'model': runner.model_runtime_identity(engine, model, args.model, transformers, torch), 'models_sha256': sha(args.models) if args.models else None, 'seed': 0, 'removal_fraction': .1, 'prompts': [], 'failures': []}
    runner.write_json(output/'run.json', run)
    for index, row in enumerate(rows):
        filename = runner.safe_filename(index, row['id'])
        record = {'status': 'running', 'manifest_row': row}
        try:
            ids = runner.tokenize_chat_prompt(tokenizer, row['prompt'])
            boundary, source = runner.build_last_prompt_boundary(engine, model, ids)
            boundary_fp, source_fp = engine.cache_fingerprint(boundary.cache), engine.cache_fingerprint(source)
            started = time.perf_counter()
            located = locate_value(row['prompt'], tokenizer)
            adapter_seconds = time.perf_counter() - started
            assert located['prompt_length'] == int(ids.shape[1])
            positions = set(located['value_positions'])
            assert bool(positions) == located['found']
            assert all(0 <= v < int(ids.shape[1])-1 for v in positions)
            runner.sync(engine, device)
            started = time.perf_counter()
            candidate_cache = engine.clone_cache(boundary.cache)
            candidate = engine.compress_knorm(candidate_cache, .1)
            masks = rescue.json_indices(candidate.kept_indices)
            runner.sync(engine, device)
            candidate_seconds = time.perf_counter() - started
            del candidate_cache
            started = time.perf_counter()
            per_head = [len(positions-set(head))/len(positions) if positions else 0.0 for layer in masks for head in layer]
            total = sum(length*len(layer) for length, layer in zip(boundary.cache_lengths, masks, strict=True))
            removed = 1 - sum(len(h) for layer in masks for h in layer)/total
            metadata = [float(not located['found']), math.log(int(ids.shape[1])), float(len(positions)), located['position_midpoint_normalized'], removed]
            features = metadata + per_head + [float(np.mean(per_head))]
            feature_seconds = time.perf_counter() - started
            assert all(math.isfinite(v) for v in features)
            if device.type == 'cuda':
                assert len(per_head) == 112
            predictions = {}
            started = time.perf_counter()
            if bundle:
                assert len(features) == bundle['feature_count']
                for name, fitted in bundle['models'].items():
                    predictions[name] = float(fitted['constant']) if 'constant' in fitted else float(fitted['estimator'].predict(np.array(features)[fitted['columns']].reshape(1, -1))[0])
            prediction_seconds = time.perf_counter() - started
            observation = {'id': row['id'], 'located': located, 'features': features, 'head_count': len(per_head), 'predictions': predictions, 'cost': {'adapter_seconds': adapter_seconds, 'candidate_clone_compress_transfer_seconds': candidate_seconds, 'feature_seconds': feature_seconds, 'prediction_seconds': prediction_seconds}}
            # This immutable per-case record is written before any paired generation.
            feature_path = output/(Path(filename).stem + '.features.json')
            assert not feature_path.exists()
            runner.write_json(feature_path, observation)
            np.savez_compressed(output/(Path(filename).stem + '.masks.npz'), kept=np.asarray(masks, dtype=np.int32))
            record.update({'observation': observation, 'feature_sha256': sha(feature_path), 'prompt_token_ids': ids[0].tolist(), 'boundary': boundary.to_dict(), 'branches': {}})
            arms = {}
            for name, fraction in (('reference', 0.0), ('noop', 0.0), ('action', .1)):
                arm = engine.continue_from_boundary(model, boundary, max_new_tokens=row.get('max_new_tokens', 128), eos_ids=eos, action=engine.ActionSpec('knorm', fraction))
                text = tokenizer.decode(list(arm.continuation.token_ids), skip_special_tokens=True)
                arms[name] = arm
                record['branches'][name] = {'continuation': runner.continuation_dict(arm.continuation, text), 'score': score_prediction(text, row['answers']), 'compression': runner.compact_compression(engine, arm.compression, boundary)}
                assert engine.cache_fingerprint(boundary.cache) == boundary_fp and engine.cache_fingerprint(source) == source_fp
            compression = record['branches']['action']['compression']
            checks = {'noop_ids_exact': arms['reference'].continuation.token_ids == arms['noop'].continuation.token_ids, 'noop_termination_exact': arms['reference'].continuation.termination_reason == arms['noop'].continuation.termination_reason, 'candidate_mask_exact': runner.index_digest(masks)[0] == compression['kept_index_hash'], 'physical_effect_exact': compression['physical_effect_exact'] and compression['strictly_reduced_for_nonzero_action'], 'source_unchanged': engine.cache_fingerprint(boundary.cache) == boundary_fp and engine.cache_fingerprint(source) == source_fp, 'features_precede_outcomes_unchanged': sha(feature_path) == record['feature_sha256']}
            assert all(checks.values()), checks
            record.update({'status': 'completed', 'checks': checks, 'signed_loss': record['branches']['reference']['score']['score_fraction'] - record['branches']['action']['score']['score_fraction']})
        except Exception as exc:
            record.update(status='failed', failure={'type': type(exc).__name__, 'error': str(exc), 'traceback': traceback.format_exc()})
            run['failures'].append({'id': row['id'], **record['failure']})
        runner.write_json(output/filename, record)
        run['prompts'].append({'id': row['id'], 'path': filename, 'status': record['status']})
        runner.write_json(output/'run.json', run)
        print(row['id'], record['status'], flush=True)
        if record['status'] == 'failed':
            break
    run['status'] = 'failed' if run['failures'] else 'completed'
    runner.write_json(output/'run.json', run)
    return int(bool(run['failures']))


if __name__ == '__main__':
    sys.exit(main())
