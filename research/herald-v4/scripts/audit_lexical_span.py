"""Independent join of frozen prompt-only features to exposed B0 evidence."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score
from transformers import AutoTokenizer

from analyze_retrieval_probe import exact_auc_permutation
from score_ruler_pilot import score_prediction

ROOT = Path(__file__).resolve().parents[1]
TOKENIZER = ROOT.parent / 'herald-v3/data/retrieval-tokenizer-a09a354'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('features', type=Path)
    args = parser.parse_args()
    payload = json.loads(args.features.read_text())
    features = payload if isinstance(payload, list) else payload['rows']
    inputs = {r['id']: r for r in json.loads((ROOT / 'results/lexical-span-input/view.json').read_text())}
    oracles = {}
    for name in ('needle-rescue-v1-first', 'needle-rescue-v1-rest', 'needle-rescue-all12-additional'):
        folder = ROOT / 'results' / name
        run = json.loads((folder / 'run.json').read_text())
        for item in run['prompts']:
            assert item['id'] not in oracles
            oracles[item['id']] = json.loads((folder / item['path']).read_text())
    assert len(features) == len(inputs) == len(oracles) == 12
    assert {r['id'] for r in features} == set(inputs) == set(oracles)
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER, local_files_only=True)
    rows = []
    max_z_error = 0.0
    for feature in features:
        row_id = feature['id']
        view, oracle = inputs[row_id], oracles[row_id]
        assert view['native_masks'] == oracle['native_baseline_masks']
        rendered = tokenizer.apply_chat_template([{'role': 'user', 'content': view['prompt']}], tokenize=False, add_generation_prompt=True)
        encoded = tokenizer(rendered, add_special_tokens=False, return_offsets_mapping=True)
        assert encoded['input_ids'] == oracle['prompt_token_ids']
        assert rendered.count(view['prompt']) == 1
        prompt_start = rendered.index(view['prompt'])
        ordered_ids = sorted(inputs)
        next_id = ordered_ids[(ordered_ids.index(row_id) + 1) % len(ordered_ids)]
        assert feature['question']['text'] == view['prompt'].rstrip().rsplit('\n', 1)[1].strip()
        assert feature['shifted_question']['source_id'] == next_id
        assert feature['shifted_question']['text'] == inputs[next_id]['prompt'].rstrip().rsplit('\n', 1)[1].strip()
        row = {'id': row_id}
        for mode in ('matched', 'shifted'):
            f = feature[mode]
            assert view['prompt'][f['raw_start']:f['raw_end']] == f['sentence']
            begin, end = prompt_start + f['raw_start'], prompt_start + f['raw_end']
            expected = [i for i, (a, b) in enumerate(encoded['offset_mapping']) if b > begin and a < end]
            assert expected == f['token_positions']
            assert expected and max(expected) < view['prompt_length'] - 1
            positions = set(expected)
            masses = [len(positions - set(head)) / len(positions) for layer in view['native_masks'] for head in layer]
            z = float(np.mean(masses))
            assert np.max(np.abs(np.array(masses) - np.array(f['per_head_fractions']).ravel())) < 1e-12
            normalized_position = float(np.mean(expected) / (view['prompt_length'] - 2))
            assert abs(normalized_position - f['normalized_position']) < 1e-12
            max_z_error = max(max_z_error, abs(z - f['z']))
            assert abs(z - f['z']) < 1e-12
            row[mode] = {'z': z, 'similarity': f['similarity'], 'position': normalized_position, 'token_count': len(expected), 'oracle_coverage': len(positions & set(oracle['span']['token_positions'])) / len(oracle['span']['token_positions']), 'extra_tokens': len(positions - set(oracle['span']['token_positions']))}
        ref = oracle['branches']['reference_shared_boundary']
        act = oracle['branches']['standard_knorm_0.10']
        scores = []
        for arm in (ref, act):
            score = score_prediction(arm['continuation']['text'], oracle['manifest_row']['answers'])
            assert score == arm['score']
            scores.append(score['score_fraction'])
        row['signed_loss'] = scores[0] - scores[1]
        assert row['signed_loss'] == act['signed_loss_vs_reference']
        rows.append(row)
    y = np.array([r['signed_loss'] > 0 for r in rows])
    assert y.sum() == 8
    matched = [r['matched']['z'] for r in rows]
    shifted = [r['shifted']['z'] for r in rows]
    auc = float(roc_auc_score(y, matched))
    control_auc = float(roc_auc_score(y, shifted))
    gates = {'all_oracle_sentences_covered_within_64_tokens': all(r['matched']['oracle_coverage'] == 1 and r['matched']['token_count'] <= 64 for r in rows), 'raw_auc_at_least_0_80': auc >= .8, 'auc_gap_at_least_0_15': auc - control_auc >= .15}
    report = {'scope': 'Exposed 12-case B0 .10 understanding audit, no fitted predictor', 'feature_sha256': hashlib.sha256(args.features.read_bytes()).hexdigest(), 'all_reconstruction_checks_pass': True, 'max_z_error': max_z_error, 'matched_auc': auc, 'shifted_auc': control_auc, 'similarity_auc': float(roc_auc_score(y, [r['matched']['similarity'] for r in rows])), 'position_auc': float(roc_auc_score(y, [r['matched']['position'] for r in rows])), 'permutation': exact_auc_permutation(matched, y), 'gates': gates, 'supports_next_design': all(gates.values()), 'rows': rows}
    output = ROOT / 'results/lexical-span-owner-audit.json'
    output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'rows'}, indent=2))


if __name__ == '__main__':
    main()
