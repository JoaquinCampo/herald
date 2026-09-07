"""Frozen058 pilot analysis from canonical records and official raw-answer F1."""
import hashlib
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'data/musique-pilot-v1/provenance/official_scorer'))
from metrics.answer import compute_f1, metric_max_over_ground_truths

manifest = [json.loads(line) for line in (ROOT / 'data/musique-pilot-v1/manifest.jsonl').read_text().splitlines()]
branches = ['plain_noop', 'instrumented_noop', 'ea_10', 'knorm_10', 'uninterrupted']
rows = []
for index, row in enumerate(manifest):
    folder = ['musique-pilot-v1-first', 'musique-pilot-v1-remaining', 'musique-pilot-v1-contract-repro'][index] if index < 3 else 'musique-pilot-v1-final13'
    paths = list((ROOT / 'results' / folder).glob(f'*{row["id"]}.json'))
    assert len(paths) == 1, (row['id'], paths)
    path = paths[0]
    record = json.loads(path.read_text())
    assert record['id'] == row['id']
    assert record['prompt']['text'] == row['prompt']
    assert record['status'] == 'completed' and all(record['gates'].values())
    c = record['continuations']
    assert c['plain_noop']['token_ids'] == c['instrumented_noop']['token_ids']
    assert c['plain_noop']['termination_reason'] == c['instrumented_noop']['termination_reason']
    assert record['config']['max_new_tokens'] == 64 and record['config']['seed'] == 0
    scores = {b: float(metric_max_over_ground_truths(compute_f1, c[b]['text'], row['answers'])) for b in branches}
    rows.append({'id': row['id'], 'group_id': row['group_id'], 'record': str(path.relative_to(ROOT)), 'record_sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'scores': scores, 'ea_loss': scores['plain_noop'] - scores['ea_10'], 'knorm_loss': scores['plain_noop'] - scores['knorm_10'], 'full_prefill_parity': c['uninterrupted']['token_ids'] == c['plain_noop']['token_ids'] and c['uninterrupted']['termination_reason'] == c['plain_noop']['termination_reason']})
assert len(rows) == 16 and len({r['id'] for r in rows}) == 16
reference_qualified = sum(r['scores']['plain_noop'] >= .8 for r in rows)
ea_levels = sorted({r['scores']['ea_10'] for r in rows})
positive = sum(r['ea_loss'] > 0 for r in rows)
gates = {'all_16_scored_and_technical': True, 'reference_at_least_12_f1_ge_0_8': reference_qualified >= 12, 'ea_at_least_3_score_levels': len(ea_levels) >= 3, 'ea_at_least_4_positive_losses': positive >= 4}
summary = {'schema': 'musique_pilot_summary.v1', 'reference_branch': 'plain_noop', 'n': len(rows), 'groups': len({r['group_id'] for r in rows}), 'means': {b: statistics.mean(r['scores'][b] for r in rows) for b in branches}, 'reference_f1_ge_0_8': reference_qualified, 'ea_score_levels': ea_levels, 'ea_signed_loss_counts': {'negative': sum(r['ea_loss'] < 0 for r in rows), 'zero': sum(r['ea_loss'] == 0 for r in rows), 'positive': positive}, 'knorm_signed_loss_counts': {'negative': sum(r['knorm_loss'] < 0 for r in rows), 'zero': sum(r['knorm_loss'] == 0 for r in rows), 'positive': sum(r['knorm_loss'] > 0 for r in rows)}, 'ea_mean_signed_loss': statistics.mean(r['ea_loss'] for r in rows), 'knorm_mean_signed_loss': statistics.mean(r['knorm_loss'] for r in rows), 'supplemental_full_prefill_parity_count': sum(r['full_prefill_parity'] for r in rows), 'gates': gates, 'feasibility_pass': all(gates.values()), 'rows': rows, 'claim_boundary': 'Fixed exploratory QA feasibility only, no predictor fit or confirmation.'}
out = ROOT / 'results/musique-pilot-summary.json'
out.write_text(json.dumps(summary, indent=2) + '\n')
print(json.dumps({k:v for k,v in summary.items() if k != 'rows'}, indent=2))
