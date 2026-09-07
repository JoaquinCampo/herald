"""Fixed exploratory comparison, composing the existing fold-local analyzer."""
import hashlib
import json
from pathlib import Path
import analyze_ea_development as analysis

ROOT = Path(__file__).resolve().parents[1]
manifest = analysis._manifest_rows(ROOT / 'data/ruler-ea-dev-v1/manifest.json')
dirs = [ROOT / 'results/ea-dev-v1-first', ROOT / 'results/ea-dev-v1-rest']
records, failures = analysis._run_records(dirs, set(manifest))
scores, score_failures = analysis._score_rows(dirs, set(manifest))
rows, coverage = analysis._materialize(manifest, records, scores)
assert not failures and not score_failures and all(not v for v in coverage.values())
prefix_rows = [json.loads(v) for v in (ROOT / 'results/prefix16-diagnostic/rows.jsonl').read_text().splitlines()]
prefix = {(r['id'], float(r['action'].split(':')[1])): int(r['prefix_disagrees']) for r in prefix_rows}
assert len(rows) == len(prefix) == 60
for row in rows:
    row['prefix16_disagreement'] = prefix[row['prompt_id'], row['action']]
features = list(analysis.FEATURES[:-1])
metrics, output = {}, {}
for model, names in [('baseline', features), ('prefix16', features + ['prefix16_disagreement'])]:
    predicted, _ = analysis._fit_oof(rows, names)
    metrics[model] = {'pooled': analysis._metric(predicted, 'prediction'),
                      'folds': {str(f): analysis._metric([r for r in predicted if r['fold'] == f], 'prediction') for f in analysis.FOLDS},
                      'spearman': analysis._spearman(predicted, 'prediction')}
    for r in predicted:
        key = r['prompt_id'], r['action']
        output.setdefault(key, {k: r[k] for k in ['prompt_id', 'task', 'fold', 'action', 'signed_loss', 'prefix16_disagreement']})
        output[key][model + '_prediction'] = r['prediction']
gain = 1 - metrics['prefix16']['pooled']['mae'] / metrics['baseline']['pooled']['mae']
wins = sum(metrics['prefix16']['folds'][str(f)]['mae'] < metrics['baseline']['folds'][str(f)]['mae'] for f in analysis.FOLDS)
gates = {'gain_at_least_10_percent': gain >= .1, 'at_least_three_fold_wins': wins >= 3,
         'positive_within_task_spearman': all(v['rho'] is not None and v['rho'] > 0 for v in metrics['prefix16']['spearman'].values())}
summary = {'metrics': metrics, 'gain_fraction': gain, 'fold_wins': wins, 'gates': gates, 'proceed': all(gates.values()),
           'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
           'reused_analyzer_sha256': hashlib.sha256(Path(analysis.__file__).read_bytes()).hexdigest(),
           'scope': 'Exploratory existing20 exposed prompts, no confirmation; counterfactual16-token-per-arm observation cost not timed.'}
out = ROOT / 'results/prefix16-model'; out.mkdir(exist_ok=True)
(out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
(out / 'predictions.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in output.values()))
print(json.dumps(summary, indent=2))
