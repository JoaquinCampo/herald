"""One frozen benchmark-conditioned audit using existing fold-local primitives."""
import hashlib
import json
from collections import Counter
from pathlib import Path
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
import analyze_ea_development as analysis

ROOT = Path(__file__).resolve().parents[1]
manifest = analysis._manifest_rows(ROOT / 'data/ruler-ea-dev-v1/manifest.json')
dirs = [ROOT / 'results/ea-dev-v1-first', ROOT / 'results/ea-dev-v1-rest']
records, failures = analysis._run_records(dirs, set(manifest))
scores, score_failures = analysis._score_rows(dirs, set(manifest))
rows, coverage = analysis._materialize(manifest, records, scores)
assert not failures and not score_failures and all(not v for v in coverage.values())
prefix = {(r['id'], float(r['action'].split(':')[1])): int(r['prefix_disagrees']) for r in map(json.loads, (ROOT / 'results/prefix16-diagnostic/rows.jsonl').read_text().splitlines())}
assert len(rows) == len(prefix) == 60
assert set(Counter(r['prompt_id'] for r in rows).values()) == {3}
for r in rows:
    r['task_indicator'] = int(r['task'] == 'niah_single_2')
    r['prefix16_disagreement'] = prefix[r['prompt_id'], r['action']]
features = list(analysis.FEATURES[:-1]) + ['task_indicator']
models = {}
for name, names in [('metadata_task', features), ('prefix16', features + ['prefix16_disagreement'])]:
    models[name], _ = analysis._fit_oof(rows, names)
models['task_action_mean'] = []
for r in rows:
    train = [v['signed_loss'] for v in rows if v['fold'] != r['fold'] and v['task'] == r['task'] and v['action'] == r['action']]
    assert train
    models['task_action_mean'].append({**r, 'prediction': float(np.mean(train))})

def metrics(items):
    y = [r['signed_loss'] for r in items]; p = [r['prediction'] for r in items]
    return {'mse': mean_squared_error(y, p), 'mae': mean_absolute_error(y, p), 'signed_bias': float(np.mean(np.asarray(p) - y)), 'prompts': len({r['prompt_id'] for r in items})}

report = {name: {'pooled': metrics(items), 'folds': {str(f): metrics([r for r in items if r['fold'] == f]) for f in analysis.FOLDS}, 'tasks': {t: metrics([r for r in items if r['task'] == t]) for t in sorted({r['task'] for r in items})}} for name, items in models.items()}
comparisons = {}
for baseline in ['task_action_mean', 'metadata_task']:
    gain = 1 - report['prefix16']['pooled']['mse'] / report[baseline]['pooled']['mse']
    wins = sum(report['prefix16']['folds'][str(f)]['mse'] < report[baseline]['folds'][str(f)]['mse'] for f in analysis.FOLDS)
    comparisons[baseline] = {'mse_gain_fraction': gain, 'fold_wins': wins, 'proceed': gain >= .1 and wins >= 3}
summary = {'scope': 'benchmark-conditioned exploratory20 exposed prompts; no confirmation', 'metrics': report, 'comparisons': comparisons, 'proceed': all(c['proceed'] for c in comparisons.values()), 'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
out = ROOT / 'results/task-aware-mse-audit'; out.mkdir(exist_ok=True)
(out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
(out / 'predictions.json').write_text(json.dumps(models, indent=2) + '\n')
print(json.dumps({'pooled': {k:v['pooled'] for k,v in report.items()}, 'comparisons': comparisons, 'proceed': summary['proceed']}, indent=2))
