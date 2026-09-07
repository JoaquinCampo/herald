"""Frozen two-coefficient residual augmentation over task/action means."""
import hashlib
import json
from pathlib import Path
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error

ROOT = Path(__file__).resolve().parents[1]
source = json.loads((ROOT / 'results/task-aware-mse-audit/predictions.json').read_text())
rows = source['task_action_mean']
assert len(rows) == 60 and len({r['prompt_id'] for r in rows}) == 20
predicted = []; coefficients = {}
for fold in range(4):
    train = [r for r in rows if r['fold'] != fold]; test = [r for r in rows if r['fold'] == fold]
    cells = {}
    for key in {(r['task'], r['action']) for r in train}:
        members = [r for r in train if (r['task'], r['action']) == key]
        bits = [r['prefix16_disagreement'] for r in members]
        cells[key] = (np.mean([r['signed_loss'] for r in members]), np.mean(bits), bool(np.var(bits) > 0))
    def design(r):
        _, bit_mean, variable = cells[r['task'], r['action']]
        x = np.zeros(2)
        if variable: x[int(r['task'] == 'niah_single_2')] = r['prefix16_disagreement'] - bit_mean
        return x
    x = np.asarray([design(r) for r in train])
    y = np.asarray([r['signed_loss'] - cells[r['task'], r['action']][0] for r in train])
    model = Ridge(alpha=1, fit_intercept=False).fit(x, y)
    coefficients[str(fold)] = model.coef_.tolist()
    correction = model.predict(np.asarray([design(r) for r in test]))
    for r, delta in zip(test, correction, strict=True):
        base = float(cells[r['task'], r['action']][0])
        predicted.append({**r, 'baseline': base, 'residual_prediction': base + float(delta)})
def metrics(items, key):
    y = [r['signed_loss'] for r in items]; p = [r[key] for r in items]
    return {'mse': mean_squared_error(y,p), 'mae': mean_absolute_error(y,p), 'bias':float(np.mean(np.asarray(p)-y))}
report = {k:{'pooled':metrics(predicted,k),'folds':{str(f):metrics([r for r in predicted if r['fold']==f],k) for f in range(4)},'tasks':{t:metrics([r for r in predicted if r['task']==t],k) for t in {r['task'] for r in predicted}}} for k in ['baseline','residual_prediction']}
gain = 1-report['residual_prediction']['pooled']['mse']/report['baseline']['pooled']['mse']
wins = sum(report['residual_prediction']['folds'][str(f)]['mse']<report['baseline']['folds'][str(f)]['mse'] for f in range(4))
summary = {'metrics':report,'gain_fraction':gain,'fold_wins':wins,'proceed':gain>=.1 and wins>=3,'coefficients':coefficients,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'scope':'existing20 exposed prompts; no confirmation'}
out = ROOT/'results/residual-prefix16';out.mkdir(exist_ok=True)
(out/'predictions.json').write_text(json.dumps(predicted,indent=2)+'\n');(out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2))
