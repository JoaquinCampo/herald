"""Fit design035 once from discovery only, serializing pipelines and fold provenance."""
import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import sklearn
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--results', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    assert not args.output.exists()
    run = json.loads((args.results/'run.json').read_text())
    assert run['status'] == 'completed' and not run['failures']
    records = [json.loads((args.results/r['path']).read_text()) for r in run['prompts']]
    assert len(records) == 48 and all(r['status'] == 'completed' and all(r['checks'].values()) for r in records)
    assert all(r['manifest_row']['provenance']['split'] == 'discovery' for r in records)
    x = np.asarray([r['observation']['features'] for r in records])
    y = np.asarray([r['signed_loss'] for r in records])
    groups = np.asarray([r['manifest_row']['group_id'] for r in records])
    assert x.shape == (48, 118) and np.isfinite(x).all() and np.isfinite(y).all()
    assert sum(y > 0) >= 6 and sum(y <= 0) >= 6, 'insufficient discovery label variation'
    folds = list(GroupKFold(4).split(x, y, groups))
    bundle = {'sklearn_version': sklearn.__version__, 'feature_count': 118, 'models': {'mean': {'constant': float(y.mean())}}}
    report = {'source_sha256': sha(__file__), 'run_sha256': sha(args.results/'run.json'), 'record_sha256': {r['path']: sha(args.results/r['path']) for r in run['prompts']}, 'ids': [r['manifest_row']['id'] for r in records], 'groups': groups.tolist(), 'features': x.tolist(), 'signed_losses': y.tolist(), 'folds': [{'train': a.tolist(), 'validation': b.tolist()} for a, b in folds], 'models': {'mean': {'constant': float(y.mean())}}, 'sklearn_version': sklearn.__version__}
    for name, columns in [('metadata', list(range(5))), ('mean_retention', list(range(5))+[117]), ('heads', list(range(117)))]:
        search = GridSearchCV(make_pipeline(StandardScaler(), Ridge()), {'ridge__alpha': [1, 10, 100, 1000]}, scoring='neg_mean_squared_error', cv=folds, error_score='raise')
        search.fit(x[:, columns], y)
        model = search.best_estimator_
        bundle['models'][name] = {'estimator': model, 'columns': columns}
        scaler, ridge = model.named_steps['standardscaler'], model.named_steps['ridge']
        report['models'][name] = {'columns': columns, 'alpha': float(ridge.alpha), 'cv_mse': (-search.cv_results_['mean_test_score']).tolist(), 'scaler_mean': scaler.mean_.tolist(), 'scaler_scale': scaler.scale_.tolist(), 'coef': ridge.coef_.tolist(), 'intercept': float(ridge.intercept_)}
    args.output.mkdir(parents=True)
    joblib.dump(bundle, args.output/'models.joblib')
    report['models_sha256'] = sha(args.output/'models.joblib')
    (args.output/'fit.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({'n': len(y), 'positive': int(sum(y>0)), 'nonpositive': int(sum(y<=0)), 'models_sha256': report['models_sha256'], 'selected_alpha': {k:v.get('alpha') for k,v in report['models'].items()}}))


if __name__ == '__main__':
    main()
