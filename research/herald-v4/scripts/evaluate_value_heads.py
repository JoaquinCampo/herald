"""Evaluate immutable design035 predictions against all paired outcomes."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, roc_auc_score


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--results', type=Path, required=True)
    p.add_argument('--fit', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    assert not args.output.exists()
    run = json.loads((args.results/'run.json').read_text())
    fit = json.loads(args.fit.read_text())
    assert run['status'] == 'completed' and not run['failures']
    assert run['models_sha256'] == fit['models_sha256']
    rows = [json.loads((args.results/r['path']).read_text()) for r in run['prompts']]
    assert len(rows) == 48 and all(r['status']=='completed' and all(r['checks'].values()) for r in rows)
    assert all(r['manifest_row']['provenance']['split']=='evaluation' for r in rows)
    assert not set(fit['groups']) & {r['manifest_row']['group_id'] for r in rows}
    y = np.array([r['signed_loss'] for r in rows])
    x = np.array([r['observation']['features'] for r in rows])
    for r, entry in zip(rows, run['prompts'], strict=True):
        f = args.results/(Path(entry['path']).stem+'.features.json')
        assert sha(f)==r['feature_sha256'] and json.loads(f.read_text())==r['observation']
    preds = {name: np.array([r['observation']['predictions'][name] for r in rows]) for name in fit['models']}
    for name, model in fit['models'].items():
        independent = np.full(48, model['constant']) if 'constant' in model else ((x[:,model['columns']]-model['scaler_mean'])/model['scaler_scale'])@np.asarray(model['coef'])+model['intercept']
        np.testing.assert_allclose(preds[name], independent, rtol=1e-10, atol=1e-10)
    variation = int(sum(y>0))>=6 and int(sum(y<=0))>=6
    report = {'run_sha256': sha(args.results/'run.json'), 'fit_sha256': sha(args.fit), 'source_sha256': sha(__file__), 'n':48, 'positive':int(sum(y>0)), 'zero':int(sum(y==0)), 'negative':int(sum(y<0)), 'label_variation_gate':variation, 'ids':[r['manifest_row']['id'] for r in rows], 'signed_losses':y.tolist(), 'predictions':{k:v.tolist() for k,v in preds.items()}, 'metrics':{}, 'comparisons':{}, 'adapter_misses':sum(not r['observation']['located']['found'] for r in rows)}
    for name, pred in preds.items():
        report['metrics'][name] = {'mse':mean_squared_error(y,pred), 'mae':mean_absolute_error(y,pred), 'signed_bias':float(np.mean(pred-y)), 'auc':roc_auc_score(y>0,pred) if len(set(y>0))==2 else None}
    squared = {k:(v-y)**2 for k,v in preds.items()}
    rng = np.random.default_rng(2026090635)
    sampled = rng.integers(0,48,size=(10000,48))
    for name in ('mean','metadata','mean_retention'):
        gain = 1-squared['heads'].mean()/squared[name].mean() if squared[name].mean()>0 else None
        wins = int(sum(squared['heads']<squared[name]))
        delta = (squared[name]-squared['heads'])[sampled].mean(axis=1)
        report['comparisons'][name]={'relative_mse_gain':gain, 'prompt_wins':wins, 'mse_improvement_bootstrap95':np.quantile(delta,[.025,.975]).tolist(), 'gate':bool(gain is not None and gain>=.10 and wins>=32)}
    report['cost_medians_seconds']={k:float(np.median([r['observation']['cost'][k] for r in rows])) for k in rows[0]['observation']['cost']}
    report['all_development_gates_pass']=bool(variation and all(v['gate'] for v in report['comparisons'].values()) and (report['metrics']['heads']['auc'] or 0)>=.8)
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k in ('positive','zero','negative','metrics','comparisons','all_development_gates_pass')}))


if __name__=='__main__':
    main()
