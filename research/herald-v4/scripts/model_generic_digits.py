"""Frozen study041 discovery fit and locked-evaluation arithmetic."""
import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import sklearn
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('mode', choices=['fit', 'evaluate'])
    p.add_argument('--results', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--models', type=Path)
    a = p.parse_args()
    assert not a.output.exists(), 'preserve existing output'
    run = json.loads((a.results/'run.json').read_text())
    assert run['status'] == 'completed' and not run['failures']
    records = [json.loads((a.results/r['path']).read_text()) for r in run['prompts']]
    assert len(records) == 48 and all(r['status']=='completed' and all(r['checks'].values()) for r in records)
    split = 'discovery' if a.mode=='fit' else 'evaluation'
    assert all(r['manifest_row']['provenance']['split']==split for r in records)
    x = np.asarray([r['observation']['features'] for r in records], dtype=float)
    y = np.asarray([r['signed_loss'] for r in records], dtype=float)
    groups = [r['manifest_row']['group_id'] for r in records]
    assert x.shape==(48,5) and np.isfinite(x).all() and np.isfinite(y).all()
    assert len(set(groups))==48
    counts = {'positive':int(sum(y>0)), 'nonpositive':int(sum(y<=0))}
    varied = counts['positive']>=6 and counts['nonpositive']>=6
    report = {'source_sha256':sha(__file__), 'run_sha256':sha(a.results/'run.json'), 'records_sha256':{r['path']:sha(a.results/r['path']) for r in run['prompts']}, 'ids':[r['manifest_row']['id'] for r in records], 'groups':groups, 'features':x.tolist(), 'signed_losses':y.tolist(), 'counts':counts, 'variation_gate':varied, 'sklearn_version':sklearn.__version__}
    a.output.mkdir(parents=True)
    if a.mode=='fit':
        if not varied:
            report['status']='closed_insufficient_discovery_variation'
        else:
            bundle={'feature_count':5,'sklearn_version':sklearn.__version__, 'discovery_groups':groups,'models':{'mean':{'constant':float(y.mean())}}}
            for name,columns in [('candidate',[0]),('structural',[3,4]),('reference_nll',[1]),('js',[2]),('joint_baseline',[1,2,3,4])]:
                estimator=make_pipeline(StandardScaler(),Ridge(alpha=1.0,fit_intercept=True)).fit(x[:,columns],y)
                bundle['models'][name]={'estimator':estimator,'columns':columns}
            joblib.dump(bundle,a.output/'models.joblib')
            report['models_sha256']=sha(a.output/'models.joblib')
            report['models']={}
            for name,item in bundle['models'].items():
                if 'constant' in item:
                    report['models'][name]=item
                else:
                    scaler,ridge=item['estimator'].named_steps.values()
                    report['models'][name]={'columns':item['columns'],'mean':scaler.mean_.tolist(),'scale':scaler.scale_.tolist(),'coef':ridge.coef_.tolist(),'intercept':float(ridge.intercept_),'alpha':1.0}
            report['status']='fitted'
    else:
        assert a.models and sha(a.models)==run['models_sha256']
        bundle=joblib.load(a.models)
        assert bundle['sklearn_version']==sklearn.__version__ and not(set(groups)&set(bundle['discovery_groups']))
        predictions={name:np.asarray([r['observation']['predictions'][name] for r in records]) for name in bundle['models']}
        for name,item in bundle['models'].items():
            expected=np.full(48,item['constant']) if 'constant' in item else item['estimator'].predict(x[:,item['columns']])
            np.testing.assert_allclose(predictions[name],expected,atol=1e-12,rtol=1e-12)
        errors={name:(pred-y)**2 for name,pred in predictions.items()}
        report['models_sha256']=sha(a.models)
        report['predictions']={k:v.tolist() for k,v in predictions.items()}
        report['metrics']={name:{'mse':float(e.mean()),'mae':float(np.abs(predictions[name]-y).mean()),'bias':float((predictions[name]-y).mean())} for name,e in errors.items()}
        report['raw_z_auc']=float(roc_auc_score(y>0,x[:,0])) if len(set(y>0))==2 else None
        rng=np.random.default_rng(2026090643)
        indices=rng.integers(0,48,size=(10000,48))
        report['comparisons']={}
        for name,e in errors.items():
            if name=='candidate':continue
            gain=1-float(errors['candidate'].mean()/e.mean()) if e.mean()>0 else None
            wins=int(sum(errors['candidate']<e))
            denom=e[indices].mean(axis=1)
            valid=denom>0
            bootstrap=1-errors['candidate'][indices].mean(axis=1)[valid]/denom[valid]
            report['comparisons'][name]={'mse_gain':gain,'strict_prompt_wins':wins,'paired_bootstrap_gain_95':np.quantile(bootstrap,[.025,.975]).tolist() if len(bootstrap) else None,'bootstrap_zero_denominator_draws':int(sum(~valid)),'passes':bool(gain is not None and gain>=.10 and wins>=32)}
        report['passes']=bool(varied and report['raw_z_auc'] is not None and report['raw_z_auc']>=.80 and all(c['passes'] for c in report['comparisons'].values()))
        report['status']='evaluated'
    (a.output/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ['status','counts','variation_gate','passes','raw_z_auc','comparisons','models_sha256'] if k in report}))


if __name__=='__main__':
    main()
