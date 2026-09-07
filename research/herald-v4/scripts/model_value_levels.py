"""Frozen value-level signed Ridge fit and prompt-level evaluation for045."""
import argparse
import hashlib
import json
from pathlib import Path
import joblib
import numpy as np
import sklearn
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def concordance(y, prediction):
    i,j=np.triu_indices(4,1)
    truth=y[:,i]-y[:,j]; guessed=prediction[:,i]-prediction[:,j]
    mixed=(truth!=0).any(axis=1)
    per_prompt=[]
    for a,b in zip(truth[mixed],guessed[mixed],strict=True):
        valid=a!=0
        per_prompt.append(float(np.mean((a[valid]*b[valid]>0)+.5*(b[valid]==0))))
    return float(np.mean(per_prompt)) if per_prompt else None


def main():
    p=argparse.ArgumentParser()
    p.add_argument('mode',choices=['fit','evaluate'])
    p.add_argument('--results',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--models',type=Path)
    a=p.parse_args();assert not a.output.exists()
    run=json.loads((a.results/'run.json').read_text())
    assert run['status']=='completed' and not run['failures']
    records=[json.loads((a.results/item['path']).read_text()) for item in run['prompts']]
    count=128 if a.mode=='fit' else 64
    split='discovery' if a.mode=='fit' else 'evaluation'
    assert len(records)==count and all(r['status']=='completed' and all(r['checks'].values()) for r in records)
    assert all(r['manifest_row']['provenance']['split']==split for r in records)
    groups=[r['manifest_row']['group_id'] for r in records]
    assert len(set(groups))==count
    x=np.asarray([r['observation']['features'] for r in records],dtype=float)
    y=np.asarray([r['per_value_signed_losses'] for r in records],dtype=float)
    assert x.shape==(count,4,118) and y.shape==(count,4) and np.isfinite(x).all() and np.isfinite(y).all()
    np.testing.assert_allclose(y.mean(axis=1),[r['signed_loss'] for r in records],atol=1e-14)
    mixed=(y.max(axis=1)!=y.min(axis=1)); variation=int(mixed.sum()) >= (24 if a.mode=='fit' else 12)
    report={'status':'pending','n':count,'mixed_prompts':int(mixed.sum()),'variation_gate':bool(variation),'groups':groups,'ids':[r['manifest_row']['id'] for r in records],'features':x.tolist(),'per_value_signed_losses':y.tolist(),'negative_contributions':int(sum(y.flatten()<0)),'source_sha256':sha(__file__),'run_sha256':sha(a.results/'run.json'),'record_sha256':{i['path']:sha(a.results/i['path']) for i in run['prompts']},'sklearn_version':sklearn.__version__}
    report['reference_quality']={'mean':float(np.mean([r['branches']['reference']['score']['score_fraction'] for r in records])),'perfect_prompts':sum(r['branches']['reference']['score']['score_fraction']==1. for r in records)}
    levels,counts=np.unique(y.mean(axis=1),return_counts=True)
    report['task_loss_distribution']={str(float(k)):int(v) for k,v in zip(levels,counts,strict=True)}
    a.output.mkdir(parents=True)
    if a.mode=='fit':
        if not variation:
            report['status']='closed_insufficient_mixed_discovery'
        else:
            flat=x.reshape(-1,118);target=y.flatten();weights=np.full(len(target),.25)
            bundle={'feature_count':118,'sklearn_version':sklearn.__version__,'discovery_groups':groups,'models':{'mean':{'constant':float(target.mean())}}}
            report['models']={'mean':bundle['models']['mean']}
            for name,columns in [('structural',list(range(4))),('aggregate',list(range(6))),('candidate',list(range(118))),('aggregate_ols',list(range(6)))]:
                regressor=LinearRegression() if name=='aggregate_ols' else Ridge(alpha=1000,fit_intercept=True)
                estimator=make_pipeline(StandardScaler(),regressor)
                fit_key='linearregression__sample_weight' if name=='aggregate_ols' else 'ridge__sample_weight'
                estimator.fit(flat[:,columns],target,**{fit_key:weights})
                bundle['models'][name]={'estimator':estimator,'columns':columns}
                scale,reg=estimator.named_steps.values()
                report['models'][name]={'columns':columns,'scale':scale.scale_.tolist(),'mean':scale.mean_.tolist(),'coef':reg.coef_.tolist(),'intercept':float(reg.intercept_),'alpha':None if name=='aggregate_ols' else 1000,'row_weight':.25}
            joblib.dump(bundle,a.output/'models.joblib')
            report.update(status='fitted',models_sha256=sha(a.output/'models.joblib'))
    else:
        assert a.models and sha(a.models)==run['models_sha256']
        bundle=joblib.load(a.models)
        assert bundle['sklearn_version']==sklearn.__version__ and not(set(groups)&set(bundle['discovery_groups']))
        predictions={};errors={};task_y=y.mean(axis=1);metrics={}
        for name,item in bundle['models'].items():
            predicted=np.asarray([r['observation']['predictions'][name] for r in records],dtype=float)
            assert predicted.shape==(64,4)
            expected=np.full((64,4),item['constant']) if 'constant' in item else item['estimator'].predict(x.reshape(-1,118)[:,item['columns']]).reshape(64,4)
            np.testing.assert_allclose(predicted,expected,atol=1e-12,rtol=1e-12)
            task_pred=predicted.mean(axis=1);errors[name]=(task_pred-task_y)**2
            predictions[name]=predicted.tolist()
            metrics[name]={'mse':float(errors[name].mean()),'mae':float(np.abs(task_pred-task_y).mean()),'bias':float((task_pred-task_y).mean()),'within_prompt_concordance':concordance(y,predicted)}
        indices=np.random.default_rng(2026090647).integers(0,64,size=(10000,64)); comparisons={}
        for name,e in errors.items():
            if name=='candidate':continue
            gain=float(1-errors['candidate'].mean()/e.mean()) if e.mean()>0 else None
            wins=int(sum(errors['candidate']<e)); denom=e[indices].mean(axis=1);valid=denom>0
            draws=1-errors['candidate'][indices].mean(axis=1)[valid]/denom[valid]
            comparisons[name]={'mse_gain':gain,'strict_prompt_wins':wins,'paired_bootstrap_gain_95':np.quantile(draws,[.025,.975]).tolist() if len(draws) else None,'zero_denominator_draws':int(sum(~valid)),'passes':bool(gain is not None and gain>=.1 and wins>=43)}
        candidate=metrics['candidate']['within_prompt_concordance'];base=metrics['structural']['within_prompt_concordance']
        rank_gate=bool(candidate is not None and base is not None and candidate>=.65 and candidate-base>=.1)
        report.update(status='evaluated',models_sha256=sha(a.models),predictions=predictions,metrics=metrics,comparisons=comparisons,concordance_gate=rank_gate,passes=bool(variation and rank_gate and all(c['passes'] for c in comparisons.values())))
    (a.output/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ['status','n','mixed_prompts','variation_gate','models_sha256','passes','metrics','comparisons','concordance_gate'] if k in report}))


if __name__=='__main__':
    main()
