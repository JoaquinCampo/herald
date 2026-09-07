"""Direct owner recomputation after the independent reviewer hit its usage limit."""
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.special import logsumexp
from sklearn.metrics import roc_auc_score
from score_ruler_pilot import score_prediction

root=Path(__file__).resolve().parents[1]
result=root/'results/generic-digit-evaluation'
run=json.loads((result/'run.json').read_text())
summary=json.loads((root/'results/generic-digit-evaluation-summary/summary.json').read_text())
fit=json.loads((root/'results/generic-digit-fit/summary.json').read_text())
assert run['status']=='completed' and len(run['prompts'])==48 and not run['failures']
y=[]; x=[]; predictions=[]
for item in run['prompts']:
 p=result/item['path'];r=json.loads(p.read_text());o=r['observation']
 assert all(r['checks'].values())
 f=p.with_name(p.stem+'.features.json');v=p.with_name(p.stem+'.vectors.npz')
 assert hashlib.sha256(f.read_bytes()).hexdigest()==r['features_sha256']
 assert hashlib.sha256(v.read_bytes()).hexdigest()==r['vectors_sha256']
 vec=np.load(v)
 z=float(np.mean(vec['reference_digit_logprobs']-vec['action_digit_logprobs']))
 nll=float(-np.mean(vec['reference_digit_logprobs']))
 a=np.asarray(vec['pending_reference_logits'],dtype=float);b=np.asarray(vec['pending_action_logits'],dtype=float)
 la=a-logsumexp(a);lb=b-logsumexp(b);lm=np.logaddexp(la,lb)-np.log(2)
 js=float(.5*(np.sum(np.exp(la)*(la-lm))+np.sum(np.exp(lb)*(lb-lm))))
 np.testing.assert_allclose(o['features'][:3],[z,nll,js],atol=2e-6,rtol=1e-6)
 scores={}
 for name,arm in r['branches'].items():
  q=score_prediction(arm['continuation']['text'],r['manifest_row']['answers'])
  assert q==arm['score'];scores[name]=q['score_fraction']
 assert scores['reference']-scores['action']==r['signed_loss']
 y.append(r['signed_loss']);x.append(o['features']);predictions.append(o['predictions'])
x=np.array(x);y=np.array(y);errs={}
for name,m in fit['models'].items():
 pred=np.full(48,m['constant']) if 'constant' in m else ((x[:,m['columns']]-m['mean'])/m['scale'])@np.array(m['coef'])+m['intercept']
 np.testing.assert_allclose(pred,[p[name] for p in predictions],atol=1e-12,rtol=1e-12)
 errs[name]=(pred-y)**2
 np.testing.assert_allclose(errs[name].mean(),summary['metrics'][name]['mse'],atol=1e-14)
for name,c in summary['comparisons'].items():
 gain=1-errs['candidate'].mean()/errs[name].mean();wins=int(sum(errs['candidate']<errs[name]))
 np.testing.assert_allclose(gain,c['mse_gain'],atol=1e-12)
 assert wins==c['strict_prompt_wins'] and bool(gain>=.1 and wins>=32)==c['passes']
auc=roc_auc_score(y>0,x[:,0]);assert auc==summary['raw_z_auc'] and not summary['passes']
out={'reviewer':'owner_direct_recomputation_not_independent_agent','records':48,'official_scores_checked':144,'feature_vectors_checked':48,'frozen_model_predictions_checked':288,'auc':float(auc),'all_checks_pass':True,'conclusion':'fixed gates failed','source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
(root/'results/generic-digit-evaluation-owner-audit.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
