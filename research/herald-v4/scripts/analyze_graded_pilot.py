"""Score the frozen graded population pilot without fitting a predictor."""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
import numpy as np
from score_ruler_pilot import score_prediction

p=argparse.ArgumentParser()
p.add_argument('--results',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
assert not a.output.exists()
run=json.loads((a.results/'run.json').read_text())
assert run['status']=='completed' and not run['failures'] and len(run['prompts'])==16
rows=[]
for item in run['prompts']:
 r=json.loads((a.results/item['path']).read_text())
 assert r['status']=='completed' and all(r['checks'].values())
 answers=r['manifest_row']['answers']; assert len(set(answers))==4
 ref=r['reference']; assert ref['reference_path']=='shared_boundary'
 arms={float(v['action']['removal_fraction']):v for v in r['arms'].values()}
 assert set(arms)=={0.,.05}
 noop=arms[0.]['continuation']; action=arms[.05]['continuation']
 assert ref['token_ids']==noop['token_ids'] and ref['termination_reason']==noop['termination_reason']
 assert arms[.05]['compression']['physical_effect_exact'] and arms[.05]['source_cache_unchanged']
 scores={n:score_prediction(v['text'],answers) for n,v in [('reference',ref),('noop',noop),('action',action)]}
 loss=scores['reference']['score_fraction']-scores['action']['score_fraction']
 rows.append({'id':r['manifest_row']['id'],'scores':scores,'signed_loss':loss,'termination':{n:v['termination_reason'] for n,v in [('reference',ref),('action',action)]},'record_sha256':hashlib.sha256((a.results/item['path']).read_bytes()).hexdigest()})
y=np.array([r['signed_loss'] for r in rows]); q=np.array([r['scores']['reference']['score_fraction'] for r in rows])
checks={'mean_reference_at_least_090':bool(q.mean()>=.9),'at_least_12_perfect_references':bool(sum(q==1)>=12),'at_least_four_partial_losses':bool(sum((y>0)&(y<1))>=4),'at_least_three_loss_levels':len(set(y))>=3,'sample_std_at_least_010':bool(y.std(ddof=1)>=.1)}
out={'n':16,'reference_mean':float(q.mean()),'perfect_references':int(sum(q==1)),'partial_positive_losses':int(sum((y>0)&(y<1))),'loss_distribution':dict(Counter(map(str,y))),'mean_signed_loss':float(y.mean()),'sample_std':float(y.std(ddof=1)),'viability_checks':checks,'viable':all(checks.values()),'rows':rows,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
a.output.write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps({k:v for k,v in out.items() if k!='rows'}))
