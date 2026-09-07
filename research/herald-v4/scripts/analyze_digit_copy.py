"""Frozen038 exploratory gate, no predictor fitting or feature selection."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--results',type=Path,nargs='+',required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    assert not args.output.exists()
    records=[]; hashes={}
    for directory in args.results:
        run=json.loads((directory/'run.json').read_text())
        assert run['status']=='completed' and not run['failures']
        for entry in run['prompts']:
            path=directory/entry['path'];r=json.loads(path.read_text())
            assert r['status']=='completed'
            obs=r['observation']
            assert all(v for v in obs['checks'].values() if v is not None)
            feature_path=path.with_name(path.stem+'.features.json')
            assert sha(feature_path)==r['features_sha256'] and json.loads(feature_path.read_text())==obs
            for arm,zkey in [('correct','z'),('control','z_control')]:
                ref=np.array(obs[arm]['reference']['logprobs']);act=np.array(obs[arm]['action']['logprobs'])
                assert ref.shape==act.shape==(7,) and np.isfinite(ref).all() and np.isfinite(act).all()
                np.testing.assert_allclose(obs[zkey],np.mean(ref-act),atol=1e-12,rtol=1e-12)
            np.testing.assert_allclose(obs['reference_nll'],-np.mean(obs['correct']['reference']['logprobs']),atol=1e-12,rtol=1e-12)
            records.append(r);hashes[str(path)]=sha(path)
    expected={f'value-head-v1-discovery-{i:03}' for i in (0,1,3,5,6,7,2,4,9,12,32,36)}
    assert len(records)==12 and {r['manifest_row']['id'] for r in records}==expected
    y=np.array([r['signed_loss'] for r in records]);assert sum(y>0)==sum(y<=0)==6
    auc={k:roc_auc_score(y>0,[r['observation'][k] for r in records]) for k in ('z','z_control','reference_nll')}
    report={'scope':'exploratory outcome-stratified exposed12 mechanism slice, no predictive validation','source_sha256':sha(__file__),'record_sha256':hashes,'n':12,'auc':auc,'understanding_gate':bool(auc['z']>=.8 and auc['z']-auc['z_control']>=.15 and auc['z']>auc['reference_nll']),'cases':[{'id':r['manifest_row']['id'],'signed_loss':r['signed_loss'],**{k:r['observation'][k] for k in ('z','z_control','reference_nll','correct','control','cost')}} for r in records]}
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ('scope','n','auc','understanding_gate')}))


if __name__=='__main__':
    main()
