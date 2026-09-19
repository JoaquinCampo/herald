"""Independent compact cross-check of the mediation text packet; stdlib only.
No model imports, fitting or forwards. Uses a fixed original-error competitor.
Run: python mediation_packet_core.py --packet HERALD_MEDIATION_TEXT_PACKET.md
"""
import argparse
import csv
import hashlib
import io
import json
import re
import struct
from pathlib import Path

EXPECTED = 'e57e53661e9e35196ca59cc669d2b9943959efddde2ac5e3d86e0d38831a7be7'


def f32(x):
    return struct.unpack('f', struct.pack('f', float(x)))[0]


def run(path):
    raw=Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest()!=EXPECTED:
        raise ValueError('Unexpected input packet; reconcile source versions explicitly')
    ds=ps=cs=None
    for kind,text in re.findall(r'^````([^\n]*)\n(.*?)^````\s*$',raw.decode(),re.M|re.S):
        if kind!='csv':continue
        rows=list(csv.DictReader(io.StringIO(text)))
        header=text.splitlines()[0]
        if header.startswith('arm,offset,t,'):ds=rows
        elif header.startswith('arm,offset,condition,candidate_ids'):ps=rows
        elif header.startswith('arm,offset,condition,recipient,token_ids'):cs=rows
    if any(x is None for x in [ds,ps,cs]):raise ValueError('Missing complete evidence tables')
    def key(r):return r['arm'],int(r['offset']),r['condition']
    di={key(r):r for r in ds};pi={key(r):r for r in ps}
    assert (len(ds),len(di),len(ps),len(pi),len(cs))==(450,450,390,390,75)
    logits={k:{int(j):f32(v) for j,v in json.loads(r['candidate_logits']).items()} for k,r in di.items()}
    for k,r in di.items():
        z=logits[k];y=int(r['target_id']);other=min((j for j in z if j!=y),key=lambda j:(-z[j],j))
        assert int(r['selected_id'])==min(z,key=lambda j:(-z[j],j))
        assert int(r['strongest_nontarget_id'])==other
        assert z[y]-z[other]==f32(r['target_margin'])
    def margin(a,o,c,j):
        r=di[a,o,c];z=logits[a,o,c]
        return z[int(r['target_id'])]-z[j]
    first=[];native=[];self_n=0
    for (a,o,c),r in di.items():
        if c!='baseline_compressed':continue
        for name in ['self_heads','self_mlp']:
            assert {k:v for k,v in r.items() if k!='condition'}=={k:v for k,v in di[a,o,name].items() if k!='condition'}
            self_n+=1
        w=int(r['selected_id']);z=logits[a,o,c];v=min((j for j in z if j!=w),key=lambda j:(-z[j],j))
        p=pi[a,o,'self_mlp'];ix={j:i for i,j in enumerate(json.loads(p['candidate_ids']))}
        votes=[]
        for l in [25,26,27]:
            u=[f32(x) for x in json.loads(p['MLP'+str(l)])]
            votes.append(u[ix[v]]-u[ix[w]])
        native.append({'id':a,'offset':o,'lead':None if r['lead']=='null' else int(r['lead']),
                       'damaged':r['native_damaged']=='true','winner':w,'runner_up':v,'late_vote':sum(votes)})
        if r['lead']!='0':continue
        m={name:margin(a,o,name,w) for name in ['baseline_compressed','baseline_reference','ref_L22G0','ref_L23G2','ref_both_heads','ref_heads_frozen_comp_MLPs','comp_heads_into_ref','comp_MLPs_into_ref']}
        first.append({'id':a,'offset':o,'original_competitor':w,'margins':m,
                      'reader_gain':m['ref_both_heads']-m['baseline_compressed'],
                      'clamped_gain':m['ref_heads_frozen_comp_MLPs']-m['baseline_compressed'],
                      'unfreeze_gain':m['ref_both_heads']-m['ref_heads_frozen_comp_MLPs'],
                      'head_interaction':m['ref_both_heads']-m['ref_L22G0']-m['ref_L23G2']+m['baseline_compressed'],
                      'recipient_interaction':m['ref_both_heads']-m['baseline_compressed']-m['baseline_reference']+m['comp_heads_into_ref']})
    assert self_n==60 and len(first)==5
    truths={'00':'3705852','01':'4860155','02':'6109204','03':'6024061'}
    first_count=any_count=0;sequences=set();metric_exceptions=[]
    for r in cs:
        ids=json.loads(r['token_ids']);sequences.add(tuple(ids));digits=''.join(str(i-15) if 15<=i<=24 else ' ' for i in ids)
        runs=digits.split();truth=truths[r['arm'][:2]];f=bool(runs and runs[0]==truth);a=truth in runs
        assert f==(r['first_number_exact']=='true') and a==bool(float(r['exact_truth_score']))
        decision=di[key(r)];assert f==(decision['selected_id']==decision['target_id'])
        first_count+=f;any_count+=a
        if f!=a:metric_exceptions.append('/'.join(map(str,key(r))))
    hits={}
    for r in native:
        if r['late_vote']<0 and r['id'] not in hits:hits[r['id']]=r
    return {'packet_sha256':EXPECTED,'decision_records':len(ds),'patch_projections':len(ps),
            'self_summary_identities':self_n,'first_errors':first,'native_conflict_hits':hits,
            'continuations':len(cs),'first_correct':first_count,'any_correct':any_count,
            'unique_complete_sequences':len(sequences),'utility_exceptions':metric_exceptions,
            'scope':'Eight prompts/four families; five error cells. Text arithmetic cross-check, not native GPU or full-cache verification.'}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--packet',type=Path,required=True)
    args=p.parse_args();print(json.dumps(run(args.packet),indent=2,sort_keys=True))
