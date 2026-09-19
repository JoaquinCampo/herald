"""Independent streaming cross-check of the GPU follow-up's main findings.
Python + NumPy only. No fitting, model imports, or forward passes.
Use --package on the directory formed by extracting all three supplied ZIPs.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def argmax(o):
    return min(i for i,z in zip(o['ids'],o['logits']) if z==max(o['logits']))


def main(package, output):
    if output.exists(): raise FileExistsError('Refusing to overwrite output')
    gpu=package/'docs/_why/artifacts/deep-20260918-gpu'
    run=gpu/'demand-all-v2';onsets={};errors=[];future={};unique=set()
    n=noop=hashes=odds=0;max_odds=0.;preserved_n=preserved_correct=0
    with (run/'trace.jsonl').open() as stream:
        for line in stream:
            r=json.loads(line)
            if 'residual' not in r:continue
            n+=1;rid=r['id'];owner=r['prefix_owner'];off=r['offset']
            unique.add((rid,r['prefix_sha256'],r['t']))
            z={}
            for side in ['ref','comp']:
                p=run/r['residual'][side]['file']
                assert hashlib.sha256(p.read_bytes()).hexdigest()==r['residual'][side]['sha256']
                hashes+=1
                with np.load(p,allow_pickle=False) as a:z[side]={k:a[k] for k in a.files}
            rr,cc=z['ref'],z['comp'];target=int(cc['target_id'])
            if rid.endswith('knorm:0.0'):
                assert all(np.array_equal(rr[k],cc[k]) for k in rr);noop+=1
            candidates=[int(i) for i in cc['candidate_ids']]
            j=min((i for i,k in enumerate(candidates) if k!=target),key=lambda i:(cc['actual_margins'][i],candidates[i]))
            direction=(rr['target_weight']-rr['candidate_weights'][j])*rr['gamma']
            mr=rr['block_output']-rr['post_attention_add'];mc=cc['block_output']-cc['post_attention_add']
            da=((cc['post_attention_add']-cc['block_input'])-(rr['post_attention_add']-rr['block_input']))@direction/float(rr['analysis_rms'])
            dm=(mc-mr)@direction/float(rr['analysis_rms'])
            if owner=='reference' and r['native_damage_label'] and argmax(r['comp_output'])!=target:
                total_loss=float(cc['actual_margins'][j]-rr['actual_margins'][j])
                errors.append(dict(cell=f'{rid}/{owner}/d{off}',mlp_larger=abs(dm.sum())>abs(da.sum()),
                    ref_mlp_positive=mr.sum(0)@direction>0,comp_mlp_negative=mc.sum(0)@direction<0,
                    late_fraction=float(dm[25:].sum()/total_loss),reader_delta=float(da[22:24].sum()),
                    alpha=float(mc.sum(0)@mr.sum(0)/(mr.sum(0)@mr.sum(0)))))
            if owner=='reference' and not r['native_damage_label']:
                preserved_n+=1;preserved_correct+=argmax(r['comp_output'])==target
            layers=r['geometry']['layers'];heads=layers[23]['heads']
            if off==0 and (owner=='compressed' or rid.endswith('knorm:0.0')):
                a=np.array([h['mass']['RR']['last2'] for h in heads]);b=np.array([h['mass']['RS']['last2'] for h in heads]);c=np.array([h['mass']['CC']['last2'] for h in heads])
                onsets[rid]=dict(damage=r['native_damage_label'],ratio=float(c.sum()/a.sum()),
                    group_zero=bool(b[14:21].sum()==0),reference_share=float(a[14:21].sum()/a.sum()),
                    lead=None if r['original_native_event'] is None else r['original_native_event']-r['t'])
            if owner=='reference' and off in [0,6]:
                v=layers[23]['fixed_final_readout_projections_not_causal']['deletion']
                future.setdefault(rid,{})[off]=dict(target=target,contrasts={k:(v[candidates.index(13)]-v[candidates.index(k)])/float(rr['analysis_rms']) for k in range(15,25)})
            for l in layers:
                for h in l['heads']:
                    for p in h['source_pair_log_odds']:
                        if 'reference' not in p:continue
                        odds+=1;assert p['deletion_only']==p['reference']
                        max_odds=max(max_odds,abs(p['compressed']-p['reference']-p['query_predicted_delta']))
    severe={k:v for k,v in onsets.items() if '/knorm:' in k and not k.endswith(':0.0') and v['ratio']<.2}
    signs=[]
    for k in severe:
        last=future[k][6]['target'];signs.append(future[k][0]['contrasts'][last]>0 and future[k][6]['contrasts'][last]<0)
    conditions=[]
    with (gpu/'coordinate-audit-v1.jsonl').open() as f:
        for line in f:
            r=json.loads(line)
            if 'condition' in r and r['id']=='113-competitor-v1-02-planted/knorm:0.1':conditions.append(r)
    def margin(o):return o['digit_logits']['2']-o['digit_logits']['7']
    base=margin(conditions[0]['baseline']);v={r['condition']:margin(r['corrected']['output']) for r in conditions}
    result=dict(comparisons=n,npz_hashes=hashes,noop=noop,unique_contexts=len(unique),
        severe_tail=len(severe),severe_damaged=sum(x['damage'] for x in severe.values()),
        severe_group_zero=sum(x['group_zero'] for x in severe.values()),
        severe_preserved_ids=[k for k,x in severe.items() if not x['damage']],
        recorded_projection_sign_reversal=sum(signs),wrong_reference_prefix_cells=len(errors),
        MLP_larger_than_attention=sum(bool(x['mlp_larger']) for x in errors),
        ref_MLP_positive=sum(bool(x['ref_mlp_positive']) for x in errors),
        comp_MLP_negative=sum(bool(x['comp_mlp_negative']) for x in errors),
        late_MLP_fraction_range=[min(x['late_fraction'] for x in errors),max(x['late_fraction'] for x in errors)],
        late_MLP_fraction_median=float(np.median([x['late_fraction'] for x in errors])),
        MLP_positive_alignment=sum(x['alpha']>0 for x in errors),preserved_reference_correct=[preserved_correct,preserved_n],
        source_odds_checks=odds,max_odds_identity_error=max_odds,coordinate_factorial=v['both']-v['needle']-v['dist']+base,
        scope='8 prompts, 4 families; repeated arms. FP32 bridge signs lack per-contrast native error bounds. Stored-update accounting is not causal attribution.')
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--package',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();main(a.package.resolve(),a.out.resolve())
