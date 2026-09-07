#!/usr/bin/env python3
"""Independent artifact audit for the frozen 045/045a value-level pipeline."""
import argparse, hashlib, json, math
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
EXPECTED={
"collect_value_levels.py":"971fba0d6e771ef3397dd67045f2742a30b85068cdf06dcfd01f4f2f4e44121b",
"value_group_adapter.py":"6b2e329f1ed51c7878c0ad58d3fbbcbf2797e3445c07b6c2f0f2a256612bfd43",
"diagnose_needle_rescue.py":"ace1754ac20955b5e292c76cb557c5abdee388bfb64810a32002131af77ede2a",
"run_pair_pilot.py":"27045a21b32139ab870bcfb379a6c62f66b28242f1acc302274a8fb42b0d7941",
"score_ruler_pilot.py":"73970846e186d1996e42b5bdc4875f44f6806eb85ea8c623dad29711335c826c",
"engine.py":"4071a4fcdb2bae4d3d7f3f05b1560d54dd2f2143d675e4e2e79a912ae67dbb52",
"model_value_levels.py":"43d0d8c8584247bc1909367f50000f18ccae20e07a7d9a19a13308f65c737f27",
}
def sha(p):
 h=hashlib.sha256()
 with open(p,"rb") as f:
  for b in iter(lambda:f.read(1<<20),b""): h.update(b)
 return h.hexdigest()
def close(a,b,t=1e-10): return bool(np.allclose(np.asarray(a,float),np.asarray(b,float),atol=t,rtol=t))
def load_batch(d):
 run=json.loads((d/"run.json").read_text())
 rows=[]
 for item in run["prompts"]:
  p=d/item["path"]; rows.append((p,json.loads(p.read_text())))
 return run,rows
def source_check(run):
 got={Path(k).name:v for k,v in run.get("source_hashes",{}).items()}
 ok=all(got.get(k)==v for k,v in EXPECTED.items() if k!="model_value_levels.py")
 current={k:sha(ROOT/"scripts"/k) for k in EXPECTED if k!="engine.py"}
 current["engine.py"]=sha(ROOT.parent/"herald-v3/src/herald_v3/engineering/engine.py")
 return ok and all(current[k]==v for k,v in EXPECTED.items()),got,current
def score_ok(branch,answers):
 s=branch["score"]; c=branch["continuation"]; post=s["postprocessed_prediction"]
 hits=[a.lower() in post.lower() for a in answers]
 return (s["raw_prediction"]==c["text"] and s["pass_vector"]==hits and
         math.isclose(s["score_fraction"],round(100*sum(hits)/len(hits),2)/100,abs_tol=1e-12))
def audit_batch(d, expected, allow_partial):
 run,items=load_batch(d); c={}
 c["run_completed"]=run["status"]=="completed" and not run["failures"]
 c["batch_complete"]=len(items)==expected and len(run["prompts"])==expected
 c["run_prompt_statuses"]=all(x["status"]=="completed" for x in run["prompts"])
 c["source_hashes_current"],got,current=source_check(run)
 recs=[]
 for path,r in items:
  recs.append(r)
  obs=r.get("observation",{}); ad=obs.get("adapter",{}); vals=ad.get("values",[])
  c[f"{r.get('manifest_row',{}).get('id','?')}:record_complete"]=r.get("status")=="completed" and all(v is True for v in r.get("checks",{}).values() if isinstance(v,bool))
  c[f"{r.get('manifest_row',{}).get('id','?')}:hashes"]=(
   path.with_name(path.stem+".features.json").is_file() and path.with_name(path.stem+".masks.npz").is_file() and
   sha(path.with_name(path.stem+".features.json"))==r.get("features_sha256") and
   sha(path.with_name(path.stem+".masks.npz"))==r.get("masks_sha256") and
   json.loads(path.with_name(path.stem+".features.json").read_text())==obs)
  c[f"{r.get('manifest_row',{}).get('id','?')}:adapter"]=(
   ad.get("found") is True and len(vals)==4 and len({v.get("value_text") for v in vals})==4 and
   all(v.get("occurrence")==i and len(v.get("value_positions",[]))==7 and
       v["value_positions"]==list(range(v["value_positions"][0],v["value_positions"][0]+7))
       for i,v in enumerate(vals)))
  try:
   with np.load(path.with_name(path.stem+".masks.npz")) as z: kept=z["kept"]
   heads=[set(map(int,h)) for h in kept.reshape(-1,kept.shape[-1])]
   pos=[set(v["value_positions"]) for v in vals]
   missing=[[len(q-h)/len(q) for h in heads] for q in pos]
   comp=obs["native_masks"]["action"]; removed=1-comp["after_bytes"]/comp["before_bytes"]
   xf=[[math.log(ad["prompt_length"]),i/3,v["position_midpoint_normalized"],removed,
        float(np.mean(missing[i])),float(np.mean(missing)),*missing[i]] for i,v in enumerate(vals)]
   c[f"{r['manifest_row']['id']}:mask_features"]=(
    kept.ndim==3 and len(heads)==obs["head_count"] and all(len(h)==kept.shape[-1] for h in heads) and
    np.asarray(obs["features"]).shape==(4,6+len(heads)) and close(obs["features"],xf,1e-12) and
    obs["feature_names"]==["log_prompt_length","occurrence_order","midpoint_normalized","global_removed_fraction","value_mean_missing","prompt_mean_missing"]+[f"head_missing_{i}" for i in range(len(heads))])
  except Exception: c[f"{r['manifest_row']['id']}:mask_features"]=False
  answers=r["manifest_row"]["answers"]; br=r.get("branches",{})
  c[f"{r['manifest_row']['id']}:branches"]=(
   set(br)=={"reference","noop","action"} and score_ok(br["reference"],answers) and score_ok(br["noop"],answers) and score_ok(br["action"],answers) and
   br["reference"]["continuation"]["token_ids"]==br["noop"]["continuation"]["token_ids"] and
   br["reference"]["continuation"]["termination_reason"]==br["noop"]["continuation"]["termination_reason"])
  per=r.get("per_value",[]); vals_text=[v["value_text"] for v in vals]
  ref=br.get("reference",{}).get("score",{}); act=br.get("action",{}).get("score",{})
  expected_losses=[float(v["reference_hit"])-float(v["action_hit"]) for v in per]
  recomputed=[float(x.lower() in br["reference"]["score"]["postprocessed_prediction"].lower())-float(x.lower() in br["action"]["score"]["postprocessed_prediction"].lower()) for x in vals_text]
  c[f"{r['manifest_row']['id']}:signed_losses"]=(
   [x["value"] for x in per]==vals_text and set(vals_text)==set(answers) and close(expected_losses,recomputed,1e-12) and
   math.isclose(np.mean(expected_losses),r["signed_loss"],abs_tol=1e-12) and
   math.isclose(r["signed_loss"],ref["score_fraction"]-act["score_fraction"],abs_tol=1e-12))
  c[f"{r['manifest_row']['id']}:controls"]=(
   r["checks"].get("boundary_source_equal") is True and r["checks"].get("boundary_source_disjoint") is True and
   r["checks"].get("source_unchanged") is True and r["checks"].get("features_precede_outcomes_unchanged") is True and
   r["checks"].get("candidate_mask_exact") is True and r["checks"].get("physical_effect_exact") is True and
   br["action"]["compression"]["after_bytes"]<br["action"]["compression"]["before_bytes"])
 X=np.asarray([r["observation"]["features"] for r in recs],float) if recs else np.empty((0,))
 Y=np.asarray([r["per_value_signed_losses"] for r in recs],float) if recs else np.empty((0,))
 c["aggregate_shapes"]=X.ndim==3 and X.shape[1]==4 and all(np.asarray(r["observation"]["features"]).shape[1]==6+r["observation"]["head_count"] for r in recs) and Y.shape==(len(recs),4) and np.isfinite(X).all() and np.isfinite(Y).all()
 return run,recs,c,{"source_hashes":got,"current_source_hashes":current,"n":len(recs),"mixed_prompts":int(np.sum(Y.max(1)!=Y.min(1))) if len(recs) else 0,"X":X,"Y":Y,"allow_partial":allow_partial}
def fit_check(summary, X,Y):
 m=summary.get("models",{}); out={}
 if summary.get("status")!="fitted" or not m: return {"fit_summary_skipped":True}
 flat=X.reshape(-1,118); y=Y.reshape(-1); w=np.full(len(y),.25)
 out["mean"]=math.isclose(m.get("mean",{}).get("constant",float("nan")),float(y.mean()),abs_tol=1e-12)
 for name in ("structural","aggregate","candidate","aggregate_ols"):
  if name not in m: out[name]=False; continue
  z=flat[:,m[name]["columns"]]; mean=z.mean(0); scale=np.where(z.std(0)==0,1,z.std(0)); zs=(z-mean)/scale
  if name=="aggregate_ols":
   q=np.linalg.lstsq(np.c_[zs,np.ones(len(y))]*np.sqrt(w)[:,None],y*np.sqrt(w),rcond=None)[0]; co,inter=q[:-1],q[-1]
  else:
   alpha=1000.; co=np.linalg.solve(zs.T@(w[:,None]*zs)+alpha*np.eye(zs.shape[1]),zs.T@(w*y)); inter=np.average(y,weights=w)
  out[name]=close(mean,m[name]["mean"]) and close(scale,m[name]["scale"]) and close(co,m[name]["coef"],1e-8) and math.isclose(inter,m[name]["intercept"],abs_tol=1e-8)
 return out
def concordance(y,p):
 i,j=np.triu_indices(4,1); t=y[:,i]-y[:,j]; q=p[:,i]-p[:,j]; vals=[]
 for a,b in zip(t,q):
  if np.any(a!=0): vals.append(np.mean((a[a!=0]*b[a!=0]>0)+.5*(b[a!=0]==0)))
 return float(np.mean(vals)) if vals else None
def prediction_check(fit,summary,X):
 out={}; flat=X.reshape(-1,118); pred=summary.get("predictions",{})
 for name,m in fit.get("models",{}).items():
  if name not in pred: out[name]=False; continue
  if "constant" in m: q=np.full((len(X),4),m["constant"])
  else:
   z=(flat[:,m["columns"]]-np.asarray(m["mean"]))/np.asarray(m["scale"])
   q=(z@np.asarray(m["coef"])+m["intercept"]).reshape(len(X),4)
  out[name]=close(q,pred[name],1e-10)
 return out
def eval_check(summary,X,Y):
 pred=summary.get("predictions",{}); out={}
 if summary.get("status")!="evaluated" or not pred: return {"eval_summary_skipped":True}
 ty=Y.mean(1); errs={}
 for name,v in pred.items():
  p=np.asarray(v,float); errs[name]=(p.mean(1)-ty)**2
  ms=summary["metrics"][name]; out[name]=(
   math.isclose(ms["mse"],errs[name].mean(),abs_tol=1e-10) and
   math.isclose(ms["mae"],np.abs(p.mean(1)-ty).mean(),abs_tol=1e-10) and
   math.isclose(ms["bias"],(p.mean(1)-ty).mean(),abs_tol=1e-10) and
   (ms["within_prompt_concordance"] is None or math.isclose(ms["within_prompt_concordance"],concordance(Y,p),abs_tol=1e-10)))
 comparison_passes=[]
 for name in ("mean","structural","aggregate","aggregate_ols"):
  e=errs.get(name); ce=errs.get("candidate")
  if e is None or ce is None: out["comparison_"+name]=False; continue
  gain=1-ce.mean()/e.mean() if e.mean()>0 else None; wins=int(np.sum(ce<e))
  got=summary["comparisons"][name]
  arithmetic=got["strict_prompt_wins"]==wins and (got["mse_gain"] is None if gain is None else math.isclose(got["mse_gain"],gain,abs_tol=1e-10))
  gate=gain is not None and gain>=.1 and wins>=43
  comparison_passes.append(gate); out["comparison_"+name]=arithmetic
 cand=summary["metrics"].get("candidate",{}).get("within_prompt_concordance")
 base=summary["metrics"].get("structural",{}).get("within_prompt_concordance")
 variation=int(np.sum(Y.max(1)!=Y.min(1)))>=12
 concordance_gate=cand is not None and base is not None and cand>=.65 and cand-base>=.1
 frozen_pass=variation and concordance_gate and all(comparison_passes)
 reported_comparisons=[summary.get("comparisons",{}).get(name,{}).get("passes") == gate for name,gate in zip(("mean","structural","aggregate","aggregate_ols"),comparison_passes)]
 out.update({"variation_gate":summary.get("variation_gate")==variation,
             "concordance_gate":summary.get("concordance_gate")==concordance_gate,
             "comparison_gates":len(reported_comparisons)==4 and all(reported_comparisons),
             "final_passes":summary.get("passes")==frozen_pass})
 return out
def main():
 p=argparse.ArgumentParser(); p.add_argument("--discovery-results",type=Path); p.add_argument("--evaluation-results",type=Path); p.add_argument("--fit-summary",type=Path); p.add_argument("--eval-summary",type=Path); p.add_argument("--output",type=Path,required=True); p.add_argument("--allow-partial",action="store_true")
 a=p.parse_args(); checks={}; details={}; batches=[]
 for d,n in ((a.discovery_results,128),(a.evaluation_results,64)):
  if d:
   run,recs,c,det=audit_batch(d,n,a.allow_partial); checks.update({("discovery:" if n==128 else "evaluation:")+k:v for k,v in c.items()}); details["discovery" if n==128 else "evaluation"]=det; batches.append((n,recs,det))
 if a.fit_summary and a.discovery_results:
  X=details["discovery"]["X"]; Y=details["discovery"]["Y"]; checks.update({"fit:"+k:v for k,v in fit_check(json.loads(a.fit_summary.read_text()),X,Y).items()})
 if a.eval_summary and a.evaluation_results:
  X=details["evaluation"]["X"]; Y=details["evaluation"]["Y"]; es=json.loads(a.eval_summary.read_text()); checks.update({"eval:"+k:v for k,v in eval_check(es,X,Y).items()})
  if a.fit_summary: checks.update({"eval_prediction:"+k:v for k,v in prediction_check(json.loads(a.fit_summary.read_text()),es,X).items()})
 if not batches: raise SystemExit("provide at least one results directory")
 blocking=any(not v for k,v in checks.items() if k.endswith("batch_complete") and not a.allow_partial)
 considered=[v for k,v in checks.items() if not (a.allow_partial and k.endswith("batch_complete"))]
 checks["all_checks_pass"]=all(considered) and not blocking
 report={"schema_version":"value_level_independent_audit.v1","verdict":"passed" if checks["all_checks_pass"] and not a.allow_partial else "passed_with_scope" if checks["all_checks_pass"] else "failed","scope":"Independent 045/045a artifact audit; no collection, regeneration, GPU work, or source edits","checks":checks,"details":{k:{x:y for x,y in v.items() if x not in ("X","Y")} for k,v in details.items()},"notes":["Feature and mask checks recompute all saved head-level missing fractions from NPZ indices and four adapter spans.","Bootstrap arithmetic is intentionally omitted because it is not part of the pass gate."]}
 a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(report,indent=2,default=lambda x:x.item() if hasattr(x,"item") else str(x))+"\n"); print(json.dumps({"verdict":report["verdict"],"false":[k for k,v in checks.items() if not v],"output":str(a.output),"sha256":sha(a.output)})); raise SystemExit(0 if checks["all_checks_pass"] else 1)
if __name__=="__main__": main()
