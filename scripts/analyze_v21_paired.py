#!/usr/bin/env python3
"""Paired per-image statistical analysis for frozen v2.1 evaluation results."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

DEFAULT_METRICS=("dice","iou","cldice","fp_pixels","component_excess")


def _load(path):
    d=json.loads(Path(path).read_text()); rows=d.get("per_image")
    if not isinstance(rows,list) or not rows: raise ValueError(f"{path}: missing per_image metrics")
    by={int(r["index"]):r for r in rows}
    return d,by


def _bootstrap(delta,seed=20260821,reps=10000):
    delta=np.asarray(delta,dtype=float); rng=np.random.default_rng(seed)
    if delta.size==0: return None
    means=np.empty(reps,dtype=float)
    for start in range(0,reps,1000):
        n=min(1000,reps-start); idx=rng.integers(0,delta.size,size=(n,delta.size)); means[start:start+n]=delta[idx].mean(axis=1)
    lo,hi=np.quantile(means,[.025,.975])
    return float(lo),float(hi)


def compare(base_path,treatment_path,metrics=DEFAULT_METRICS,reps=10000):
    _,base=_load(base_path); _,treat=_load(treatment_path)
    ids=sorted(set(base)&set(treat))
    if len(ids)!=len(base) or len(ids)!=len(treat): raise ValueError("paired evaluation rows do not align exactly")
    out={"base":str(base_path),"treatment":str(treatment_path),"paired_images":len(ids),"metrics":{}}
    for metric in metrics:
        pairs=[(base[i].get(metric),treat[i].get(metric)) for i in ids]
        pairs=[(a,b) for a,b in pairs if a is not None and b is not None]
        if not pairs: continue
        delta=np.asarray([float(b)-float(a) for a,b in pairs])
        ci=_bootstrap(delta,reps=reps)
        out["metrics"][metric]={"n":int(delta.size),"mean_delta":float(delta.mean()),"std_delta":float(delta.std(ddof=1)) if delta.size>1 else 0.0,"median_delta":float(np.median(delta)),"positive_fraction":float((delta>0).mean()),"negative_fraction":float((delta<0).mean()),"bootstrap_95ci_mean":list(ci) if ci else None}
    return out


def main():
    p=argparse.ArgumentParser(); p.add_argument("--s0",required=True); p.add_argument("--s1",required=True); p.add_argument("--s2",required=True); p.add_argument("--s3",required=True); p.add_argument("--bootstrap-reps",type=int,default=10000); p.add_argument("--out",required=True); a=p.parse_args()
    contrasts={"S1-S0":compare(a.s0,a.s1,reps=a.bootstrap_reps),"S2-S0":compare(a.s0,a.s2,reps=a.bootstrap_reps),"S3-S0":compare(a.s0,a.s3,reps=a.bootstrap_reps),"S3-S2":compare(a.s2,a.s3,reps=a.bootstrap_reps)}
    result={"schema":"oasis-rc-v2.1-paired-stats-v1","bootstrap_reps":a.bootstrap_reps,"contrasts":contrasts,"interpretation":"Report all preregistered contrasts; do not select only positive metrics."}
    Path(a.out).write_text(json.dumps(result,indent=2)); print(json.dumps(result,indent=2))


if __name__=="__main__": main()
