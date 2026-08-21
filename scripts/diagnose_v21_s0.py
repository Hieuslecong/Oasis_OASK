#!/usr/bin/env python3
"""Diagnose and optionally gate v2.1 RC gradients on a trained S0 manifold."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from oasis_rc_v2.checkpoint import validate_student_checkpoint, validate_critic_checkpoint
from oasis_rc_v2.corruptions import make_corrupted_mask
from oasis_rc_v2.critic import OASISRCv2Critic
from oasis_rc_v2.energy_qualification import gradient_alignment_diagnostics, summarize_energy_trajectory
from oasis_rc_v2.losses import oasis_rc_student_loss_v2, segmentation_loss
from oasis_rc_v2.protocol import dataset_content_sha256
from oasis_cycle_aosk.data import ManifestDataset
from oasis_cycle_aosk.evaluate_rc import build, manifest_splits


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--s0-checkpoint",required=True)
    p.add_argument("--critic-checkpoint",required=True)
    p.add_argument("--manifest",required=True)
    p.add_argument("--full-gate0-certificate",required=True)
    p.add_argument("--device",default="cuda")
    p.add_argument("--max-batches",type=int,default=16)
    p.add_argument("--out",required=True)
    p.add_argument("--margin",type=float,default=0.10)
    p.add_argument("--path-margin",type=float,default=0.02)
    p.add_argument("--require-pass",action="store_true")
    p.add_argument("--min-rc-grad",type=float,default=1e-8)
    p.add_argument("--min-grad-ratio",type=float,default=1e-4)
    p.add_argument("--max-grad-ratio",type=float,default=100.0)
    p.add_argument("--min-cosine",type=float,default=-0.95)
    p.add_argument("--min-energy-gap-fraction",type=float,default=0.70)
    p.add_argument("--min-path-order-fraction",type=float,default=0.65)
    a=p.parse_args()

    splits=manifest_splits(a.manifest)
    if "test" in splits or "normal_test" in splits:
        raise ValueError("S0 diagnostic refuses manifests containing test rows")
    if "val" not in splits: raise ValueError("validation split required")
    device=torch.device(a.device)

    s0=torch.load(a.s0_checkpoint,map_location="cpu",weights_only=False)
    validate_student_checkpoint(s0)
    if s0.get("mode") != "control": raise ValueError("--s0-checkpoint must be a control checkpoint")
    cks=torch.load(a.critic_checkpoint,map_location="cpu",weights_only=False)
    cfg=cks.get("config",{})
    validate_critic_checkpoint(
        cks,a.manifest,cfg,cks["normal_fraction"],cks["normal_critic_weight"],
        dataset_content_sha256_value=dataset_content_sha256(a.manifest),
        full_gate0_certificate=a.full_gate0_certificate,
    )

    student=build(s0["student_kind"],int(s0["student_width"])).to(device)
    student.load_state_dict(s0["student"]); student.eval()
    critic=OASISRCv2Critic(width=int(cks["width"])).to(device)
    critic.load_state_dict(cks["critic"]); critic.eval()
    for q in critic.parameters(): q.requires_grad_(False)

    size=int(s0["effective_config"]["image_size"])
    loader=DataLoader(
        ManifestDataset(a.manifest,"val",size,return_is_normal=True),
        batch_size=2,shuffle=False,num_workers=0,
    )
    generator=torch.Generator(device=device).manual_seed(81273)
    rows=[]; energy_rows=[]
    for bi,batch in enumerate(loader):
        if bi>=a.max_batches: break
        x,y,is_normal=[z.to(device) for z in batch]
        wrong,_=make_corrupted_mask(y,true_normal=is_normal.bool(),generator=generator,image=x)
        logits=student(x)
        seg=segmentation_loss(logits,y)
        pred=logits.sigmoid()
        with torch.no_grad(): gt=critic(x,y); co=critic(x,wrong)
        rc,_=oasis_rc_student_loss_v2(critic(x,pred),gt,co,pred,y,margin=a.margin)
        rows.append(gradient_alignment_diagnostics(seg,rc,logits))
        energy_rows.append(summarize_energy_trajectory(critic,x,y,wrong,margin=a.path_margin))
    if not rows: raise RuntimeError("no validation batches diagnosed")

    def avg(key):
        vals=[float(r[key]) for r in rows if r.get(key) is not None]
        return float(np.mean(vals)) if vals else None

    result={
        "batches":len(rows),
        "seg_grad_norm_mean":avg("seg_grad_norm"),
        "rc_grad_norm_mean":avg("aux_grad_norm"),
        "rc_to_seg_grad_ratio_mean":avg("aux_to_seg_norm_ratio"),
        "seg_rc_cosine_mean":avg("cosine_similarity"),
        "all_gradients_finite":all(bool(r.get("finite")) for r in rows),
        "energy_positive_gap_fraction_mean":float(np.mean([float(r["positive_energy_gap_fraction"]) for r in energy_rows])),
        "energy_path_order_fraction_mean":float(np.mean([float(r["continuous_path_order_fraction"]) for r in energy_rows])),
        "canonical_test_opened":False,
        "s0_checkpoint":str(Path(a.s0_checkpoint).resolve()),
        "critic_checkpoint":str(Path(a.critic_checkpoint).resolve()),
    }
    failures=[]
    if result["all_gradients_finite"] is not True: failures.append("all_gradients_finite=true")
    if result["rc_grad_norm_mean"] is None or result["rc_grad_norm_mean"] <= a.min_rc_grad: failures.append(f"rc_grad_norm_mean>{a.min_rc_grad}")
    ratio=result["rc_to_seg_grad_ratio_mean"]
    if ratio is None or ratio < a.min_grad_ratio or ratio > a.max_grad_ratio: failures.append(f"grad_ratio_in_[{a.min_grad_ratio},{a.max_grad_ratio}]")
    cosine=result["seg_rc_cosine_mean"]
    if cosine is None or cosine < a.min_cosine: failures.append(f"seg_rc_cosine_mean>={a.min_cosine}")
    if result["energy_positive_gap_fraction_mean"] < a.min_energy_gap_fraction: failures.append(f"energy_positive_gap_fraction_mean>={a.min_energy_gap_fraction}")
    if result["energy_path_order_fraction_mean"] < a.min_path_order_fraction: failures.append(f"energy_path_order_fraction_mean>={a.min_path_order_fraction}")
    result["development_gate_failures"]=failures
    result["development_gate_pass"]=not failures
    Path(a.out).write_text(json.dumps(result,indent=2)); print(json.dumps(result,indent=2))
    if a.require_pass and failures:
        raise SystemExit("S0 RC diagnostic failed: "+", ".join(failures))


if __name__=="__main__": main()
