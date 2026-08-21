"""OASIS-RC-v2.1 scientific training runner.

Separate from reconstructed v2.0.4. The canonical test is never accepted by
this entrypoint. Connected arms require both critic representation qualification
and continuous relation-energy qualification.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml

from oasis_rc_v2.checkpoint import (
    CHECKPOINT_SCHEMA, EXPERIMENT_ID, IMPLEMENTATION_VERSION, METHOD_VERSION,
    sha256_file, validate_critic_checkpoint,
)
from oasis_rc_v2.corruptions import build_targets, make_corrupted_mask
from oasis_rc_v2.critic import OASISRCv2Critic
from oasis_rc_v2.energy_qualification import summarize_energy_trajectory
from oasis_rc_v2.losses import (
    adversarial_pair_student_loss, continuous_relation_path_loss,
    oasis_rc_critic_loss, oasis_rc_student_loss_v2, segmentation_loss,
)
from oasis_rc_v2.protocol import verify_gate0_certificate
from oasis_rc_v2.qualification import connected_gate_failures
from .aosk import oriented_consistency_loss
from .topology_loss import centerline_cldice_loss
from .train_oasis_rc_v2 import (
    configure_determinism, critic_metrics, load_student_init, make_generator,
    make_loader, make_student, make_train_loader, manifest_has_split,
    manifest_splits, runtime_metadata, seed_all, select_threshold,
)

ARMS={"control","connected","aosk","aosk_connected","cldice","adversarial"}
AOSK_VARIANT="oriented-consistency-v1-isotropic-flat"


def _optimizer(params,args):
    return torch.optim.AdamW(
        params, lr=float(args.lr), weight_decay=float(args.weight_decay),
        betas=(float(args.beta1),float(args.beta2)), eps=float(args.adam_eps),
    )


def _unpack(batch):
    if len(batch)==3: return batch
    x,y=batch; return x,y,y.flatten(1).sum(1)==0


def _critic_hparams(args,cfg,determinism):
    return {
        "lr":float(args.lr),"weight_decay":float(args.weight_decay),
        "betas":[float(args.beta1),float(args.beta2)],"adam_eps":float(args.adam_eps),
        "critic_epochs":int(args.critic_epochs),"critic_width":int(args.critic_width),
        "batch_size":int(cfg["batch_size"]),"crack_dice_weight":float(args.crack_dice_weight),
        "mismatch_weight":float(args.mismatch_weight),"pair_weight":float(args.pair_weight),
        "normal_critic_weight":float(args.normal_critic_weight),"normal_fraction":float(args.normal_fraction),
        "path_weight":float(args.path_weight),"path_margin":float(args.path_margin),
        "path_levels":[0.0,0.25,0.5,0.75,1.0],"determinism_mode":determinism,
        "method_spec":"METHOD_SPEC_V2_1.md",
    }


def train_critic(args,cfg,device,out,determinism):
    loader,sampler=make_train_loader(
        args.manifest,cfg["image_size"],cfg["batch_size"],args.normal_fraction,
        int(cfg["seed"]),cfg.get("num_workers",0),
    )
    critic=OASISRCv2Critic(width=args.critic_width).to(device)
    optimizer=_optimizer(critic.parameters(),args)
    generator=make_generator(device,int(cfg["seed"])+30002)
    history=[]
    for epoch in range(args.critic_epochs):
        if sampler: sampler.set_epoch(epoch)
        critic.train(); rows=[]
        for batch in loader:
            x,y,is_normal=_unpack(batch); x,y=x.to(device),y.to(device); is_normal=is_normal.to(device,dtype=torch.bool)
            wrong,invalid=make_corrupted_mask(y,true_normal=is_normal,generator=generator,image=x)
            semantic,mismatch,pair_valid=build_targets(wrong,invalid)
            cls,terms=oasis_rc_critic_loss(
                critic(x,wrong),semantic,mismatch,pair_valid,
                crack_dice_weight=args.crack_dice_weight,
                mismatch_weight=args.mismatch_weight,pair_weight=args.pair_weight,
            )
            path,path_terms=continuous_relation_path_loss(
                critic,x,y,wrong,pair_weight=args.student_pair_weight,margin=args.path_margin,
            )
            loss=cls+float(args.path_weight)*path
            optimizer.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(critic.parameters(),float(args.grad_clip)); optimizer.step()
            rows.append({"loss":float(loss.detach()),"classification":float(cls.detach()),"path":float(path.detach()),"path_order_fraction":float(path_terms["path_order_fraction"])})
        history.append({
            "epoch":epoch,
            "loss":sum(r["loss"] for r in rows)/max(1,len(rows)),
            "classification":sum(r["classification"] for r in rows)/max(1,len(rows)),
            "path":sum(r["path"] for r in rows)/max(1,len(rows)),
            "path_order_fraction":sum(r["path_order_fraction"] for r in rows)/max(1,len(rows)),
        })
    ck={
        "checkpoint_schema":CHECKPOINT_SCHEMA,"experiment_id":EXPERIMENT_ID,
        "method_version":METHOD_VERSION,"implementation_version":IMPLEMENTATION_VERSION,
        "seed":int(cfg["seed"]),"critic":critic.state_dict(),"width":int(args.critic_width),
        "config":dict(cfg),"manifest_file_sha256":sha256_file(args.manifest),
        "dataset_content_sha256":args._dataset_content_sha256,
        "full_gate0_certificate_sha256":sha256_file(args.full_gate0_certificate),
        "normal_fraction":float(args.normal_fraction),"normal_critic_weight":float(args.normal_critic_weight),
        "training_hparams":_critic_hparams(args,cfg,determinism),
        "runtime":runtime_metadata(device,determinism),
    }
    torch.save(ck,out/"critic.pt"); (out/"critic_history.json").write_text(json.dumps(history,indent=2))
    return critic


@torch.no_grad()
def energy_qualification(critic,loader,device,pair_weight,margin):
    critic.eval(); rows=[]; generator=make_generator(device,99173)
    for batch in loader:
        x,y,is_normal=_unpack(batch); x,y=x.to(device),y.to(device); is_normal=is_normal.to(device,dtype=torch.bool)
        wrong,_=make_corrupted_mask(y,true_normal=is_normal,generator=generator,image=x)
        rows.append(summarize_energy_trajectory(
            critic,x,y,wrong,pair_weight=pair_weight,
            levels=(0.0,0.25,0.5,0.75,1.0),margin=margin,
        ))
    if not rows: return {"energy_samples":0,"energy_finite":False}
    total=sum(int(r.get("energy_samples",0)) for r in rows)
    result={"energy_samples":total,"energy_finite":all(bool(r.get("energy_finite",False)) for r in rows)}
    for key in ("positive_energy_gap_fraction","continuous_path_order_fraction","mean_energy_gap","median_energy_gap"):
        vals=[(float(r[key]),int(r.get("energy_samples",0))) for r in rows if r.get(key) is not None]
        if vals: result[key]=sum(v*n for v,n in vals)/max(1,sum(n for _,n in vals))
    return result


def _validate_loaded_critic(saved,args,cfg):
    return validate_critic_checkpoint(
        saved,args.manifest,cfg,args.normal_fraction,args.normal_critic_weight,
        dataset_content_sha256_value=args._dataset_content_sha256,
        expected_hparams=_critic_hparams(args,cfg,args._determinism_mode),
        full_gate0_certificate=args.full_gate0_certificate,
    )


def qualify_critic(critic,args,cfg,device,out):
    val=make_loader(
        args.manifest,"val",cfg["image_size"],cfg["batch_size"],False,
        cfg.get("num_workers",0),seed=int(cfg["seed"]),return_is_normal=True,
    )
    normal_loader=None
    if args.normal_fraction>0 and manifest_has_split(args.manifest,"normal_val"):
        normal_loader=make_loader(
            args.manifest,"normal_val",cfg["image_size"],cfg["batch_size"],False,
            cfg.get("num_workers",0),seed=int(cfg["seed"]),return_is_normal=True,
        )
    elif args.normal_fraction>0 and manifest_has_split(args.manifest,"normal_train"):
        # Development fallback only. The report records this so it cannot be
        # confused with held-out normal evidence.
        normal_loader=make_loader(
            args.manifest,"normal_train",cfg["image_size"],cfg["batch_size"],False,
            cfg.get("num_workers",0),seed=int(cfg["seed"]),return_is_normal=True,
        )
    representation=critic_metrics(critic,val,device,normal_loader=normal_loader)
    energy=energy_qualification(critic,val,device,args.student_pair_weight,args.path_margin)
    report={"classification":representation,"energy":energy,"normal_split":"normal_val" if manifest_has_split(args.manifest,"normal_val") else ("normal_train" if normal_loader is not None else None)}
    failures=connected_gate_failures(representation,energy)
    report["failures"]=failures; report["pass"]=not failures
    (out/"critic_qualification_v21.json").write_text(json.dumps(report,indent=2))
    return report


def train_student(args,cfg,device,out,critic=None):
    seed=int(cfg["seed"]); seed_all(seed)
    loader,sampler=make_train_loader(args.manifest,cfg["image_size"],cfg["batch_size"],args.normal_fraction,seed,cfg.get("num_workers",0))
    val_loader=make_loader(args.manifest,"val",cfg["image_size"],cfg["batch_size"],False,cfg.get("num_workers",0),seed=seed)
    student=make_student(args.student_kind,args.student_width).to(device); setattr(student,"_oasis_width",int(args.student_width)); load_student_init(student,args.student_init_checkpoint,seed)
    optimizer=_optimizer(student.parameters(),args); generator=make_generator(device,seed+20001)
    if critic is not None:
        critic.eval(); [p.requires_grad_(False) for p in critic.parameters()]
    history=[]; best=None; best_state=None
    for epoch in range(args.epochs):
        if sampler: sampler.set_epoch(epoch)
        student.train(); rows=[]
        for batch in loader:
            x,y,is_normal=_unpack(batch); x,y=x.to(device),y.to(device); is_normal=is_normal.to(device,dtype=torch.bool)
            logits=student(x); seg=segmentation_loss(logits,y); total=seg; aux=logits.new_zeros(()); structural=logits.new_zeros(())
            if args.mode in {"connected","aosk_connected"}:
                pred=logits.sigmoid(); wrong,_=make_corrupted_mask(y,true_normal=is_normal,generator=generator,image=x)
                with torch.no_grad(): gt=critic(x,y); corrupt=critic(x,wrong)
                aux,_=oasis_rc_student_loss_v2(
                    critic(x,pred),gt,corrupt,pred,y,margin=args.student_margin,
                    pair_weight=args.student_pair_weight,
                    corrupted_rank_weight=args.corrupted_rank_weight,fp_weight=args.fp_weight,
                ); total=total+float(args.lambda_oasis)*aux
            elif args.mode=="adversarial":
                pred=logits.sigmoid(); aux=adversarial_pair_student_loss(critic(x,pred)); total=total+float(args.lambda_adversarial)*aux
            if args.mode in {"aosk","aosk_connected"}:
                structural=oriented_consistency_loss(logits,x,y); total=total+float(args.lambda_aosk)*structural
            elif args.mode=="cldice":
                structural=centerline_cldice_loss(logits,y); total=total+float(args.lambda_cldice)*structural
            optimizer.zero_grad(set_to_none=True); total.backward(); torch.nn.utils.clip_grad_norm_(student.parameters(),float(args.grad_clip)); optimizer.step()
            rows.append((float(total.detach()),float(seg.detach()),float(aux.detach()),float(structural.detach())))
        validation=select_threshold(student,val_loader,device); key=float(validation["dice"])
        history.append({"epoch":epoch,"loss":sum(r[0] for r in rows)/max(1,len(rows)),"seg":sum(r[1] for r in rows)/max(1,len(rows)),"aux":sum(r[2] for r in rows)/max(1,len(rows)),"structural":sum(r[3] for r in rows)/max(1,len(rows)),"val":validation})
        if best is None or key>best:
            best=key; best_state={k:v.detach().cpu().clone() for k,v in student.state_dict().items()}
    if best_state is None: raise RuntimeError("no student checkpoint selected")
    student.load_state_dict(best_state); validation=select_threshold(student,val_loader,device)
    effective={
        "method_version":METHOD_VERSION,"seed":seed,"image_size":int(cfg["image_size"]),
        "batch_size":int(cfg["batch_size"]),"epochs":int(args.epochs),"mode":args.mode,
        "student_kind":args.student_kind,"student_width":int(args.student_width),
        "lr":float(args.lr),"weight_decay":float(args.weight_decay),
        "betas":[float(args.beta1),float(args.beta2)],"adam_eps":float(args.adam_eps),
        "grad_clip":float(args.grad_clip),"student_margin":float(args.student_margin),
        "student_pair_weight":float(args.student_pair_weight),"corrupted_rank_weight":float(args.corrupted_rank_weight),
        "fp_weight":float(args.fp_weight),"lambda_oasis":float(args.lambda_oasis),
        "lambda_aosk":float(args.lambda_aosk),"lambda_cldice":float(args.lambda_cldice),
        "lambda_adversarial":float(args.lambda_adversarial),"normal_fraction":float(args.normal_fraction),
        "checkpoint_selection":"max-validation-micro-dice",
        "threshold_selection":"max-validation-micro-dice-normal-fp-tiebreak",
    }
    ck={
        "checkpoint_schema":CHECKPOINT_SCHEMA,"experiment_id":EXPERIMENT_ID,
        "method_version":METHOD_VERSION,"implementation_version":IMPLEMENTATION_VERSION,
        "seed":seed,"student":student.state_dict(),"student_kind":args.student_kind,
        "student_width":int(args.student_width),"mode":args.mode,"effective_config":effective,
        "manifest_file_sha256":sha256_file(args.manifest),"dataset_content_sha256":args._dataset_content_sha256,
        "training_view_dataset_sha256":args._dataset_content_sha256,
        "gate0_certificate_sha256":sha256_file(args.gate0_certificate),
        "full_gate0_certificate_sha256":sha256_file(args.full_gate0_certificate),
        "student_init_sha256":sha256_file(args.student_init_checkpoint),
        "critic_checkpoint_sha256":sha256_file(args.critic_checkpoint),
        "threshold_validation":float(validation["threshold"]),
        "runtime":runtime_metadata(device,args._determinism_mode),
        "inference_contract":"RGB -> crack logits only",
    }
    torch.save(ck,out/"student_only.pt"); (out/"history.json").write_text(json.dumps(history,indent=2)); (out/"validation.json").write_text(json.dumps(validation,indent=2)); (out/"effective_config.json").write_text(json.dumps(effective,indent=2))
    return student,validation


def parser():
    p=argparse.ArgumentParser()
    p.add_argument("--config",required=True); p.add_argument("--manifest",required=True); p.add_argument("--gate0-certificate",required=True); p.add_argument("--full-gate0-certificate",required=True); p.add_argument("--out",required=True); p.add_argument("--mode",choices=sorted(ARMS|{"critic"}),required=True)
    p.add_argument("--student-init-checkpoint"); p.add_argument("--critic-checkpoint"); p.add_argument("--student-kind",default="mobilenetv3"); p.add_argument("--student-width",type=int,default=16)
    p.add_argument("--normal-fraction",type=float,default=0.0); p.add_argument("--normal-critic-weight",type=float,default=1.0); p.add_argument("--critic-epochs",type=int,default=10); p.add_argument("--epochs",type=int,default=12); p.add_argument("--critic-width",type=int,default=8)
    p.add_argument("--lr",type=float,default=2e-4); p.add_argument("--weight-decay",type=float,default=.01); p.add_argument("--beta1",type=float,default=.9); p.add_argument("--beta2",type=float,default=.999); p.add_argument("--adam-eps",type=float,default=1e-8); p.add_argument("--grad-clip",type=float,default=5.0)
    p.add_argument("--crack-dice-weight",type=float,default=1.0); p.add_argument("--mismatch-weight",type=float,default=1.0); p.add_argument("--pair-weight",type=float,default=.25); p.add_argument("--path-weight",type=float,default=.25); p.add_argument("--path-margin",type=float,default=.02)
    p.add_argument("--student-margin",type=float,default=.10); p.add_argument("--student-pair-weight",type=float,default=.25); p.add_argument("--corrupted-rank-weight",type=float,default=1.0); p.add_argument("--fp-weight",type=float,default=1.0)
    p.add_argument("--lambda-oasis",type=float,default=.001); p.add_argument("--lambda-aosk",type=float,default=.01); p.add_argument("--lambda-cldice",type=float,default=.1); p.add_argument("--lambda-adversarial",type=float,default=.001); p.add_argument("--determinism-mode",choices=("off","best_effort","strict"),default="strict")
    return p


def main():
    args=parser().parse_args(); cfg=yaml.safe_load(Path(args.config).read_text())
    for key in ("seed","image_size","batch_size","device"):
        if key not in cfg: raise ValueError(f"config missing {key}")
    if "test" in manifest_splits(args.manifest) or "normal_test" in manifest_splits(args.manifest): raise ValueError("v2.1 trainer refuses canonical/held-out test rows")
    cert=verify_gate0_certificate(args.gate0_certificate,args.manifest,int(cfg["image_size"]),"train" if args.normal_fraction>0 else "none",args.full_gate0_certificate)
    args._dataset_content_sha256=cert["dataset_content_sha256"]; args._determinism_mode=args.determinism_mode
    device=torch.device(cfg["device"]); seed_all(cfg["seed"]); configure_determinism(args.determinism_mode,device.type)
    out=Path(args.out); out.mkdir(parents=True,exist_ok=True); critic=None; report=None
    if args.mode=="critic": critic=train_critic(args,cfg,device,out,args.determinism_mode)
    elif args.mode in {"connected","aosk_connected","adversarial"}:
        if not args.critic_checkpoint: raise ValueError(f"{args.mode} requires --critic-checkpoint")
        saved=torch.load(args.critic_checkpoint,map_location=device,weights_only=False); _validate_loaded_critic(saved,args,cfg); critic=OASISRCv2Critic(width=int(saved["width"])).to(device); critic.load_state_dict(saved["critic"])
    if critic is not None:
        report=qualify_critic(critic,args,cfg,device,out)
        if report["failures"]: raise RuntimeError("v2.1 critic qualification failed: "+", ".join(report["failures"]))
        if args.mode=="critic":
            payload=torch.load(out/"critic.pt",map_location="cpu",weights_only=False); payload["qualification_v21"]=report; torch.save(payload,out/"critic.pt"); return
    if not args.student_init_checkpoint: raise ValueError("student runs require --student-init-checkpoint")
    return train_student(args,cfg,device,out,critic)


if __name__=="__main__": main()
