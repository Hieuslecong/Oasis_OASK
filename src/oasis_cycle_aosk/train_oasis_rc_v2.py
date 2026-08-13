"""Canonical OASIS-RC v2 trainer.

Official runs accept only a train/val manifest bound to a Gate-0 certificate.
Canonical test images/masks are never opened by this process.
"""
from __future__ import annotations
import argparse, json, random, shlex, sys
from pathlib import Path
import numpy as np
import torch
import yaml
from torch.utils.data import ConcatDataset, DataLoader

from oasis_rc_v2.checkpoint import (
    CHECKPOINT_SCHEMA, EXPERIMENT_ID, IMPLEMENTATION_VERSION, METHOD_VERSION,
    sha256_file, validate_critic_checkpoint,
)
from oasis_rc_v2.corruptions import build_targets, make_corrupted_mask, shift_zero
from oasis_rc_v2.critic import OASISRCv2Critic
from oasis_rc_v2.losses import oasis_rc_critic_loss, oasis_rc_student_loss_v2, segmentation_loss
from oasis_rc_v2.protocol import verify_gate0_certificate
from oasis_rc_v2.qualification import critic_gate_passes
from .aosk import oriented_consistency_loss
from .data import ManifestDataset
from .models import (
    BiSeNetTiny, DSUNetLite, FastSCNNLite, LightweightSegmenter,
    MobileNetV3SmallSegmenter, MultiScaleLightweightSegmenter,
)
from .samplers import MixedBatchSampler


def seed_all(seed):
    seed=int(seed); random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def configure_determinism(enabled):
    enabled=bool(enabled)
    if hasattr(torch.backends,"cudnn"):
        torch.backends.cudnn.deterministic=enabled
        if enabled:
            torch.backends.cudnn.benchmark=False
            if hasattr(torch.backends.cudnn,"allow_tf32"): torch.backends.cudnn.allow_tf32=False
    if enabled and hasattr(torch.backends,"cuda") and hasattr(torch.backends.cuda,"matmul"):
        torch.backends.cuda.matmul.allow_tf32=False
    torch.use_deterministic_algorithms(enabled,warn_only=False)


def make_generator(device,seed): return torch.Generator(device=device).manual_seed(int(seed))


def augment(x,y,generator=None):
    def r(): return torch.rand((),device=x.device,generator=generator)
    if r()<.5: x,y=x.flip(-1),y.flip(-1)
    if r()<.5: x,y=x.flip(-2),y.flip(-2)
    if r()<.35:
        scale=.90+.20*r(); bias=(r()-.5)*.08; x=(x*scale+bias).clamp(-1,1)
    return x,y


def make_loader(manifest,split,size,batch,shuffle,num_workers=0,seed=1337,return_is_normal=False):
    ds=ManifestDataset(manifest,split,size,return_is_normal=return_is_normal)
    gen=torch.Generator().manual_seed(int(seed)) if shuffle else None
    return DataLoader(ds,batch_size=batch,shuffle=shuffle,generator=gen,num_workers=num_workers,
                      drop_last=False,pin_memory=(num_workers>0))


def make_train_loader(manifest,size,batch,normal_fraction,seed,num_workers=0):
    crack=ManifestDataset(manifest,"train",size,return_is_normal=True)
    if float(normal_fraction)<=0:
        gen=torch.Generator().manual_seed(int(seed))
        return DataLoader(crack,batch_size=batch,shuffle=True,generator=gen,num_workers=num_workers,
                          drop_last=False,pin_memory=(num_workers>0)),None
    normal=ManifestDataset(manifest,"normal_train",size,return_is_normal=True)
    sampler=MixedBatchSampler(len(crack),len(normal),batch,normal_fraction,seed=seed)
    return DataLoader(ConcatDataset([crack,normal]),batch_sampler=sampler,num_workers=num_workers,
                      pin_memory=(num_workers>0)),sampler


def manifest_splits(manifest):
    return {json.loads(line).get("split") for line in Path(manifest).read_text().splitlines() if line.strip()}


def manifest_has_split(manifest,split): return split in manifest_splits(manifest)


def make_student(kind,width):
    if kind=="lightweight": return LightweightSegmenter(width=width)
    if kind=="mobilenetv3": return MobileNetV3SmallSegmenter()
    if kind=="dsunet": return DSUNetLite(width=width)
    if kind=="fastscnn": return FastSCNNLite(width=width)
    if kind=="bisenet": return BiSeNetTiny(width=width)
    return MultiScaleLightweightSegmenter(width=width)


def type_name_for_student(student):
    m={LightweightSegmenter:"lightweight",MultiScaleLightweightSegmenter:"multiscale",
       MobileNetV3SmallSegmenter:"mobilenetv3",DSUNetLite:"dsunet",FastSCNNLite:"fastscnn",BiSeNetTiny:"bisenet"}
    return m.get(type(student),type(student).__name__)


def load_student_init(student,checkpoint,expected_seed=None):
    if not checkpoint: return
    saved=torch.load(checkpoint,map_location="cpu",weights_only=False)
    if isinstance(saved,dict):
        if saved.get("student_kind") is not None and saved["student_kind"]!=type_name_for_student(student):
            raise ValueError("student init kind mismatch")
        if saved.get("student_width") is not None and hasattr(student,"_oasis_width") and int(saved["student_width"])!=int(student._oasis_width):
            raise ValueError("student init width mismatch")
        if expected_seed is not None and saved.get("seed") is not None and int(saved["seed"])!=int(expected_seed):
            raise ValueError(f"student init seed mismatch: checkpoint={saved['seed']} run={expected_seed}")
        state=saved.get("student",saved)
    else: state=saved
    student.load_state_dict(state)


def validate_loaded_critic(saved,args,cfg):
    return validate_critic_checkpoint(saved,args.manifest,cfg,args.normal_fraction,args.normal_critic_weight)


def _mean(v): return float(np.mean(v)) if v else 0.0


@torch.no_grad()
def segmentation_metrics(model,loader,device,threshold):
    model.eval(); tp=fp=fn=0.0; normal=[]
    for x,y in loader:
        x,y=x.to(device),y.to(device); pred=(model(x).sigmoid()>=threshold).float()
        tp+=float((pred*y).sum()); fp+=float((pred*(1-y)).sum()); fn+=float(((1-pred)*y).sum())
        normal.extend(float(pred[j].sum()) for j in range(y.shape[0]) if y[j].sum()==0)
    return {"precision":tp/(tp+fp+1e-8),"recall":tp/(tp+fn+1e-8),
            "dice":2*tp/(2*tp+fp+fn+1e-8),"iou":tp/(tp+fp+fn+1e-8),
            "normal_fp_pixels_mean":float(np.mean(normal)) if normal else None,
            "normal_fp_images":int(sum(v>0 for v in normal))}


@torch.no_grad()
def select_threshold(model,loader,device):
    best=None
    for t in np.arange(.05,.951,.01):
        m=segmentation_metrics(model,loader,device,float(t)); m["threshold"]=float(t)
        key=(m["dice"],-m["normal_fp_pixels_mean"] if m["normal_fp_pixels_mean"] is not None else 0.0)
        if best is None or key>best[0]: best=(key,m)
    return best[1]


@torch.no_grad()
def critic_metrics(critic,loader,device):
    gen=make_generator(device,1729); critic.eval()
    crack_tp=crack_fn=invalid_tp=invalid_fn=0.0; valid_crack_predictions=0
    rgb_good=[]; rgb_bad=[]; mask_good=[]; mask_bad=[]
    for x,y in loader:
        x,y=x.to(device),y.to(device); crack_rows=y.flatten(1).sum(1)>0
        wrong,invalid=make_corrupted_mask(y,generator=gen)
        for valid,mask,inv in ((True,y,torch.zeros_like(y)),(False,wrong,invalid)):
            sem,_,_=build_targets(mask,inv); pred=critic(x,mask)["semantic"].argmax(1)
            if valid:
                valid_crack_predictions+=int((pred==1).sum()); crack_tp+=float(((pred==1)&(sem==1)).sum()); crack_fn+=float(((pred!=1)&(sem==1)).sum())
            else:
                invalid_tp+=float(((pred==2)&(sem==2)).sum()); invalid_fn+=float(((pred!=2)&(sem==2)).sum())
        if crack_rows.any():
            xc,yc=x[crack_rows],y[crack_rows]
            rgb_good.extend(critic(xc,yc)["pair"].sigmoid().flatten().cpu().tolist())
            rgb_bad.extend(critic(xc.flip(-1),yc)["pair"].sigmoid().flatten().cpu().tolist())
            mb=yc.flip(-1); changed=(mb-yc).abs().flatten(1).sum(1)>0
            if changed.any():
                mask_good.extend(critic(xc[changed],yc[changed])["pair"].sigmoid().flatten().cpu().tolist())
                mask_bad.extend(critic(xc[changed],mb[changed])["pair"].sigmoid().flatten().cpu().tolist())
    mg=lambda v: float(np.mean(v)) if v else None
    rg,rb,mgood,mbad=mg(rgb_good),mg(rgb_bad),mg(mask_good),mg(mask_bad)
    return {"valid_crack_recall":crack_tp/(crack_tp+crack_fn+1e-8),
            "invalid_recall":invalid_tp/(invalid_tp+invalid_fn+1e-8),
            "valid_crack_predictions":valid_crack_predictions,
            "rgb_pair_drop":None if rg is None or rb is None else rg-rb,
            "mask_pair_drop":None if mgood is None or mbad is None else mgood-mbad,
            "rgb_pair_samples":len(rgb_bad),"mask_pair_samples":len(mask_bad)}


def _critic_gate_passes(metrics): return critic_gate_passes(metrics)


def _critic_term(critic,x,mask,invalid,args):
    sem,mm,pv=build_targets(mask,invalid)
    return oasis_rc_critic_loss(critic(x,mask),sem,mm,pv,
        crack_dice_weight=args.crack_dice_weight,mismatch_weight=args.mismatch_weight,pair_weight=args.pair_weight)[0]


def train_critic(args,cfg,device,out):
    loader,sampler=make_train_loader(args.manifest,cfg["image_size"],cfg["batch_size"],args.normal_fraction,int(cfg["seed"]),cfg.get("num_workers",0))
    critic=OASISRCv2Critic(width=args.critic_width).to(device); opt=torch.optim.AdamW(critic.parameters(),lr=args.lr)
    aug_gen=make_generator(device,int(cfg["seed"])+30001); corrupt_gen=make_generator(device,int(cfg["seed"])+30002); history=[]
    for epoch in range(args.critic_epochs):
        if sampler: sampler.set_epoch(epoch)
        losses=[]; counts={}; normals=0; critic.train()
        for x,y,is_normal in loader:
            x,y=x.to(device),y.to(device); is_normal=is_normal.to(device,dtype=torch.bool); x,y=augment(x,y,aug_gen); normals+=int(is_normal.sum())
            wrong,invalid,meta=make_corrupted_mask(y,true_normal=is_normal,generator=corrupt_gen,return_meta=True)
            for z in meta: counts[z["kind"]]=counts.get(z["kind"],0)+1
            loss=.5*(_critic_term(critic,x,y,torch.zeros_like(y),args)+_critic_term(critic,x,wrong,invalid,args))
            relational=[]; crack_rows=(~is_normal)&(y.flatten(1).sum(1)>0)
            if crack_rows.any():
                xc,yc=x[crack_rows],y[crack_rows]
                sem,mm,pv=build_targets(yc,torch.zeros_like(yc)); pv=torch.zeros_like(pv)
                relational.append(oasis_rc_critic_loss(critic(xc.flip(-1),yc),sem,mm,pv,
                    crack_dice_weight=args.crack_dice_weight,mismatch_weight=args.mismatch_weight,pair_weight=args.pair_weight)[0])
                mb=yc.flip(-1); ch=(mb-yc).abs().flatten(1).sum(1)>0
                if ch.any(): relational.append(_critic_term(critic,xc[ch],mb[ch],(mb[ch]-yc[ch]).abs(),args))
            if relational: loss=loss+args.rgb_mask_weight*torch.stack(relational).mean()
            if is_normal.any() and crack_rows.any():
                xn=x[is_normal]; cm=y[crack_rows]; idx=torch.randint(0,cm.shape[0],(xn.shape[0],),device=y.device,generator=corrupt_gen); dm=cm[idx]
                loss=loss+args.normal_critic_weight*_critic_term(critic,xn,dm,dm.clone(),args)
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(critic.parameters(),5.0); opt.step(); losses.append(float(loss.detach()))
        if args.normal_fraction>0 and normals<=0: raise RuntimeError("normal supervision requested but critic saw zero true-normal samples")
        history.append({"epoch":epoch,"critic_loss":_mean(losses),"corruption_counts":counts,"normal_samples_seen":normals}); print(history[-1],flush=True)
    torch.save({"checkpoint_schema":CHECKPOINT_SCHEMA,"experiment_id":EXPERIMENT_ID,"method_version":METHOD_VERSION,
        "implementation_version":IMPLEMENTATION_VERSION,"critic":critic.state_dict(),"width":int(args.critic_width),"config":dict(cfg),
        "manifest_file_sha256":sha256_file(args.manifest),"normal_fraction":float(args.normal_fraction),"normal_critic_weight":float(args.normal_critic_weight)},out/"critic.pt")
    (out/"critic_history.json").write_text(json.dumps(history,indent=2)); return critic


def train_student(args,cfg,device,out,critic=None,aosk=False):
    seed=int(cfg["seed"]); seed_all(seed)
    loader,sampler=make_train_loader(args.manifest,cfg["image_size"],cfg["batch_size"],args.normal_fraction,seed,cfg.get("num_workers",0))
    val_loader=make_loader(args.manifest,"val",cfg["image_size"],cfg["batch_size"],False,cfg.get("num_workers",0))
    student=make_student(args.student_kind,args.student_width).to(device); setattr(student,"_oasis_width",int(args.student_width)); load_student_init(student,args.student_init_checkpoint,seed)
    opt=torch.optim.AdamW(student.parameters(),lr=args.lr); aug_gen=make_generator(device,seed+10001); rc_gen=make_generator(device,seed+20001)
    if critic:
        critic.eval()
        for p in critic.parameters(): p.requires_grad_(False)
    history=[]; best_key=None; best_state=None
    for epoch in range(args.epochs):
        if sampler: sampler.set_epoch(epoch)
        student.train(); vals=[]; rc_ramp=0.0
        for x,y,is_normal in loader:
            x,y=x.to(device),y.to(device); is_normal=is_normal.to(device,dtype=torch.bool); x,y=augment(x,y,aug_gen)
            logits=student(x); loss=segmentation_loss(logits,y)
            if critic is not None and epoch>=args.warmup:
                pred=logits.sigmoid(); wrong,_=make_corrupted_mask(y,true_normal=is_normal,generator=rc_gen)
                with torch.no_grad(): gt=critic(x,y); corr=critic(x,wrong)
                rc,_=oasis_rc_student_loss_v2(critic(x,pred),gt,corr,pred,y,pair_weight=args.student_pair_weight,corrupted_rank_weight=args.corrupted_rank_weight)
                rc_ramp=min(1.0,(epoch-args.warmup+1)/max(1,args.ramp_epochs)); loss=loss+args.lambda_oasis*rc_ramp*rc
            if aosk: loss=loss+args.lambda_aosk*oriented_consistency_loss(logits,x,y)
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(student.parameters(),5.0); opt.step(); vals.append(float(loss.detach()))
        val=select_threshold(student,val_loader,device); history.append({"epoch":epoch,"loss":_mean(vals),"rc_ramp":rc_ramp,"val":val}); print(history[-1],flush=True)
        key=(val["dice"],val["iou"])
        if best_key is None or key>best_key: best_key=key; best_state={k:v.detach().cpu().clone() for k,v in student.state_dict().items()}
    if best_state is None: raise RuntimeError("no student checkpoint selected")
    student.load_state_dict(best_state); validation=select_threshold(student,val_loader,device)
    effective={"seed":seed,"device":str(device),"image_size":int(cfg["image_size"]),"batch_size":int(cfg["batch_size"]),"epochs":int(args.epochs),
               "mode":args.mode,"student_kind":args.student_kind,"student_width":int(args.student_width),"lambda_oasis":float(args.lambda_oasis),"lambda_aosk":float(args.lambda_aosk),
               "normal_fraction":float(args.normal_fraction),"warmup":int(args.warmup),"ramp_epochs":int(args.ramp_epochs)}
    ck={"checkpoint_schema":CHECKPOINT_SCHEMA,"experiment_id":EXPERIMENT_ID,"method_version":METHOD_VERSION,"implementation_version":IMPLEMENTATION_VERSION,
        "student":student.state_dict(),"student_kind":args.student_kind,"student_width":int(args.student_width),"config":dict(cfg),"effective_config":effective,
        "mode":args.mode,"manifest_file_sha256":sha256_file(args.manifest),"gate0_certificate_sha256":sha256_file(args.gate0_certificate),
        "student_init_sha256":sha256_file(args.student_init_checkpoint),"threshold_validation":float(validation["threshold"]),"inference_contract":"RGB -> crack logits only"}
    torch.save(ck,out/"student_only.pt"); (out/"history.json").write_text(json.dumps(history,indent=2)); (out/"validation.json").write_text(json.dumps(validation,indent=2)); (out/"effective_config.json").write_text(json.dumps(effective,indent=2))
    (out/"run_metadata.json").write_text(json.dumps({"checkpoint_schema":CHECKPOINT_SCHEMA,"experiment_id":EXPERIMENT_ID,"method_version":METHOD_VERSION,"implementation_version":IMPLEMENTATION_VERSION,
        "exact_command":" ".join(shlex.quote(i) for i in sys.argv),"manifest_file_sha256":sha256_file(args.manifest),"gate0_certificate_sha256":sha256_file(args.gate0_certificate),
        "critic_checkpoint_sha256":sha256_file(args.critic_checkpoint),"inference_contract":"RGB -> crack logits only"},indent=2))
    return student,validation


def _build_parser():
    p=argparse.ArgumentParser(); p.add_argument("--config",required=True); p.add_argument("--manifest",required=True); p.add_argument("--gate0-certificate",default=None); p.add_argument("--allow-uncertified-manifest",action="store_true"); p.add_argument("--out",required=True)
    p.add_argument("--mode",choices=("control","critic","connected","aosk","aosk_connected"),required=True); p.add_argument("--normal-fraction",type=float,default=0.0); p.add_argument("--normal-critic-weight",type=float,default=1.0)
    p.add_argument("--critic-epochs",type=int,default=10); p.add_argument("--epochs",type=int,default=12); p.add_argument("--warmup",type=int,default=4); p.add_argument("--ramp-epochs",type=int,default=3)
    p.add_argument("--lambda-aosk",type=float,default=.01); p.add_argument("--lambda-oasis",type=float,default=None); p.add_argument("--critic-width",type=int,default=None); p.add_argument("--crack-dice-weight",type=float,default=1.0)
    p.add_argument("--mismatch-weight",type=float,default=1.0); p.add_argument("--pair-weight",type=float,default=.25); p.add_argument("--rgb-mask-weight",type=float,default=1.0); p.add_argument("--student-pair-weight",type=float,default=.25); p.add_argument("--corrupted-rank-weight",type=float,default=1.0)
    p.add_argument("--student-width",type=int,default=16); p.add_argument("--lr",type=float,default=2e-4); p.add_argument("--deterministic",action="store_true"); p.add_argument("--allow-random-init",action="store_true"); p.add_argument("--allow-inline-critic",action="store_true")
    p.add_argument("--student-kind",choices=("multiscale","lightweight","mobilenetv3","dsunet","fastscnn","bisenet"),default="multiscale"); p.add_argument("--critic-checkpoint",default=None); p.add_argument("--student-init-checkpoint",default=None); return p


def main():
    args=_build_parser().parse_args()
    if not 0<=args.normal_fraction<1: raise ValueError("--normal-fraction must satisfy 0 <= f < 1")
    if min(args.normal_critic_weight,args.crack_dice_weight,args.mismatch_weight,args.pair_weight,args.rgb_mask_weight)<0: raise ValueError("critic loss weights must be non-negative")
    cfg=yaml.safe_load(Path(args.config).read_text())
    for k in ("seed","image_size","batch_size","device"):
        if k not in cfg: raise ValueError(f"config missing required field: {k}")
    if args.lambda_oasis is None: args.lambda_oasis=float(cfg.get("lambda_oasis",.001))
    if args.critic_width is None: args.critic_width=int(cfg.get("critic_width",8))
    splits=manifest_splits(args.manifest)
    if "test" in splits: raise ValueError("official trainer refuses manifests containing test rows")
    if not {"train","val"}.issubset(splits): raise ValueError("training manifest must contain train and val")
    normal_policy="train" if args.normal_fraction>0 else "none"
    if not args.allow_uncertified_manifest: verify_gate0_certificate(args.gate0_certificate,args.manifest,int(cfg["image_size"]),normal_policy)
    if args.mode!="critic" and not args.student_init_checkpoint and not args.allow_random_init: raise ValueError("official student runs require --student-init-checkpoint")
    if args.mode in ("connected","aosk_connected") and not args.critic_checkpoint and not args.allow_inline_critic: raise ValueError("connected arms require one frozen --critic-checkpoint shared by S1/S3")
    seed_all(cfg["seed"]); configure_determinism(args.deterministic); device=torch.device(cfg["device"])
    if device.type=="cuda" and not torch.cuda.is_available(): raise RuntimeError("config requests CUDA but torch.cuda.is_available() is false")
    out=Path(args.out); out.mkdir(parents=True,exist_ok=True); critic=None
    if args.mode in ("critic","connected","aosk_connected"):
        if args.critic_checkpoint:
            saved=torch.load(args.critic_checkpoint,map_location=device,weights_only=False); validate_loaded_critic(saved,args,cfg); critic=OASISRCv2Critic(width=int(saved["width"])).to(device); critic.load_state_dict(saved["critic"]); torch.save(saved,out/"critic.pt")
        else: critic=train_critic(args,cfg,device,out)
        metrics=critic_metrics(critic,make_loader(args.manifest,"val",cfg["image_size"],cfg["batch_size"],False,cfg.get("num_workers",0)),device); (out/"critic_validation.json").write_text(json.dumps(metrics,indent=2)); print({"critic_validation":metrics},flush=True)
        if args.mode=="critic": return
        if not critic_gate_passes(metrics): raise RuntimeError("OASIS-RC v2 quality gate failed; connected training is blocked")
    return train_student(args,cfg,device,out,critic if args.mode in ("connected","aosk_connected") else None,args.mode in ("aosk","aosk_connected"))


if __name__=="__main__": main()
