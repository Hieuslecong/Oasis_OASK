#!/usr/bin/env python3
"""Small reference trainer; full real training is gated by data and renderer qualification."""
import argparse, hashlib, json, platform, random, sys
from pathlib import Path
import numpy as np
import torch
import yaml
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from crack_stress.checkpoint import load_checkpoint, save_checkpoint
from crack_stress.datasets import DatasetRegistry
from crack_stress.losses import segmentation_loss
from crack_stress.models import build_segmenter
from crack_stress.renderer import ToyStressRenderer
from crack_stress.search import HardNuisanceSearcher, RandomNuisanceSampler
from crack_stress.constraints import CrackGeometryValidator, ValidStressEnvelope
from crack_stress.types import NuisanceVector
from crack_stress.metrics import binary_metrics, cldice

def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--config",required=True); ap.add_argument("--mode",default=None); ap.add_argument("--resume",default=None); a=ap.parse_args()
    cfg=yaml.safe_load(Path(a.config).read_text()); seed_all(int(cfg.get("seed",1337))); device=torch.device(cfg.get("device","cpu"))
    registry=DatasetRegistry.from_config(cfg); loaders=registry.build_loaders(cfg); model=build_segmenter(cfg.get("model",{}).get("name","unet"), width=cfg.get("model",{}).get("width",16)).to(device); opt=torch.optim.Adam(model.parameters(),lr=float(cfg.get("optimizer",{}).get("lr",1e-3)))
    start=0
    if a.resume: start,_=load_checkpoint(a.resume,model,opt,map_location=device); start+=1
    mode=a.mode or cfg.get("mode","baseline"); renderer=ToyStressRenderer() if cfg.get("renderer",{}).get("backend","none")=="toy" else None
    if mode != "baseline" and renderer is None: raise RuntimeError("stress mode requires an explicitly configured qualified renderer; no DP-GAN checkpoint/backend is present")
    envelope=ValidStressEnvelope(CrackGeometryValidator()); searcher=HardNuisanceSearcher(RandomNuisanceSampler(active=["illumination","contrast","texture"],seed=int(cfg.get("seed",1337))),candidates=int(cfg.get("stress",{}).get("candidates",4)),loss_fn=segmentation_loss)
    run=Path(cfg.get("run_dir","runs/dev")); run.mkdir(parents=True,exist_ok=True)
    epochs=int(cfg.get("epochs",1)); metrics=[]; best={"f1":-1.0,"iou":-1.0,"cldice":-1.0}
    for epoch in range(start,epochs):
        model.train(); total=0.0
        for batch in loaders["train"]:
            x,y=batch["image"].to(device),batch["mask"].to(device); opt.zero_grad(); loss=segmentation_loss(model(x),y)
            if mode != "baseline":
                hard_images, hard_targets = [], []
                for sample_index in range(y.shape[0]):
                    hard,diag=searcher.search(model,y[sample_index:sample_index+1],renderer,envelope)
                    if hard is not None:
                        hard_images.append(hard.image); hard_targets.append(y[sample_index:sample_index+1])
                if hard_images:
                    hard_logits = model(torch.cat(hard_images, 0))
                    loss=loss+float(cfg.get("stress",{}).get("hard_weight",.5))*segmentation_loss(hard_logits,torch.cat(hard_targets, 0))
            loss.backward()
            opt.step(); total+=float(loss.detach())
        row={"epoch":epoch,"train_loss":total/max(len(loaders["train"]),1)}
        model.eval(); val_rows=[]
        with torch.no_grad():
            for batch in loaders.get("val", []):
                pred=model(batch["image"].to(device)).sigmoid().cpu().numpy(); target=batch["mask"].numpy()
                for p,t in zip(pred,target): val_rows.append({**binary_metrics(p,t),"cldice":cldice(p,t)})
        if val_rows:
            for key in ("f1","iou","cldice"): row["val_"+key]=float(np.mean([r[key] for r in val_rows]))
        metrics.append(row); save_checkpoint(run/"last.pt",model,opt,epoch,config=cfg)
        for key in ("f1","iou","cldice"):
            if row.get("val_"+key,-1)>best[key]: best[key]=row["val_"+key]; save_checkpoint(run/f"best_{key}.pt",model,opt,epoch,config=cfg)
    env={"python":platform.python_version(),"torch":torch.__version__,"device":str(device),"config_sha256":hashlib.sha256(Path(a.config).read_bytes()).hexdigest(),"seed":cfg.get("seed",1337)}
    (run/"environment.json").write_text(json.dumps(env,indent=2)+"\n")
    (run/"metrics.json").write_text(json.dumps(metrics,indent=2)+"\n"); print(json.dumps(metrics,indent=2))
if __name__ == "__main__": main()
