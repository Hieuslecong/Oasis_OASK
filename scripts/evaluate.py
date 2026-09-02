#!/usr/bin/env python3
import argparse, json, sys
from pathlib import Path
import numpy as np, torch, yaml
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from crack_stress.checkpoint import load_checkpoint
from crack_stress.datasets import DatasetRegistry
from crack_stress.metrics import binary_metrics, cldice
from crack_stress.models import build_segmenter

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--config",required=True); ap.add_argument("--checkpoint",required=True); ap.add_argument("--split",default="test"); a=ap.parse_args(); cfg=yaml.safe_load(Path(a.config).read_text());
    device=torch.device(cfg.get("device","cpu")); model=build_segmenter(cfg.get("model",{}).get("name","unet"),width=cfg.get("model",{}).get("width",16)).to(device); load_checkpoint(a.checkpoint,model,map_location=device); model.eval(); rows=[]
    with torch.no_grad():
        for b in DatasetRegistry.from_config(cfg).build_loaders(cfg)[a.split]:
            p=model(b["image"].to(device)).sigmoid().cpu().numpy(); y=b["mask"].numpy()
            for pi,yi in zip(p,y): rows.append({**binary_metrics(pi,yi),"cldice":cldice(pi,yi)})
    keys=rows[0].keys() if rows else []; out={k:float(np.mean([r[k] for r in rows])) for k in keys}; print(json.dumps(out,indent=2))
if __name__ == "__main__": main()
