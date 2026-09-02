#!/usr/bin/env python3
import argparse, json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from crack_stress.metrics import binary_metrics, cldice
from crack_stress.renderer import ToyStressRenderer
from crack_stress.types import NuisanceVector
import torch

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--output", default="reports/nuisance_control.json"); a=ap.parse_args()
    mask=torch.zeros(1,1,32,32); mask[:,:,8:24,15:17]=1; renderer=ToyStressRenderer(); rows=[]
    for factor in ("illumination","contrast","texture"):
        for value in (0,.25,.5,.75,1):
            n=NuisanceVector({"illumination":.5,"contrast":.5,"texture":.5}).with_value(factor,value)
            out=renderer.render(mask,n); rgb=(out.image+1)/2; bg=rgb*(1-mask.repeat(1,3,1,1));
            feature=float(bg.mean()) if factor=="illumination" else float(bg.std()) if factor=="contrast" else float(torch.abs(bg[:,:,1:,1:]-bg[:,:,:-1,:-1]).mean())
            rows.append({"factor":factor,"value":value,"target_feature":feature,"cldice":cldice(out.rendered_mask,mask)})
    Path(a.output).parent.mkdir(parents=True,exist_ok=True); Path(a.output).write_text(json.dumps(rows,indent=2)+"\n"); print(json.dumps(rows,indent=2))
if __name__ == "__main__": main()
