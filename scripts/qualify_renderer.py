#!/usr/bin/env python3
import argparse, json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from crack_stress.constraints import CrackGeometryValidator, ValidStressEnvelope
from crack_stress.renderer import ToyStressRenderer
from crack_stress.types import NuisanceVector
import torch

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--output", default="reports/renderer_qualification.md"); a=ap.parse_args()
    mask=torch.zeros(1,1,32,32); mask[:,:,8:24,15:17]=1; renderer=ToyStressRenderer(); out=renderer.render(mask,NuisanceVector({"illumination":.5,"contrast":.5}))
    result=ValidStressEnvelope(CrackGeometryValidator()).accept(mask,out.image,NuisanceVector({"illumination":.5}),out.rendered_mask)
    report={"renderer":"ToyStressRenderer","status":"PASS" if result.valid else "FAIL","qualification_scope":"pipeline-only; not evidence for DP-GAN realism"}
    Path(a.output).parent.mkdir(parents=True,exist_ok=True); Path(a.output).write_text("# Renderer Qualification\n\n"+json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2))
if __name__ == "__main__": main()
