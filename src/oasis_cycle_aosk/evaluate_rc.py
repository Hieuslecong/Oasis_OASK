"""Validation/test evaluation for RGB-only deployment checkpoints."""
import argparse, json
from pathlib import Path
import numpy as np, torch
from torch.utils.data import DataLoader
from .models import BiSeNetTiny, DSUNetLite, FastSCNNLite, LightweightSegmenter, MobileNetV3SmallSegmenter, MultiScaleLightweightSegmenter
from .data import ManifestDataset


@torch.no_grad()
def evaluate(model, loader, threshold):
    model.eval(); tp=fp=fn=0.0; normal=[]
    for x,y in loader:
        pred=(model(x).sigmoid()>=threshold).float()
        tp += float((pred*y).sum()); fp += float((pred*(1-y)).sum()); fn += float(((1-pred)*y).sum())
        for j in range(y.size(0)):
            if y[j].sum()==0: normal.append(float(pred[j].sum()))
    p=tp/(tp+fp+1e-8); r=tp/(tp+fn+1e-8)
    return {"precision":p,"recall":r,"dice_f1":2*tp/(2*tp+fp+fn+1e-8),"iou":tp/(tp+fp+fn+1e-8),
            "normal_fp_pixels_mean":float(np.mean(normal)) if normal else None,
            "normal_fp_images":int(sum(v>0 for v in normal)),"threshold":threshold}


def main():
    p=argparse.ArgumentParser(); p.add_argument("--checkpoint",required=True); p.add_argument("--manifest",required=True)
    p.add_argument("--split",required=True); p.add_argument("--size",type=int,default=128); p.add_argument("--threshold",type=float,default=None); p.add_argument("--out",required=True)
    a=p.parse_args(); ck=torch.load(a.checkpoint,map_location="cpu",weights_only=False)
    kind=ck.get("student_kind", "multiscale"); width=int(ck.get("student_width", 16))
    if kind=="lightweight": model=LightweightSegmenter(width=width)
    elif kind=="mobilenetv3": model=MobileNetV3SmallSegmenter()
    elif kind=="dsunet": model=DSUNetLite(width=width)
    elif kind=="fastscnn": model=FastSCNNLite(width=width)
    elif kind=="bisenet": model=BiSeNetTiny(width=width)
    else: model=MultiScaleLightweightSegmenter(width=width)
    model.load_state_dict(ck["student"]); model.eval()
    threshold=a.threshold if a.threshold is not None else float(ck.get("threshold_validation",0.5))
    result=evaluate(model,DataLoader(ManifestDataset(a.manifest,a.split,a.size),batch_size=4,shuffle=False,num_workers=0),threshold)
    result["split"]=a.split; result["checkpoint_mode"]=ck.get("mode"); result["student_kind"]=kind; Path(a.out).write_text(json.dumps(result,indent=2)); print(json.dumps(result,indent=2))


if __name__=="__main__": main()
