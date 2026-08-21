#!/usr/bin/env python3
"""Open the canonical v2.1 test once and evaluate the whole frozen bundle.

Do not use during development. The ledger key is the immutable bundle_id and is
stored beside the locked canonical manifest, so relocating a Gate0 certificate
cannot create a fresh opening namespace.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from oasis_rc_v2.checkpoint import validate_student_checkpoint, sha256_file
from oasis_rc_v2.final_bundle import validate_final_bundle
from oasis_cycle_aosk.data import ManifestDataset
from oasis_cycle_aosk.evaluate_rc import build
from oasis_cycle_aosk.evaluate_v21 import evaluate


def _atomic_create(path, payload):
    path.parent.mkdir(parents=True,exist_ok=True)
    flags=os.O_WRONLY|os.O_CREAT|os.O_EXCL
    fd=os.open(path,flags,0o600)
    with os.fdopen(fd,"w") as f:
        json.dump(payload,f,indent=2); f.flush(); os.fsync(f.fileno())


def main():
    p=argparse.ArgumentParser(); p.add_argument("--bundle",required=True); p.add_argument("--out-dir",required=True); p.add_argument("--device",default="cuda"); a=p.parse_args()
    bundle=validate_final_bundle(a.bundle)
    manifest=Path(bundle["manifest"]).resolve()
    ledger_dir=manifest.parent/".oasis_rc_v21_final_ledger"
    marker=ledger_dir/f"{bundle['bundle_id']}.json"
    opened={"state":"OPENED","bundle_id":bundle["bundle_id"],"bundle":str(Path(a.bundle).resolve()),"bundle_sha256":sha256_file(a.bundle),"manifest":str(manifest),"opened_utc":dt.datetime.now(dt.timezone.utc).isoformat(),"entries":len(bundle["entries"])}
    try: _atomic_create(marker,opened)
    except FileExistsError: raise SystemExit(f"REFUSE: bundle already opened: {marker}")
    # From this exact point the canonical test is considered opened even if evaluation fails.
    device=torch.device(a.device); out_dir=Path(a.out_dir); out_dir.mkdir(parents=True,exist_ok=True); results=[]
    try:
        for entry in bundle["entries"]:
            ck=torch.load(entry["checkpoint"],map_location="cpu",weights_only=False); validate_student_checkpoint(ck)
            if int(ck["seed"]) != int(entry["seed"]): raise ValueError("bundle seed/checkpoint mismatch")
            if abs(float(ck["threshold_validation"])-float(entry["threshold"]))>1e-12: raise ValueError("bundle threshold/checkpoint mismatch")
            model=build(ck["student_kind"],int(ck["student_width"])).to(device); model.load_state_dict(ck["student"])
            size=int(ck["effective_config"]["image_size"]); loader=DataLoader(ManifestDataset(manifest,"test",size),batch_size=4,shuffle=False,num_workers=0)
            pred_dir=out_dir/f"pred_{entry['arm']}_seed{entry['seed']}"
            r=evaluate(model,loader,float(entry["threshold"]),device,pred_dir); r.update({"arm":entry["arm"],"seed":int(entry["seed"]),"checkpoint_sha256":entry["checkpoint_sha256"],"method_version":ck["method_version"]})
            path=out_dir/f"{entry['arm']}_seed{entry['seed']}.json"; path.write_text(json.dumps(r,indent=2)); results.append({"arm":entry["arm"],"seed":int(entry["seed"]),"result":str(path.resolve()),"result_sha256":sha256_file(path)})
        summary={"bundle_id":bundle["bundle_id"],"state":"DONE","completed_utc":dt.datetime.now(dt.timezone.utc).isoformat(),"results":results}
        (out_dir/"bundle_results.json").write_text(json.dumps(summary,indent=2))
        tmp=marker.with_suffix(".tmp"); tmp.write_text(json.dumps({**opened,**summary},indent=2)); os.replace(tmp,marker)
        print(json.dumps(summary,indent=2))
    except Exception as exc:
        failed={**opened,"state":"FAILED_AFTER_OPEN","failed_utc":dt.datetime.now(dt.timezone.utc).isoformat(),"error":repr(exc)}
        tmp=marker.with_suffix(".tmp"); tmp.write_text(json.dumps(failed,indent=2)); os.replace(tmp,marker)
        raise


if __name__=="__main__": main()
