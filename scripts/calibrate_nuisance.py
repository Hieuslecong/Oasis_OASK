#!/usr/bin/env python3
import argparse, json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from crack_stress.calibration import CalibrationModel, NuisanceExtractor
from crack_stress.datasets import ManifestDataset

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--manifest", required=True); ap.add_argument("--split", default="train"); ap.add_argument("--size", type=int, default=128); ap.add_argument("--output", default="artifacts/nuisance_calibration.json"); a=ap.parse_args()
    ds=ManifestDataset(a.manifest, a.split, a.size); ext=NuisanceExtractor(); records=[]
    for i in range(len(ds)):
        item=ds[i]; records.append(ext.extract(item["image"].numpy(), item["mask"].numpy()))
    model=CalibrationModel.fit(records); model.save(a.output)
    print(json.dumps({"output":a.output,"count":len(records),"factors":model.stats["_meta"]["factors"]}, indent=2))
if __name__ == "__main__": main()
