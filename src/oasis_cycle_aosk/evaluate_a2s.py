"""Checkpoint evaluation for OASIS-A2S v0.1 deployment models.

By default TEST/final split names are rejected. A future frozen final evaluation
must opt in explicitly with --allow-final-test; development scripts never pass it.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .a2s import METHOD_VERSION, OASISA2SDiscriminator
from .data import ManifestDataset
from .train_a2s import _FORBIDDEN_DEV_SPLITS, _sha256, evaluate_logits


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True); p.add_argument("--manifest", required=True)
    p.add_argument("--split", required=True); p.add_argument("--batch", type=int, default=8)
    p.add_argument("--workers", type=int, default=0); p.add_argument("--device", default="cpu")
    p.add_argument("--allow-final-test", action="store_true"); p.add_argument("--out", required=True)
    a = p.parse_args()
    if a.split.strip().lower() in _FORBIDDEN_DEV_SPLITS and not a.allow_final_test:
        raise ValueError("final/test evaluation is sealed; use --allow-final-test only after protocol freeze")
    ck = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
    if ck.get("method") != METHOD_VERSION: raise ValueError(f"not an {METHOD_VERSION} checkpoint")
    if "segmenter" not in ck: raise ValueError("deployment evaluation requires a Stage-II/A0 segmenter checkpoint")
    forbidden = {"generator", "discriminator", "critic", "aosk"}.intersection(ck)
    if forbidden: raise ValueError(f"deployment checkpoint contains training-only state: {sorted(forbidden)}")
    device = torch.device(a.device)
    if device.type == "cuda" and not torch.cuda.is_available(): raise RuntimeError("CUDA requested but unavailable")
    width, size = int(ck["width"]), int(ck["image_size"])
    model = OASISA2SDiscriminator(width, 2).to(device); model.load_state_dict(ck["segmenter"])
    loader = DataLoader(ManifestDataset(a.manifest, a.split, size), batch_size=a.batch, shuffle=False, num_workers=a.workers)
    result = evaluate_logits(model, loader, device)
    result.update({"method":METHOD_VERSION,"arm":ck.get("arm"),"split":a.split,"checkpoint_sha256":_sha256(a.checkpoint),"image_size":size,"width":width,"inference_contract":ck.get("inference_contract")})
    Path(a.out).write_text(json.dumps(result, indent=2)); print(json.dumps(result, indent=2))


if __name__ == "__main__": main()
