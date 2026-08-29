"""Train/evaluate OASIS-A2S v0.1 arms A0/A1/A2 without opening TEST.

A0: same 2-class D architecture, supervised from scratch.
A1: Stage-I 3-class OASIS discriminator evaluated directly on real classes.
A2: Stage-I discriminator -> exact 2-class transfer -> real-only fine-tuning.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .a2s import (
    METHOD_VERSION,
    OASISA2SDiscriminator,
    OASISA2SGenerator,
    parameter_count,
    stage1_discriminator_loss,
    stage1_generator_loss,
    stage1_real_class_logits,
    stage2_segmentation_loss,
    transfer_to_segmenter,
)
from .data import ManifestDataset

_FORBIDDEN_DEV_SPLITS = {"test", "final", "holdout", "external_test", "final_test"}


def _assert_dev_split(name: str) -> None:
    if name.strip().lower() in _FORBIDDEN_DEV_SPLITS:
        raise ValueError(f"OASIS-A2S v0.1 development firewall rejects split={name!r}")


def _seed_everything(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def _sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""): h.update(chunk)
    return h.hexdigest()


def _atomic_torch_save(obj, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp"); torch.save(obj, tmp); tmp.replace(path)


def _loader(manifest, split, size, batch, workers, shuffle, seed=None):
    _assert_dev_split(split)
    gen = torch.Generator().manual_seed(int(seed if seed is not None else 0)) if shuffle else None
    return DataLoader(
        ManifestDataset(manifest, split, size), batch_size=batch, shuffle=shuffle,
        num_workers=workers, pin_memory=False, drop_last=False, generator=gen,
    )


@torch.no_grad()
def evaluate_logits(model, loader, device, *, stage1: bool = False):
    model.eval(); tp = fp = fn = tn = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = stage1_real_class_logits(model, x) if stage1 else model(x)
        pred = logits.argmax(1, keepdim=True).float()
        tp += float((pred * y).sum()); fp += float((pred * (1-y)).sum())
        fn += float(((1-pred) * y).sum()); tn += float(((1-pred) * (1-y)).sum())
    p = tp/(tp+fp+1e-8); r = tp/(tp+fn+1e-8)
    return {
        "precision": p, "recall": r,
        "dice_f1": 2*tp/(2*tp+fp+fn+1e-8),
        "iou": tp/(tp+fp+fn+1e-8),
        "accuracy": (tp+tn)/(tp+tn+fp+fn+1e-8),
    }


def train_stage1(d, g, loader, device, epochs, lr_d, lr_g, lambda_labelmix, seed):
    d.train(); g.train()
    opt_d = torch.optim.Adam(d.parameters(), lr=lr_d, betas=(0.0, 0.999))
    opt_g = torch.optim.Adam(g.parameters(), lr=lr_g, betas=(0.0, 0.999))
    history = []
    mix_gen = torch.Generator(device=device.type if device.type == "cuda" else "cpu"); mix_gen.manual_seed(seed + 991)
    for epoch in range(int(epochs)):
        sums = {"d":0.0,"dr":0.0,"df":0.0,"mix":0.0,"g":0.0}; n = 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            with torch.no_grad(): fake_d = g(y)
            opt_d.zero_grad(set_to_none=True)
            ld, lr, lf, lm = stage1_discriminator_loss(d, x, y, fake_d, lambda_labelmix, mix_gen)
            ld.backward(); opt_d.step()
            for p in d.parameters(): p.requires_grad_(False)
            opt_g.zero_grad(set_to_none=True); fake_g = g(y); lg = stage1_generator_loss(d, fake_g, y)
            lg.backward(); opt_g.step()
            for p in d.parameters(): p.requires_grad_(True)
            for k, v in zip(sums, (ld, lr, lf, lm, lg)): sums[k] += float(v.detach())
            n += 1
        history.append({"epoch":epoch, **{k:v/max(n,1) for k,v in sums.items()}})
    return history


def train_stage2(model, loader, device, epochs, lr, dice_weight):
    model.train(); opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4); hist = []
    for epoch in range(int(epochs)):
        total = 0.0; n = 0
        for x, y in loader:
            x, y = x.to(device), y.to(device); opt.zero_grad(set_to_none=True)
            loss = stage2_segmentation_loss(model(x), y, dice_weight)
            loss.backward(); opt.step(); total += float(loss.detach()); n += 1
        hist.append({"epoch":epoch,"loss":total/max(n,1)})
    return hist


def _checkpoint_common(a):
    return {
        "method": METHOD_VERSION, "seed": a.seed, "image_size": a.size,
        "width": a.width, "manifest_sha256": _sha256(a.manifest),
        "train_split": a.train_split, "val_split": a.val_split,
        "test_firewall": "closed",
    }


def run(a):
    _assert_dev_split(a.train_split); _assert_dev_split(a.val_split); _seed_everything(a.seed)
    device = torch.device(a.device)
    if device.type == "cuda" and not torch.cuda.is_available(): raise RuntimeError("CUDA requested but unavailable")
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    stage1_loader = _loader(a.manifest, a.train_split, a.size, a.batch, a.workers, True, a.seed + 101)
    a0_loader = _loader(a.manifest, a.train_split, a.size, a.batch, a.workers, True, a.seed + 202)
    a2_loader = _loader(a.manifest, a.train_split, a.size, a.batch, a.workers, True, a.seed + 202)
    val_loader = _loader(a.manifest, a.val_split, a.size, a.batch, a.workers, False)
    common = _checkpoint_common(a); results = {"method":METHOD_VERSION,"arms":{}}

    _seed_everything(a.seed)
    a0 = OASISA2SDiscriminator(a.width, 2).to(device)
    h0 = train_stage2(a0, a0_loader, device, a.stage2_epochs, a.stage2_lr, a.dice_weight); m0 = evaluate_logits(a0, val_loader, device)
    a0_path = out/"a0_supervised.pt"
    _atomic_torch_save({**common,"arm":"A0","segmenter":a0.state_dict(),"history":h0,"inference_contract":"RGB -> 2-class OASIS-A2S D"}, a0_path)
    results["arms"]["A0"] = {**m0,"checkpoint":str(a0_path),"params":parameter_count(a0)}

    _seed_everything(a.seed)
    d = OASISA2SDiscriminator(a.width, 3).to(device); g = OASISA2SGenerator(a.generator_width, a.noise_channels).to(device)
    h1 = train_stage1(d, g, stage1_loader, device, a.stage1_epochs, a.lr_d, a.lr_g, a.lambda_labelmix, a.seed)
    stage1_path = out/"stage1_oasis.pt"
    _atomic_torch_save({**common,"arm":"Stage-I","discriminator":d.state_dict(),"generator":g.state_dict(),"generator_width":a.generator_width,"noise_channels":a.noise_channels,"history":h1,"training_only_generator":True}, stage1_path)
    m1 = evaluate_logits(d, val_loader, device, stage1=True)
    results["arms"]["A1"] = {**m1,"checkpoint":str(stage1_path),"params":parameter_count(d),"inference":"first two real-class logits only"}

    a2 = transfer_to_segmenter(d).to(device)
    h2 = train_stage2(a2, a2_loader, device, a.stage2_epochs, a.stage2_lr, a.dice_weight); m2 = evaluate_logits(a2, val_loader, device)
    a2_path = out/"a2_transferred.pt"
    _atomic_torch_save({**common,"arm":"A2","segmenter":a2.state_dict(),"history":h2,"source_stage1_sha256":_sha256(stage1_path),"inference_contract":"RGB -> transferred 2-class OASIS-A2S D","generator_in_checkpoint":False}, a2_path)
    results["arms"]["A2"] = {**m2,"checkpoint":str(a2_path),"params":parameter_count(a2)}
    results["delta_A2_minus_A0"] = {k:m2[k]-m0[k] for k in ("precision","recall","dice_f1","iou","accuracy")}
    results["decision_gate"] = "continue" if m2["dice_f1"] > m0["dice_f1"] and m2["iou"] > m0["iou"] else "stop_or_inconclusive"
    (out/"results.json").write_text(json.dumps(results, indent=2)); print(json.dumps(results, indent=2)); return results


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", required=True); p.add_argument("--out", required=True)
    p.add_argument("--train-split", default="train"); p.add_argument("--val-split", default="val")
    p.add_argument("--size", type=int, default=256); p.add_argument("--batch", type=int, default=8)
    p.add_argument("--workers", type=int, default=0); p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=1337); p.add_argument("--width", type=int, default=24)
    p.add_argument("--generator-width", type=int, default=32); p.add_argument("--noise-channels", type=int, default=4)
    p.add_argument("--stage1-epochs", type=int, default=30); p.add_argument("--stage2-epochs", type=int, default=30)
    p.add_argument("--lr-d", type=float, default=2e-4); p.add_argument("--lr-g", type=float, default=2e-4)
    p.add_argument("--stage2-lr", type=float, default=2e-4); p.add_argument("--lambda-labelmix", type=float, default=10.0)
    p.add_argument("--dice-weight", type=float, default=1.0)
    return p


def main(): run(build_parser().parse_args())


if __name__ == "__main__": main()
