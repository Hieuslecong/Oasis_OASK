#!/usr/bin/env python3
"""Measure auxiliary-gradient strength/alignment on the certified train view only."""
import argparse, json, math
from pathlib import Path
import numpy as np
import torch
import yaml

from oasis_cycle_aosk.aosk import oriented_consistency_loss
from oasis_cycle_aosk.train_oasis_rc_v2 import (
    make_corrupted_mask,
    make_generator,
    make_student,
    make_train_loader,
    seed_all,
)
from oasis_rc_v2.checkpoint import validate_critic_checkpoint
from oasis_rc_v2.critic import OASISRCv2Critic
from oasis_rc_v2.losses import segmentation_loss, oasis_rc_student_loss_v2
from oasis_rc_v2.protocol import verify_gate0_certificate


def _load_student(model, path):
    saved = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(saved.get("student", saved) if isinstance(saved, dict) else saved)


def _load_critic(
    path, manifest, cfg, normal_fraction, normal_critic_weight,
    full_gate0_certificate, device,
):
    if not path:
        return None
    saved = torch.load(path, map_location=device, weights_only=False)
    validate_critic_checkpoint(
        saved, manifest, cfg, normal_fraction, normal_critic_weight,
        full_gate0_certificate=full_gate0_certificate,
    )
    critic = OASISRCv2Critic(width=int(saved["width"])).to(device)
    critic.load_state_dict(saved["critic"])
    critic.eval()
    for p in critic.parameters():
        p.requires_grad_(False)
    return critic


def _grads(loss, params, retain_graph):
    raw = torch.autograd.grad(loss, params, retain_graph=retain_graph, allow_unused=True)
    return [torch.zeros_like(p) if g is None else g.detach() for p, g in zip(params, raw)]


def _norm(grads):
    return math.sqrt(sum(float((g.double() ** 2).sum()) for g in grads))


def _dot(a, b):
    return sum(float((x.double() * y.double()).sum()) for x, y in zip(a, b))


def _cosine(a, b):
    na, nb = _norm(a), _norm(b)
    return None if na == 0.0 or nb == 0.0 else _dot(a, b) / (na * nb)


def _scale(grads, weight):
    return [g * float(weight) for g in grads]


def _summary(values):
    vals = [v for v in values if v is not None and np.isfinite(v)]
    return {"mean": float(np.mean(vals)) if vals else None, "std": float(np.std(vals)) if vals else None}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--gate0-certificate", required=True)
    p.add_argument("--full-gate0-certificate", required=True)
    p.add_argument("--student-init-checkpoint", required=True)
    p.add_argument("--critic-checkpoint", default=None)
    p.add_argument("--student-kind", default="multiscale")
    p.add_argument("--student-width", type=int, default=16)
    p.add_argument("--lambda-oasis", type=float, default=0.001)
    p.add_argument("--lambda-aosk", type=float, default=0.01)
    p.add_argument("--student-pair-weight", type=float, default=0.25)
    p.add_argument("--corrupted-rank-weight", type=float, default=1.0)
    p.add_argument("--rc-ramp", type=float, default=1.0)
    p.add_argument("--normal-fraction", type=float, default=0.25)
    p.add_argument("--normal-critic-weight", type=float, default=1.0)
    p.add_argument("--batches", type=int, default=50)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    normal_policy = "train" if args.normal_fraction > 0 else "none"
    verify_gate0_certificate(
        args.gate0_certificate,
        args.manifest,
        int(cfg["image_size"]),
        normal_policy,
        args.full_gate0_certificate,
    )
    seed = int(cfg["seed"])
    seed_all(seed)
    device = torch.device(cfg["device"])
    student = make_student(args.student_kind, args.student_width).to(device)
    _load_student(student, args.student_init_checkpoint)
    student.eval()
    critic = _load_critic(
        args.critic_checkpoint,
        args.manifest,
        cfg,
        args.normal_fraction,
        args.normal_critic_weight,
        args.full_gate0_certificate,
        device,
    )
    corruption_gen = make_generator(device, seed + 20001)
    loader, sampler = make_train_loader(
        args.manifest,
        int(cfg["image_size"]),
        int(cfg["batch_size"]),
        args.normal_fraction,
        seed,
        int(cfg.get("num_workers", 0)),
    )
    if sampler is not None:
        sampler.set_epoch(0)

    params = [p for p in student.parameters() if p.requires_grad]
    rows = []
    for batch_idx, (x, y, is_normal) in enumerate(loader):
        if batch_idx >= args.batches:
            break
        x, y = x.to(device), y.to(device)
        is_normal = is_normal.to(device, dtype=torch.bool)
        logits = student(x)
        seg = segmentation_loss(logits, y)
        aosk = oriented_consistency_loss(logits, x, y)
        rc = ex = None
        if critic is not None:
            pred = logits.sigmoid()
            wrong, _ = make_corrupted_mask(y, true_normal=is_normal, generator=corruption_gen)
            with torch.no_grad():
                gt = critic(x, y)
                corrupt = critic(x, wrong)
            rc, ex = oasis_rc_student_loss_v2(
                critic(x, pred),
                gt,
                corrupt,
                pred,
                y,
                pair_weight=args.student_pair_weight,
                corrupted_rank_weight=args.corrupted_rank_weight,
            )
        g_seg = _grads(seg, params, True)
        g_aosk = _grads(aosk, params, rc is not None)
        seg_norm = _norm(g_seg)
        row = {
            "batch": batch_idx,
            "loss_seg": float(seg.detach()),
            "loss_aosk": float(aosk.detach()),
            "grad_seg": seg_norm,
            "ratio_aosk_to_seg": _norm(_scale(g_aosk, args.lambda_aosk)) / max(seg_norm, 1e-30),
            "cosine_seg_aosk": _cosine(g_seg, g_aosk),
        }
        if rc is not None:
            g_rc = _grads(rc, params, False)
            row.update(
                {
                    "loss_rc": float(rc.detach()),
                    "ratio_rc_to_seg": _norm(_scale(g_rc, args.lambda_oasis * args.rc_ramp)) / max(seg_norm, 1e-30),
                    "cosine_seg_rc": _cosine(g_seg, g_rc),
                    "cosine_rc_aosk": _cosine(g_rc, g_aosk),
                    "e_pred": float(ex["e_pred"]),
                    "e_gt": float(ex["e_gt"]),
                    "e_corrupted": float(ex["e_corrupted"]),
                }
            )
        rows.append(row)
    if not rows:
        raise RuntimeError("no diagnostic batches were processed")
    names = sorted({k for row in rows for k in row if k != "batch"})
    result = {
        "summary": {
            "batches": len(rows),
            "seed": seed,
            "test_split_read": False,
            "metrics": {name: _summary([row.get(name) for row in rows]) for name in names},
        },
        "rows": rows,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
