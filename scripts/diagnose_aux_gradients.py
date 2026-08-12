#!/usr/bin/env python3
"""Measure auxiliary-gradient strength/alignment without updating the model.

This script never reads the test split and never calls optimizer.step().
It uses the same RC-v2 and AOSK objectives as training.
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
import yaml

from oasis_cycle_aosk.aosk import oriented_consistency_loss
from oasis_cycle_aosk.losses_v2 import segmentation_loss, oasis_rc_student_loss_v2
from oasis_cycle_aosk.models import RelationalOASISRC
from oasis_cycle_aosk.train_oasis_rc_v2 import (
    make_corrupted_mask,
    make_student,
    make_train_loader,
    seed_all,
)


def _load_student(model, path):
    if not path:
        return
    saved = torch.load(path, map_location="cpu", weights_only=False)
    state = saved.get("student", saved) if isinstance(saved, dict) else saved
    model.load_state_dict(state)


def _load_critic(path, width, device):
    if not path:
        return None
    saved = torch.load(path, map_location=device, weights_only=False)
    critic = RelationalOASISRC(width=int(saved.get("width", width))).to(device)
    critic.load_state_dict(saved["critic"])
    critic.eval()
    for p in critic.parameters():
        p.requires_grad_(False)
    return critic


def _grads(loss, params, retain_graph):
    raw = torch.autograd.grad(
        loss,
        params,
        retain_graph=retain_graph,
        allow_unused=True,
        create_graph=False,
    )
    return [torch.zeros_like(p) if g is None else g.detach() for p, g in zip(params, raw)]


def _norm(grads):
    value = sum(float((g.double() ** 2).sum()) for g in grads)
    return math.sqrt(value)


def _dot(a, b):
    return sum(float((x.double() * y.double()).sum()) for x, y in zip(a, b))


def _cosine(a, b):
    na, nb = _norm(a), _norm(b)
    if na == 0.0 or nb == 0.0:
        return None
    return _dot(a, b) / (na * nb)


def _scale(grads, weight):
    return [g * float(weight) for g in grads]


def _mean(values):
    vals = [v for v in values if v is not None and np.isfinite(v)]
    return float(np.mean(vals)) if vals else None


def _std(values):
    vals = [v for v in values if v is not None and np.isfinite(v)]
    return float(np.std(vals)) if vals else None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--student-init-checkpoint", default=None)
    p.add_argument("--critic-checkpoint", default=None)
    p.add_argument("--student-kind", default="multiscale")
    p.add_argument("--student-width", type=int, default=16)
    p.add_argument("--critic-width", type=int, default=8)
    p.add_argument("--lambda-oasis", type=float, default=0.001)
    p.add_argument("--lambda-aosk", type=float, default=0.01)
    p.add_argument("--student-pair-weight", type=float, default=0.25)
    p.add_argument("--corrupted-rank-weight", type=float, default=1.0)
    p.add_argument("--rc-ramp", type=float, default=1.0)
    p.add_argument("--normal-fraction", type=float, default=0.25)
    p.add_argument("--batches", type=int, default=50)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    if not 0.0 <= args.normal_fraction < 1.0:
        raise ValueError("--normal-fraction must satisfy 0 <= f < 1")
    if not 0.0 <= args.rc_ramp <= 1.0:
        raise ValueError("--rc-ramp must satisfy 0 <= r <= 1")

    cfg = yaml.safe_load(Path(args.config).read_text())
    seed = int(cfg["seed"])
    seed_all(seed)
    device = torch.device(cfg["device"])
    student = make_student(args.student_kind, args.student_width).to(device)
    _load_student(student, args.student_init_checkpoint)
    student.train()
    critic = _load_critic(args.critic_checkpoint, args.critic_width, device)

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
    for batch_idx, (x, y) in enumerate(loader):
        if batch_idx >= args.batches:
            break
        x, y = x.to(device), y.to(device)
        logits = student(x)
        seg = segmentation_loss(logits, y)
        aosk = oriented_consistency_loss(logits, x, y)

        rc = None
        ex = None
        if critic is not None:
            pred_mask = logits.sigmoid()
            wrong_mask, _ = make_corrupted_mask(y)
            with torch.no_grad():
                gt_out = critic(x, y)
                corrupted_out = critic(x, wrong_mask)
            pred_out = critic(x, pred_mask)
            rc, ex = oasis_rc_student_loss_v2(
                pred_out,
                gt_out,
                corrupted_out,
                pred_mask,
                y,
                pair_weight=args.student_pair_weight,
                corrupted_rank_weight=args.corrupted_rank_weight,
            )

        g_seg = _grads(seg, params, retain_graph=True)
        g_aosk = _grads(aosk, params, retain_graph=(rc is not None))
        g_aosk_w = _scale(g_aosk, args.lambda_aosk)
        seg_norm = _norm(g_seg)

        row = {
            "batch": batch_idx,
            "loss_seg": float(seg.detach()),
            "loss_aosk": float(aosk.detach()),
            "grad_seg": seg_norm,
            "grad_aosk_raw": _norm(g_aosk),
            "grad_aosk_weighted": _norm(g_aosk_w),
            "ratio_aosk_to_seg": _norm(g_aosk_w) / max(seg_norm, 1e-30),
            "cosine_seg_aosk": _cosine(g_seg, g_aosk),
        }

        if rc is not None:
            g_rc = _grads(rc, params, retain_graph=False)
            rc_weight = args.lambda_oasis * args.rc_ramp
            g_rc_w = _scale(g_rc, rc_weight)
            row.update(
                {
                    "loss_rc": float(rc.detach()),
                    "grad_rc_raw": _norm(g_rc),
                    "grad_rc_weighted": _norm(g_rc_w),
                    "ratio_rc_to_seg": _norm(g_rc_w) / max(seg_norm, 1e-30),
                    "cosine_seg_rc": _cosine(g_seg, g_rc),
                    "cosine_rc_aosk": _cosine(g_rc, g_aosk),
                    "e_pred": float(ex["e_pred"]),
                    "e_gt": float(ex["e_gt"]),
                    "e_corrupted": float(ex["e_corrupted"]),
                    "delta_pred_gt": float(ex["delta_pred_gt"]),
                    "delta_pred_corrupted": float(ex["delta_pred_corrupted"]),
                }
            )
        rows.append(row)

    if not rows:
        raise RuntimeError("no diagnostic batches were processed")

    metric_names = sorted({k for row in rows for k in row if k != "batch"})
    summary = {
        "batches": len(rows),
        "seed": seed,
        "normal_fraction": args.normal_fraction,
        "lambda_oasis": args.lambda_oasis,
        "lambda_aosk": args.lambda_aosk,
        "rc_ramp": args.rc_ramp,
        "critic_loaded": critic is not None,
        "metrics": {
            name: {
                "mean": _mean([row.get(name) for row in rows]),
                "std": _std([row.get(name) for row in rows]),
            }
            for name in metric_names
        },
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
