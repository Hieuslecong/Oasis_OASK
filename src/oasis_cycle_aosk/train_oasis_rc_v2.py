"""OASIS-RC-v2 reconstructed controlled experiment.

Training may mix canonical crack samples with external true-normal RGB samples,
while canonical validation/test remain unchanged. Critic and AOSK are
training-only; deployment remains RGB -> student -> crack logits.
"""
import argparse
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import ConcatDataset, DataLoader

from .audit import audit
from .aosk import oriented_consistency_loss
from .data import ManifestDataset
from .losses_v2 import segmentation_loss, oasis_rc_critic_loss, oasis_rc_student_loss_v2
from .models import (
    BiSeNetTiny,
    DSUNetLite,
    FastSCNNLite,
    LightweightSegmenter,
    MobileNetV3SmallSegmenter,
    MultiScaleLightweightSegmenter,
    RelationalOASISRC,
)
from .samplers import MixedBatchSampler


def sha256_file(path):
    if not path:
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def augment(x, y):
    if torch.rand(()) < 0.5:
        x, y = x.flip(-1), y.flip(-1)
    if torch.rand(()) < 0.5:
        x, y = x.flip(-2), y.flip(-2)
    if torch.rand(()) < 0.35:
        scale = 0.90 + 0.20 * torch.rand((), device=x.device)
        bias = (torch.rand((), device=x.device) - 0.5) * 0.08
        x = (x * scale + bias).clamp(-1, 1)
    return x, y


def shift_zero(mask, dx=3, dy=0):
    b, c, h, w = mask.shape
    out = torch.zeros_like(mask)
    xs0, xs1 = max(0, dx), min(w, w + dx)
    ys0, ys1 = max(0, dy), min(h, h + dy)
    src_x0, src_x1 = max(0, -dx), min(w, w - dx)
    src_y0, src_y1 = max(0, -dy), min(h, h - dy)
    out[..., ys0:ys1, xs0:xs1] = mask[..., src_y0:src_y1, src_x0:src_x1]
    return out


def make_corrupted_mask(mask):
    shifted = shift_zero(mask, dx=3)
    dilated = F.max_pool2d(mask, 5, 1, 2)
    eroded = -F.max_pool2d(-mask, 5, 1, 2)
    keep = (
        F.max_pool2d((torch.rand_like(mask) > 0.985).float(), 9, 1, 4) < 0.5
    ).float()
    broken = mask * keep
    noise = (torch.rand_like(mask) > 0.992).float()
    blob = (F.max_pool2d(noise, 11, 1, 5) > 0.5).float()
    donor = mask[torch.randperm(mask.shape[0], device=mask.device)]
    normal = (mask.sum((1, 2, 3)) == 0).view(-1, 1, 1, 1)
    choices = torch.randint(0, 5, (mask.shape[0], 1, 1, 1), device=mask.device)
    candidates = torch.stack([shifted, dilated, eroded, broken, donor], dim=1)
    idx = choices.expand(-1, 1, mask.shape[-2], mask.shape[-1])
    wrong = torch.gather(candidates, 1, idx.unsqueeze(1)).squeeze(1)
    wrong = torch.where(normal, blob, wrong)
    wrong = (wrong > 0.5).float()
    invalid = (wrong - mask).abs().clamp(0, 1)
    return wrong, invalid


def build_targets(mask, invalid):
    semantic = mask[:, 0].long()
    semantic = torch.where(
        invalid[:, 0] > 0.5, torch.full_like(semantic, 2), semantic
    )
    mismatch = invalid
    pair_valid = (invalid.flatten(1).sum(1) == 0).float().unsqueeze(1)
    return semantic, mismatch, pair_valid


def make_loader(manifest, split, size, batch, shuffle, num_workers=0, seed=1337):
    ds = ManifestDataset(manifest, split, size)
    generator = torch.Generator().manual_seed(int(seed)) if shuffle else None
    return DataLoader(
        ds,
        batch_size=batch,
        shuffle=shuffle,
        generator=generator,
        num_workers=num_workers,
        drop_last=False,
        pin_memory=(num_workers > 0),
    )


def make_train_loader(manifest, size, batch, normal_fraction, seed, num_workers=0):
    crack_ds = ManifestDataset(manifest, "train", size)
    if float(normal_fraction) <= 0:
        generator = torch.Generator().manual_seed(int(seed))
        return (
            DataLoader(
                crack_ds,
                batch_size=batch,
                shuffle=True,
                generator=generator,
                num_workers=num_workers,
                drop_last=False,
                pin_memory=(num_workers > 0),
            ),
            None,
        )
    normal_ds = ManifestDataset(manifest, "normal_train", size)
    joined = ConcatDataset([crack_ds, normal_ds])
    sampler = MixedBatchSampler(
        len(crack_ds), len(normal_ds), batch, normal_fraction, seed=seed
    )
    return (
        DataLoader(
            joined,
            batch_sampler=sampler,
            num_workers=num_workers,
            pin_memory=(num_workers > 0),
        ),
        sampler,
    )


def make_student(kind, width):
    if kind == "lightweight":
        return LightweightSegmenter(width=width)
    if kind == "mobilenetv3":
        return MobileNetV3SmallSegmenter()
    if kind == "dsunet":
        return DSUNetLite(width=width)
    if kind == "fastscnn":
        return FastSCNNLite(width=width)
    if kind == "bisenet":
        return BiSeNetTiny(width=width)
    return MultiScaleLightweightSegmenter(width=width)


def load_student_init(student, checkpoint):
    if not checkpoint:
        return
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = saved.get("student", saved) if isinstance(saved, dict) else saved
    student.load_state_dict(state)


@torch.no_grad()
def segmentation_metrics(model, loader, device, threshold):
    model.eval()
    tp = fp = fn = 0.0
    normal_fp = []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = (model(x).sigmoid() >= threshold).float()
        tp += float((pred * y).sum())
        fp += float((pred * (1 - y)).sum())
        fn += float(((1 - pred) * y).sum())
        for j in range(y.shape[0]):
            if y[j].sum() == 0:
                normal_fp.append(float(pred[j].sum()))
    p = tp / (tp + fp + 1e-8)
    r = tp / (tp + fn + 1e-8)
    return {
        "precision": p,
        "recall": r,
        "dice": 2 * tp / (2 * tp + fp + fn + 1e-8),
        "iou": tp / (tp + fp + fn + 1e-8),
        "normal_fp_pixels_mean": float(np.mean(normal_fp)) if normal_fp else None,
        "normal_fp_images": int(sum(v > 0 for v in normal_fp)),
    }


@torch.no_grad()
def select_threshold(model, loader, device):
    best = None
    for t in np.arange(0.05, 0.951, 0.01):
        m = segmentation_metrics(model, loader, device, float(t))
        m["threshold"] = float(t)
        key = (
            m["dice"],
            -m["normal_fp_pixels_mean"]
            if m["normal_fp_pixels_mean"] is not None
            else 0.0,
        )
        if best is None or key > best[0]:
            best = (key, m)
    return best[1]


@torch.no_grad()
def critic_metrics(critic, loader, device):
    rng_state = torch.random.get_rng_state()
    torch.manual_seed(1729)
    critic.eval()
    sem_correct = sem_total = crack_tp = crack_fn = invalid_tp = invalid_fn = 0.0
    pair_correct = pair_total = 0.0
    mismatch_scores, mismatch_labels = [], []
    valid_pair_scores, rgb_pair_scores, mask_pair_scores = [], [], []
    rgb_tp = rgb_fn = mask_tp = mask_fn = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        wrong, invalid = make_corrupted_mask(y)
        pairs = [
            (
                "valid",
                x,
                y,
                torch.zeros_like(y),
                torch.ones(x.size(0), 1, device=device),
            ),
            (
                "wrong",
                x,
                wrong,
                invalid,
                torch.zeros(x.size(0), 1, device=device),
            ),
        ]
        for pair_kind, image, mask, inv, pv in pairs:
            sem, mm, _ = build_targets(mask, inv)
            out = critic(image, mask)
            pred = out["semantic"].argmax(1)
            sem_correct += float((pred == sem).sum())
            sem_total += float(sem.numel())
            if pair_kind == "valid":
                crack_tp += float(((pred == 1) & (sem == 1)).sum())
                crack_fn += float(((pred != 1) & (sem == 1)).sum())
                valid_pair_scores.extend(out["pair"].sigmoid().flatten().cpu().tolist())
            else:
                invalid_tp += float(((pred == 2) & (sem == 2)).sum())
                invalid_fn += float(((pred != 2) & (sem == 2)).sum())
            pp = (out["pair"].sigmoid() >= 0.5).float()
            pair_correct += float((pp == pv).sum())
            pair_total += float(pv.numel())
            mismatch_scores.extend(out["mismatch"].sigmoid().flatten().cpu().tolist())
            mismatch_labels.extend(mm.flatten().cpu().tolist())
        rgb_pair = critic(x.flip(-1), y)["pair"].sigmoid()
        rgb_pair_scores.extend(rgb_pair.flatten().cpu().tolist())
        rgb_tp += float((rgb_pair < 0.5).sum())
        rgb_fn += float((rgb_pair >= 0.5).sum())
        mask_bad = y.flip(-1)
        mask_pair = critic(x, mask_bad)["pair"].sigmoid()
        mask_pair_scores.extend(mask_pair.flatten().cpu().tolist())
        mask_tp += float((mask_pair < 0.5).sum())
        mask_fn += float((mask_pair >= 0.5).sum())

    def auc(scores, labels):
        order = np.argsort(np.asarray(scores))
        yy = np.asarray(labels)[order]
        pos = yy.sum()
        neg = len(yy) - pos
        if pos == 0 or neg == 0:
            return None
        rank = np.arange(1, len(yy) + 1)
        return float((rank[yy == 1].sum() - pos * (pos + 1) / 2) / (pos * neg))

    torch.random.set_rng_state(rng_state)
    vp = float(np.mean(valid_pair_scores)) if valid_pair_scores else 0.0
    rp = float(np.mean(rgb_pair_scores)) if rgb_pair_scores else 0.0
    mp = float(np.mean(mask_pair_scores)) if mask_pair_scores else 0.0
    return {
        "semantic_accuracy": sem_correct / (sem_total + 1e-8),
        "valid_crack_recall": crack_tp / (crack_tp + crack_fn + 1e-8),
        "invalid_recall": invalid_tp / (invalid_tp + invalid_fn + 1e-8),
        "pair_accuracy": pair_correct / (pair_total + 1e-8),
        "mismatch_auc": auc(mismatch_scores, mismatch_labels),
        "rgb_mismatch_recall": rgb_tp / (rgb_tp + rgb_fn + 1e-8),
        "mask_mismatch_recall": mask_tp / (mask_tp + mask_fn + 1e-8),
        "valid_pair_score": vp,
        "rgb_pair_score": rp,
        "mask_pair_score": mp,
        "rgb_pair_drop": vp - rp,
        "mask_pair_drop": vp - mp,
    }


def _mean(values):
    return float(np.mean(values)) if values else 0.0


def train_critic(args, cfg, device, out):
    loader, mixed_sampler = make_train_loader(
        args.manifest,
        cfg["image_size"],
        cfg["batch_size"],
        args.normal_fraction,
        int(cfg["seed"]),
        cfg.get("num_workers", 0),
    )
    critic = RelationalOASISRC(width=args.critic_width).to(device)
    opt = torch.optim.AdamW(critic.parameters(), lr=args.lr)
    history = []
    for epoch in range(args.critic_epochs):
        if mixed_sampler is not None:
            mixed_sampler.set_epoch(epoch)
        critic.train()
        epoch_losses = []
        normal_samples_seen = normal_batches_seen = normal_donor_pairs = 0
        for x, y in loader:
            x, y = augment(x.to(device), y.to(device))
            wrong, invalid = make_corrupted_mask(y)
            sem, mm, pv = build_targets(y, torch.zeros_like(y))
            sem_w, mm_w, pv_w = build_targets(wrong, invalid)
            rgb_bad = x.flip(-1)
            sem_rgb, mm_rgb, pv_rgb = build_targets(y, torch.zeros_like(y))
            pv_rgb = torch.zeros_like(pv_rgb)
            mask_bad = y.flip(-1)
            mask_invalid = (mask_bad - y).abs().clamp(0, 1)
            sem_m, mm_m, pv_m = build_targets(mask_bad, mask_invalid)
            opt.zero_grad()
            terms = [
                oasis_rc_critic_loss(
                    critic(x, y), sem, mm, pv, pair_weight=args.pair_weight
                ),
                oasis_rc_critic_loss(
                    critic(x, wrong),
                    sem_w,
                    mm_w,
                    pv_w,
                    pair_weight=args.pair_weight,
                ),
                oasis_rc_critic_loss(
                    critic(rgb_bad, y),
                    sem_rgb,
                    mm_rgb,
                    pv_rgb,
                    pair_weight=args.pair_weight,
                ),
                oasis_rc_critic_loss(
                    critic(x, mask_bad),
                    sem_m,
                    mm_m,
                    pv_m,
                    pair_weight=args.pair_weight,
                ),
            ]
            normal_rows = y.flatten(1).sum(1) == 0
            crack_rows = ~normal_rows
            if normal_rows.any():
                normal_batches_seen += 1
                normal_samples_seen += int(normal_rows.sum())
            if normal_rows.any() and crack_rows.any():
                x_n = x[normal_rows]
                crack_masks = y[crack_rows]
                donor_idx = torch.randint(
                    0, crack_masks.shape[0], (x_n.shape[0],), device=y.device
                )
                m_n = crack_masks[donor_idx]
                sem_n, mm_n, pv_n = build_targets(m_n, m_n)
                pv_n = torch.zeros_like(pv_n)
                terms.append(
                    oasis_rc_critic_loss(
                        critic(x_n, m_n),
                        sem_n,
                        mm_n,
                        pv_n,
                        pair_weight=args.pair_weight,
                    )
                )
                normal_donor_pairs += int(x_n.shape[0])
            loss = sum(terms) / len(terms)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(critic.parameters(), 5.0)
            opt.step()
            epoch_losses.append(float(loss.detach()))
        row = {
            "epoch": epoch,
            "critic_loss": _mean(epoch_losses),
            "normal_fraction": float(args.normal_fraction),
            "critic_normal_samples_seen": normal_samples_seen,
            "critic_normal_batches_seen": normal_batches_seen,
            "normal_donor_negative_pairs": normal_donor_pairs,
        }
        history.append(row)
        print(row, flush=True)
    if args.normal_fraction > 0 and not any(
        r["critic_normal_samples_seen"] > 0 for r in history
    ):
        raise RuntimeError("normal supervision requested but critic saw zero normal samples")
    torch.save(
        {"critic": critic.state_dict(), "width": args.critic_width, "config": cfg},
        out / "critic.pt",
    )
    (out / "critic_history.json").write_text(json.dumps(history, indent=2))
    return critic


def train_student(args, cfg, device, out, critic=None, aosk=False):
    seed_all(int(cfg["seed"]))
    train_loader, mixed_sampler = make_train_loader(
        args.manifest,
        cfg["image_size"],
        cfg["batch_size"],
        args.normal_fraction,
        int(cfg["seed"]),
        cfg.get("num_workers", 0),
    )
    val_loader = make_loader(
        args.manifest,
        "val",
        cfg["image_size"],
        cfg["batch_size"],
        False,
        cfg.get("num_workers", 0),
    )
    student = make_student(args.student_kind, args.student_width).to(device)
    load_student_init(student, args.student_init_checkpoint)
    opt = torch.optim.AdamW(student.parameters(), lr=args.lr)
    history, best, best_state = [], None, None
    if critic is not None:
        critic.eval()
        for p in critic.parameters():
            p.requires_grad_(False)

    for epoch in range(args.epochs):
        if mixed_sampler is not None:
            mixed_sampler.set_epoch(epoch)
        student.train()
        logs = {
            name: []
            for name in (
                "loss_total",
                "loss_seg",
                "rc_total_raw",
                "rc_total_weighted",
                "rank_gt",
                "rank_corrupted",
                "rc_fp",
                "e_pred",
                "e_gt",
                "e_corrupted",
                "delta_pred_gt",
                "delta_pred_corrupted",
                "aosk_raw",
                "aosk_weighted",
            )
        }
        rc_ramp = 0.0
        for x, y in train_loader:
            x, y = augment(x.to(device), y.to(device))
            logits = student(x)
            seg = segmentation_loss(logits, y)
            loss = seg
            logs["loss_seg"].append(float(seg.detach()))
            if critic is not None and epoch >= args.warmup:
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
                rc_ramp = min(
                    1.0,
                    (epoch - args.warmup + 1) / max(1, args.ramp_epochs),
                )
                rc_weighted = args.lambda_oasis * rc_ramp * rc
                loss = loss + rc_weighted
                logs["rc_total_raw"].append(float(rc.detach()))
                logs["rc_total_weighted"].append(float(rc_weighted.detach()))
                mapping = {
                    "rank_gt": "rank_gt",
                    "rank_corrupted": "rank_corrupted",
                    "rc_fp": "fp",
                    "e_pred": "e_pred",
                    "e_gt": "e_gt",
                    "e_corrupted": "e_corrupted",
                    "delta_pred_gt": "delta_pred_gt",
                    "delta_pred_corrupted": "delta_pred_corrupted",
                }
                for dst, src in mapping.items():
                    logs[dst].append(float(ex[src]))
            if aosk:
                cons = oriented_consistency_loss(logits, x, y)
                aosk_weighted = args.lambda_aosk * cons
                loss = loss + aosk_weighted
                logs["aosk_raw"].append(float(cons.detach()))
                logs["aosk_weighted"].append(float(aosk_weighted.detach()))
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 5.0)
            opt.step()
            logs["loss_total"].append(float(loss.detach()))

        val = select_threshold(student, val_loader, device)
        row = {
            "epoch": epoch,
            "losses": {name: _mean(values) for name, values in logs.items()},
            "lambda_oasis": float(args.lambda_oasis),
            "lambda_aosk": float(args.lambda_aosk),
            "rc_ramp": float(rc_ramp),
            "normal_fraction": float(args.normal_fraction),
            "val": val,
        }
        history.append(row)
        print(row, flush=True)
        key = (val["dice"], val["iou"])
        if best is None or key > best:
            best = key
            best_state = {
                k: v.detach().cpu().clone() for k, v in student.state_dict().items()
            }

    if best_state is None:
        raise RuntimeError("no student checkpoint selected")
    student.load_state_dict(best_state)
    threshold = select_threshold(student, val_loader, device)
    init_sha = sha256_file(args.student_init_checkpoint)
    manifest_sha = sha256_file(args.manifest)
    checkpoint = {
        "student": student.state_dict(),
        "student_kind": args.student_kind,
        "student_width": args.student_width,
        "config": cfg,
        "mode": args.mode,
        "method_version": "OASIS-RC-v2-reconstructed",
        "implementation_version": "2.2.0-normal-rgb-diagnostics",
        "lambda_oasis": args.lambda_oasis,
        "lambda_aosk": args.lambda_aosk,
        "normal_fraction": args.normal_fraction,
        "student_init_sha256": init_sha,
        "manifest_file_sha256": manifest_sha,
        "threshold_validation": threshold["threshold"],
        "inference_contract": "RGB -> crack logits only",
    }
    torch.save(checkpoint, out / "student_only.pt")
    (out / "history.json").write_text(json.dumps(history, indent=2))
    (out / "validation.json").write_text(json.dumps(threshold, indent=2))
    metadata = {
        "method_version": checkpoint["method_version"],
        "implementation_version": checkpoint["implementation_version"],
        "mode": args.mode,
        "seed": int(cfg["seed"]),
        "image_size": int(cfg["image_size"]),
        "batch_size": int(cfg["batch_size"]),
        "epochs": int(args.epochs),
        "normal_fraction": float(args.normal_fraction),
        "lambda_oasis": float(args.lambda_oasis),
        "lambda_aosk": float(args.lambda_aosk),
        "student_init_checkpoint": args.student_init_checkpoint,
        "student_init_sha256": init_sha,
        "manifest": str(Path(args.manifest).resolve()),
        "manifest_file_sha256": manifest_sha,
        "critic_checkpoint": args.critic_checkpoint,
        "critic_checkpoint_sha256": sha256_file(args.critic_checkpoint),
        "inference_contract": checkpoint["inference_contract"],
    }
    (out / "run_metadata.json").write_text(json.dumps(metadata, indent=2))
    return student, threshold


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--out", required=True)
    p.add_argument(
        "--mode",
        choices=("control", "critic", "connected", "aosk", "aosk_connected"),
        required=True,
    )
    p.add_argument("--test-split", default="test")
    p.add_argument("--normal-fraction", type=float, default=0.0)
    p.add_argument("--critic-epochs", type=int, default=10)
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--warmup", type=int, default=4)
    p.add_argument("--ramp-epochs", type=int, default=3)
    p.add_argument("--lambda-aosk", type=float, default=0.01)
    p.add_argument("--lambda-oasis", type=float, default=None)
    p.add_argument("--critic-width", type=int, default=None)
    p.add_argument("--pair-weight", type=float, default=0.25)
    p.add_argument("--student-pair-weight", type=float, default=0.25)
    p.add_argument("--corrupted-rank-weight", type=float, default=1.0)
    p.add_argument("--student-width", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument(
        "--student-kind",
        choices=(
            "multiscale",
            "lightweight",
            "mobilenetv3",
            "dsunet",
            "fastscnn",
            "bisenet",
        ),
        default="multiscale",
    )
    p.add_argument("--critic-checkpoint", default=None)
    p.add_argument("--student-init-checkpoint", default=None)
    args = p.parse_args()
    if not 0.0 <= args.normal_fraction < 1.0:
        raise ValueError("--normal-fraction must satisfy 0 <= f < 1")
    cfg = yaml.safe_load(Path(args.config).read_text())
    if args.lambda_oasis is None:
        args.lambda_oasis = float(cfg.get("lambda_oasis", 0.001))
    if args.critic_width is None:
        args.critic_width = int(cfg.get("critic_width", 8))
    normal_policy = "train" if args.normal_fraction > 0 else "none"
    errors = audit(
        args.manifest, test_split=args.test_split, normal_policy=normal_policy
    )
    if errors:
        raise RuntimeError("G0 FAIL:\n" + "\n".join(errors))
    seed_all(int(cfg["seed"]))
    device = torch.device(cfg["device"])
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    critic = None
    use_aosk = args.mode in ("aosk", "aosk_connected")
    if args.mode in ("critic", "connected", "aosk_connected"):
        if args.critic_checkpoint:
            saved = torch.load(
                args.critic_checkpoint, map_location=device, weights_only=False
            )
            critic = RelationalOASISRC(
                width=int(saved.get("width", args.critic_width))
            ).to(device)
            critic.load_state_dict(saved["critic"])
            torch.save(saved, out / "critic.pt")
        else:
            critic = train_critic(args, cfg, device, out)
        val_critic = critic_metrics(
            critic,
            make_loader(
                args.manifest,
                "val",
                cfg["image_size"],
                cfg["batch_size"],
                False,
                cfg.get("num_workers", 0),
            ),
            device,
        )
        (out / "critic_validation.json").write_text(
            json.dumps(val_critic, indent=2)
        )
        print({"critic_validation": val_critic}, flush=True)
        if args.mode == "critic":
            return
        if (
            val_critic["valid_crack_recall"] < 0.80
            or val_critic["invalid_recall"] < 0.90
            or val_critic["rgb_pair_drop"] < 0.05
            or val_critic["mask_pair_drop"] < 0.05
        ):
            raise RuntimeError(
                "OASIS-RC quality gate failed; connected training is blocked"
            )
    train_student(
        args,
        cfg,
        device,
        out,
        critic if args.mode in ("connected", "aosk_connected") else None,
        aosk=use_aosk,
    )


if __name__ == "__main__":
    main()
