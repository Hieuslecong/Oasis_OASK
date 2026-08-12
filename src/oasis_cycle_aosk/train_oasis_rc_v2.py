"""OASIS-RC-v2 reconstructed canonical controlled experiment.

This entrypoint never loads a test split while training.  It trains the critic
on online corrupted masks, qualifies it on validation, and then optionally
connects the frozen critic to an RGB-only multi-scale student.
"""
import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

from .audit import audit
from .losses_v2 import segmentation_loss, oasis_rc_critic_loss, oasis_rc_student_loss_v2
from .models import BiSeNetTiny, DSUNetLite, FastSCNNLite, LightweightSegmenter, MobileNetV3SmallSegmenter, MultiScaleLightweightSegmenter, RelationalOASISRC
from .data import ManifestDataset


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


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
    """Zero-padded translation; intentionally no circular wrap-around."""
    b, c, h, w = mask.shape
    out = torch.zeros_like(mask)
    xs0, xs1 = max(0, dx), min(w, w + dx)
    ys0, ys1 = max(0, dy), min(h, h + dy)
    src_x0, src_x1 = max(0, -dx), min(w, w - dx)
    src_y0, src_y1 = max(0, -dy), min(h, h - dy)
    out[..., ys0:ys1, xs0:xs1] = mask[..., src_y0:src_y1, src_x0:src_x1]
    return out


def make_corrupted_mask(mask):
    """Create hard negatives in-memory only, with no dataset expansion."""
    shifted = shift_zero(mask, dx=3)
    dilated = F.max_pool2d(mask, 5, 1, 2)
    eroded = -F.max_pool2d(-mask, 5, 1, 2)
    # Randomly remove local crack fragments.
    keep = (F.max_pool2d((torch.rand_like(mask) > 0.985).float(), 9, stride=1, padding=4) < 0.5).float()
    broken = mask * keep
    # Texture-like blobs are generated only inside the current training batch.
    noise = (torch.rand_like(mask) > 0.992).float()
    blob = (F.max_pool2d(noise, 11, stride=1, padding=5) > 0.5).float()
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
    semantic = torch.where(invalid[:, 0] > 0.5, torch.full_like(semantic, 2), semantic)
    mismatch = invalid
    pair_valid = (invalid.flatten(1).sum(1) == 0).float().unsqueeze(1)
    return semantic, mismatch, pair_valid


def make_loader(manifest, split, size, batch, shuffle):
    ds = ManifestDataset(manifest, split, size)
    return DataLoader(ds, batch_size=batch, shuffle=shuffle, num_workers=0, drop_last=False)


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


@torch.no_grad()
def segmentation_metrics(model, loader, device, threshold):
    model.eval(); tp = fp = fn = 0.0; normal_fp = []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = (model(x).sigmoid() >= threshold).float()
        tp += float((pred * y).sum()); fp += float((pred * (1-y)).sum()); fn += float(((1-pred) * y).sum())
        for j in range(y.shape[0]):
            if y[j].sum() == 0: normal_fp.append(float(pred[j].sum()))
    p = tp / (tp + fp + 1e-8); r = tp / (tp + fn + 1e-8)
    return {"precision": p, "recall": r, "dice": 2*tp/(2*tp+fp+fn+1e-8),
            "iou": tp/(tp+fp+fn+1e-8), "normal_fp_pixels_mean": float(np.mean(normal_fp)) if normal_fp else None,
            "normal_fp_images": int(sum(v > 0 for v in normal_fp))}


@torch.no_grad()
def select_threshold(model, loader, device):
    best = None
    for t in np.arange(0.25, 0.81, 0.05):
        m = segmentation_metrics(model, loader, device, float(t)); m["threshold"] = float(t)
        key = (m["dice"], -m["normal_fp_pixels_mean"] if m["normal_fp_pixels_mean"] is not None else 0.0)
        if best is None or key > best[0]: best = (key, m)
    return best[1]


@torch.no_grad()
def critic_metrics(critic, loader, device):
    # Validation corruption must be repeatable; do not let metric evaluation
    # consume the training RNG stream.
    rng_state = torch.random.get_rng_state()
    torch.manual_seed(1729)
    critic.eval(); sem_correct = sem_total = crack_tp = crack_fn = invalid_tp = invalid_fn = 0.0
    pair_correct = pair_total = 0.0; mismatch_scores = []; mismatch_labels = []
    valid_pair_scores = []; rgb_pair_scores = []; mask_pair_scores = []
    rgb_tp = rgb_fn = mask_tp = mask_fn = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device); wrong, invalid = make_corrupted_mask(y)
        for pair_kind, image, mask, inv, pv in [("valid", x, y, torch.zeros_like(y), torch.ones(x.size(0), 1, device=device)),
                                     ("wrong", x, wrong, invalid, torch.zeros(x.size(0), 1, device=device))]:
            sem, mm, pair = build_targets(mask, inv)
            out = critic(image, mask)
            pred = out["semantic"].argmax(1)
            sem_correct += float((pred == sem).sum()); sem_total += float(sem.numel())
            if pair_kind == "valid":
                crack_tp += float(((pred == 1) & (sem == 1)).sum()); crack_fn += float(((pred != 1) & (sem == 1)).sum())
            else:
                invalid_tp += float(((pred == 2) & (sem == 2)).sum()); invalid_fn += float(((pred != 2) & (sem == 2)).sum())
            pp = (out["pair"].sigmoid() >= 0.5).float()
            pair_correct += float((pp == pv).sum()); pair_total += float(pv.numel())
            if pair_kind == "valid": valid_pair_scores.extend(out["pair"].sigmoid().flatten().cpu().tolist())
            mismatch_scores.extend(out["mismatch"].sigmoid().flatten().cpu().tolist())
            mismatch_labels.extend(mm.flatten().cpu().tolist())
        rgb_bad = x.flip(-1)
        rgb_pair = critic(rgb_bad, y)["pair"].sigmoid()
        rgb_pair_scores.extend(rgb_pair.flatten().cpu().tolist())
        rgb_tp += float((rgb_pair < 0.5).sum())
        rgb_fn += float((rgb_pair >= 0.5).sum())
        mask_bad = y.flip(-1); mask_inv = (mask_bad-y).abs()
        mask_pair = critic(x, mask_bad)["pair"].sigmoid()
        mask_pair_scores.extend(mask_pair.flatten().cpu().tolist())
        mask_tp += float((mask_pair < 0.5).sum())
        mask_fn += float((mask_pair >= 0.5).sum())
    def auc(scores, labels):
        order = np.argsort(np.asarray(scores)); y = np.asarray(labels)[order]; pos = y.sum(); neg = len(y)-pos
        if pos == 0 or neg == 0: return None
        rank = np.arange(1, len(y)+1); return float((rank[y == 1].sum() - pos*(pos+1)/2)/(pos*neg))
    torch.random.set_rng_state(rng_state)
    vp = float(np.mean(valid_pair_scores)) if valid_pair_scores else 0.0
    rp = float(np.mean(rgb_pair_scores)) if rgb_pair_scores else 0.0
    mp = float(np.mean(mask_pair_scores)) if mask_pair_scores else 0.0
    return {"semantic_accuracy": sem_correct/(sem_total+1e-8), "valid_crack_recall": crack_tp/(crack_tp+crack_fn+1e-8),
            "invalid_recall": invalid_tp/(invalid_tp+invalid_fn+1e-8), "pair_accuracy": pair_correct/(pair_total+1e-8),
            "mismatch_auc": auc(mismatch_scores, mismatch_labels),
            "rgb_mismatch_recall": rgb_tp/(rgb_tp+rgb_fn+1e-8),
            "mask_mismatch_recall": mask_tp/(mask_tp+mask_fn+1e-8),
            "valid_pair_score": vp, "rgb_pair_score": rp, "mask_pair_score": mp,
            "rgb_pair_drop": vp-rp, "mask_pair_drop": vp-mp}


def train_critic(args, cfg, device, out):
    loader = make_loader(args.manifest, "train", cfg["image_size"], cfg["batch_size"], True)
    critic = RelationalOASISRC(width=args.critic_width).to(device)
    opt = torch.optim.AdamW(critic.parameters(), lr=args.lr)
    history = []
    for epoch in range(args.critic_epochs):
        critic.train(); epoch_losses = []
        for x, y in loader:
            x, y = augment(x.to(device), y.to(device)); wrong, invalid = make_corrupted_mask(y)
            sem, mm, pv = build_targets(y, torch.zeros_like(y))
            sem_w, mm_w, pv_w = build_targets(wrong, invalid)
            # Explicit RGB mismatch: mask stays fixed while image is mirrored.
            rgb_bad = x.flip(-1)
            # RGB-only corruption has no pixel-accurate invalid map.  It is a
            # pair-level negative; assigning crack pixels to class=invalid was
            # found to teach the critic that every crack is invalid.
            sem_rgb, mm_rgb, pv_rgb = build_targets(y, torch.zeros_like(y))
            pv_rgb = torch.zeros_like(pv_rgb)
            # Explicit mask mismatch is a separate control, not a hidden test.
            mask_bad = y.flip(-1); mask_invalid = (mask_bad - y).abs().clamp(0, 1)
            sem_m, mm_m, pv_m = build_targets(mask_bad, mask_invalid)
            opt.zero_grad()
            terms = [oasis_rc_critic_loss(critic(x, y), sem, mm, pv, pair_weight=args.pair_weight)
                    + oasis_rc_critic_loss(critic(x, wrong), sem_w, mm_w, pv_w, pair_weight=args.pair_weight)
                    + oasis_rc_critic_loss(critic(rgb_bad, y), sem_rgb, mm_rgb, pv_rgb, pair_weight=args.pair_weight)
                    + oasis_rc_critic_loss(critic(x, mask_bad), sem_m, mm_m, pv_m, pair_weight=args.pair_weight)]
            # Explicitly train on a normal RGB image paired with a crack mask.
            # This pair is absent when a batch contains no normal image; it is
            # kept online and is never materialized as a separate dataset.
            normal_rows = (y.flatten(1).sum(1) == 0)
            donor = y[torch.randperm(y.shape[0], device=y.device)]
            if normal_rows.any() and donor[normal_rows].sum() > 0:
                x_n = x[normal_rows]
                m_n = donor[normal_rows]
                inv_n = m_n
                sem_n, mm_n, pv_n = build_targets(m_n, inv_n)
                pv_n = torch.zeros_like(pv_n)
                terms.append(oasis_rc_critic_loss(critic(x_n, m_n), sem_n, mm_n, pv_n,
                                                  pair_weight=args.pair_weight))
            loss = sum(terms) / len(terms)
            loss.backward(); torch.nn.utils.clip_grad_norm_(critic.parameters(), 5.0); opt.step(); epoch_losses.append(float(loss.detach()))
        row = {"epoch": epoch, "critic_loss": float(np.mean(epoch_losses))}; history.append(row); print(row, flush=True)
    torch.save({"critic": critic.state_dict(), "width": args.critic_width, "config": cfg}, out / "critic.pt")
    (out / "critic_history.json").write_text(json.dumps(history, indent=2))
    return critic


def train_student(args, cfg, device, out, critic=None):
    # Critic construction/training must not change the student's initialization
    # or augmentation RNG stream.  This is required for paired control runs.
    seed_all(int(cfg["seed"]))
    train_loader = make_loader(args.manifest, "train", cfg["image_size"], cfg["batch_size"], True)
    val_loader = make_loader(args.manifest, "val", cfg["image_size"], cfg["batch_size"], False)
    student = make_student(args.student_kind, args.student_width).to(device)
    opt = torch.optim.AdamW(student.parameters(), lr=args.lr)
    history = []; best = None; best_state = None
    if critic is not None:
        critic.eval()
        for p in critic.parameters(): p.requires_grad_(False)
    for epoch in range(args.epochs):
        student.train(); losses = []
        for x, y in train_loader:
            x, y = augment(x.to(device), y.to(device)); logits = student(x); loss = segmentation_loss(logits, y)
            extras = {}
            if critic is not None and epoch >= args.warmup:
                pred_mask = logits.sigmoid()
                wrong_mask, _ = make_corrupted_mask(y)
                with torch.no_grad():
                    gt_out = critic(x, y)
                    corrupted_out = critic(x, wrong_mask)
                pred_out = critic(x, pred_mask)
                rc, extras = oasis_rc_student_loss_v2(
                    pred_out, gt_out, corrupted_out, pred_mask, y,
                    pair_weight=args.student_pair_weight,
                    corrupted_rank_weight=args.corrupted_rank_weight,
                )
                ramp = min(1.0, (epoch - args.warmup + 1) / max(1, args.ramp_epochs))
                loss = loss + args.lambda_oasis * ramp * rc
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(student.parameters(), 5.0); opt.step(); losses.append(float(loss.detach()))
        val = select_threshold(student, val_loader, device)
        row = {"epoch": epoch, "train_loss": float(np.mean(losses)), "val": val, "oasis": {k: float(v) for k,v in extras.items()}}
        history.append(row); print(row, flush=True)
        key = (val["dice"], val["iou"])
        if best is None or key > best:
            best = key; best_state = {k: v.detach().cpu().clone() for k,v in student.state_dict().items()}
    student.load_state_dict(best_state)
    threshold = select_threshold(student, val_loader, device)
    torch.save({"student": student.state_dict(), "student_kind": args.student_kind, "student_width": args.student_width, "config": cfg, "mode": args.mode,
                "method_version": "OASIS-RC-v2-reconstructed",
                "lambda_oasis": args.lambda_oasis, "threshold_validation": threshold["threshold"],
                "inference_contract": "RGB -> crack logits only"}, out / "student_only.pt")
    (out / "history.json").write_text(json.dumps(history, indent=2))
    (out / "validation.json").write_text(json.dumps(threshold, indent=2))
    return student, threshold


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True); p.add_argument("--manifest", required=True); p.add_argument("--out", required=True)
    p.add_argument("--mode", choices=("control", "critic", "connected"), required=True)
    p.add_argument("--test-split", default="test_debug")
    p.add_argument("--critic-epochs", type=int, default=10); p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--warmup", type=int, default=4); p.add_argument("--ramp-epochs", type=int, default=3)
    p.add_argument("--lambda-oasis", type=float, default=0.001); p.add_argument("--critic-width", type=int, default=8)
    p.add_argument("--pair-weight", type=float, default=0.25,
                   help="Pair-consistency weight while training the critic")
    p.add_argument("--student-pair-weight", type=float, default=0.25,
                   help="Pair-energy weight in the frozen-critic student loss")
    p.add_argument("--corrupted-rank-weight", type=float, default=1.0,
                   help="Weight for prediction-vs-corrupted-mask ranking")
    p.add_argument("--student-width", type=int, default=16); p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--student-kind", choices=("multiscale", "lightweight", "mobilenetv3", "dsunet", "fastscnn", "bisenet"), default="multiscale")
    p.add_argument("--critic-checkpoint", default=None)
    args = p.parse_args(); cfg = yaml.safe_load(Path(args.config).read_text())
    errors = audit(args.manifest, test_split=args.test_split)
    if errors: raise RuntimeError("G0 FAIL:\n" + "\n".join(errors))
    seed_all(int(cfg["seed"])); device = torch.device(cfg["device"]); out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    critic = None
    if args.mode in ("critic", "connected"):
        if args.critic_checkpoint:
            saved = torch.load(args.critic_checkpoint, map_location=device, weights_only=False)
            critic = RelationalOASISRC(width=int(saved.get("width", args.critic_width))).to(device)
            critic.load_state_dict(saved["critic"])
            torch.save(saved, out / "critic.pt")
        else:
            critic = train_critic(args, cfg, device, out)
        val_critic = critic_metrics(critic, make_loader(args.manifest, "val", cfg["image_size"], cfg["batch_size"], False), device)
        (out / "critic_validation.json").write_text(json.dumps(val_critic, indent=2)); print({"critic_validation": val_critic}, flush=True)
        if args.mode == "critic": return
        if (val_critic["valid_crack_recall"] < 0.80 or
                val_critic["invalid_recall"] < 0.90 or
                val_critic["rgb_pair_drop"] < 0.05 or
                val_critic["mask_pair_drop"] < 0.05):
            raise RuntimeError("OASIS-RC quality gate failed; connected training is blocked")
    train_student(args, cfg, device, out, critic if args.mode == "connected" else None)


if __name__ == "__main__": main()
