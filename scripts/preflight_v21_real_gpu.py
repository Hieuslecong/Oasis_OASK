#!/usr/bin/env python3
"""Fail-closed v2.1-dev2 CUDA/data/backward preflight on real training rows."""
from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
from pathlib import Path

import torch
import yaml

from oasis_cycle_aosk.aosk import oriented_consistency_loss
from oasis_cycle_aosk.data import ManifestDataset
from oasis_cycle_aosk.topology_loss import centerline_cldice_loss
from oasis_cycle_aosk.train_oasis_rc_v2 import (
    configure_determinism,
    make_student,
    manifest_splits,
)
from oasis_rc_v2.corruptions import build_targets, make_corrupted_mask
from oasis_rc_v2.critic import OASISRCv2Critic
from oasis_rc_v2.losses import (
    adversarial_pair_student_loss,
    continuous_relation_path_loss,
    critic_endpoint_energy_loss,
    oasis_rc_critic_loss,
    oasis_rc_student_loss_v2,
    segmentation_loss,
)
from oasis_rc_v2.protocol import verify_gate0_certificate


def command(*args):
    try:
        return subprocess.check_output(
            args, text=True, stderr=subprocess.STDOUT
        ).strip()
    except Exception as exc:
        return f"UNAVAILABLE: {exc}"


def finite_grad(module):
    grads = [p.grad for p in module.parameters() if p.grad is not None]
    return bool(grads) and all(torch.isfinite(g).all() for g in grads)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--gate0-certificate", required=True)
    p.add_argument("--full-gate0-certificate", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--student-kind", default="mobilenetv3")
    p.add_argument("--student-width", type=int, default=16)
    p.add_argument("--critic-width", type=int, default=8)
    p.add_argument("--normal-fraction", type=float, choices=(0.0, 0.25), default=0.0)
    p.add_argument(
        "--determinism-mode",
        choices=("off", "best_effort", "strict"),
        default="strict",
    )
    p.add_argument(
        "--min-gpu-gib",
        type=float,
        default=0.0,
        help="Optional deployment-site capacity policy; 0 disables arbitrary model-size gating.",
    )
    p.add_argument(
        "--min-disk-gib",
        type=float,
        default=1.0,
        help="Operational output-space floor, recorded in the report.",
    )
    a = p.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cfg = yaml.safe_load(Path(a.config).read_text())
    device = torch.device(cfg["device"])
    if device.type != "cuda" or not torch.cuda.is_available():
        raise SystemExit("v2.1 real GPU preflight requires available CUDA")
    configure_determinism(a.determinism_mode, device.type)

    splits = manifest_splits(a.manifest)
    if "test" in splits or "normal_test" in splits:
        raise SystemExit("preflight refuses training manifests containing final-test rows")
    if a.normal_fraction > 0 and "normal_val" not in splits:
        raise SystemExit("N25 preflight requires held-out normal_val rows")

    cert = verify_gate0_certificate(
        a.gate0_certificate,
        a.manifest,
        int(cfg["image_size"]),
        "train" if a.normal_fraction > 0 else "none",
        a.full_gate0_certificate,
    )
    torch.cuda.reset_peak_memory_stats(device)

    crack_dataset = ManifestDataset(
        a.manifest, "train", cfg["image_size"], return_is_normal=True
    )
    crack_rows = []
    for i in range(len(crack_dataset)):
        sample = crack_dataset[i]
        if not bool(sample[2]) and float(sample[1].sum()) > 0:
            crack_rows.append(sample)
        if len(crack_rows) >= 2:
            break
    if len(crack_rows) < 2:
        raise SystemExit("preflight requires >=2 crack-positive training rows")

    normal_rows = []
    if a.normal_fraction > 0:
        normal_dataset = ManifestDataset(
            a.manifest, "normal_train", cfg["image_size"], return_is_normal=True
        )
        for i in range(min(2, len(normal_dataset))):
            sample = normal_dataset[i]
            if bool(sample[2]) and float(sample[1].sum()) == 0:
                normal_rows.append(sample)
        if not normal_rows:
            raise SystemExit("N25 preflight requires true-normal normal_train row")

    samples = crack_rows[:2] + (normal_rows[:1] if a.normal_fraction > 0 else [])
    x = torch.stack([s[0] for s in samples]).to(device)
    y = torch.stack([s[1] for s in samples]).to(device)
    is_normal = torch.stack([s[2] for s in samples]).to(device, dtype=torch.bool)
    generator = torch.Generator(device=device).manual_seed(1729)
    wrong, invalid = make_corrupted_mask(
        y, true_normal=is_normal, generator=generator, image=x
    )

    critic = OASISRCv2Critic(width=a.critic_width).to(device)
    sem, mis, pair = build_targets(wrong, invalid)
    repr_loss, _ = oasis_rc_critic_loss(critic(x, wrong), sem, mis, pair)
    endpoint, _ = critic_endpoint_energy_loss(critic, x, y, wrong)
    path, _ = continuous_relation_path_loss(critic, x, y, wrong)
    critic_total = repr_loss + endpoint + path
    critic.zero_grad(set_to_none=True)
    critic_total.backward()
    if not torch.isfinite(critic_total) or not finite_grad(critic):
        raise SystemExit("v2.1 critic backward is non-finite/empty")
    critic.zero_grad(set_to_none=True)
    critic.eval()
    for q in critic.parameters():
        q.requires_grad_(False)

    arm_results = {}
    for mode in (
        "control",
        "cldice",
        "aosk",
        "connected",
        "aosk_connected",
        "adversarial",
    ):
        student = make_student(a.student_kind, a.student_width).to(device)
        logits = student(x)
        seg = segmentation_loss(logits, y)
        total = seg
        if mode == "cldice":
            total = total + 0.1 * centerline_cldice_loss(logits, y)
        if mode in {"aosk", "aosk_connected"}:
            total = total + 0.01 * oriented_consistency_loss(logits, x, y)
        if mode in {"connected", "aosk_connected"}:
            pred = logits.sigmoid()
            with torch.no_grad():
                gt = critic(x, y)
                corrupted = critic(x, wrong)
            rc, _ = oasis_rc_student_loss_v2(
                critic(x, pred), gt, corrupted, pred, y
            )
            total = total + 0.001 * rc
        if mode == "adversarial":
            total = total + 0.001 * adversarial_pair_student_loss(
                critic(x, logits.sigmoid())
            )
        student.zero_grad(set_to_none=True)
        critic.zero_grad(set_to_none=True)
        total.backward()
        if not torch.isfinite(total) or not finite_grad(student):
            raise SystemExit(f"{mode} backward is non-finite/empty")
        if any(q.grad is not None for q in critic.parameters()):
            raise SystemExit(f"{mode} leaked gradients into frozen critic")
        arm_results[mode] = {
            "loss": float(total.detach()),
            "student_grad_finite": True,
            "critic_frozen": True,
        }
        del student, logits, total

    props = torch.cuda.get_device_properties(device)
    disk = shutil.disk_usage(out.parent)
    required_gpu = int(max(0.0, a.min_gpu_gib) * 1024**3)
    required_disk = int(max(0.0, a.min_disk_gib) * 1024**3)
    if required_gpu and props.total_memory < required_gpu:
        raise SystemExit(
            f"GPU capacity policy failed: {props.total_memory} < {required_gpu} bytes"
        )
    if required_disk and disk.free < required_disk:
        raise SystemExit(
            f"output storage policy failed: {disk.free} < {required_disk} bytes"
        )

    git_head = command("git", "-C", str(repo_root), "rev-parse", "HEAD")
    git_status = command("git", "-C", str(repo_root), "status", "--porcelain")
    if git_head.startswith("UNAVAILABLE:") or git_status.startswith("UNAVAILABLE:"):
        raise SystemExit("preflight requires readable git provenance")
    if git_status:
        raise SystemExit("preflight requires a clean git worktree")

    report = {
        "status": "PASS",
        "method": "OASIS-RC-v2.1-dev2",
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "gpu_name": props.name,
        "gpu_total_memory": props.total_memory,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "capacity_policy": {
            "min_gpu_gib": a.min_gpu_gib,
            "min_disk_gib": a.min_disk_gib,
        },
        "driver": command(
            "nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"
        ),
        "git_head": git_head,
        "git_dirty": False,
        "dataset_content_sha256": cert["dataset_content_sha256"],
        "student_kind": a.student_kind,
        "student_width": a.student_width,
        "normal_fraction": a.normal_fraction,
        "determinism_mode": a.determinism_mode,
        "critic_backward_loss": float(critic_total.detach()),
        "arms": arm_results,
        "disk_free_bytes": disk.free,
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "canonical_test_opened": False,
    }
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
