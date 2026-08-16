#!/usr/bin/env python3
"""Fail-closed CUDA/data preflight for an official real-data run."""
import argparse
import json
import os
import platform
import shutil
import subprocess
import tempfile
from pathlib import Path

import torch
import yaml

from oasis_cycle_aosk.data import ManifestDataset
from oasis_cycle_aosk.train_oasis_rc_v2 import make_student
from oasis_rc_v2.corruptions import CORRUPTION_NAMES, make_corrupted_mask
from oasis_rc_v2.critic import OASISRCv2Critic
from oasis_rc_v2.losses import segmentation_loss
from oasis_rc_v2.protocol import verify_gate0_certificate


def command(*args):
    try:
        return subprocess.check_output(args, text=True, stderr=subprocess.STDOUT).strip()
    except Exception as exc:
        return f"UNAVAILABLE: {exc}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--gate0-certificate", required=True)
    parser.add_argument("--full-gate0-certificate", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--student-kind", default="multiscale")
    parser.add_argument("--student-width", type=int, default=16)
    parser.add_argument("--normal-fraction", type=float, choices=(0.0, 0.25), default=0.0)
    args = parser.parse_args()
    repo_root = Path(__file__).resolve().parents[1]

    cfg = yaml.safe_load(Path(args.config).read_text())
    if torch.device(cfg["device"]).type != "cuda" or not torch.cuda.is_available():
        raise SystemExit("official GPU preflight requires an available CUDA device")
    certificate = verify_gate0_certificate(
        args.gate0_certificate,
        args.manifest,
        cfg["image_size"],
        "train" if args.normal_fraction > 0 else "none",
        args.full_gate0_certificate,
    )
    device = torch.device(cfg["device"])
    dataset = ManifestDataset(
        args.manifest, "train", cfg["image_size"], return_is_normal=True
    )
    crack_rows, normal_rows = [], []
    for index in range(len(dataset)):
        sample = dataset[index]
        target = normal_rows if bool(sample[2]) else crack_rows
        if len(target) < 2:
            target.append(sample)
        if len(crack_rows) >= 2 and (
            args.normal_fraction == 0.0 or len(normal_rows) >= 1
        ):
            break
    if len(crack_rows) < 2:
        raise SystemExit("preflight requires two crack-positive training rows for C7")
    if args.normal_fraction > 0 and not normal_rows:
        raise SystemExit("N25 preflight requires a true-normal training row for C8")
    samples = crack_rows + (normal_rows[:1] if args.normal_fraction > 0 else [])
    x = torch.stack([sample[0] for sample in samples])
    y = torch.stack([sample[1] for sample in samples])
    true_normal = torch.stack([sample[2] for sample in samples])
    x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
    student = make_student(args.student_kind, args.student_width).to(device)
    critic = OASISRCv2Critic(width=int(cfg.get("critic_width", 8))).to(device)
    logits = student(x)
    loss = segmentation_loss(logits, y)
    loss.backward()
    if not torch.isfinite(loss):
        raise SystemExit("non-finite student loss")
    with torch.no_grad():
        clean = critic(x, y)
        generator = torch.Generator(device=device).manual_seed(1729)
        observed = set()
        for kind in range(len(CORRUPTION_NAMES)):
            wrong, invalid, meta = make_corrupted_mask(
                y,
                true_normal=true_normal.to(device),
                generator=generator,
                forced_kinds=[kind] * y.shape[0],
                image=x,
                return_meta=True,
            )
            if not torch.isfinite(wrong).all() or not torch.isfinite(invalid).all():
                raise SystemExit(f"non-finite corruption request {kind}")
            observed.update(item["kind"] for item in meta)
            for value in critic(x, wrong).values():
                if not torch.isfinite(value).all():
                    raise SystemExit(f"non-finite critic output for request {kind}")
        if not all(torch.isfinite(value).all() for value in clean.values()):
            raise SystemExit("non-finite critic clean output")
    expected = set(CORRUPTION_NAMES)
    if args.normal_fraction == 0.0:
        expected.remove("C8_crack_on_normal")
    missing = sorted(expected - observed)
    if missing:
        raise SystemExit("preflight did not exercise corruption kinds: " + ", ".join(missing))

    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=output.parent, suffix=".pt") as handle:
        torch.save({"student": student.state_dict()}, handle.name)
        torch.load(handle.name, map_location="cpu", weights_only=False)
    props = torch.cuda.get_device_properties(device)
    disk = shutil.disk_usage(output.parent)
    if props.total_memory < 20 * 1024**3:
        raise SystemExit("official A30 run requires at least 20 GiB GPU memory")
    if disk.free < 20 * 1024**3:
        raise SystemExit("official run requires at least 20 GiB free output storage")
    git_head = command("git", "-C", str(repo_root), "rev-parse", "HEAD")
    git_status = command("git", "-C", str(repo_root), "status", "--porcelain")
    if git_head.startswith("UNAVAILABLE:") or git_status.startswith("UNAVAILABLE:"):
        raise SystemExit("official run requires readable git provenance")
    if git_status:
        raise SystemExit("official run requires a clean git worktree")
    report = {
        "status": "PASS",
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "gpu_name": props.name,
        "gpu_uuid": getattr(props, "uuid", None),
        "gpu_total_memory": props.total_memory,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "driver": command("nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"),
        "git_head": git_head,
        "git_dirty": False,
        "dataset_content_sha256": certificate["dataset_content_sha256"],
        "batch_size": cfg["batch_size"],
        "num_workers": cfg.get("num_workers", 0),
        "disk_free_bytes": disk.free,
        "observed_corruption_kinds": sorted(observed),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    }
    output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
