#!/usr/bin/env python3
"""Actual end-to-end CPU smoke for the v2.1 CLI.

This is intentionally NOT a unit-test wrapper. It creates real PNG files and
JSONL manifests on disk, issues Gate0 certificates through the production audit
CLI, invokes the production v2.1 trainer in subprocesses for critic and S0-S3,
then invokes the production v2.1 evaluator on the resulting deployment
checkpoints. Canonical test is never created or opened; the only held-out split
is named ``smoke_holdout`` and is used only to make the parent full-benchmark
certificate structurally realistic.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def run(cmd, cwd):
    cmd = [str(x) for x in cmd]
    print("RUN:", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=cwd, check=True)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def make_case(root: Path, index: int, split: str, size: int, rng: np.random.Generator):
    # High-entropy texture prevents accidental Gate0 near-duplicate matches while
    # preserving a strong RGB-mask relation that the critic can learn quickly.
    yy, xx = np.mgrid[0:size, 0:size]
    base = rng.integers(18, 58, size=(size, size, 3), dtype=np.uint8)
    gradient = ((xx * (3 + index % 5) + yy * (1 + index % 3)) % 37).astype(np.uint8)
    image = np.clip(base.astype(np.int16) + gradient[..., None], 0, 150).astype(np.uint8)

    mask_img = Image.new("L", (size, size), 0)
    draw = ImageDraw.Draw(mask_img)
    x0 = 4 + (index * 7) % 13
    x1 = 18 + (index * 5) % 11
    x2 = 31 + (index * 3) % 12
    x3 = 48 + (index * 2) % 10
    points = [
        (x0, 4),
        (x1, 18 + index % 5),
        (x2, 34 + (index * 2) % 7),
        (x3, size - 5),
    ]
    draw.line(points, fill=255, width=2 + (index % 2))
    # Add an asymmetric short branch so horizontal flips are never self-identical.
    bx, by = points[2]
    draw.line([(bx, by), (min(size - 3, bx + 8 + index % 5), max(3, by - 7))], fill=255, width=2)
    mask = np.asarray(mask_img, dtype=np.uint8) > 127

    # Render the annotated crack as a bright image structure, with a mild halo.
    image[mask] = np.array([238, 244, 250], dtype=np.uint8)
    halo = np.asarray(mask_img.resize((size, size), Image.Resampling.NEAREST), dtype=np.uint8) > 0
    for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        shifted = np.roll(np.roll(halo, dy, axis=0), dx, axis=1)
        image[shifted & ~mask] = np.maximum(image[shifted & ~mask], 150)

    image_path = root / "images" / f"{split}_{index:04d}.png"
    mask_path = root / "masks" / f"{split}_{index:04d}.png"
    Image.fromarray(image, mode="RGB").save(image_path)
    mask_img.save(mask_path)
    return {
        "image": str(image_path.resolve()),
        "mask": str(mask_path.resolve()),
        "split": split,
        "source_id": f"smoke_source_{index:04d}",
        "lineage_id": f"smoke_lineage_{index:04d}",
        "is_normal": False,
    }


def write_jsonl(path: Path, rows):
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--critic-epochs", type=int, default=24)
    ap.add_argument("--student-epochs", type=int, default=2)
    ap.add_argument("--image-size", type=int, default=64)
    ap.add_argument("--train-samples", type=int, default=48)
    ap.add_argument("--val-samples", type=int, default=24)
    ap.add_argument("--holdout-samples", type=int, default=8)
    args = ap.parse_args()

    repo = Path(__file__).resolve().parents[1]
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    (out / "images").mkdir(exist_ok=True)
    (out / "masks").mkdir(exist_ok=True)

    rng = np.random.default_rng(1337)
    rows = []
    cursor = 0
    for split, count in (
        ("train", args.train_samples),
        ("val", args.val_samples),
        ("smoke_holdout", args.holdout_samples),
    ):
        for _ in range(count):
            rows.append(make_case(out, cursor, split, args.image_size, rng))
            cursor += 1

    full_manifest = out / "manifest_full.jsonl"
    train_manifest = out / "manifest_trainval.jsonl"
    write_jsonl(full_manifest, rows)
    write_jsonl(train_manifest, [r for r in rows if r["split"] in {"train", "val"}])

    config = out / "smoke_cpu_v21.yaml"
    config.write_text(
        "\n".join(
            [
                "seed: 1337",
                f"image_size: {args.image_size}",
                "batch_size: 8",
                "device: cpu",
                "num_workers: 0",
                "",
            ]
        )
    )

    py = sys.executable
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo / "src") + os.pathsep + env.get("PYTHONPATH", "")

    # subprocess.run above inherits the current environment, so set PYTHONPATH
    # for all subsequent CLI invocations in this process.
    os.environ.update(env)

    full_gate = out / "gate0_full.json"
    train_gate = out / "gate0_training.json"
    run(
        [
            py, "-m", "oasis_cycle_aosk.audit",
            "--manifest", full_manifest,
            "--test-split", "smoke_holdout",
            "--required-splits", "train", "val", "smoke_holdout",
            "--resize-size", str(args.image_size),
            "--normal-policy", "none",
            "--certificate-out", full_gate,
            "--certificate-scope", "full_benchmark",
        ],
        repo,
    )
    run(
        [
            py, "-m", "oasis_cycle_aosk.audit",
            "--manifest", train_manifest,
            "--test-split", "smoke_holdout",
            "--required-splits", "train", "val",
            "--resize-size", str(args.image_size),
            "--normal-policy", "none",
            "--certificate-out", train_gate,
            "--certificate-scope", "training_view",
            "--parent-full-certificate", full_gate,
        ],
        repo,
    )

    init = out / "student_init.pt"
    run(
        [
            py, repo / "scripts" / "create_student_init.py",
            "--out", init,
            "--seed", "1337",
            "--student-kind", "mobilenetv3",
            "--student-width", "16",
        ],
        repo,
    )

    critic_dir = out / "critic"
    common = [
        "--config", config,
        "--manifest", train_manifest,
        "--gate0-certificate", train_gate,
        "--full-gate0-certificate", full_gate,
        "--normal-fraction", "0.0",
        "--critic-epochs", str(args.critic_epochs),
        "--critic-width", "8",
        "--lr", "0.003",
        "--weight-decay", "0.0",
        "--determinism-mode", "strict",
    ]
    run(
        [py, "-m", "oasis_cycle_aosk.train_oasis_rc_v21", *common, "--out", critic_dir, "--mode", "critic"],
        repo,
    )
    critic_ckpt = critic_dir / "critic.pt"
    if not critic_ckpt.exists():
        raise RuntimeError("critic CLI completed without critic.pt")

    arms = {
        "S0": "control",
        "S1": "connected",
        "S2": "aosk",
        "S3": "aosk_connected",
    }
    results = {}
    init_hashes = set()
    for arm, mode in arms.items():
        arm_dir = out / arm
        command = [
            py, "-m", "oasis_cycle_aosk.train_oasis_rc_v21",
            *common,
            "--out", arm_dir,
            "--mode", mode,
            "--student-init-checkpoint", init,
            "--student-kind", "mobilenetv3",
            "--student-width", "16",
            "--epochs", str(args.student_epochs),
            "--warmup", "0",
            "--ramp-epochs", "1",
        ]
        if mode in {"connected", "aosk_connected"}:
            command += ["--critic-checkpoint", critic_ckpt]
        run(command, repo)

        checkpoint = arm_dir / "student_only.pt"
        if not checkpoint.exists():
            raise RuntimeError(f"{arm} CLI completed without student_only.pt")
        eval_path = arm_dir / "eval_val.json"
        prediction_dir = arm_dir / "predictions_val"
        run(
            [
                py, "-m", "oasis_cycle_aosk.evaluate_v21",
                "--checkpoint", checkpoint,
                "--manifest", train_manifest,
                "--split", "val",
                "--device", "cpu",
                "--out", eval_path,
                "--prediction-dir", prediction_dir,
            ],
            repo,
        )
        evaluation = json.loads(eval_path.read_text())
        # Read only metadata using torch here after the CLI has produced a real checkpoint.
        import torch
        ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
        init_hashes.add(ck.get("student_init_sha256"))
        if ck.get("inference_contract") != "RGB -> crack logits only":
            raise RuntimeError(f"{arm} inference contract changed")
        if ck.get("mode") != mode:
            raise RuntimeError(f"{arm} checkpoint mode mismatch")
        results[arm] = {
            "mode": mode,
            "checkpoint_sha256": sha256(checkpoint),
            "checkpoint_bytes": checkpoint.stat().st_size,
            "threshold": evaluation["threshold"],
            "dice": evaluation["dice"],
            "iou": evaluation["iou"],
            "cldice": evaluation["cldice"],
            "precision": evaluation["precision"],
            "recall": evaluation["recall"],
            "image_count": evaluation["image_count"],
        }

    if len(init_hashes) != 1 or next(iter(init_hashes)) != sha256(init):
        raise RuntimeError("S0-S3 did not share exactly one student initialization")

    qualification = json.loads((critic_dir / "critic_qualification_v21.json").read_text())
    if qualification.get("pass") is not True:
        raise RuntimeError("critic qualification did not pass despite critic CLI success")

    summary = {
        "status": "PASS",
        "kind": "actual-cli-integration-smoke",
        "implementation": "OASIS-RC-v2.1",
        "device": "cpu",
        "image_size": args.image_size,
        "train_samples": args.train_samples,
        "val_samples": args.val_samples,
        "smoke_holdout_samples": args.holdout_samples,
        "critic_epochs": args.critic_epochs,
        "student_epochs": args.student_epochs,
        "student_kind": "mobilenetv3",
        "shared_student_init_sha256": sha256(init),
        "critic_checkpoint_sha256": sha256(critic_ckpt),
        "critic_qualification": qualification,
        "arms": results,
        "canonical_test_created": False,
        "canonical_test_opened": False,
    }
    (out / "ACTUAL_SMOKE_SUMMARY.json").write_text(json.dumps(summary, indent=2))
    print("ACTUAL_V21_SMOKE_RESULT=" + json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
