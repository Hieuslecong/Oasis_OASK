#!/usr/bin/env python3
"""Actual disk/CLI smoke for OASIS-RC-v2.1-dev2.

This is not a pytest wrapper. It creates real PNGs/manifests, issues Gate0
certificates through the production audit CLI, trains a critic and all six arms
for both N0 and N25, evaluates crack validation and N25 normal validation, and
verifies the RGB-only deployment checkpoints. No canonical project test data is
created or opened; ``smoke_holdout`` is synthetic and never evaluated.
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


def _background(size, index, rng):
    yy, xx = np.mgrid[0:size, 0:size]
    base = rng.integers(18, 58, size=(size, size, 3), dtype=np.uint8)
    gradient = ((xx * (3 + index % 5) + yy * (1 + index % 3)) % 37).astype(np.uint8)
    return np.clip(base.astype(np.int16) + gradient[..., None], 0, 150).astype(np.uint8)


def make_crack_case(root, index, split, size, rng):
    image = _background(size, index, rng)
    mask_img = Image.new("L", (size, size), 0)
    draw = ImageDraw.Draw(mask_img)
    x0 = 4 + (index * 7) % 13
    x1 = 18 + (index * 5) % 11
    x2 = 31 + (index * 3) % 12
    x3 = min(size - 5, 48 + (index * 2) % 10)
    points = [
        (x0, 4),
        (x1, min(size - 5, 18 + index % 5)),
        (x2, min(size - 5, 34 + (index * 2) % 7)),
        (x3, size - 5),
    ]
    draw.line(points, fill=255, width=2 + (index % 2))
    bx, by = points[2]
    draw.line(
        [(bx, by), (min(size - 3, bx + 8 + index % 5), max(3, by - 7))],
        fill=255,
        width=2,
    )
    mask = np.asarray(mask_img, dtype=np.uint8) > 127
    image[mask] = np.array([238, 244, 250], dtype=np.uint8)
    halo = np.asarray(mask_img, dtype=np.uint8) > 0
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
        "source_id": f"smoke_crack_source_{index:04d}",
        "lineage_id": f"smoke_crack_lineage_{index:04d}",
        "is_normal": False,
    }


def make_normal_case(root, index, split, size, rng):
    image = _background(size, 10000 + index, rng)
    # Add non-crack texture structures so normals are not trivial flat images.
    yy, xx = np.mgrid[0:size, 0:size]
    wave = (8.0 * np.sin((xx + 2 * yy + index) / 7.0))[..., None]
    image = np.clip(image.astype(np.float32) + wave, 0, 180).astype(np.uint8)
    path = root / "images" / f"{split}_{index:04d}.png"
    Image.fromarray(image, mode="RGB").save(path)
    return {
        "image": str(path.resolve()),
        "mask": None,
        "split": split,
        "source_id": "smoke_external_true_normal",
        "lineage_id": f"smoke_normal_session_{index:04d}",
        "is_normal": True,
        "semantic_qc_status": "synthetic-known-empty",
    }


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))


def issue_gate0(py, repo, manifest, full_manifest, out, image_size, normal_policy):
    full_gate = out / "gate0_full.json"
    train_gate = out / "gate0_training.json"
    run(
        [
            py,
            "-m",
            "oasis_cycle_aosk.audit",
            "--manifest",
            full_manifest,
            "--test-split",
            "smoke_holdout",
            "--required-splits",
            "train",
            "val",
            "smoke_holdout",
            "--resize-size",
            str(image_size),
            "--normal-policy",
            normal_policy,
            "--certificate-out",
            full_gate,
            "--certificate-scope",
            "full_benchmark",
        ],
        repo,
    )
    run(
        [
            py,
            "-m",
            "oasis_cycle_aosk.audit",
            "--manifest",
            manifest,
            "--test-split",
            "smoke_holdout",
            "--required-splits",
            "train",
            "val",
            "--resize-size",
            str(image_size),
            "--normal-policy",
            normal_policy,
            "--certificate-out",
            train_gate,
            "--certificate-scope",
            "training_view",
            "--parent-full-certificate",
            full_gate,
        ],
        repo,
    )
    return full_gate, train_gate


def run_protocol(
    name,
    rows,
    out,
    repo,
    py,
    config,
    init,
    image_size,
    critic_epochs,
    student_epochs,
    normal_fraction,
):
    root = out / name
    root.mkdir(parents=True, exist_ok=True)
    keep = {"train", "val"}
    if normal_fraction > 0:
        keep |= {"normal_train", "normal_val"}
    train_rows = [r for r in rows if r["split"] in keep]
    full_manifest = root / "manifest_full.jsonl"
    train_manifest = root / "manifest_trainval.jsonl"
    write_jsonl(full_manifest, rows)
    write_jsonl(train_manifest, train_rows)
    policy = "train_and_aux_val" if normal_fraction > 0 else "none"
    full_gate, train_gate = issue_gate0(
        py, repo, train_manifest, full_manifest, root, image_size, policy
    )

    common = [
        "--config",
        config,
        "--manifest",
        train_manifest,
        "--gate0-certificate",
        train_gate,
        "--full-gate0-certificate",
        full_gate,
        "--normal-fraction",
        str(normal_fraction),
        "--critic-epochs",
        str(critic_epochs),
        "--critic-width",
        "8",
        "--lr",
        "0.003",
        "--weight-decay",
        "0.0",
        "--determinism-mode",
        "strict",
    ]
    critic_dir = root / "critic"
    run(
        [
            py,
            "-m",
            "oasis_cycle_aosk.train_oasis_rc_v21",
            *common,
            "--out",
            critic_dir,
            "--mode",
            "critic",
        ],
        repo,
    )
    critic_ckpt = critic_dir / "critic.pt"
    if not critic_ckpt.exists():
        raise RuntimeError(f"{name}: critic CLI completed without critic.pt")
    qualification = json.loads((critic_dir / "critic_qualification_v21.json").read_text())
    if qualification.get("pass") is not True:
        raise RuntimeError(f"{name}: critic qualification did not pass")
    if normal_fraction > 0 and qualification.get("normal_split") != "normal_val":
        raise RuntimeError(f"{name}: N25 critic did not qualify on normal_val")

    arms = {
        "B0": "control",
        "B1": "cldice",
        "B2": "adversarial",
        "S1": "connected",
        "S2": "aosk",
        "S3": "aosk_connected",
    }
    results = {}
    init_hashes = set()
    for arm, mode in arms.items():
        arm_dir = root / arm
        command = [
            py,
            "-m",
            "oasis_cycle_aosk.train_oasis_rc_v21",
            *common,
            "--out",
            arm_dir,
            "--mode",
            mode,
            "--student-init-checkpoint",
            init,
            "--student-kind",
            "mobilenetv3",
            "--student-width",
            "16",
            "--epochs",
            str(student_epochs),
            "--warmup",
            "0",
            "--ramp-epochs",
            "1",
        ]
        if mode in {"connected", "aosk_connected", "adversarial"}:
            command += ["--critic-checkpoint", critic_ckpt]
        run(command, repo)

        checkpoint = arm_dir / "student_only.pt"
        if not checkpoint.exists():
            raise RuntimeError(f"{name}/{arm}: missing student_only.pt")
        crack_eval = arm_dir / "eval_val.json"
        run(
            [
                py,
                "-m",
                "oasis_cycle_aosk.evaluate_v21",
                "--checkpoint",
                checkpoint,
                "--manifest",
                train_manifest,
                "--split",
                "val",
                "--device",
                "cpu",
                "--out",
                crack_eval,
                "--prediction-dir",
                arm_dir / "predictions_val",
            ],
            repo,
        )
        evaluation = json.loads(crack_eval.read_text())
        normal_eval = None
        if normal_fraction > 0:
            normal_path = arm_dir / "eval_normal_val.json"
            run(
                [
                    py,
                    "-m",
                    "oasis_cycle_aosk.evaluate_v21",
                    "--checkpoint",
                    checkpoint,
                    "--manifest",
                    train_manifest,
                    "--split",
                    "normal_val",
                    "--device",
                    "cpu",
                    "--out",
                    normal_path,
                ],
                repo,
            )
            normal_eval = json.loads(normal_path.read_text())

        import torch

        ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
        init_hashes.add(ck.get("student_init_sha256"))
        if ck.get("inference_contract") != "RGB -> crack logits only":
            raise RuntimeError(f"{name}/{arm}: inference contract changed")
        if ck.get("mode") != mode:
            raise RuntimeError(f"{name}/{arm}: checkpoint mode mismatch")
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
            "normal_any_fp_rate": (
                None if normal_eval is None else normal_eval["normal_any_fp_rate"]
            ),
            "normal_fp_pixels_mean": (
                None if normal_eval is None else normal_eval["normal_fp_pixels_mean"]
            ),
        }

    if len(init_hashes) != 1 or next(iter(init_hashes)) != sha256(init):
        raise RuntimeError(f"{name}: six arms did not share exactly one initialization")
    return {
        "normal_fraction": normal_fraction,
        "full_gate0": str(full_gate),
        "training_gate0": str(train_gate),
        "critic_checkpoint_sha256": sha256(critic_ckpt),
        "critic_qualification": qualification,
        "arms": results,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--critic-epochs", type=int, default=24)
    ap.add_argument("--student-epochs", type=int, default=2)
    ap.add_argument("--image-size", type=int, default=64)
    ap.add_argument("--train-samples", type=int, default=48)
    ap.add_argument("--val-samples", type=int, default=24)
    ap.add_argument("--holdout-samples", type=int, default=8)
    ap.add_argument("--normal-train-samples", type=int, default=24)
    ap.add_argument("--normal-val-samples", type=int, default=16)
    args = ap.parse_args()

    repo = Path(__file__).resolve().parents[1]
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    (out / "images").mkdir(exist_ok=True)
    (out / "masks").mkdir(exist_ok=True)

    rng = np.random.default_rng(1337)
    crack_rows = []
    cursor = 0
    for split, count in (
        ("train", args.train_samples),
        ("val", args.val_samples),
        ("smoke_holdout", args.holdout_samples),
    ):
        for _ in range(count):
            crack_rows.append(make_crack_case(out, cursor, split, args.image_size, rng))
            cursor += 1

    normal_rows = []
    normal_cursor = 0
    for split, count in (
        ("normal_train", args.normal_train_samples),
        ("normal_val", args.normal_val_samples),
    ):
        for _ in range(count):
            normal_rows.append(
                make_normal_case(out, normal_cursor, split, args.image_size, rng)
            )
            normal_cursor += 1

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
    os.environ.update(env)

    init = out / "student_init.pt"
    run(
        [
            py,
            repo / "scripts" / "create_student_init.py",
            "--out",
            init,
            "--seed",
            "1337",
            "--student-kind",
            "mobilenetv3",
            "--student-width",
            "16",
        ],
        repo,
    )

    protocols = {
        "N0": run_protocol(
            "N0",
            crack_rows,
            out,
            repo,
            py,
            config,
            init,
            args.image_size,
            args.critic_epochs,
            args.student_epochs,
            0.0,
        ),
        "N25": run_protocol(
            "N25",
            crack_rows + normal_rows,
            out,
            repo,
            py,
            config,
            init,
            args.image_size,
            args.critic_epochs,
            args.student_epochs,
            0.25,
        ),
    }
    expected_arms = {"B0", "B1", "B2", "S1", "S2", "S3"}
    for name, result in protocols.items():
        if set(result["arms"]) != expected_arms:
            raise RuntimeError(f"{name}: incomplete six-arm smoke")

    summary = {
        "status": "PASS",
        "kind": "actual-cli-integration-smoke-dev2",
        "implementation": "OASIS-RC-v2.1-dev2",
        "device": "cpu",
        "image_size": args.image_size,
        "student_kind": "mobilenetv3",
        "shared_student_init_sha256": sha256(init),
        "protocols": protocols,
        "canonical_test_created": False,
        "canonical_test_opened": False,
        "synthetic_smoke_holdout_evaluated": False,
    }
    (out / "ACTUAL_SMOKE_SUMMARY.json").write_text(json.dumps(summary, indent=2))
    print("ACTUAL_V21_DEV2_SMOKE_RESULT=" + json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
