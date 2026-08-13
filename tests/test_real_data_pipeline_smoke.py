import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from PIL import Image

pytest.importorskip("torch")

REPO = Path(__file__).resolve().parents[1]


def _rgb(path, color):
    Image.new("RGB", (32, 32), color=color).save(path)


def _mask(path, kind):
    im = Image.new("L", (32, 32), 0)
    px = im.load()
    if kind == "h":
        for x in range(4, 28):
            px[x, 8] = 255
    elif kind == "v":
        for y in range(4, 28):
            px[20, y] = 255
    else:
        for i in range(5, 27):
            px[i, i] = 255
    im.save(path)


def _run(args, env=None):
    merged = os.environ.copy()
    merged["PYTHONPATH"] = str(REPO / "src") + os.pathsep + merged.get("PYTHONPATH", "")
    if env:
        merged.update({k: str(v) for k, v in env.items()})
    return subprocess.run(
        [str(x) for x in args],
        cwd=REPO,
        env=merged,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=True,
    )


def test_real_file_pipeline_reaches_optimizer_and_checkpoint(tmp_path):
    data = tmp_path / "raw"
    data.mkdir()
    rows = []
    specs = [
        ("train", (110, 15, 15), "h"),
        ("val", (15, 110, 15), "v"),
        ("test", (15, 15, 110), "d"),
    ]
    for idx, (split, color, mkind) in enumerate(specs):
        image = data / f"{split}.png"
        mask = data / f"{split}_mask.png"
        _rgb(image, color)
        _mask(mask, mkind)
        rows.append(
            {
                "image": str(image),
                "mask": str(mask),
                "split": split,
                "source_id": f"synthetic_{split}",
                "lineage_id": f"parent_{idx}",
                "is_normal": False,
            }
        )
    canonical = tmp_path / "canonical.jsonl"
    canonical.write_text("\n".join(json.dumps(r) for r in rows) + "\n")

    normals = tmp_path / "normals"
    normals.mkdir()
    _rgb(normals / "normal_a.png", (160, 160, 160))
    _rgb(normals / "normal_b.png", (205, 205, 205))

    derived = tmp_path / "derived"
    _run(
        ["bash", REPO / "scripts" / "prepare_real_data.sh"],
        {
            "DATA_ROOT": derived,
            "CANONICAL_MANIFEST": canonical,
            "NORMAL_ROOT": normals,
            "PYTHON": sys.executable,
        },
    )

    full_manifest = derived / "manifest_full_with_normal.jsonl"
    train_manifest = derived / "manifest_trainval_with_normal.jsonl"
    assert full_manifest.exists() and train_manifest.exists()
    train_rows = [json.loads(x) for x in train_manifest.read_text().splitlines() if x.strip()]
    assert {r["split"] for r in train_rows} == {"train", "val", "normal_train"}
    assert all(r["split"] != "test" for r in train_rows)
    assert json.loads((derived / "gate0_full.json").read_text())["status"] == "PASS"

    cert32 = tmp_path / "gate0_train_32.json"
    _run(
        [
            sys.executable,
            "-m",
            "oasis_cycle_aosk.audit",
            "--manifest",
            train_manifest,
            "--resize-size",
            "32",
            "--normal-policy",
            "train",
            "--required-splits",
            "train",
            "val",
            "--certificate-out",
            cert32,
            "--certificate-scope",
            "training_view",
        ]
    )

    config = tmp_path / "cpu32.yaml"
    config.write_text(
        "seed: 1337\nimage_size: 32\nbatch_size: 2\ndevice: cpu\n"
        "num_workers: 0\nlambda_oasis: 0.001\ncritic_width: 4\n"
    )
    init = tmp_path / "student_init.pt"
    _run(
        [
            sys.executable,
            REPO / "scripts" / "create_student_init.py",
            "--seed",
            "1337",
            "--student-kind",
            "multiscale",
            "--student-width",
            "4",
            "--out",
            init,
        ]
    )

    critic_out = tmp_path / "critic"
    _run(
        [
            sys.executable,
            "-m",
            "oasis_cycle_aosk.train_oasis_rc_v2",
            "--config",
            config,
            "--manifest",
            train_manifest,
            "--gate0-certificate",
            cert32,
            "--out",
            critic_out,
            "--mode",
            "critic",
            "--critic-width",
            "4",
            "--critic-epochs",
            "1",
            "--normal-fraction",
            "0.5",
            "--deterministic",
        ]
    )
    assert (critic_out / "critic.pt").exists()
    assert (critic_out / "critic_validation.json").exists()

    student_out = tmp_path / "control"
    _run(
        [
            sys.executable,
            "-m",
            "oasis_cycle_aosk.train_oasis_rc_v2",
            "--config",
            config,
            "--manifest",
            train_manifest,
            "--gate0-certificate",
            cert32,
            "--out",
            student_out,
            "--mode",
            "control",
            "--student-kind",
            "multiscale",
            "--student-width",
            "4",
            "--student-init-checkpoint",
            init,
            "--epochs",
            "1",
            "--normal-fraction",
            "0.5",
            "--deterministic",
        ]
    )
    assert (student_out / "student_only.pt").exists()
    validation = json.loads((student_out / "validation.json").read_text())
    assert 0.05 <= float(validation["threshold"]) <= 0.95
