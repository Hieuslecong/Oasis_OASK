import json

import pytest
import torch
from PIL import Image

from oasis_cycle_aosk.aosk import oriented_consistency_loss
from oasis_cycle_aosk.audit import audit
from oasis_cycle_aosk.data import ManifestDataset
from oasis_cycle_aosk.samplers import MixedBatchSampler


def _save_rgb(path, value=127, size=(8, 8)):
    Image.new("RGB", size, color=(value, value, value)).save(path)


def _save_mask(path, cracked=True, size=(8, 8)):
    im = Image.new("L", size, color=0)
    if cracked:
        px = im.load()
        for i in range(min(size)):
            px[i, i] = 255
    im.save(path)


def test_true_normal_row_produces_virtual_zero_mask(tmp_path):
    image = tmp_path / "normal.png"
    _save_rgb(image)
    manifest = tmp_path / "m.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "image": str(image),
                "mask": None,
                "split": "normal_train",
                "source_id": "walls",
                "lineage_id": "walls::normal.png",
                "is_normal": True,
            }
        )
        + "\n"
    )
    x, y, is_normal = ManifestDataset(
        manifest, "normal_train", 16, return_is_normal=True
    )[0]
    assert x.shape == (3, 16, 16)
    assert y.shape == (1, 16, 16)
    assert float(y.sum()) == 0.0
    assert bool(is_normal) is True


def test_crack_row_missing_mask_is_not_silently_normal(tmp_path):
    image = tmp_path / "crack.png"
    _save_rgb(image)
    manifest = tmp_path / "m.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "image": str(image),
                "mask": None,
                "split": "train",
                "source_id": "crack",
                "lineage_id": "crack::1",
                "is_normal": False,
            }
        )
        + "\n"
    )
    ds = ManifestDataset(manifest, "train", 16)
    with pytest.raises(ValueError, match="missing mask"):
        ds[0]


def test_explicit_crack_identity_survives_empty_resized_tensor(tmp_path):
    image = tmp_path / "crack.png"
    mask = tmp_path / "mask.png"
    _save_rgb(image, size=(8, 8))
    im = Image.new("L", (8, 8), color=0)
    im.putpixel((0, 0), 255)
    im.save(mask)
    manifest = tmp_path / "m.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "image": str(image),
                "mask": str(mask),
                "split": "train",
                "source_id": "crack",
                "lineage_id": "crack::tiny",
                "is_normal": False,
            }
        )
        + "\n"
    )
    _, y, is_normal = ManifestDataset(
        manifest, "train", 1, return_is_normal=True
    )[0]
    assert float(y.sum()) == 0.0
    assert bool(is_normal) is False


def test_mixed_sampler_has_fixed_composition_and_is_deterministic():
    s1 = MixedBatchSampler(12, 20, batch_size=8, normal_fraction=0.25, seed=1337)
    s2 = MixedBatchSampler(12, 20, batch_size=8, normal_fraction=0.25, seed=1337)
    b1 = list(s1)
    b2 = list(s2)
    assert b1 == b2
    assert b1
    assert s1.realized_normal_fraction == 0.25
    for batch in b1:
        normal = sum(i >= 12 for i in batch)
        crack = sum(i < 12 for i in batch)
        assert normal == 2
        assert crack == 6


def test_normal_fraction_does_not_increase_optimizer_steps_per_epoch():
    # Crack-only baseline with 24 samples, batch 8 has exactly 3 updates.
    mixed = MixedBatchSampler(24, 100, batch_size=8, normal_fraction=0.25, seed=1)
    assert len(mixed) == 3
    assert mixed.samples_per_epoch == 24
    assert mixed.crack_samples_per_epoch == 18
    assert mixed.normal_samples_per_epoch == 6


def test_mixed_sampler_rejects_requested_fraction_that_rounds_to_zero():
    with pytest.raises(ValueError, match="too small"):
        MixedBatchSampler(12, 20, batch_size=4, normal_fraction=0.10, seed=1337)


def test_audit_rejects_split_qualified_lineage_and_cross_split_duplicate(tmp_path):
    image = tmp_path / "same.png"
    image_copy = tmp_path / "same_copy.png"
    mask = tmp_path / "mask.png"
    _save_rgb(image)
    image_copy.write_bytes(image.read_bytes())
    _save_mask(mask)
    rows = [
        {
            "image": str(image),
            "mask": str(mask),
            "split": "train",
            "source_id": "s",
            "lineage_id": "train::parent",
            "is_normal": False,
        },
        {
            "image": str(image_copy),
            "mask": str(mask),
            "split": "val",
            "source_id": "s2",
            "lineage_id": "parent2",
            "is_normal": False,
        },
        {
            "image": str(tmp_path / "test.png"),
            "mask": str(mask),
            "split": "test",
            "source_id": "s3",
            "lineage_id": "parent3",
            "is_normal": False,
        },
    ]
    _save_rgb(tmp_path / "test.png", value=64)
    manifest = tmp_path / "m.jsonl"
    manifest.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    errors = audit(manifest)
    assert any("split-qualified" in e for e in errors)
    assert any("duplicate across splits" in e for e in errors)


def test_actual_aosk_loss_backpropagates_to_logits():
    logits = torch.randn(2, 1, 16, 16, requires_grad=True)
    image = torch.randn(2, 3, 16, 16)
    mask = torch.zeros(2, 1, 16, 16)
    mask[:, :, 7:9, :] = 1.0
    loss = oriented_consistency_loss(logits, image, mask)
    assert torch.isfinite(loss)
    assert loss.requires_grad
    loss.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
    assert float(logits.grad.abs().sum()) > 0.0


def test_exclude_file_removes_cross_label_duplicate_from_normal_train(tmp_path):
    from scripts.add_normal_rgb_to_manifest import main as build_manifest  # type: ignore

    normal_root = tmp_path / "Non-cracked"
    normal_root.mkdir()
    keep = normal_root / "keep.png"
    dup = normal_root / "dup.png"
    _save_rgb(keep)
    _save_rgb(dup)
    canonical = tmp_path / "canonical.jsonl"
    canonical.write_text(
        json.dumps(
            {
                "image": str(tmp_path / "crack.png"),
                "mask": str(tmp_path / "mask.png"),
                "split": "train",
                "source_id": "s",
                "lineage_id": "s::1",
                "is_normal": False,
            }
        )
        + "\n"
    )
    _save_rgb(tmp_path / "crack.png")
    _save_mask(tmp_path / "mask.png")
    exclude = tmp_path / "exclude.json"
    exclude.write_text(
        json.dumps({"excluded_normal_candidates": [{"path": str(dup)}]})
    )
    out = tmp_path / "manifest_with_normal.jsonl"
    import sys

    sys.argv = [
        "add_normal_rgb_to_manifest",
        "--canonical-manifest",
        str(canonical),
        "--normal-root",
        str(normal_root),
        "--out",
        str(out),
        "--train-ratio",
        "1.0",
        "--exclude-file",
        str(exclude),
    ]
    build_manifest()
    rows = [json.loads(l) for l in out.read_text().splitlines() if l.strip()]
    normal_rows = [r for r in rows if r.get("is_normal")]
    assert len(normal_rows) == 1
    assert normal_rows[0]["image"] == str(keep)
    assert all(r["image"] != str(dup) for r in normal_rows)
