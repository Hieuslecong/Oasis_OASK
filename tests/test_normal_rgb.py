import json

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
    x, y = ManifestDataset(manifest, "normal_train", 16)[0]
    assert x.shape == (3, 16, 16)
    assert y.shape == (1, 16, 16)
    assert float(y.sum()) == 0.0


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
    try:
        ds[0]
    except ValueError as exc:
        assert "missing mask" in str(exc)
    else:
        raise AssertionError("crack row without mask must fail")


def test_mixed_sampler_has_fixed_composition_and_is_deterministic():
    s1 = MixedBatchSampler(12, 20, batch_size=8, normal_fraction=0.25, seed=1337)
    s2 = MixedBatchSampler(12, 20, batch_size=8, normal_fraction=0.25, seed=1337)
    b1 = list(s1)
    b2 = list(s2)
    assert b1 == b2
    assert b1
    for batch in b1:
        normal = sum(i >= 12 for i in batch)
        crack = sum(i < 12 for i in batch)
        assert normal == 2
        assert crack == 6


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
