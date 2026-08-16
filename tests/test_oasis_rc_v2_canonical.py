import hashlib
import inspect
import json

import pytest
from PIL import Image

torch = pytest.importorskip("torch")

from oasis_rc_v2.checkpoint import (
    CHECKPOINT_SCHEMA,
    EXPERIMENT_ID,
    IMPLEMENTATION_VERSION,
    METHOD_VERSION,
    validate_student_checkpoint,
)
from oasis_rc_v2.corruptions import CORRUPTION_NAMES, make_corrupted_mask
from oasis_rc_v2.critic import OASISRCv2Critic
from oasis_rc_v2.losses import (
    balanced_semantic_cross_entropy,
    oasis_rc_critic_loss,
)
from oasis_rc_v2.protocol import dataset_content_sha256, verify_gate0_certificate


def _batch():
    mask = torch.zeros(3, 1, 32, 32)
    mask[0, 0, 8:24, 15:17] = 1
    mask[1, 0, 10:22, 5:7] = 1
    return mask, torch.tensor([False, False, True])


def test_c1_c9_are_nonempty_and_no_torch_roll():
    mask, normal = _batch()
    image = torch.rand(3, 3, 32, 32)
    g = torch.Generator().manual_seed(7)
    seen = set()
    for kind in range(9):
        forced = [kind, kind, 7 if kind == 7 else 8]
        _, invalid, meta = make_corrupted_mask(
            mask,
            normal,
            generator=g,
            forced_kinds=forced,
            return_meta=True,
            image=image,
        )
        assert (invalid.flatten(1).sum(1) > 0).all()
        assert meta[0]["changed_pixels"] > 0
        seen.add(meta[0]["kind"])
        if kind == 8:
            assert meta[0]["texture_guided"] is True
    assert set(CORRUPTION_NAMES).issubset(seen | {"C8_crack_on_normal"})
    source = inspect.getsource(__import__("oasis_rc_v2.corruptions", fromlist=["x"]))
    assert "torch.roll(" not in source


def test_donor_is_nonself_and_crack_positive():
    mask, normal = _batch()
    _, _, meta = make_corrupted_mask(
        mask,
        normal,
        generator=torch.Generator().manual_seed(9),
        forced_kinds=[6, 6, 7],
        return_meta=True,
    )
    assert meta[0]["donor_index"] in {1}
    assert meta[1]["donor_index"] in {0}
    assert meta[0]["donor_index"] != 0
    assert meta[1]["donor_index"] != 1
    assert meta[2]["kind"] == "C8_crack_on_normal"
    assert meta[2]["donor_index"] in {0, 1}


def test_semantic_loss_balances_active_classes_not_pixel_frequency():
    logits = torch.tensor(
        [[[[2.0, 2.0]], [[0.0, 0.0]], [[-1.0, -1.0]]]],
        requires_grad=True,
    )
    target = torch.tensor([[[0, 1]]])
    base = balanced_semantic_cross_entropy(logits, target)
    repeated_logits = torch.cat([logits[:, :, :1, :1].expand(-1, -1, 1, 20), logits[:, :, :1, 1:]], dim=3)
    repeated_target = torch.cat(
        [torch.zeros(1, 1, 20, dtype=torch.long), torch.ones(1, 1, 1, dtype=torch.long)],
        dim=2,
    )
    repeated = balanced_semantic_cross_entropy(repeated_logits, repeated_target)
    assert torch.allclose(base, repeated)


def test_critic_loss_contains_valid_crack_dice():
    mask, _ = _batch()
    critic = OASISRCv2Critic(width=4)
    out = critic(torch.rand(3, 3, 32, 32), mask)
    sem = mask[:, 0].long()
    mm = torch.zeros_like(mask)
    pv = torch.ones(3, 1)
    loss, parts = oasis_rc_critic_loss(out, sem, mm, pv)
    loss.backward()
    assert torch.isfinite(loss)
    assert "valid_crack_dice" in parts and torch.isfinite(parts["valid_crack_dice"])
    assert float(parts["rgb_shuffle_pair_only"]) == 0.0


def test_rgb_shuffle_term_is_pair_only_not_mask_semantic_supervision():
    semantic = torch.randn(2, 3, 8, 8, requires_grad=True)
    crack = torch.randn(2, 1, 8, 8, requires_grad=True)
    mismatch = torch.randn(2, 1, 8, 8, requires_grad=True)
    pair = torch.randn(2, 1, requires_grad=True)
    out = {"semantic": semantic, "crack": crack, "mismatch": mismatch, "pair": pair}
    sem_target = torch.zeros(2, 8, 8, dtype=torch.long)
    mm = torch.zeros(2, 1, 8, 8)
    pv = torch.zeros(2, 1)
    loss, parts = oasis_rc_critic_loss(out, sem_target, mm, pv)
    loss.backward()
    assert float(parts["rgb_shuffle_pair_only"]) == 1.0
    assert pair.grad is not None and float(pair.grad.abs().sum()) > 0
    assert semantic.grad is not None and float(semantic.grad.abs().sum()) == 0.0
    assert crack.grad is not None and float(crack.grad.abs().sum()) == 0.0
    assert mismatch.grad is not None and float(mismatch.grad.abs().sum()) == 0.0


def test_gate0_certificate_binds_dataset_bytes(tmp_path):
    image = tmp_path / "image.png"
    mask = tmp_path / "mask.png"
    Image.new("RGB", (8, 8), (10, 20, 30)).save(image)
    Image.new("L", (8, 8), 0).save(mask)
    manifest = tmp_path / "trainval.jsonl"
    row = {
        "image": str(image),
        "mask": str(mask),
        "split": "train",
        "source_id": "s",
        "lineage_id": "l",
        "is_normal": False,
        "empty_target_status": "verified_no_crack",
    }
    manifest.write_text(json.dumps(row) + "\n")
    full = tmp_path / "gate0_full.json"
    full.write_text(json.dumps({"status": "PASS", "scope": "full_benchmark"}))
    cert = tmp_path / "gate0.json"
    cert.write_text(
        json.dumps(
            {
                "status": "PASS",
                "scope": "training_view",
                "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
                "dataset_content_sha256": dataset_content_sha256(manifest),
                "resize_size": 256,
                "normal_policy": "none",
                "parent_full_gate0_certificate_sha256": hashlib.sha256(
                    full.read_bytes()
                ).hexdigest(),
            }
        )
    )
    verify_gate0_certificate(cert, manifest, 256, "none", full)
    other_full = tmp_path / "other_full.json"
    other_full.write_text(
        json.dumps({"status": "PASS", "scope": "full_benchmark", "other": True})
    )
    with pytest.raises(ValueError, match="parent full Gate 0 mismatch"):
        verify_gate0_certificate(cert, manifest, 256, "none", other_full)
    Image.new("RGB", (8, 8), (11, 20, 30)).save(image)
    with pytest.raises(ValueError, match="dataset-content SHA256"):
        verify_gate0_certificate(cert, manifest, 256, "none", full)


def test_student_checkpoint_rejects_legacy_and_wrong_implementation():
    with pytest.raises(ValueError, match="legacy checkpoint rejected"):
        validate_student_checkpoint({"student": {}})
    good = {
        "checkpoint_schema": CHECKPOINT_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "method_version": METHOD_VERSION,
        "implementation_version": IMPLEMENTATION_VERSION,
        "student": {},
        "student_kind": "multiscale",
        "student_width": 16,
        "seed": 1337,
        "mode": "control",
        "effective_config": {
            "image_size": 256,
            "student_kind": "multiscale",
            "student_width": 16,
            "seed": 1337,
        },
        "threshold_validation": 0.5,
        "manifest_file_sha256": "a" * 64,
        "dataset_content_sha256": "b" * 64,
        "training_view_dataset_sha256": "b" * 64,
        "gate0_certificate_sha256": "c" * 64,
        "full_gate0_certificate_sha256": "d" * 64,
        "student_init_sha256": "e" * 64,
        "inference_contract": "RGB -> crack logits only",
    }
    validate_student_checkpoint(good)
    bad = dict(good)
    bad["implementation_version"] = "legacy"
    with pytest.raises(ValueError, match="implementation_version"):
        validate_student_checkpoint(bad)
