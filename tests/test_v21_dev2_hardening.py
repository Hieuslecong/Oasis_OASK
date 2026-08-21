import math

import pytest
import torch

from oasis_cycle_aosk.aosk import oriented_consistency_loss, structure_tensor_tangent
from oasis_cycle_aosk.evaluate_v21 import FINAL_SPLITS, _row_metrics
from oasis_rc_v2.checkpoint import (
    CHECKPOINT_SCHEMA,
    EXPERIMENT_ID,
    IMPLEMENTATION_VERSION,
    METHOD_VERSION,
    sha256_file,
    validate_critic_checkpoint,
)
from oasis_rc_v2.final_bundle import canonical_bundle_id
from oasis_rc_v2.protocol import _verify_normal_policy


def _ridge(angle_deg, size=65, sigma=1.5):
    angle = math.radians(angle_deg)
    y, x = torch.meshgrid(
        torch.arange(size, dtype=torch.float32),
        torch.arange(size, dtype=torch.float32),
        indexing="ij",
    )
    x = x - (size - 1) / 2
    y = y - (size - 1) / 2
    # Tangent=(cos,sin); normal=(-sin,cos).
    normal_coord = -math.sin(angle) * x + math.cos(angle) * y
    ridge = torch.exp(-0.5 * (normal_coord / sigma).square())
    return ridge[None, None].repeat(1, 3, 1, 1)


@pytest.mark.parametrize("angle", [0, 30, 45, 60, 90])
def test_structure_tensor_tracks_arbitrary_crack_angles(angle):
    image = _ridge(angle)
    tx, ty, coherence = structure_tensor_tangent(image, window=7)
    true_x = math.cos(math.radians(angle))
    true_y = math.sin(math.radians(angle))
    # Ignore the outer border and weight by orientation confidence. Direction
    # has a 180-degree ambiguity, therefore compare |dot|.
    sl = (slice(None), slice(None), slice(8, -8), slice(8, -8))
    dot = (tx[sl] * true_x + ty[sl] * true_y).abs()
    weight = coherence[sl]
    score = float((dot * weight).sum() / weight.sum().clamp_min(1e-6))
    assert score >= math.cos(math.radians(15.0))


def test_structure_tensor_aosk_is_finite_and_differentiable_on_diagonal():
    image = _ridge(45)
    logits = torch.randn(1, 1, 65, 65, requires_grad=True)
    mask = (image[:, :1] > 0.5).float()
    loss = oriented_consistency_loss(logits, image, mask)
    loss.backward()
    assert torch.isfinite(loss)
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
    assert float(logits.grad.abs().sum()) > 0


def test_n25_accepts_stronger_train_and_aux_val_policy():
    splits = {"train", "val", "normal_train", "normal_val"}
    _verify_normal_policy("train", "train_and_aux_val", splits)
    with pytest.raises(ValueError):
        _verify_normal_policy("none", "train_and_aux_val", splits)
    with pytest.raises(ValueError):
        _verify_normal_policy("train", "train_and_aux_val", {"train", "val", "normal_train"})


def test_both_canonical_splits_are_firewalled():
    assert FINAL_SPLITS == {"test", "normal_test"}


def test_empty_target_does_not_create_fake_zero_dice():
    pred = torch.zeros(8, 8, dtype=torch.uint8).numpy()
    target = torch.zeros(8, 8, dtype=torch.uint8).numpy()
    row = _row_metrics(pred, target)
    assert row["is_normal"] is True
    assert row["dice"] is None
    assert row["iou"] is None
    assert row["fp_pixels"] == 0
    assert row["any_fp"] is False


def test_final_bundle_identity_is_path_relocation_invariant():
    base = {
        "schema": "oasis-rc-v2.1-final-bundle-v1",
        "manifest": "/a/manifest.jsonl",
        "manifest_sha256": "m" * 64,
        "dataset_content_sha256": "d" * 64,
        "full_gate0_certificate": "/a/gate.json",
        "full_gate0_certificate_sha256": "g" * 64,
        "method_spec": "/a/spec.md",
        "method_spec_sha256": "s" * 64,
        "protocol": "/a/protocol.json",
        "protocol_sha256": "p" * 64,
        "evaluator": "/a/eval.py",
        "evaluator_sha256": "e" * 64,
        "metric_spec_sha256": "x" * 64,
        "git_commit_sha": "c" * 40,
        "entries": [
            {
                "arm": "B0",
                "seed": 2027,
                "checkpoint": "/a/B0.pt",
                "checkpoint_sha256": "0" * 64,
                "threshold": 0.42,
            }
        ],
    }
    moved = {**base, "manifest": "/b/manifest.jsonl", "evaluator": "/b/eval.py"}
    moved["entries"] = [{**base["entries"][0], "checkpoint": "/b/B0.pt"}]
    assert canonical_bundle_id(base) == canonical_bundle_id(moved)


def test_critic_validator_requires_qualification_and_full_consumer_contract(tmp_path):
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text("{}\n")
    full = tmp_path / "full.json"
    full.write_text("{}")
    cfg = {"image_size": 256, "seed": 1337}
    saved = {
        "checkpoint_schema": CHECKPOINT_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "method_version": METHOD_VERSION,
        "implementation_version": IMPLEMENTATION_VERSION,
        "critic": {},
        "config": cfg,
        "manifest_file_sha256": sha256_file(manifest),
        "dataset_content_sha256": "data",
        "normal_fraction": 0.0,
        "normal_critic_weight": 1.0,
        "training_hparams": {
            "energy_head_contract": "dedicated-scalar-lower-is-better-v1",
            "mask_variant_contract": "operator-preserved-v1",
            "rgb_shuffle_pair_only": True,
            "mask_flip_training": False,
            "method_spec": "METHOD_SPEC_V2_1.md",
        },
        "width": 8,
        "seed": 1337,
        "full_gate0_certificate_sha256": sha256_file(full),
        "energy_head_contract": "dedicated-scalar-lower-is-better-v1",
        "qualification_v21": {"pass": True, "failures": []},
    }
    with pytest.raises(ValueError, match="full v2.1 contract"):
        validate_critic_checkpoint(
            saved,
            manifest,
            cfg,
            0.0,
            1.0,
            full_gate0_certificate=full,
            expected_hparams={"energy_head_contract": "dedicated-scalar-lower-is-better-v1"},
        )

    bad = {**saved, "qualification_v21": {"pass": False, "failures": ["energy"]}}
    with pytest.raises(ValueError, match="not v2.1-qualified"):
        validate_critic_checkpoint(
            bad,
            manifest,
            cfg,
            0.0,
            1.0,
            full_gate0_certificate=full,
            expected_hparams=saved["training_hparams"],
        )
