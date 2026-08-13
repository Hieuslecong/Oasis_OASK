import inspect
import json
import pytest

torch = pytest.importorskip("torch")

from oasis_rc_v2.checkpoint import (
    CHECKPOINT_SCHEMA,
    EXPERIMENT_ID,
    METHOD_VERSION,
    validate_student_checkpoint,
)
from oasis_rc_v2.corruptions import CORRUPTION_NAMES, make_corrupted_mask
from oasis_rc_v2.critic import OASISRCv2Critic
from oasis_rc_v2.losses import oasis_rc_critic_loss
from oasis_rc_v2.protocol import verify_gate0_certificate


def _batch():
    m = torch.zeros(3, 1, 32, 32)
    m[0, 0, 8:24, 15:17] = 1
    m[1, 0, 10:22, 5:7] = 1
    return m, torch.tensor([False, False, True])


def test_c1_c9_are_nonempty_and_no_torch_roll():
    mask, normal = _batch()
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
        )
        assert (invalid.flatten(1).sum(1) > 0).all()
        assert meta[0]["changed_pixels"] > 0
        seen.add(meta[0]["kind"])
    assert set(CORRUPTION_NAMES).issubset(seen | {"C8_crack_on_normal"})
    source = inspect.getsource(__import__("oasis_rc_v2.corruptions", fromlist=["x"]))
    assert "torch.roll(" not in source


def test_donor_is_nonself():
    mask, normal = _batch()
    _, _, meta = make_corrupted_mask(
        mask,
        normal,
        generator=torch.Generator().manual_seed(9),
        forced_kinds=[6, 6, 7],
        return_meta=True,
    )
    assert meta[0]["donor_index"] is not None and meta[0]["donor_index"] != 0
    assert meta[1]["donor_index"] is not None and meta[1]["donor_index"] != 1


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


def test_gate0_certificate_binds_training_manifest(tmp_path):
    m = tmp_path / "trainval.jsonl"
    m.write_text('{"split":"train"}\n')
    import hashlib

    h = hashlib.sha256(m.read_bytes()).hexdigest()
    cert = tmp_path / "gate0.json"
    cert.write_text(
        json.dumps(
            {
                "status": "PASS",
                "scope": "training_view",
                "manifest_sha256": h,
                "resize_size": 256,
                "normal_policy": "train",
            }
        )
    )
    verify_gate0_certificate(cert, m, 256, "train")
    m.write_text('{"split":"val"}\n')
    with pytest.raises(ValueError, match="SHA256"):
        verify_gate0_certificate(cert, m, 256, "train")


def test_student_checkpoint_rejects_legacy():
    with pytest.raises(ValueError, match="legacy checkpoint rejected"):
        validate_student_checkpoint({"student": {}})
    good = {
        "checkpoint_schema": CHECKPOINT_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "method_version": METHOD_VERSION,
        "student": {},
    }
    validate_student_checkpoint(good)
