from types import SimpleNamespace

import pytest
import torch

from oasis_cycle_aosk import train_oasis_rc_v21 as v21
from oasis_rc_v2.checkpoint import (
    CHECKPOINT_SCHEMA,
    EXPERIMENT_ID,
    IMPLEMENTATION_VERSION,
    METHOD_VERSION,
    TRAINER_CONTRACT,
    validate_student_checkpoint,
)


def _student_checkpoint():
    return {
        "checkpoint_schema": CHECKPOINT_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "implementation_version": IMPLEMENTATION_VERSION,
        "method_version": METHOD_VERSION,
        "student": {},
        "student_kind": "mobilenetv3",
        "student_width": 16,
        "seed": 2027,
        "mode": "control",
        "effective_config": {
            "student_kind": "mobilenetv3",
            "student_width": 16,
            "seed": 2027,
        },
        "threshold_validation": 0.5,
        "manifest_file_sha256": "m",
        "dataset_content_sha256": "d",
        "training_view_dataset_sha256": "d",
        "gate0_certificate_sha256": "g",
        "full_gate0_certificate_sha256": "f",
        "student_init_sha256": "i",
        "inference_contract": "RGB -> crack logits only",
        "trainer_contract": TRAINER_CONTRACT,
    }


def test_legacy_student_checkpoint_cannot_masquerade_as_v21():
    ck = _student_checkpoint()
    ck.pop("trainer_contract")
    with pytest.raises(ValueError, match="trainer_contract"):
        validate_student_checkpoint(ck)


def test_v21_default_student_budget_is_not_twelve_epochs():
    defaults = {action.dest: action.default for action in v21.parser()._actions}
    assert defaults["epochs"] == 100


def test_n25_normal_energy_failure_is_fail_closed(tmp_path, monkeypatch):
    monkeypatch.setattr(v21, "make_loader", lambda manifest, split, *a, **k: split)
    monkeypatch.setattr(v21, "manifest_has_split", lambda *a, **k: True)
    monkeypatch.setattr(v21, "critic_metrics", lambda *a, **k: {"ok": True})
    monkeypatch.setattr(v21, "connected_gate_failures", lambda *a, **k: [])
    monkeypatch.setattr(
        v21,
        "energy_qualification",
        lambda critic, loader, device, margin: {"loader": loader},
    )
    monkeypatch.setattr(
        v21,
        "normal_donor_energy_qualification",
        lambda critic, crack_loader, normal_loader, device, margin: {
            "loader": "normal_donor"
        },
    )
    monkeypatch.setattr(
        v21,
        "relation_energy_gate_failures",
        lambda metrics: ["gap"]
        if metrics["loader"] in {"normal_val", "normal_donor"}
        else [],
    )
    args = SimpleNamespace(manifest="m", normal_fraction=0.25, path_margin=0.02)
    cfg = {"image_size": 256, "batch_size": 4, "seed": 1337, "num_workers": 0}
    report = v21.qualify_critic(None, args, cfg, "cpu", tmp_path)
    assert report["pass"] is False
    assert report["failures"] == ["normal_texture_gap", "normal_donor_gap"]


def test_c8_normal_donor_energy_uses_heldout_normal_rgb(monkeypatch):
    crack_y = torch.zeros(3, 1, 8, 8)
    crack_y[0, 0, 2:6, 3] = 1
    crack_y[1, 0, 4, 1:7] = 1
    normal_y = torch.zeros(2, 1, 8, 8)
    crack_loader = [
        (torch.zeros(3, 3, 8, 8), crack_y, torch.zeros(3, dtype=torch.bool))
    ]
    normal_loader = [
        (torch.zeros(2, 3, 8, 8), normal_y, torch.ones(2, dtype=torch.bool))
    ]
    seen = {}

    def trajectory(critic, x, y, wrong, t_values):
        seen["wrong_sum"] = float(wrong.sum())
        gap = wrong.flatten(1).mean(1)
        return torch.stack([gap * float(t) for t in t_values], dim=1)

    monkeypatch.setattr(v21, "relation_energy_trajectory", trajectory)
    monkeypatch.setattr(
        v21,
        "summarize_energy_trajectories",
        lambda e, t_values, margin: {
            "energy_samples": int(e.shape[0]),
            "energy_finite": bool(torch.isfinite(e).all()),
            "positive_energy_gap_fraction": float(
                (e[:, -1] > e[:, 0]).float().mean()
            ),
        },
    )
    result = v21.normal_donor_energy_qualification(
        None, crack_loader, normal_loader, "cpu", 0.02
    )
    assert result["energy_samples"] == 2
    assert result["energy_finite"] is True
    assert result["positive_energy_gap_fraction"] == 1.0
    assert seen["wrong_sum"] > 0
