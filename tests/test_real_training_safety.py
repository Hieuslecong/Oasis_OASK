import json

import pytest
from PIL import Image

torch = pytest.importorskip("torch")
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from oasis_cycle_aosk.train_oasis_rc_v2 import (
    configure_determinism,
    threshold_sweep_metrics,
)
from oasis_rc_v2.checkpoint import sha256_file
from oasis_rc_v2.protocol import dataset_content_sha256, verify_final_test_authorization
from oasis_rc_v2.qualification import critic_gate_failures


class CountingModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, x):
        self.calls += 1
        return x[:, :1]


def test_threshold_sweep_forwards_once_per_validation_batch():
    x = torch.randn(10, 3, 8, 8)
    y = torch.zeros(10, 1, 8, 8)
    y[:, :, 2:6, 3:5] = 1
    loader = DataLoader(TensorDataset(x, y), batch_size=4, shuffle=False)
    model = CountingModel()
    results = threshold_sweep_metrics(
        model,
        loader,
        torch.device("cpu"),
        thresholds=[0.1, 0.2, 0.3, 0.4, 0.5],
        chunk_size=2,
    )
    assert model.calls == len(loader)
    assert len(results) == 5
    assert all(0.0 <= row["threshold"] <= 1.0 for row in results)


def test_best_effort_determinism_does_not_enable_strict_throwing_on_cpu():
    previous_enabled = torch.are_deterministic_algorithms_enabled()
    previous_warn = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        configure_determinism("best_effort", "cpu")
        assert torch.are_deterministic_algorithms_enabled()
        assert torch.is_deterministic_algorithms_warn_only_enabled()
        configure_determinism("strict", "cpu")
        assert torch.are_deterministic_algorithms_enabled()
        assert not torch.is_deterministic_algorithms_warn_only_enabled()
    finally:
        torch.use_deterministic_algorithms(previous_enabled, warn_only=previous_warn)


def _make_full_manifest(tmp_path):
    image = tmp_path / "test.png"
    mask = tmp_path / "test_mask.png"
    Image.new("RGB", (8, 8), (20, 40, 60)).save(image)
    m = Image.new("L", (8, 8), 0)
    m.putpixel((4, 4), 255)
    m.save(mask)
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "image": str(image),
                "mask": str(mask),
                "split": "test",
                "source_id": "test_source",
                "lineage_id": "test_lineage",
                "is_normal": False,
            }
        )
        + "\n"
    )
    return manifest


def test_final_test_authorization_binds_checkpoint_manifest_data_and_threshold(tmp_path):
    checkpoint = tmp_path / "student.pt"
    checkpoint.write_bytes(b"checkpoint-bytes")
    manifest = _make_full_manifest(tmp_path)
    auth = tmp_path / "opened.json"
    payload = {
        "state": "OPENED",
        "checkpoint_sha256": sha256_file(checkpoint),
        "manifest_sha256": sha256_file(manifest),
        "dataset_content_sha256": dataset_content_sha256(manifest),
        "threshold": 0.42,
    }
    auth.write_text(json.dumps(payload))
    verify_final_test_authorization(auth, checkpoint, manifest, 0.42)

    payload["threshold"] = 0.43
    auth.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="threshold mismatch"):
        verify_final_test_authorization(auth, checkpoint, manifest, 0.42)


def _good_critic_metrics():
    names = (
        "C1_translation",
        "C2_erosion",
        "C3_dilation",
        "C4_local_break",
        "C5_wrong_width",
        "C6_wrong_connection",
        "C7_donor_mask",
        "C8_crack_on_normal",
        "C9_texture_fp_blob",
    )
    return {
        "valid_crack_recall": 0.90,
        "invalid_recall": 0.95,
        "rgb_pair_drop": 0.10,
        "mask_pair_drop": 0.10,
        "valid_normal_bg_recall": 0.90,
        "normal_pair_valid_mean": 0.80,
        "normal_invalid_rate": 0.05,
        "min_corruption_invalid_recall": 0.80,
        "rgb_pair_samples": 10,
        "mask_pair_samples": 10,
        "normal_samples": 10,
        "valid_crack_predictions": 20,
        "corruption_invalid_recall": {name: 0.80 for name in names},
    }


def test_critic_gate_requires_true_normal_and_all_c1_c9():
    metrics = _good_critic_metrics()
    assert critic_gate_failures(metrics) == []

    no_normal = dict(metrics)
    no_normal["normal_samples"] = 0
    assert "normal_samples>0" in critic_gate_failures(no_normal)

    missing_kind = dict(metrics)
    missing_kind["corruption_invalid_recall"] = dict(metrics["corruption_invalid_recall"])
    missing_kind["corruption_invalid_recall"].pop("C6_wrong_connection")
    assert "C6_wrong_connection:samples>0" in critic_gate_failures(missing_kind)
