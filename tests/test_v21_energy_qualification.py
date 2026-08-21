import pytest

torch = pytest.importorskip("torch")

from oasis_rc_v2.energy_qualification import (
    gradient_diagnostics,
    interpolate_mask,
    summarize_energy_trajectories,
)
from oasis_rc_v2.qualification import (
    relation_energy_gate_failures,
    relation_energy_gate_passes,
)


def test_interpolate_mask_endpoints_and_soft_midpoint():
    gt = torch.zeros(1, 1, 2, 2)
    corrupted = torch.ones_like(gt)
    assert torch.equal(interpolate_mask(gt, corrupted, 0.0), gt)
    assert torch.equal(interpolate_mask(gt, corrupted, 1.0), corrupted)
    assert torch.allclose(
        interpolate_mask(gt, corrupted, 0.5),
        torch.full_like(gt, 0.5),
    )


def test_energy_summary_passes_for_monotone_positive_paths():
    energies = torch.tensor(
        [
            [0.10, 0.20, 0.30, 0.40, 0.50],
            [0.12, 0.18, 0.29, 0.35, 0.44],
        ]
        * 8,
        dtype=torch.float32,
    )
    metrics = summarize_energy_trajectories(energies)
    assert metrics["energy_samples"] == 16
    assert metrics["energy_finite"] is True
    assert metrics["positive_energy_gap_fraction"] == pytest.approx(1.0)
    assert metrics["continuous_path_order_fraction"] == pytest.approx(1.0)
    assert relation_energy_gate_passes(metrics)


def test_energy_gate_rejects_inverted_or_too_small_sample():
    energies = torch.tensor(
        [[0.8, 0.6, 0.4, 0.2, 0.1]] * 8,
        dtype=torch.float32,
    )
    metrics = summarize_energy_trajectories(energies)
    failures = relation_energy_gate_failures(metrics)
    assert any("energy_samples" in item for item in failures)
    assert any("positive_energy_gap_fraction" in item for item in failures)
    assert any("median_energy_gap" in item for item in failures)


def test_gradient_diagnostics_reports_alignment_and_ratio():
    seg = torch.tensor([1.0, 0.0, 1.0])
    rc = torch.tensor([0.5, 0.0, 0.5])
    result = gradient_diagnostics(seg, rc)
    assert result["gradient_finite"] is True
    assert result["rc_to_seg_grad_norm_ratio"] == pytest.approx(0.5)
    assert result["seg_rc_grad_cosine"] == pytest.approx(1.0)
