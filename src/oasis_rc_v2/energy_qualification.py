from __future__ import annotations

import math

import torch

from .losses import relation_energy


DEFAULT_TRAJECTORY_T = (0.0, 0.25, 0.50, 0.75, 1.0)


def interpolate_mask(gt_mask, corrupted_mask, t):
    """Continuous mask path from GT (t=0) to structured corruption (t=1)."""
    t = float(t)
    if not 0.0 <= t <= 1.0:
        raise ValueError("trajectory t must be in [0,1]")
    return (1.0 - t) * gt_mask + t * corrupted_mask


@torch.no_grad()
def relation_energy_trajectory(
    critic,
    image,
    gt_mask,
    corrupted_mask,
    pair_weight=0.25,
    t_values=DEFAULT_TRAJECTORY_T,
):
    """Return [B,T] energy tensor along a continuous GT->corruption path."""
    values = []
    for t in t_values:
        mask = interpolate_mask(gt_mask, corrupted_mask, t)
        values.append(relation_energy(critic(image, mask), pair_weight=pair_weight))
    return torch.stack(values, dim=1)


def summarize_energy_trajectories(energies):
    """Summarize [N,T] trajectories for v2.1 energy qualification."""
    if energies.ndim != 2 or energies.shape[1] < 2:
        raise ValueError("energies must have shape [N,T] with T>=2")
    if energies.shape[0] == 0:
        return {
            "energy_samples": 0,
            "energy_finite": False,
            "positive_energy_gap_fraction": None,
            "mean_energy_gap": None,
            "median_energy_gap": None,
            "continuous_path_order_fraction": None,
        }

    finite = bool(torch.isfinite(energies).all().item())
    gap = energies[:, -1] - energies[:, 0]
    adjacent = energies[:, 1:] >= energies[:, :-1]
    strict_endpoint = gap > 0

    return {
        "energy_samples": int(energies.shape[0]),
        "energy_finite": finite,
        "positive_energy_gap_fraction": float(strict_endpoint.float().mean().item()),
        "mean_energy_gap": float(gap.mean().item()),
        "median_energy_gap": float(gap.median().item()),
        "continuous_path_order_fraction": float(adjacent.float().mean().item()),
    }


def gradient_diagnostics(seg_grad, rc_grad, eps=1e-12):
    """Compute norm ratio and cosine for flattened student gradients.

    This utility is intentionally model-agnostic so the diagnostic script can
    aggregate arbitrary student parameter gradients without changing the method.
    """
    seg_grad = seg_grad.reshape(-1)
    rc_grad = rc_grad.reshape(-1)
    if seg_grad.numel() != rc_grad.numel():
        raise ValueError("gradient vectors must have equal size")
    seg_norm = torch.linalg.vector_norm(seg_grad)
    rc_norm = torch.linalg.vector_norm(rc_grad)
    denom = (seg_norm * rc_norm).clamp_min(float(eps))
    cosine = torch.dot(seg_grad, rc_grad) / denom
    return {
        "seg_grad_norm": float(seg_norm.detach().cpu()),
        "rc_grad_norm": float(rc_norm.detach().cpu()),
        "rc_to_seg_grad_norm_ratio": float((rc_norm / seg_norm.clamp_min(float(eps))).detach().cpu()),
        "seg_rc_grad_cosine": float(cosine.detach().cpu()),
        "gradient_finite": bool(
            math.isfinite(float(seg_norm))
            and math.isfinite(float(rc_norm))
            and math.isfinite(float(cosine))
        ),
    }
