"""Donor-mask helpers for relational crack training."""
from __future__ import annotations

from .mask_geometry import randint


def nonself_crack_donor(mask, index, crack_indices, generator):
    candidates = crack_indices[crack_indices != index]
    if candidates.numel() == 0:
        return None, None
    donor_index = int(
        candidates[randint(0, candidates.numel(), mask.device, generator)].item()
    )
    return mask[donor_index : donor_index + 1].clone(), donor_index
