import numpy as np
import torch


def pair_diagnostics(critic, x, y):
    good = critic(x, y)["pair"].sigmoid().flatten().cpu().tolist()
    rgb_bad = critic(x.flip(-1), y)["pair"].sigmoid().flatten().cpu().tolist()
    flipped = y.flip(-1)
    changed = (flipped - y).abs().flatten(1).sum(1) > 0
    mask_good, mask_bad = [], []
    if changed.any():
        mask_good = critic(x[changed], y[changed])["pair"].sigmoid().flatten().cpu().tolist()
        mask_bad = critic(x[changed], flipped[changed])["pair"].sigmoid().flatten().cpu().tolist()
    mean = lambda values: float(np.mean(values)) if values else None
    g, r, mg, mb = mean(good), mean(rgb_bad), mean(mask_good), mean(mask_bad)
    return {
        "rgb_good": good,
        "rgb_bad": rgb_bad,
        "mask_good": mask_good,
        "mask_bad": mask_bad,
        "rgb_drop": None if g is None or r is None else g - r,
        "mask_drop": None if mg is None or mb is None else mg - mb,
    }
