import torch

from .topology_ops import soft_centerline

AOSK_TOPOLOGY_VARIANT = "centerline-cldice-v1"


def centerline_cldice_loss(logits, mask, iterations=10, smooth=1e-6):
    if logits.ndim != 4 or logits.shape[1] != 1 or mask.shape != logits.shape:
        raise ValueError("expected matching Bx1xHxW logits and mask")
    target = (mask > 0.5).float()
    rows = target.flatten(1).sum(1) > 0
    if not rows.any():
        return logits.sum() * 0.0
    pred, target = logits[rows].sigmoid(), target[rows]
    pred_center = soft_centerline(pred, iterations)
    target_center = soft_centerline(target, iterations)
    dims = (1, 2, 3)
    precision = ((pred_center * target).sum(dims) + smooth) / (
        pred_center.sum(dims) + smooth
    )
    sensitivity = ((target_center * pred).sum(dims) + smooth) / (
        target_center.sum(dims) + smooth
    )
    score = 2 * precision * sensitivity / (precision + sensitivity + smooth)
    return (1 - score).mean()
