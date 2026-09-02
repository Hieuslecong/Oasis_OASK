import torch
import torch.nn.functional as F


def segmentation_loss(logits, target, weights=None):
    weights = weights or {"bce": 1.0, "dice": 1.0}
    bce = F.binary_cross_entropy_with_logits(logits, target)
    p = torch.sigmoid(logits)
    dice = 1 - (2 * (p * target).sum((1,2,3)) + 1e-6) / (p.sum((1,2,3)) + target.sum((1,2,3)) + 1e-6)
    return float(weights.get("bce", 1.0)) * bce + float(weights.get("dice", 1.0)) * dice.mean()
