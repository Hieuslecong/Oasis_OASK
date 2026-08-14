"""Texture-guided false-positive mask helper."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .mask_geometry import randint


def random_blob(single, generator, kernel=11):
    noise = torch.rand(single.shape, device=single.device, generator=generator)
    seed = (noise > 0.995).float()
    blob = F.max_pool2d(seed, kernel, 1, kernel // 2)
    if float(blob.sum()) == 0.0:
        h, w = single.shape[-2:]
        y = randint(0, h, single.device, generator)
        x = randint(0, w, single.device, generator)
        blob[..., y, x] = 1.0
    return (blob > 0.5).float()


def texture_fp_blob(single, image_single, generator, kernel=11):
    if image_single is None:
        return random_blob(single, generator, kernel=kernel)
    if image_single.ndim != 4 or image_single.shape[0] != 1 or image_single.shape[1] != 3:
        raise ValueError("image_single must be 1x3xHxW")
    gray = image_single.mean(1, keepdim=True)
    gx = F.pad((gray[..., 1:] - gray[..., :-1]).abs(), (0, 1, 0, 0))
    gy = F.pad((gray[..., 1:, :] - gray[..., :-1, :]).abs(), (0, 0, 0, 1))
    background = 1.0 - F.max_pool2d(single, 7, 1, 3).clamp(0, 1)
    score = ((gx + gy) * background).flatten()
    if float(score.max()) <= 0.0:
        return random_blob(single, generator, kernel=kernel)
    k = min(64, max(1, int((score > 0).sum().item())))
    top = torch.topk(score, k=k, largest=True).indices
    selected = int(top[randint(0, top.numel(), score.device, generator)].item())
    _, _, _, w = single.shape
    y, x = divmod(selected, w)
    seed = torch.zeros_like(single)
    seed[..., y, x] = 1.0
    return (F.max_pool2d(seed, kernel, 1, kernel // 2) > 0.5).float()
