"""Small geometry helpers for segmentation-mask training variants."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def randint(low, high, device, generator):
    return int(torch.randint(low, high, (), device=device, generator=generator).item())


def shift_zero(mask, dx=3, dy=0):
    if mask.ndim != 4:
        raise ValueError("mask must be Bx1xHxW")
    _, _, h, w = mask.shape
    out = torch.zeros_like(mask)
    xs0, xs1 = max(0, dx), min(w, w + dx)
    ys0, ys1 = max(0, dy), min(h, h + dy)
    src_x0, src_x1 = max(0, -dx), min(w, w - dx)
    src_y0, src_y1 = max(0, -dy), min(h, h - dy)
    if xs1 > xs0 and ys1 > ys0:
        out[..., ys0:ys1, xs0:xs1] = mask[..., src_y0:src_y1, src_x0:src_x1]
    return out


def local_break(single, generator):
    out = single.clone()
    coords = torch.nonzero(single[0, 0] > 0.5, as_tuple=False)
    if coords.numel() == 0:
        return out
    y, x = (int(v) for v in coords[randint(0, coords.shape[0], single.device, generator)].tolist())
    h, w = single.shape[-2:]
    # Scale the break so the removed region is large enough to register as a
    # real corruption against the acceptance gate (iou <= 0.95 / min_diff_pixels).
    # A fixed tiny radius fails on thin, long real cracks (only ~40-60 px removed).
    n_crack = int((single[0, 0] > 0.5).sum())
    target = max(1, int(0.08 * n_crack))
    radius = 1
    while True:
        y1, y2 = max(0, y - radius), min(h, y + radius + 1)
        x1, x2 = max(0, x - radius), min(w, x + radius + 1)
        candidate = out.clone()
        candidate[..., y1:y2, x1:x2] = 0.0
        if int((candidate != single).sum()) >= target or radius >= max(h, w):
            return candidate
        radius *= 2



def wrong_connection(single, generator):
    out = single.clone()
    coords = torch.nonzero(single[0, 0] > 0.5, as_tuple=False)
    if coords.shape[0] < 2:
        return out
    first = coords[randint(0, coords.shape[0], single.device, generator)]
    distance = (coords.float() - first.float()).abs().sum(1)
    far = torch.topk(distance, k=max(1, coords.shape[0] // 4), largest=True).indices
    second = coords[int(far[randint(0, far.numel(), single.device, generator)].item())]
    y1, x1 = (int(v) for v in first.tolist())
    y2, x2 = (int(v) for v in second.tolist())
    steps = max(abs(y2-y1), abs(x2-x1)) + 1
    ys = torch.linspace(y1, y2, steps, device=single.device).round().long()
    xs = torch.linspace(x1, x2, steps, device=single.device).round().long()
    out[0, 0, ys, xs] = 1.0
    return (F.max_pool2d(out, 3, 1, 1) > 0.5).float()
