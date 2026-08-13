from __future__ import annotations

import numpy as np
import torch


@torch.no_grad()
def threshold_sweep_metrics(model, loader, device, thresholds=None, chunk_size=16):
    model.eval()
    if thresholds is None:
        thresholds = np.arange(0.05, 0.951, 0.01)
    thresholds = [float(t) for t in thresholds]
    ts_all = torch.tensor(thresholds, device=device, dtype=torch.float32)
    n = len(thresholds)
    tp = torch.zeros(n, device=device, dtype=torch.float64)
    fp = torch.zeros(n, device=device, dtype=torch.float64)
    fn = torch.zeros(n, device=device, dtype=torch.float64)
    normal_fp_pixels = torch.zeros(n, device=device, dtype=torch.float64)
    normal_fp_images = torch.zeros(n, device=device, dtype=torch.float64)
    normal_count = 0
    batches = 0
    for batch in loader:
        batches += 1
        x, y = batch[:2]
        x, y = x.to(device), y.to(device)
        probability = model(x).sigmoid()
        target = y > 0.5
        normal_rows = target.flatten(1).sum(1) == 0
        normal_count += int(normal_rows.sum().item())
        for start in range(0, n, int(chunk_size)):
            stop = min(n, start + int(chunk_size))
            ts = ts_all[start:stop].view(1, -1, 1, 1, 1)
            pred = probability.unsqueeze(1) >= ts
            truth = target.unsqueeze(1)
            dims = (0, 2, 3, 4)
            tp[start:stop] += (pred & truth).sum(dims, dtype=torch.float64)
            fp[start:stop] += (pred & ~truth).sum(dims, dtype=torch.float64)
            fn[start:stop] += (~pred & truth).sum(dims, dtype=torch.float64)
            if normal_rows.any():
                p = pred[normal_rows]
                normal_fp_pixels[start:stop] += p.sum((0, 2, 3, 4), dtype=torch.float64)
                normal_fp_images[start:stop] += p.flatten(2).any(-1).sum(0, dtype=torch.float64)
    if batches == 0:
        raise RuntimeError("validation loader produced no batches")
    result = []
    for i, threshold in enumerate(thresholds):
        a, b, c = float(tp[i]), float(fp[i]), float(fn[i])
        result.append({
            "precision": a / (a + b + 1e-8),
            "recall": a / (a + c + 1e-8),
            "dice": 2 * a / (2 * a + b + c + 1e-8),
            "iou": a / (a + b + c + 1e-8),
            "normal_fp_pixels_mean": float(normal_fp_pixels[i]) / normal_count if normal_count else None,
            "normal_fp_images": int(normal_fp_images[i].item()),
            "normal_image_count": normal_count,
            "threshold": threshold,
        })
    return result


@torch.no_grad()
def segmentation_metrics(model, loader, device, threshold):
    return threshold_sweep_metrics(model, loader, device, [float(threshold)])[0]


@torch.no_grad()
def select_threshold(model, loader, device):
    best = None
    for metrics in threshold_sweep_metrics(model, loader, device):
        key = (
            metrics["dice"],
            -metrics["normal_fp_pixels_mean"] if metrics["normal_fp_pixels_mean"] is not None else 0.0,
        )
        if best is None or key > best[0]:
            best = (key, metrics)
    return best[1]
