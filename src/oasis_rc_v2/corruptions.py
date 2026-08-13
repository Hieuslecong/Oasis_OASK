"""Canonical online C1--C9 hard-negative corruptions for OASIS-RC v2."""
from __future__ import annotations

import torch
import torch.nn.functional as F

CORRUPTION_NAMES = (
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


def _randint(low, high, device, generator):
    return int(torch.randint(low, high, (), device=device, generator=generator).item())


def _random_blob(single, generator, kernel=11):
    noise = torch.rand(single.shape, device=single.device, generator=generator)
    seed = (noise > 0.995).float()
    blob = F.max_pool2d(seed, kernel, 1, kernel // 2)
    if float(blob.sum()) == 0.0:
        h, w = single.shape[-2:]
        y = _randint(0, h, single.device, generator)
        x = _randint(0, w, single.device, generator)
        blob[..., y, x] = 1.0
    return (blob > 0.5).float()


def _texture_fp_blob(single, image_single, generator, kernel=11):
    """Place a false-positive blob preferentially on high-texture RGB background."""
    if image_single is None:
        return _random_blob(single, generator, kernel=kernel)
    if image_single.ndim != 4 or image_single.shape[0] != 1 or image_single.shape[1] != 3:
        raise ValueError("image_single must be 1x3xHxW")
    gray = image_single.mean(1, keepdim=True)
    gx = F.pad((gray[..., 1:] - gray[..., :-1]).abs(), (0, 1, 0, 0))
    gy = F.pad((gray[..., 1:, :] - gray[..., :-1, :]).abs(), (0, 0, 0, 1))
    texture = gx + gy
    background = 1.0 - F.max_pool2d(single, 7, 1, 3).clamp(0, 1)
    score = texture * background
    jitter = torch.rand(score.shape, device=score.device, generator=generator) * 1e-6
    flat = (score + jitter).flatten()
    if float(score.max()) <= 0.0:
        return _random_blob(single, generator, kernel=kernel)
    index = int(flat.argmax().item())
    h, w = single.shape[-2:]
    y, x = divmod(index, w)
    seed = torch.zeros_like(single)
    seed[..., y, x] = 1.0
    blob = F.max_pool2d(seed, kernel, 1, kernel // 2)
    return (blob > 0.5).float()


def _local_break(single, generator):
    out = single.clone()
    coords = torch.nonzero(single[0, 0] > 0.5, as_tuple=False)
    if coords.numel() == 0:
        return out
    idx = _randint(0, coords.shape[0], single.device, generator)
    y, x = (int(v) for v in coords[idx].tolist())
    h, w = single.shape[-2:]
    radius = max(1, min(h, w) // 32)
    y0, y1 = max(0, y - radius), min(h, y + radius + 1)
    x0, x1 = max(0, x - radius * 2), min(w, x + radius * 2 + 1)
    out[..., y0:y1, x0:x1] = 0.0
    return out


def _wrong_connection(single, generator):
    out = single.clone()
    coords = torch.nonzero(single[0, 0] > 0.5, as_tuple=False)
    if coords.shape[0] < 2:
        return out
    first = coords[_randint(0, coords.shape[0], single.device, generator)]
    distances = (coords.float() - first.float()).abs().sum(1)
    second = coords[int(distances.argmax().item())]
    y1, x1 = (int(v) for v in first.tolist())
    y2, x2 = (int(v) for v in second.tolist())
    steps = max(abs(y2 - y1), abs(x2 - x1)) + 1
    ys = torch.linspace(y1, y2, steps, device=single.device).round().long()
    xs = torch.linspace(x1, x2, steps, device=single.device).round().long()
    out[0, 0, ys, xs] = 1.0
    out = F.max_pool2d(out, 3, 1, 1)
    return (out > 0.5).float()


def _nonself_crack_donor(mask, index, crack_indices, generator):
    candidates = crack_indices[crack_indices != index]
    if candidates.numel() == 0:
        return None, None
    pos = _randint(0, candidates.numel(), mask.device, generator)
    donor_index = int(candidates[pos].item())
    return mask[donor_index : donor_index + 1].clone(), donor_index


def _iou(a, b):
    aa = a > 0.5
    bb = b > 0.5
    inter = (aa & bb).sum().float()
    union = (aa | bb).sum().float()
    if float(union) == 0.0:
        return 1.0
    return float((inter / union).item())


def _acceptable(candidate, original, max_iou, min_diff_pixels):
    diff = int((candidate != original).sum().item())
    return diff >= int(min_diff_pixels) or _iou(candidate, original) <= float(max_iou)


def _force_difference(candidate, original, generator):
    if int((candidate != original).sum().item()) > 0:
        return candidate
    out = candidate.clone()
    h, w = out.shape[-2:]
    y = _randint(0, h, out.device, generator)
    x = _randint(0, w, out.device, generator)
    out[..., y, x] = 1.0 - out[..., y, x]
    return out


def _apply(kind, mask, i, crack_indices, generator, image=None):
    original = mask[i : i + 1]
    h, w = original.shape[-2:]
    donor_index = None

    if kind == 0:
        dx = _randint(2, max(3, min(9, w // 8 + 2)), mask.device, generator)
        if _randint(0, 2, mask.device, generator) == 0:
            dx = -dx
        candidate = shift_zero(original, dx=dx, dy=0)
    elif kind == 1:
        candidate = -F.max_pool2d(-original, 3, 1, 1)
    elif kind == 2:
        candidate = F.max_pool2d(original, 3, 1, 1)
    elif kind == 3:
        candidate = _local_break(original, generator)
    elif kind == 4:
        candidate = (
            F.max_pool2d(original, 7, 1, 3)
            if _randint(0, 2, mask.device, generator) == 0
            else -F.max_pool2d(-original, 7, 1, 3)
        )
    elif kind == 5:
        candidate = _wrong_connection(original, generator)
    elif kind in (6, 7):
        candidate, donor_index = _nonself_crack_donor(mask, i, crack_indices, generator)
        if candidate is None:
            candidate = _wrong_connection(original, generator)
            if int((candidate != original).sum().item()) == 0:
                candidate = _random_blob(original, generator, kernel=7)
    else:
        image_single = None if image is None else image[i : i + 1]
        candidate = torch.maximum(
            original,
            _texture_fp_blob(original, image_single, generator, kernel=11),
        )

    candidate = (candidate > 0.5).float()
    return candidate, donor_index


def make_corrupted_mask(
    mask,
    true_normal=None,
    generator=None,
    max_iou=0.95,
    min_diff_ratio=0.001,
    max_attempts=12,
    forced_kinds=None,
    return_meta=False,
    image=None,
):
    """Generate one certified hard negative per sample.

    Invariants: online C1--C9 only, non-self crack donors, non-empty changes,
    IoU/minimum-difference qualification, no circular ``torch.roll``, and a
    texture-guided C9 when RGB is supplied.
    """
    if mask.ndim != 4 or mask.shape[1] != 1:
        raise ValueError("mask must be Bx1xHxW")
    if image is not None and (image.ndim != 4 or image.shape[0] != mask.shape[0]):
        raise ValueError("image must be Bx3xHxW with the same batch size")
    mask = (mask > 0.5).float()
    b, _, h, w = mask.shape
    if true_normal is None:
        normal = mask.flatten(1).sum(1) == 0
    else:
        normal = true_normal.to(mask.device, dtype=torch.bool).view(-1)
        if normal.numel() != b:
            raise ValueError("true_normal must have one flag per batch row")
    crack_indices = torch.nonzero(
        ~normal & (mask.flatten(1).sum(1) > 0), as_tuple=False
    ).flatten()
    min_diff_pixels = max(1, int(round(h * w * float(min_diff_ratio))))

    wrong = torch.empty_like(mask)
    meta = []
    for i in range(b):
        if forced_kinds is not None:
            kind = int(
                forced_kinds[i]
                if isinstance(forced_kinds, (list, tuple))
                else forced_kinds
            )
        elif bool(normal[i]):
            kind = 7 if crack_indices.numel() > 0 else 8
        else:
            pool = (0, 1, 2, 3, 4, 5, 6, 8)
            kind = pool[_randint(0, len(pool), mask.device, generator)]

        candidate = None
        donor_index = None
        attempts = 0
        for attempts in range(1, int(max_attempts) + 1):
            candidate, donor_index = _apply(
                kind, mask, i, crack_indices, generator, image=image
            )
            candidate = _force_difference(candidate, mask[i : i + 1], generator)
            if _acceptable(candidate, mask[i : i + 1], max_iou, min_diff_pixels):
                break
            if forced_kinds is None:
                if bool(normal[i]):
                    kind = 7 if crack_indices.numel() > 0 else 8
                else:
                    pool = (0, 1, 2, 3, 4, 5, 6, 8)
                    kind = pool[_randint(0, len(pool), mask.device, generator)]
        candidate = _force_difference(candidate, mask[i : i + 1], generator)
        if not _acceptable(candidate, mask[i : i + 1], max_iou, min_diff_pixels):
            flat = candidate.view(-1).clone()
            original_flat = mask[i : i + 1].view(-1)
            count = min(int(min_diff_pixels), flat.numel())
            idxs = torch.randperm(
                flat.numel(), device=flat.device, generator=generator
            )[:count]
            flat[idxs] = 1.0 - original_flat[idxs]
            candidate = flat.view_as(candidate)
        if int((candidate != mask[i : i + 1]).sum().item()) == 0:
            raise RuntimeError("corruption regeneration produced a no-op")
        if not _acceptable(candidate, mask[i : i + 1], max_iou, min_diff_pixels):
            raise RuntimeError("corruption failed IoU/minimum-difference qualification")
        if donor_index is not None and donor_index == i:
            raise RuntimeError("donor corruption selected self")
        wrong[i : i + 1] = candidate
        meta.append(
            {
                "kind": CORRUPTION_NAMES[kind],
                "kind_index": kind,
                "attempts": attempts,
                "changed_pixels": int(
                    (candidate != mask[i : i + 1]).sum().item()
                ),
                "iou": _iou(candidate, mask[i : i + 1]),
                "donor_index": donor_index,
                "texture_guided": bool(kind == 8 and image is not None),
            }
        )

    invalid = (wrong - mask).abs().clamp(0, 1)
    if (invalid.flatten(1).sum(1) <= 0).any():
        raise RuntimeError("all OASIS-RC v2 corruptions must be non-empty")
    return (wrong, invalid, meta) if return_meta else (wrong, invalid)


def build_targets(mask, invalid):
    semantic = mask[:, 0].long()
    semantic = torch.where(
        invalid[:, 0] > 0.5,
        torch.full_like(semantic, 2),
        semantic,
    )
    mismatch = invalid
    pair_valid = (invalid.flatten(1).sum(1) == 0).float().unsqueeze(1)
    return semantic, mismatch, pair_valid
