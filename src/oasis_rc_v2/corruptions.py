"""Canonical online C1--C9 hard-negative mask variants for OASIS-RC v2."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .mask_donor import nonself_crack_donor
from .mask_geometry import local_break, randint, shift_zero, wrong_connection
from .mask_texture import texture_fp_blob

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


def _iou(a, b):
    aa, bb = a > 0.5, b > 0.5
    inter = (aa & bb).sum().float()
    union = (aa | bb).sum().float()
    return 1.0 if float(union) == 0.0 else float((inter / union).item())


def _acceptable(candidate, original, max_iou, min_diff_pixels):
    diff = int((candidate != original).sum().item())
    return diff > 0 and (
        diff >= int(min_diff_pixels) or _iou(candidate, original) <= float(max_iou)
    )


def _kernel(device, generator, choices):
    return int(choices[randint(0, len(choices), device, generator)])


def _apply(kind, mask, i, crack_indices, generator, image=None, is_normal=False):
    original = mask[i : i + 1]
    _, _, _, w = original.shape
    donor_index = None
    if kind == 0:
        dx = randint(2, max(3, min(9, w // 8 + 2)), mask.device, generator)
        dx *= -1 if randint(0, 2, mask.device, generator) == 0 else 1
        candidate = shift_zero(original, dx=dx)
    elif kind == 1:
        k = _kernel(mask.device, generator, (3, 5, 7))
        candidate = -F.max_pool2d(-original, k, 1, k // 2)
    elif kind == 2:
        k = _kernel(mask.device, generator, (3, 5, 7))
        candidate = F.max_pool2d(original, k, 1, k // 2)
    elif kind == 3:
        candidate = local_break(original, generator)
    elif kind == 4:
        k = _kernel(mask.device, generator, (5, 7, 9))
        candidate = (
            F.max_pool2d(original, k, 1, k // 2)
            if randint(0, 2, mask.device, generator) == 0
            else -F.max_pool2d(-original, k, 1, k // 2)
        )
    elif kind == 5:
        candidate = wrong_connection(original, generator)
    elif kind == 6:
        if is_normal:
            raise ValueError("C7_donor_mask requires a crack-positive RGB row")
        candidate, donor_index = nonself_crack_donor(mask, i, crack_indices, generator)
        if candidate is None:
            raise RuntimeError("C7_donor_mask requires a non-self crack donor")
    elif kind == 7:
        if not is_normal:
            raise ValueError("C8_crack_on_normal requires true-normal RGB")
        if float(original.sum()) != 0.0:
            raise ValueError("true-normal C8 row must have an empty target mask")
        candidate, donor_index = nonself_crack_donor(mask, i, crack_indices, generator)
        if candidate is None:
            raise RuntimeError("C8_crack_on_normal requires an available crack donor")
    elif kind == 8:
        rgb = None if image is None else image[i : i + 1]
        candidate = torch.maximum(original, texture_fp_blob(original, rgb, generator))
    else:
        raise ValueError(f"unknown corruption kind: {kind}")
    return (candidate > 0.5).float(), donor_index


def _eligible(is_normal, crack_count):
    if is_normal:
        return (7, 8) if int(crack_count) > 0 else (8,)
    kinds = [0, 1, 2, 3, 4, 5, 8]
    if int(crack_count) > 1:
        kinds.append(6)
    return tuple(kinds)


def _ordered(kinds, device, generator):
    if len(kinds) < 2:
        return list(kinds)
    order = torch.randperm(len(kinds), device=device, generator=generator).tolist()
    return [kinds[j] for j in order]


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
    """Return one qualified C1--C9 variant per row without changing operator identity.

    A no-op operator is retried with its own stochastic parameters. A forced kind
    is treated as a request: if that operator is semantically illegal for the row
    (for example C8 on crack RGB), an eligible operator is resampled and metadata
    records the actual operator. No random pixel-toggle fallback is permitted.
    """
    if mask.ndim != 4 or mask.shape[1] != 1:
        raise ValueError("mask must be Bx1xHxW")
    if image is not None and (
        image.ndim != 4 or image.shape[0] != mask.shape[0] or image.shape[1] != 3
    ):
        raise ValueError("image must be Bx3xHxW with the same batch size")
    if int(max_attempts) <= 0:
        raise ValueError("max_attempts must be positive")

    mask = (mask > 0.5).float()
    b, _, h, w = mask.shape
    normal = (
        mask.flatten(1).sum(1) == 0
        if true_normal is None
        else true_normal.to(mask.device, dtype=torch.bool).view(-1)
    )
    if normal.numel() != b:
        raise ValueError("true_normal must have one flag per batch row")
    crack_indices = torch.nonzero(
        ~normal & (mask.flatten(1).sum(1) > 0), as_tuple=False
    ).flatten()
    min_diff_pixels = max(1, int(round(h * w * float(min_diff_ratio))))

    wrong = torch.empty_like(mask)
    meta = []
    for i in range(b):
        eligible = _eligible(bool(normal[i]), crack_indices.numel())
        requested_kind = None
        if forced_kinds is None:
            kinds = _ordered(eligible, mask.device, generator)
        else:
            requested_kind = int(
                forced_kinds[i]
                if isinstance(forced_kinds, (list, tuple))
                else forced_kinds
            )
            if not 0 <= requested_kind < len(CORRUPTION_NAMES):
                raise ValueError(f"forced corruption kind out of range: {requested_kind}")
            kinds = (
                [requested_kind]
                if requested_kind in eligible
                else _ordered(eligible, mask.device, generator)
            )

        accepted = None
        accepted_kind = accepted_donor = accepted_attempt = None
        total_attempts = 0
        last_error = None
        for kind in kinds:
            for attempt in range(1, int(max_attempts) + 1):
                total_attempts += 1
                try:
                    candidate, donor = _apply(
                        kind,
                        mask,
                        i,
                        crack_indices,
                        generator,
                        image=image,
                        is_normal=bool(normal[i]),
                    )
                except RuntimeError as exc:
                    last_error = exc
                    continue
                if _acceptable(candidate, mask[i : i + 1], max_iou, min_diff_pixels):
                    accepted, accepted_kind = candidate, kind
                    accepted_donor, accepted_attempt = donor, attempt
                    break
            if accepted is not None:
                break

        if accepted is None:
            name = CORRUPTION_NAMES[kinds[0]] if len(kinds) == 1 else "eligible C1-C9 set"
            detail = f"; last error: {last_error}" if last_error else ""
            raise RuntimeError(
                f"{name} could not produce a qualified mask after {total_attempts} attempts{detail}"
            )
        if accepted_donor is not None and accepted_donor == i:
            raise RuntimeError("donor operator selected self")
        if accepted_kind in (6, 7) and accepted_donor is None:
            raise RuntimeError("donor operator completed without a crack donor")

        wrong[i : i + 1] = accepted
        meta.append(
            {
                "kind": CORRUPTION_NAMES[accepted_kind],
                "kind_index": accepted_kind,
                "requested_kind_index": requested_kind,
                "request_resampled": (
                    requested_kind is not None and requested_kind != accepted_kind
                ),
                "attempts": total_attempts,
                "operator_attempt": accepted_attempt,
                "changed_pixels": int((accepted != mask[i : i + 1]).sum().item()),
                "iou": _iou(accepted, mask[i : i + 1]),
                "donor_index": accepted_donor,
                "texture_guided": bool(accepted_kind == 8 and image is not None),
                "operator_preserved": True,
            }
        )

    invalid = (wrong - mask).abs().clamp(0, 1)
    if (invalid.flatten(1).sum(1) <= 0).any():
        raise RuntimeError("all OASIS-RC v2 variants must be non-empty")
    return (wrong, invalid, meta) if return_meta else (wrong, invalid)


def build_targets(mask, invalid):
    semantic = mask[:, 0].long()
    semantic = torch.where(invalid[:, 0] > 0.5, torch.full_like(semantic, 2), semantic)
    mismatch = invalid
    pair_valid = (invalid.flatten(1).sum(1) == 0).float().unsqueeze(1)
    return semantic, mismatch, pair_valid
