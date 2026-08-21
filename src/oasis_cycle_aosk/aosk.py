import torch
from torch import nn
import torch.nn.functional as F


def orientation_weights(ex, ey, eps=1e-6, flat_threshold=1e-6):
    """Legacy axis-aligned weights retained only for compatibility diagnostics."""
    total = ex + ey
    wx = ey / total.clamp_min(float(eps))
    wy = ex / total.clamp_min(float(eps))
    flat = total <= float(flat_threshold)
    half = torch.full_like(wx, 0.5)
    wx = torch.where(flat, half, wx)
    wy = torch.where(flat, half, wy)
    return wx, wy


def _sobel(gray):
    dtype, device = gray.dtype, gray.device
    kx = torch.tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
        dtype=dtype,
        device=device,
    ).view(1, 1, 3, 3) / 8.0
    ky = kx.transpose(-1, -2)
    gx = F.conv2d(gray, kx, padding=1)
    gy = F.conv2d(gray, ky, padding=1)
    return gx, gy


def structure_tensor_tangent(image, window=5, eps=1e-6):
    """Estimate arbitrary-angle tangent direction and orientation coherence.

    The dominant structure-tensor eigenvector is the local image-gradient
    direction (normal to an elongated crack edge).  The returned tangent is its
    perpendicular. Direction sign is immaterial because AOSK samples at +/-t.
    """
    if image.ndim != 4 or image.shape[1] != 3:
        raise ValueError("image must be Bx3xHxW")
    if window < 3 or window % 2 == 0:
        raise ValueError("structure-tensor window must be an odd integer >= 3")
    gray = image.mean(1, keepdim=True)
    gx, gy = _sobel(gray)
    pad = window // 2
    jxx = F.avg_pool2d(gx.square(), window, stride=1, padding=pad)
    jyy = F.avg_pool2d(gy.square(), window, stride=1, padding=pad)
    jxy = F.avg_pool2d(gx * gy, window, stride=1, padding=pad)

    # Principal (gradient-normal) orientation of a symmetric 2x2 tensor.
    theta_n = 0.5 * torch.atan2(2.0 * jxy, jxx - jyy)
    tx = -torch.sin(theta_n)
    ty = torch.cos(theta_n)
    anisotropy = torch.sqrt((jxx - jyy).square() + 4.0 * jxy.square())
    coherence = (anisotropy / (jxx + jyy + float(eps))).clamp(0.0, 1.0)
    finite = torch.isfinite(tx) & torch.isfinite(ty) & torch.isfinite(coherence)
    tx = torch.where(finite, tx, torch.zeros_like(tx))
    ty = torch.where(finite, ty, torch.zeros_like(ty))
    coherence = torch.where(finite, coherence, torch.zeros_like(coherence))
    return tx, ty, coherence


def _isotropic_four_neighbor(logits):
    left = torch.cat(
        [logits[..., :1].expand(-1, -1, -1, 1), logits[..., :-1]], -1
    )
    right = torch.cat(
        [logits[..., 1:], logits[..., -1:].expand(-1, -1, -1, 1)], -1
    )
    up = torch.cat(
        [logits[..., :1, :].expand(-1, -1, 1, -1), logits[..., :-1, :]], -2
    )
    down = torch.cat(
        [logits[..., 1:, :], logits[..., -1:, :].expand(-1, -1, 1, -1)], -2
    )
    return (left + right + up + down) / 4.0


def _sample_along_tangent(logits, tx, ty, distance=1.0):
    b, _, h, w = logits.shape
    ys = torch.linspace(-1.0, 1.0, h, dtype=logits.dtype, device=logits.device)
    xs = torch.linspace(-1.0, 1.0, w, dtype=logits.dtype, device=logits.device)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    base = torch.stack((xx, yy), dim=-1).unsqueeze(0).expand(b, -1, -1, -1)
    dx = 2.0 * float(distance) * tx[:, 0] / max(w - 1, 1)
    dy = 2.0 * float(distance) * ty[:, 0] / max(h - 1, 1)
    delta = torch.stack((dx, dy), dim=-1)
    plus = F.grid_sample(
        logits,
        base + delta,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )
    minus = F.grid_sample(
        logits,
        base - delta,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )
    return 0.5 * (plus + minus)


class AOSKSoft(nn.Module):
    """Legacy reconstruction helper; not used by the canonical v2.1-dev2 loss."""

    def __init__(self, max_shift=6):
        super().__init__()
        self.max_shift = max_shift
        self.gate = nn.Sequential(
            nn.Conv2d(4, 8, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(8, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, image, mask):
        if image.ndim != 4 or mask.shape[:2] != (image.shape[0], 1):
            raise ValueError("expected Bx3xHxW image and Bx1xHxW mask")
        gx = image[..., :, 1:] - image[..., :, :-1]
        gy = image[..., 1:, :] - image[..., :-1, :]
        ex = F.pad(gx.abs().mean(1, keepdim=True), (0, 1))
        ey = F.pad(gy.abs().mean(1, keepdim=True), (0, 0, 0, 1))
        wx, wy = orientation_weights(ex, ey)

        left = right = up = down = image
        for shift in range(1, self.max_shift + 1):
            left = left + torch.cat(
                [image[..., :1].expand(-1, -1, -1, shift), image[..., :-shift]],
                dim=-1,
            )
            right = right + torch.cat(
                [image[..., shift:], image[..., -1:].expand(-1, -1, -1, shift)],
                dim=-1,
            )
            up = up + torch.cat(
                [image[..., :1, :].expand(-1, -1, shift, -1), image[..., :-shift, :]],
                dim=-2,
            )
            down = down + torch.cat(
                [image[..., shift:, :], image[..., -1:, :].expand(-1, -1, shift, -1)],
                dim=-2,
            )
        denom = float(self.max_shift + 1)
        horizontal = (left + right) / (2 * denom)
        vertical = (up + down) / (2 * denom)
        reconstructed = wx * horizontal + wy * vertical
        gate = self.gate(torch.cat([image, mask], 1)) * mask
        return image * (1.0 - gate) + reconstructed * gate


def seam_loss(original, reconstructed, mask):
    delta = (original - reconstructed).abs().mean(1, keepdim=True)
    boundary = F.max_pool2d(mask, 3, 1, 1) - (
        1.0 - F.max_pool2d(1.0 - mask, 3, 1, 1)
    )
    return (delta * boundary.clamp_min(0)).sum() / boundary.clamp_min(0).sum().clamp_min(1)


def oriented_consistency_loss(
    logits,
    image,
    mask,
    tensor_window=5,
    coherence_threshold=0.10,
    sample_distance=1.0,
):
    """Training-only structure-tensor-steered local consistency.

    This is still *not* a topology loss.  It supports arbitrary local angles via
    bilinear +/-tangent sampling.  Low-coherence regions smoothly fall back to
    an isotropic four-neighbour target instead of inventing a confident
    orientation from numerical noise.
    """
    if logits.ndim != 4 or logits.shape[1] != 1:
        raise ValueError("logits must be Bx1xHxW")
    if image.shape[0] != logits.shape[0] or image.shape[-2:] != logits.shape[-2:]:
        raise ValueError("image/logit spatial shape mismatch")
    if mask.shape != logits.shape:
        raise ValueError("mask/logit shape mismatch")
    if not 0.0 <= coherence_threshold < 1.0:
        raise ValueError("coherence_threshold must be in [0,1)")

    tx, ty, coherence = structure_tensor_tangent(image, window=tensor_window)
    directional = _sample_along_tangent(
        logits, tx, ty, distance=sample_distance
    )
    isotropic = _isotropic_four_neighbor(logits)
    confidence = (
        (coherence - float(coherence_threshold))
        / max(1.0 - float(coherence_threshold), 1e-6)
    ).clamp(0.0, 1.0)
    target = confidence * directional + (1.0 - confidence) * isotropic
    band = F.max_pool2d(mask, 3, 1, 1)
    return ((logits - target).abs() * band).sum() / band.sum().clamp_min(1)
