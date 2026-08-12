import torch
from torch import nn
import torch.nn.functional as F


class AOSKSoft(nn.Module):
    """Differentiable approximation: mirror side textures across a soft crack band.

    A small learned confidence gate controls the reconstruction. It is training-only and
    never imported by the deployment inference entrypoint.
    """
    def __init__(self, max_shift=6):
        super().__init__(); self.max_shift = max_shift
        self.gate = nn.Sequential(nn.Conv2d(4, 8, 3, padding=1), nn.SiLU(), nn.Conv2d(8, 1, 1), nn.Sigmoid())
    def forward(self, image, mask):
        if image.ndim != 4 or mask.shape[:2] != (image.shape[0], 1): raise ValueError("expected Bx3xHxW image and Bx1xHxW mask")
        # Structure tensor selects a local dominant axis; this compact version samples both
        # horizontal and vertical flanks then blends them by gradient energy.
        gx = image[..., :, 1:] - image[..., :, :-1]; gy = image[..., 1:, :] - image[..., :-1, :]
        ex = F.pad(gx.abs().mean(1, keepdim=True), (0, 1)); ey = F.pad(gy.abs().mean(1, keepdim=True), (0, 0, 0, 1))
        wx = ey / (ex + ey + 1e-6); wy = 1.0 - wx
        # Replicate padding prevents the wrap-around artifact caused by torch.roll.
        left = right = up = down = image
        for shift in range(1, self.max_shift + 1):
            # Explicit edge replication works on every supported PyTorch build; it also
            # avoids the circular wrap-around introduced by torch.roll.
            left = left + torch.cat([image[..., :1].expand(-1, -1, -1, shift), image[..., :-shift]], dim=-1)
            right = right + torch.cat([image[..., shift:], image[..., -1:].expand(-1, -1, -1, shift)], dim=-1)
            up = up + torch.cat([image[..., :1, :].expand(-1, -1, shift, -1), image[..., :-shift, :]], dim=-2)
            down = down + torch.cat([image[..., shift:, :], image[..., -1:, :].expand(-1, -1, shift, -1)], dim=-2)
        denom = float(self.max_shift + 1)
        horizontal = (left + right) / (2 * denom); vertical = (up + down) / (2 * denom)
        reconstructed = wx * horizontal + wy * vertical
        gate = self.gate(torch.cat([image, mask], 1)) * mask
        return image * (1.0 - gate) + reconstructed * gate


def seam_loss(original, reconstructed, mask):
    delta = (original - reconstructed).abs().mean(1, keepdim=True)
    boundary = F.max_pool2d(mask, 3, 1, 1) - (1.0 - F.max_pool2d(1.0 - mask, 3, 1, 1))
    return (delta * boundary.clamp_min(0)).sum() / boundary.clamp_min(0).sum().clamp_min(1)

def oriented_consistency_loss(logits, image, mask):
    """Training-only local consistency along the two image-gradient-selected flanks."""
    gx = image[..., :, 1:] - image[..., :, :-1]; gy = image[..., 1:, :] - image[..., :-1, :]
    ex = F.pad(gx.abs().mean(1, keepdim=True), (0, 1)); ey = F.pad(gy.abs().mean(1, keepdim=True), (0, 0, 0, 1))
    wx = ey / (ex + ey + 1e-6); wy = 1.0 - wx
    left = torch.cat([logits[..., :1].expand(-1, -1, -1, 1), logits[..., :-1]], -1)
    right = torch.cat([logits[..., 1:], logits[..., -1:].expand(-1, -1, -1, 1)], -1)
    up = torch.cat([logits[..., :1, :].expand(-1, -1, 1, -1), logits[..., :-1, :]], -2)
    down = torch.cat([logits[..., 1:, :], logits[..., -1:, :].expand(-1, -1, 1, -1)], -2)
    target = wx * (left + right) / 2 + wy * (up + down) / 2
    band = F.max_pool2d(mask, 3, 1, 1)
    return ((logits - target).abs() * band).sum() / band.sum().clamp_min(1)
