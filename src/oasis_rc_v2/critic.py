import torch
from torch import nn
import torch.nn.functional as F


class _ConvBlock(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        groups = 4 if cout % 4 == 0 else 1
        self.net = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1),
            nn.GroupNorm(groups, cout),
            nn.SiLU(),
            nn.Conv2d(cout, cout, 3, padding=1),
            nn.SiLU(),
        )

    def forward(self, x):
        return self.net(x)


class OASISRCv2Critic(nn.Module):
    """Training-only relational RGB-mask critic for canonical OASIS-RC v2.

    Input: RGB image (B,3,H,W) and soft/binary mask (B,1,H,W).
    Output: semantic logits (valid background / valid crack / invalid pair),
    pixel mismatch logits and a pair-validity logit.
    """

    def __init__(self, width=8):
        super().__init__()
        self.i1 = _ConvBlock(3, width)
        self.m1 = _ConvBlock(1, width)
        self.f1 = _ConvBlock(width * 4, width)
        self.i2 = _ConvBlock(width, width * 2)
        self.m2 = _ConvBlock(width, width * 2)
        self.f2 = _ConvBlock(width * 8, width * 2)
        self.i3 = _ConvBlock(width * 2, width * 4)
        self.m3 = _ConvBlock(width * 2, width * 4)
        self.f3 = _ConvBlock(width * 16, width * 4)
        self.pool = nn.AvgPool2d(2)
        self.u2 = _ConvBlock(width * 6, width * 2)
        self.u1 = _ConvBlock(width * 3, width)
        self.crack_head = nn.Conv2d(width, 1, 1)
        self.mismatch_head = nn.Conv2d(width, 1, 1)
        self.pair_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(width * 4, width * 2),
            nn.SiLU(),
            nn.Linear(width * 2, 1),
        )

    @staticmethod
    def _relational(fi, fm):
        return torch.cat([fi, fm, fi * fm, (fi - fm).abs()], dim=1)

    def forward(self, image, mask):
        if image.ndim != 4 or image.shape[1] != 3:
            raise ValueError("image must be Bx3xHxW")
        if mask.ndim != 4 or mask.shape[1] != 1 or image.shape[0] != mask.shape[0]:
            raise ValueError("mask must be Bx1xHxW with matching batch")
        if image.shape[-2:] != mask.shape[-2:]:
            raise ValueError("image and mask spatial sizes must match")

        i1, m1 = self.i1(image), self.m1(mask)
        z1 = self.f1(self._relational(i1, m1))
        i2, m2 = self.i2(self.pool(i1)), self.m2(self.pool(m1))
        z2 = self.f2(self._relational(i2, m2))
        i3, m3 = self.i3(self.pool(i2)), self.m3(self.pool(m2))
        z3 = self.f3(self._relational(i3, m3))
        u2 = self.u2(torch.cat([
            F.interpolate(z3, size=z2.shape[-2:], mode="bilinear", align_corners=False),
            z2,
        ], 1))
        u1 = self.u1(torch.cat([
            F.interpolate(u2, size=z1.shape[-2:], mode="bilinear", align_corners=False),
            z1,
        ], 1))
        crack = F.interpolate(
            self.crack_head(u1), size=image.shape[-2:], mode="bilinear", align_corners=False
        )
        mismatch = F.interpolate(
            self.mismatch_head(u1), size=image.shape[-2:], mode="bilinear", align_corners=False
        )
        semantic = torch.cat([-crack - mismatch, crack - mismatch, mismatch], dim=1)
        return {
            "semantic": semantic,
            "crack": crack,
            "mismatch": mismatch,
            "invalid": mismatch,
            "pair": self.pair_head(z3).flatten(1),
        }
