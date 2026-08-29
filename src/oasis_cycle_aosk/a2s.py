"""OASIS-A2S v0.1 core models and losses.

OASIS-A2S keeps the central OASIS formulation: a semantic discriminator predicts
N real semantic classes plus one FAKE class at every pixel. For binary crack
segmentation N=2 -> {background, crack, fake}. Stage II discards the fake output
and fine-tunes the same discriminator as the deployable crack segmenter.

The discriminator is a compact U-Net-style adaptation rather than a byte-for-byte
copy of the archived OASIS implementation. The N+1 loss, inverse-frequency class
balancing, class-aware LabelMix semantics and semantic/noise-conditioned generator
follow the OASIS training principles needed by the A2S hypothesis.
"""
from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor, nn
import torch.nn.functional as F


METHOD_VERSION = "OASIS-A2S-v0.1"
IMPLEMENTATION_REVISION = "0.1.2"
REAL_CLASSES = 2
FAKE_CLASS = 2


def _valid_width(width: int) -> int:
    width = int(width)
    if width < 4:
        raise ValueError("width must be >= 4")
    return width


class ConvNormAct(nn.Module):
    def __init__(self, cin: int, cout: int, stride: int = 1):
        super().__init__()
        groups = 4 if cout % 4 == 0 else 1
        self.net = nn.Sequential(
            nn.Conv2d(cin, cout, 3, stride=stride, padding=1, bias=False),
            nn.GroupNorm(groups, cout),
            nn.SiLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False),
            nn.GroupNorm(groups, cout),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class OASISA2SDiscriminator(nn.Module):
    """Compact U-Net semantic discriminator / segmenter backbone.

    out_classes=3 is Stage I ({bg, crack, fake}); out_classes=2 is the
    transferred Stage-II deployable crack segmenter ({bg, crack}).
    """

    def __init__(self, width: int = 24, out_classes: int = 3):
        super().__init__()
        width = _valid_width(width)
        if out_classes not in (2, 3):
            raise ValueError("out_classes must be 2 or 3")
        self.width, self.out_classes = width, int(out_classes)
        self.e1 = ConvNormAct(3, width)
        self.e2 = ConvNormAct(width, width * 2, stride=2)
        self.e3 = ConvNormAct(width * 2, width * 4, stride=2)
        self.bottleneck = ConvNormAct(width * 4, width * 4, stride=2)
        self.d3 = ConvNormAct(width * 8, width * 2)
        self.d2 = ConvNormAct(width * 4, width)
        self.d1 = ConvNormAct(width * 2, width)
        self.head = nn.Conv2d(width, self.out_classes, 1)

    @staticmethod
    def _up(x: Tensor, ref: Tensor) -> Tensor:
        return F.interpolate(x, size=ref.shape[-2:], mode="bilinear", align_corners=False)

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 4 or x.shape[1] != 3:
            raise ValueError("image must be Bx3xHxW")
        e1 = self.e1(x)
        e2 = self.e2(e1)
        e3 = self.e3(e2)
        z = self.bottleneck(e3)
        d3 = self.d3(torch.cat([self._up(z, e3), e3], 1))
        d2 = self.d2(torch.cat([self._up(d3, e2), e2], 1))
        d1 = self.d1(torch.cat([self._up(d2, e1), e1], 1))
        return self.head(d1)


class SpatialAffineNorm(nn.Module):
    """Small spatially conditioned normalization used by training-only G."""

    def __init__(self, channels: int, condition_channels: int, hidden: int = 32):
        super().__init__()
        self.condition_channels = int(condition_channels)
        self.norm = nn.GroupNorm(1, channels, affine=False)
        self.shared = nn.Sequential(
            nn.Conv2d(self.condition_channels, hidden, 3, padding=1),
            nn.SiLU(inplace=True),
        )
        self.gamma = nn.Conv2d(hidden, channels, 3, padding=1)
        self.beta = nn.Conv2d(hidden, channels, 3, padding=1)

    def forward(self, x: Tensor, condition: Tensor) -> Tensor:
        if condition.ndim != 4 or condition.shape[1] != self.condition_channels:
            raise ValueError(
                f"condition must be Bx{self.condition_channels}xHxW"
            )
        condition = F.interpolate(condition, size=x.shape[-2:], mode="nearest")
        h = self.shared(condition)
        return self.norm(x) * (1.0 + self.gamma(h)) + self.beta(h)


class ConditionalResBlock(nn.Module):
    def __init__(self, cin: int, cout: int, condition_channels: int):
        super().__init__()
        self.n1 = SpatialAffineNorm(cin, condition_channels)
        self.n2 = SpatialAffineNorm(cout, condition_channels)
        self.c1 = nn.Conv2d(cin, cout, 3, padding=1)
        self.c2 = nn.Conv2d(cout, cout, 3, padding=1)
        self.skip = nn.Identity() if cin == cout else nn.Conv2d(cin, cout, 1)

    def forward(self, x: Tensor, condition: Tensor) -> Tensor:
        h = self.c1(F.silu(self.n1(x, condition)))
        h = self.c2(F.silu(self.n2(h, condition)))
        return h + self.skip(x)


class OASISA2SGenerator(nn.Module):
    """Training-only crack-mask-conditioned generator with 3D noise conditioning.

    Default training samples one global latent vector per image and spatially
    expands it, matching the upstream OASIS training path. Explicit full-resolution
    noise is also accepted for controlled local-sampling diagnostics.
    """

    def __init__(self, width: int = 32, noise_channels: int = 4):
        super().__init__()
        width = _valid_width(width)
        if noise_channels < 1:
            raise ValueError("noise_channels must be >= 1")
        self.width, self.noise_channels = width, int(noise_channels)
        condition_channels = REAL_CLASSES + self.noise_channels
        self.condition_channels = condition_channels
        self.in_conv = nn.Conv2d(condition_channels, width, 3, padding=1)
        self.b1 = ConditionalResBlock(width, width, condition_channels)
        self.down = nn.Conv2d(width, width * 2, 4, stride=2, padding=1)
        self.b2 = ConditionalResBlock(width * 2, width * 2, condition_channels)
        self.b3 = ConditionalResBlock(width * 2, width * 2, condition_channels)
        self.out = nn.Sequential(
            nn.Conv2d(width * 3, width, 3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(width, 3, 1),
            nn.Tanh(),
        )

    @staticmethod
    def semantic_one_hot(mask: Tensor) -> Tensor:
        if mask.ndim != 4 or mask.shape[1] != 1:
            raise ValueError("mask must be Bx1xHxW")
        crack = (mask > 0.5).to(mask.dtype)
        return torch.cat([1.0 - crack, crack], 1)

    def _noise_map(self, semantic: Tensor, noise: Optional[Tensor]) -> Tensor:
        b, _, h, w = semantic.shape
        if noise is None:
            noise = torch.randn(
                b,
                self.noise_channels,
                1,
                1,
                device=semantic.device,
                dtype=semantic.dtype,
            )
        if noise.ndim != 4 or noise.shape[:2] != (b, self.noise_channels):
            raise ValueError(
                f"noise must start with shape {(b, self.noise_channels)}, got {tuple(noise.shape)}"
            )
        if noise.shape[-2:] == (1, 1):
            return noise.expand(b, self.noise_channels, h, w)
        if noise.shape[-2:] == (h, w):
            return noise
        raise ValueError(
            f"noise spatial shape must be (1,1) or {(h, w)}, got {tuple(noise.shape[-2:])}"
        )

    def forward(self, mask: Tensor, noise: Optional[Tensor] = None) -> Tensor:
        semantic = self.semantic_one_hot(mask)
        noise_map = self._noise_map(semantic, noise)
        condition = torch.cat([semantic, noise_map], 1)
        x0 = self.in_conv(condition)
        x1 = self.b1(x0, condition)
        x2 = self.b3(self.b2(self.down(x1), condition), condition)
        up = F.interpolate(x2, size=x1.shape[-2:], mode="bilinear", align_corners=False)
        return self.out(torch.cat([up, x1], 1))


def semantic_target(mask: Tensor) -> Tensor:
    if mask.ndim != 4 or mask.shape[1] != 1:
        raise ValueError("mask must be Bx1xHxW")
    return (mask[:, 0] > 0.5).long()


def _balanced_weight_map(target: Tensor) -> Tensor:
    """OASIS inverse-frequency balancing, normalized over present classes."""
    if target.ndim != 3:
        raise ValueError("target must be BxHxW")
    counts = torch.stack([(target == c).sum() for c in range(REAL_CLASSES)]).float()
    present = counts > 0
    n_present = int(present.sum())
    if n_present == 0:
        raise ValueError("target has no semantic pixels")
    coeff = torch.zeros_like(counts)
    total = float(target.numel())
    coeff[present] = total / (n_present * counts[present])
    return coeff.to(target.device)[target]


def balanced_semantic_ce(logits: Tensor, target: Tensor) -> Tensor:
    if logits.ndim != 4 or logits.shape[1] < REAL_CLASSES:
        raise ValueError("logits must be BxCxHxW with C>=2")
    if logits.shape[0] != target.shape[0] or logits.shape[-2:] != target.shape[-2:]:
        raise ValueError("logits/target shape mismatch")
    loss = F.cross_entropy(logits, target, reduction="none")
    return (loss * _balanced_weight_map(target)).mean()


def fake_ce(logits_fake: Tensor) -> Tensor:
    if logits_fake.ndim != 4 or logits_fake.shape[1] != REAL_CLASSES + 1:
        raise ValueError("Stage-I discriminator logits must have 3 channels")
    target = torch.full(
        (logits_fake.shape[0], *logits_fake.shape[-2:]),
        FAKE_CLASS,
        dtype=torch.long,
        device=logits_fake.device,
    )
    return F.cross_entropy(logits_fake, target)


def labelmix_mask(target: Tensor, generator: Optional[torch.Generator] = None) -> Tensor:
    """Class-aware LabelMix mask matching the official OASIS sampling semantics.

    One Bernoulli decision is sampled for each semantic class present in the
    complete batch target map, so a semantic class is never split into real/fake
    identities by LabelMix itself.
    """
    if target.ndim != 3:
        raise ValueError("target must be BxHxW")
    out = torch.zeros(target.shape, device=target.device, dtype=torch.float32)
    for c in torch.unique(target):
        if bool(torch.rand((), generator=generator, device=target.device) >= 0.5):
            out[target == c] = 1.0
    return out[:, None]


def labelmix_consistency(
    discriminator: nn.Module,
    real: Tensor,
    fake_detached: Tensor,
    logits_real: Tensor,
    logits_fake: Tensor,
    mix_mask: Tensor,
) -> Tensor:
    if mix_mask.shape != real[:, :1].shape:
        raise ValueError("mix_mask must be Bx1xHxW")
    mixed = mix_mask * real + (1.0 - mix_mask) * fake_detached
    expected = mix_mask * logits_real + (1.0 - mix_mask) * logits_fake
    return F.mse_loss(discriminator(mixed), expected)


def stage1_discriminator_loss(
    discriminator: OASISA2SDiscriminator,
    real: Tensor,
    mask: Tensor,
    fake_detached: Tensor,
    lambda_labelmix: float = 10.0,
    mix_generator: Optional[torch.Generator] = None,
):
    if discriminator.out_classes != 3:
        raise ValueError("Stage I requires a 3-class discriminator")
    target = semantic_target(mask)
    logits_real = discriminator(real)
    logits_fake = discriminator(fake_detached)
    loss_real = balanced_semantic_ce(logits_real, target)
    loss_fake = fake_ce(logits_fake)
    mix = labelmix_mask(target, mix_generator)
    loss_mix = labelmix_consistency(
        discriminator, real, fake_detached, logits_real, logits_fake, mix
    )
    return (
        loss_real + loss_fake + float(lambda_labelmix) * loss_mix,
        loss_real,
        loss_fake,
        loss_mix,
    )


def stage1_generator_loss(
    discriminator: OASISA2SDiscriminator, fake: Tensor, mask: Tensor
) -> Tensor:
    if discriminator.out_classes != 3:
        raise ValueError("Stage I requires a 3-class discriminator")
    return balanced_semantic_ce(discriminator(fake), semantic_target(mask))


def soft_dice_loss(logits: Tensor, target: Tensor, eps: float = 1e-6) -> Tensor:
    if logits.ndim != 4 or logits.shape[1] != 2:
        raise ValueError("Stage-II logits must be Bx2xHxW")
    probs = logits.softmax(1)[:, 1]
    truth = target.float()
    inter = (probs * truth).sum((1, 2))
    denom = probs.sum((1, 2)) + truth.sum((1, 2))
    return 1.0 - ((2.0 * inter + eps) / (denom + eps)).mean()


def stage2_segmentation_loss(
    logits: Tensor, mask: Tensor, dice_weight: float = 1.0
) -> Tensor:
    target = semantic_target(mask)
    return balanced_semantic_ce(logits, target) + float(dice_weight) * soft_dice_loss(
        logits, target
    )


def transfer_to_segmenter(
    discriminator: OASISA2SDiscriminator,
) -> OASISA2SDiscriminator:
    """Create a 2-class segmenter and preserve all BG/CRACK weights exactly."""
    if discriminator.out_classes != 3:
        raise ValueError("transfer source must be a 3-class Stage-I discriminator")
    segmenter = OASISA2SDiscriminator(discriminator.width, out_classes=2)
    src, dst = discriminator.state_dict(), segmenter.state_dict()
    for name in dst:
        dst[name].copy_(src[name][:REAL_CLASSES] if name.startswith("head.") else src[name])
    segmenter.load_state_dict(dst)
    return segmenter


def stage1_real_class_logits(
    discriminator: OASISA2SDiscriminator, image: Tensor
) -> Tensor:
    if discriminator.out_classes != 3:
        raise ValueError("A1 requires a 3-class Stage-I discriminator")
    return discriminator(image)[:, :REAL_CLASSES]


def parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
