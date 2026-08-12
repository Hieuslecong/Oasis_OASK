import torch
from torch import nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1), nn.GroupNorm(4, cout), nn.SiLU(), nn.Conv2d(cout, cout, 3, padding=1), nn.SiLU())
    def forward(self, x): return self.net(x)


class LightweightSegmenter(nn.Module):
    """RGB-only deployment model; no GAN/AOSK dependency."""
    def __init__(self, width=24):
        super().__init__()
        self.e1, self.e2 = ConvBlock(3, width), ConvBlock(width, width * 2)
        self.pool = nn.MaxPool2d(2)
        self.d = ConvBlock(width * 3, width)
        self.head = nn.Conv2d(width, 1, 1)
    def forward(self, x):
        a = self.e1(x); b = self.e2(self.pool(a))
        y = F.interpolate(b, size=a.shape[-2:], mode="bilinear", align_corners=False)
        return self.head(self.d(torch.cat([a, y], 1)))


class MultiScaleLightweightSegmenter(nn.Module):
    """RGB-only multi-scale student for thin-crack detail preservation."""
    def __init__(self, width=16):
        super().__init__()
        self.f1 = ConvBlock(3, width)
        self.f2 = ConvBlock(width, width * 2)
        self.f3 = ConvBlock(width * 2, width * 4)
        self.f4 = ConvBlock(width * 4, width * 4)
        self.pool = nn.MaxPool2d(2)
        self.d3 = ConvBlock(width * 8, width * 2)
        self.d2 = ConvBlock(width * 4, width)
        self.d1 = ConvBlock(width * 2, width)
        self.head = nn.Conv2d(width, 1, 1)
    def forward(self, x):
        f1 = self.f1(x)
        f2 = self.f2(self.pool(f1))
        f3 = self.f3(self.pool(f2))
        f4 = self.f4(self.pool(f3))
        d3 = self.d3(torch.cat([F.interpolate(f4, size=f3.shape[-2:], mode="bilinear", align_corners=False), f3], 1))
        d2 = self.d2(torch.cat([F.interpolate(d3, size=f2.shape[-2:], mode="bilinear", align_corners=False), f2], 1))
        d1 = self.d1(torch.cat([F.interpolate(d2, size=f1.shape[-2:], mode="bilinear", align_corners=False), f1], 1))
        return self.head(d1)


class HSwish(nn.Module):
    def __init__(self, inplace=True):
        super().__init__(); self.inplace = inplace
    def forward(self, x):
        return x * F.relu6(x + 3.0, inplace=self.inplace) / 6.0


class SqueezeExcite(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()
        hidden = max(8, channels // reduction)
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Conv2d(channels, hidden, 1), nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1), nn.Hardsigmoid(inplace=True)
        )
    def forward(self, x): return x * self.net(x)


class MobileNetV3Block(nn.Module):
    def __init__(self, cin, hidden, cout, stride, use_se, activation):
        super().__init__()
        self.use_residual = stride == 1 and cin == cout
        act = nn.ReLU if activation == "relu" else HSwish
        layers = [nn.Conv2d(cin, hidden, 1, bias=False), nn.BatchNorm2d(hidden), act(inplace=True)]
        layers += [nn.Conv2d(hidden, hidden, 3, stride=stride, padding=1, groups=hidden, bias=False),
                   nn.BatchNorm2d(hidden), act(inplace=True)]
        if use_se: layers.append(SqueezeExcite(hidden))
        layers += [nn.Conv2d(hidden, cout, 1, bias=False), nn.BatchNorm2d(cout)]
        self.block = nn.Sequential(*layers)
    def forward(self, x):
        y = self.block(x)
        return x + y if self.use_residual else y


class MobileNetV3SmallSegmenter(nn.Module):
    """MobileNetV3-Small-style RGB-only student with a compact decoder."""
    def __init__(self, width_mult=1.0):
        super().__init__()
        def c(v): return max(8, int(round(v * width_mult)))
        self.stem = nn.Sequential(nn.Conv2d(3, c(16), 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(c(16)), HSwish())
        self.b1 = MobileNetV3Block(c(16), c(16), c(16), 2, False, "relu")   # 1/4
        self.b2 = MobileNetV3Block(c(16), c(72), c(24), 2, False, "relu")   # 1/8
        self.b3 = MobileNetV3Block(c(24), c(88), c(24), 1, False, "hs")
        self.b4 = MobileNetV3Block(c(24), c(96), c(40), 2, True, "hs")      # 1/16
        self.b5 = MobileNetV3Block(c(40), c(240), c(40), 1, True, "hs")
        self.b6 = MobileNetV3Block(c(40), c(120), c(48), 1, True, "hs")
        self.b7 = MobileNetV3Block(c(48), c(288), c(96), 2, True, "hs")     # 1/32
        self.b8 = MobileNetV3Block(c(96), c(576), c(96), 1, True, "hs")
        self.d16 = ConvBlock(c(96) + c(48), c(48))
        self.d8 = ConvBlock(c(48) + c(24), c(32))
        self.d4 = ConvBlock(c(32) + c(16), c(16))
        self.head = nn.Conv2d(c(16), 1, 1)

    @staticmethod
    def _up(x, ref):
        return F.interpolate(x, size=ref.shape[-2:], mode="bilinear", align_corners=False)

    def forward(self, x):
        s2 = self.stem(x)
        s4 = self.b1(s2)
        s8 = self.b3(self.b2(s4))
        s16 = self.b6(self.b5(self.b4(s8)))
        s32 = self.b8(self.b7(s16))
        d16 = self.d16(torch.cat([self._up(s32, s16), s16], 1))
        d8 = self.d8(torch.cat([self._up(d16, s8), s8], 1))
        d4 = self.d4(torch.cat([self._up(d8, s4), s4], 1))
        return self.head(F.interpolate(d4, size=x.shape[-2:], mode="bilinear", align_corners=False))


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, cin, cout, stride=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cin, cin, 3, stride=stride, padding=1, groups=cin, bias=False),
            nn.GroupNorm(max(1, min(4, cin)), cin), nn.SiLU(),
            nn.Conv2d(cin, cout, 1, bias=False),
            nn.GroupNorm(max(1, min(4, cout)), cout), nn.SiLU(),
        )
    def forward(self, x): return self.net(x)


class DSUNetLite(nn.Module):
    """Depthwise-separable U-Net-style student for thin crack detail."""
    def __init__(self, width=16):
        super().__init__()
        self.e1 = DepthwiseSeparableConv(3, width, 1)
        self.e2 = DepthwiseSeparableConv(width, width * 2, 2)
        self.e3 = DepthwiseSeparableConv(width * 2, width * 4, 2)
        self.e4 = DepthwiseSeparableConv(width * 4, width * 6, 2)
        self.d3 = DepthwiseSeparableConv(width * 10, width * 4)
        self.d2 = DepthwiseSeparableConv(width * 6, width * 2)
        self.d1 = DepthwiseSeparableConv(width * 3, width)
        self.head = nn.Conv2d(width, 1, 1)
    def forward(self, x):
        e1 = self.e1(x); e2 = self.e2(e1); e3 = self.e3(e2); e4 = self.e4(e3)
        d3 = self.d3(torch.cat([F.interpolate(e4, size=e3.shape[-2:], mode="bilinear", align_corners=False), e3], 1))
        d2 = self.d2(torch.cat([F.interpolate(d3, size=e2.shape[-2:], mode="bilinear", align_corners=False), e2], 1))
        d1 = self.d1(torch.cat([F.interpolate(d2, size=e1.shape[-2:], mode="bilinear", align_corners=False), e1], 1))
        return self.head(d1)


class FastSCNNLite(nn.Module):
    """Compact Fast-SCNN-style high-resolution/detail fusion student."""
    def __init__(self, width=16):
        super().__init__()
        self.down = nn.Sequential(DepthwiseSeparableConv(3, width, 2), DepthwiseSeparableConv(width, width * 2, 2))
        self.low = nn.Sequential(DepthwiseSeparableConv(width * 2, width * 3, 2), DepthwiseSeparableConv(width * 3, width * 4, 2))
        self.fuse = DepthwiseSeparableConv(width * 6, width * 2)
        self.head = nn.Conv2d(width * 2, 1, 1)
    def forward(self, x):
        detail = self.down(x)
        low = self.low(detail)
        fused = self.fuse(torch.cat([F.interpolate(low, size=detail.shape[-2:], mode="bilinear", align_corners=False), detail], 1))
        return self.head(F.interpolate(fused, size=x.shape[-2:], mode="bilinear", align_corners=False))


class BiSeNetTiny(nn.Module):
    """Tiny detail/semantic dual-branch student inspired by BiSeNet."""
    def __init__(self, width=16):
        super().__init__()
        self.detail = nn.Sequential(DepthwiseSeparableConv(3, width, 2), DepthwiseSeparableConv(width, width * 2, 2))
        self.semantic = nn.Sequential(ConvBlock(3, width, ), nn.MaxPool2d(2), ConvBlock(width, width * 2), nn.MaxPool2d(2), ConvBlock(width * 2, width * 4), nn.MaxPool2d(2), ConvBlock(width * 4, width * 4))
        self.fuse = ConvBlock(width * 6, width * 2)
        self.head = nn.Conv2d(width * 2, 1, 1)
    def forward(self, x):
        d = self.detail(x); s = self.semantic(x)
        z = self.fuse(torch.cat([d, F.interpolate(s, size=d.shape[-2:], mode="bilinear", align_corners=False)], 1))
        return self.head(F.interpolate(z, size=x.shape[-2:], mode="bilinear", align_corners=False))


class TranslationGenerator(nn.Module):
    """Image-to-image generator. semantic_map is background-only or crack mask."""
    def __init__(self, width=32):
        super().__init__()
        self.body = nn.Sequential(ConvBlock(4, width), ConvBlock(width, width), nn.Conv2d(width, 3, 1), nn.Tanh())
    def forward(self, image, semantic_map):
        return self.body(torch.cat([image, semantic_map], 1))


class OASISDiscriminator(nn.Module):
    """Semantic discriminator with classes background=0, crack=1, fake=2."""
    def __init__(self, width=32):
        super().__init__()
        self.net = nn.Sequential(ConvBlock(3, width), nn.AvgPool2d(2), ConvBlock(width, width * 2), nn.Conv2d(width * 2, 3, 1))
    def forward(self, image):
        return F.interpolate(self.net(image), size=image.shape[-2:], mode="bilinear", align_corners=False)


class ConditionalOASISCritic(nn.Module):
    """Training-only RGB+mask semantic consistency critic."""
    def __init__(self, width=16):
        super().__init__()
        # Separate encoders plus multiplicative interaction prevent the critic
        # from solving the task from mask shape alone.
        self.image_encoder = ConvBlock(3, width)
        self.mask_encoder = ConvBlock(1, width)
        self.net = nn.Sequential(ConvBlock(width * 3, width * 2), nn.AvgPool2d(2), ConvBlock(width * 2, width * 2), nn.AvgPool2d(2), ConvBlock(width * 2, width * 2), nn.Conv2d(width * 2, 3, 1))
    def forward(self, image, mask):
        if image.shape[0] != mask.shape[0] or mask.shape[1] != 1: raise ValueError("expected RGB image and Bx1xHxW mask")
        fi = self.image_encoder(image); fm = self.mask_encoder(mask)
        return F.interpolate(self.net(torch.cat([fi, fm, fi * fm], 1)), size=image.shape[-2:], mode="bilinear", align_corners=False)


class RelationalOASISRC(nn.Module):
    """Training-only RGB--mask relational critic.

    OASIS-RC has three outputs: semantic pixel classes (background/crack/invalid),
    a pixel mismatch map, and a pair-level validity logit.  The pair-level head
    prevents the critic from being only a local mask-shape classifier, while the
    relational fusion explicitly exposes image/mask agreement features.
    """
    def __init__(self, width=8):
        super().__init__()
        self.i1 = ConvBlock(3, width)
        self.m1 = ConvBlock(1, width)
        self.f1 = ConvBlock(width * 4, width)
        self.i2 = ConvBlock(width, width * 2)
        self.m2 = ConvBlock(width, width * 2)
        self.f2 = ConvBlock(width * 8, width * 2)
        self.i3 = ConvBlock(width * 2, width * 4)
        self.m3 = ConvBlock(width * 2, width * 4)
        self.f3 = ConvBlock(width * 16, width * 4)
        self.pool = nn.AvgPool2d(2)
        self.u2 = ConvBlock(width * 6, width * 2)
        self.u1 = ConvBlock(width * 3, width)
        # Hierarchical heads: valid crack/background semantics are learned
        # separately from RGB--mask invalidity, then composed into 3 classes.
        self.crack_head = nn.Conv2d(width, 1, 1)
        self.mismatch_head = nn.Conv2d(width, 1, 1)
        self.pair_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(width * 4, width * 2), nn.SiLU(), nn.Linear(width * 2, 1)
        )

    @staticmethod
    def _relational(fi, fm):
        return torch.cat([fi, fm, fi * fm, (fi - fm).abs()], dim=1)

    def forward(self, image, mask):
        if image.ndim != 4 or image.shape[1] != 3:
            raise ValueError("image must be Bx3xHxW")
        if mask.ndim != 4 or mask.shape[1] != 1 or image.shape[0] != mask.shape[0]:
            raise ValueError("mask must be Bx1xHxW with matching batch")
        i1, m1 = self.i1(image), self.m1(mask)
        z1 = self.f1(self._relational(i1, m1))
        i2, m2 = self.i2(self.pool(i1)), self.m2(self.pool(m1))
        z2 = self.f2(self._relational(i2, m2))
        i3, m3 = self.i3(self.pool(i2)), self.m3(self.pool(m2))
        z3 = self.f3(self._relational(i3, m3))
        u2 = self.u2(torch.cat([F.interpolate(z3, size=z2.shape[-2:], mode="bilinear", align_corners=False), z2], 1))
        u1 = self.u1(torch.cat([F.interpolate(u2, size=z1.shape[-2:], mode="bilinear", align_corners=False), z1], 1))
        crack = F.interpolate(self.crack_head(u1), size=image.shape[-2:], mode="bilinear", align_corners=False)
        invalid = F.interpolate(self.mismatch_head(u1), size=image.shape[-2:], mode="bilinear", align_corners=False)
        # For valid pairs, crack > background is controlled by crack.  For
        # invalid pairs, invalid wins over either valid semantic class.
        semantic = torch.cat([-crack - invalid, crack - invalid, invalid], dim=1)
        return {
            "semantic": semantic,
            "crack": crack,
            "mismatch": invalid,
            "invalid": invalid,
            "pair": self.pair_head(z3).flatten(1),
        }
