import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, cin, cout):
        super().__init__(); self.net = nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1), nn.GroupNorm(max(1, min(4, cout)), cout), nn.SiLU(), nn.Conv2d(cout, cout, 3, padding=1), nn.SiLU())
    def forward(self, x): return self.net(x)


class UNet(nn.Module):
    def __init__(self, width=16):
        super().__init__(); self.e1=ConvBlock(3,width); self.e2=ConvBlock(width,width*2); self.e3=ConvBlock(width*2,width*4); self.d2=ConvBlock(width*6,width*2); self.d1=ConvBlock(width*3,width); self.head=nn.Conv2d(width,1,1); self.pool=nn.MaxPool2d(2)
    def forward(self,x):
        e1=self.e1(x); e2=self.e2(self.pool(e1)); e3=self.e3(self.pool(e2)); d2=self.d2(__import__('torch').cat([F.interpolate(e3,size=e2.shape[-2:],mode='bilinear',align_corners=False),e2],1)); d1=self.d1(__import__('torch').cat([F.interpolate(d2,size=e1.shape[-2:],mode='bilinear',align_corners=False),e1],1)); return self.head(d1)


def build_segmenter(name, **kwargs):
    if name in {"unet", "s0"}: return UNet(**kwargs)
    if name in {"legacy_lightweight", "s1"}:
        from oasis_cycle_aosk.models import LightweightSegmenter
        return LightweightSegmenter(**kwargs)
    raise ValueError(f"unsupported segmenter {name!r}; available: unet, legacy_lightweight")
