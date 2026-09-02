import importlib
from pathlib import Path

import torch
from torch import nn

from .types import NuisanceVector, RenderOutput


class StressRenderer(nn.Module):
    factorized = False

    def render(self, mask, nuisance, noise=None, return_features=False):
        raise NotImplementedError


class ToyStressRenderer(StressRenderer):
    """Deterministic renderer used only for pipeline tests, never paper claims."""
    factorized = True

    def render(self, mask, nuisance, noise=None, return_features=False):
        if not isinstance(nuisance, NuisanceVector): nuisance = NuisanceVector(nuisance)
        m = mask.float()
        illum = nuisance.values.get("illumination", .5)
        contrast = nuisance.values.get("contrast", .5)
        texture = nuisance.values.get("texture", .0)
        noise_map = torch.rand_like(m) - .5
        b = (illum * .8 + .1) + noise_map * texture * .35
        b = (b - .5) * (0.25 + 1.5 * contrast) + .5
        crack = torch.cat([m, m, m], 1) * (0.15 + .75 * contrast)
        image = (b.repeat(1, 3, 1, 1) * (1 - m) + crack * m).clamp(0, 1) * 2 - 1
        return RenderOutput(image=image, rendered_mask=m, features={"illumination": float(illum), "contrast": float(contrast), "texture": float(texture)})


class DPGANStressRenderer(StressRenderer):
    """Adapter for an externally supplied DP-GAN generator.

    The adapter deliberately reports ``factorized=False`` unless the backend
    explicitly exposes a nuisance-conditioned render method. It never claims
    latent noise is disentangled.
    """
    def __init__(self, backend, checkpoint=None, device="cpu", nuisance_order=None):
        super().__init__(); self.backend = backend.to(device); self.device = device
        self.nuisance_order = list(nuisance_order or [])
        if checkpoint:
            path = Path(checkpoint)
            if not path.exists(): raise FileNotFoundError(f"DP-GAN checkpoint not found: {path}")
            payload = torch.load(path, map_location=device)
            state = payload.get("generator", payload.get("state_dict", payload)) if isinstance(payload, dict) else payload
            self.backend.load_state_dict(state, strict=False)
        self.eval()
        self.factorized = bool(hasattr(self.backend, "render_nuisance"))

    def render(self, mask, nuisance, noise=None, return_features=False):
        if not isinstance(nuisance, NuisanceVector): nuisance = NuisanceVector(nuisance)
        with torch.no_grad():
            if self.factorized:
                image = self.backend.render_nuisance(mask.to(self.device), nuisance.as_tensor(self.nuisance_order, self.device), noise=noise)
                metadata = {"variant": "G1", "factorized": True}
            else:
                # G0 compatibility path: latent/noise rendering is supported,
                # but the output is explicitly marked non-factorized.
                image = self.backend(mask.to(self.device), noise) if noise is not None else self.backend(mask.to(self.device))
                metadata = {"variant": "G0", "factorized": False}
        return RenderOutput(image=image, rendered_mask=mask, metadata=metadata)


def load_backend(spec):
    module_name, attr = spec.rsplit(":", 1)
    return getattr(importlib.import_module(module_name), attr)()
