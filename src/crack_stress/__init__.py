"""Renderer-agnostic crack stress learning utilities.

DP-GAN is an optional training-time renderer. Deployment remains RGB -> segmenter.
"""

from .types import NuisanceVector, RenderOutput, ValidationResult

__all__ = ["NuisanceVector", "RenderOutput", "ValidationResult"]
