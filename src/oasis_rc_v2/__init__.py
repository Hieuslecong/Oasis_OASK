"""Canonical OASIS-RC v2 implementation.

This package is intentionally independent of legacy OASIS/RC training code.
Only the RGB student/data utilities live in the shared crack-segmentation package.
"""
from .checkpoint import (
    CHECKPOINT_SCHEMA,
    EXPERIMENT_ID,
    IMPLEMENTATION_VERSION,
    METHOD_VERSION,
)
from .critic import OASISRCv2Critic

__all__ = [
    "CHECKPOINT_SCHEMA",
    "EXPERIMENT_ID",
    "IMPLEMENTATION_VERSION",
    "METHOD_VERSION",
    "OASISRCv2Critic",
]
