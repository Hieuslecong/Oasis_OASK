"""Training-only critic API.

The deployment path must not import this module. The implementation lives in
models.py so the architecture remains explicit and checkpoint export can keep
only the RGB student.
"""
from .models import ConditionalOASISCritic

__all__ = ["ConditionalOASISCritic"]
