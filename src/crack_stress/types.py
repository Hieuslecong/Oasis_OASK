from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True)
class NuisanceVector:
    """Dynamic, config-driven nuisance vector.

    Unknown dimensions are allowed so adding a factor does not require changing
    model code. Values are kept in the normalized [0, 1] space by callers.
    """

    values: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self):
        clean = {str(k): float(v) for k, v in self.values.items()}
        for key, value in clean.items():
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"nuisance {key!r} must be in [0, 1], got {value}")
        object.__setattr__(self, "values", clean)

    @classmethod
    def from_config(cls, active, values=None):
        values = values or {}
        return cls({name: float(values.get(name, 0.5)) for name in active})

    def with_value(self, name: str, value: float):
        updated = dict(self.values)
        updated[name] = float(value)
        return NuisanceVector(updated)

    def as_tensor(self, order=None, device=None):
        import torch

        order = list(order or self.values.keys())
        return torch.tensor([self.values.get(k, 0.0) for k in order], dtype=torch.float32, device=device)


@dataclass
class RenderOutput:
    image: Any
    rendered_mask: Any | None = None
    features: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationResult:
    valid: bool
    skeleton_score: float = 0.0
    width_score: float = 0.0
    connectivity_score: float = 0.0
    realism_score: float = 0.0
    violations: list[str] = field(default_factory=list)
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def as_dict(self):
        return {
            "valid": bool(self.valid),
            "skeleton_score": float(self.skeleton_score),
            "width_score": float(self.width_score),
            "connectivity_score": float(self.connectivity_score),
            "realism_score": float(self.realism_score),
            "violations": list(self.violations),
            "diagnostics": self.diagnostics,
        }
