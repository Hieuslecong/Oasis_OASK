import numpy as np

from .metrics import _bin, _skeleton, connected_components
from .types import RenderOutput, ValidationResult


class CrackGeometryValidator:
    def __init__(self, skeleton_min=.98, width_tolerance=.25, connectivity_min=.9):
        self.skeleton_min, self.width_tolerance, self.connectivity_min = skeleton_min, width_tolerance, connectivity_min

    def validate(self, original_mask, rendered_mask):
        y, g = _bin(original_mask), _bin(rendered_mask)
        sy, sg = _skeleton(y), _skeleton(g)
        skeleton_score = float((sy == sg).mean())
        width_score = float(1.0 - abs(y.sum() - g.sum()) / max(float(y.sum()), 1.0))
        cy, cg = connected_components(y), connected_components(g)
        connectivity_score = 1.0 if cy == cg else float(min(cy, cg) / max(cy, cg, 1))
        violations = []
        if skeleton_score < self.skeleton_min: violations.append("skeleton")
        if abs(y.sum() - g.sum()) / max(float(y.sum()), 1.0) > self.width_tolerance: violations.append("width")
        if connectivity_score < self.connectivity_min: violations.append("connectivity")
        return skeleton_score, width_score, connectivity_score, violations


class ValidStressEnvelope:
    def __init__(self, geometry, realism_min=0.0):
        self.geometry, self.realism_min = geometry, float(realism_min)

    def accept(self, original_mask, generated_image, nuisance_vector, rendered_mask=None, realism_score=1.0):
        if rendered_mask is None:
            return ValidationResult(False, realism_score=realism_score, violations=["renderer_did_not_return_semantic_mask"])
        ss, ws, cs, violations = self.geometry.validate(original_mask, rendered_mask)
        if realism_score < self.realism_min: violations.append("realism")
        return ValidationResult(not violations, ss, ws, cs, float(realism_score), violations)
