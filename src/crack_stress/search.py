from dataclasses import dataclass

import torch

from .types import NuisanceVector


@dataclass
class Candidate:
    image: torch.Tensor
    nuisance: NuisanceVector
    difficulty: float
    validity: object


class RandomNuisanceSampler:
    def __init__(self, calibration=None, active=None, seed=0):
        import numpy as np
        self.calibration, self.active, self.rng = calibration, list(active or []), np.random.default_rng(seed)

    def sample(self):
        values = self.calibration.sample(self.rng) if self.calibration else {k: float(self.rng.random()) for k in self.active}
        return NuisanceVector(values)


class HardNuisanceSearcher:
    def __init__(self, sampler, candidates=8, loss_fn=None):
        self.sampler, self.candidates, self.loss_fn = sampler, int(candidates), loss_fn

    def search(self, model, mask, renderer, envelope):
        was_training = model.training; model.eval(); found = []
        with torch.no_grad():
            for _ in range(self.candidates):
                nuisance = self.sampler.sample(); output = renderer.render(mask, nuisance)
                validity = envelope.accept(mask, output.image, nuisance, output.rendered_mask)
                if not validity.valid: continue
                pred = model(output.image)
                difficulty = float((self.loss_fn(pred, mask) if self.loss_fn else torch.sigmoid(pred).mean()).detach().cpu())
                found.append(Candidate(output.image, nuisance, difficulty, validity))
        if was_training: model.train()
        if not found: return None, {"candidate_count": self.candidates, "valid_candidate_count": 0}
        best = max(found, key=lambda c: c.difficulty)
        return best, {"candidate_count": self.candidates, "valid_candidate_count": len(found), "selected_hardness": best.difficulty}
