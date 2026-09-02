import numpy as np


class RealismValidator:
    """Statistics-based realism screen calibrated on real training images."""
    def __init__(self, calibration, factors=("illumination", "contrast", "roughness"), min_score=0.0):
        self.calibration, self.factors, self.min_score = calibration, tuple(factors), float(min_score)

    def score(self, features):
        distances = []
        for key in self.factors:
            if key not in features or key not in self.calibration.stats: continue
            stats = self.calibration.stats[key]
            scale = max(stats.get("mad", 0.0) * 1.4826, stats.get("std", 0.0), .02)
            distances.append(abs(float(features[key]) - stats["median"]) / scale)
        return float(np.exp(-np.mean(distances))) if distances else 0.0

    def valid(self, features):
        value = self.score(features)
        return value >= self.min_score, value
