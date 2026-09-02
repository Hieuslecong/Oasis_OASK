import json
from pathlib import Path

import numpy as np


class NuisanceExtractor:
    """Extract appearance statistics primarily from background pixels."""

    def extract(self, image, mask=None):
        x = np.asarray(image, dtype=np.float32)
        if x.ndim == 3 and x.shape[0] in (1, 3):
            x = np.moveaxis(x, 0, -1)
        if x.max(initial=0) > 1.5:
            x = x / 255.0
        if x.shape[-1] == 1:
            x = np.repeat(x, 3, axis=-1)
        gray = x.mean(axis=-1)
        bg = np.ones(gray.shape, bool) if mask is None else np.asarray(mask).squeeze() < 0.5
        if not bg.any():
            bg = np.ones(gray.shape, bool)
        vals = gray[bg]
        gx = np.diff(gray, axis=1, prepend=gray[:, :1])
        gy = np.diff(gray, axis=0, prepend=gray[:1, :])
        edge = np.sqrt(gx * gx + gy * gy)
        local = gray - self._blur(gray)
        return {
            "illumination": float(np.clip(vals.mean(), 0, 1)),
            "contrast": float(np.clip(vals.std() * 4, 0, 1)),
            "local_contrast": float(np.clip(local[bg].std() * 8, 0, 1)),
            "saturation": float(np.clip((x.max(-1) - x.min(-1))[bg].mean(), 0, 1)),
            "edge_density": float(np.clip((edge[bg] > 0.08).mean(), 0, 1)),
            "background_frequency": float(np.clip(edge[bg].mean() * 4, 0, 1)),
            "background_entropy": self._entropy(vals),
            "roughness": float(np.clip(np.abs(local[bg]).mean() * 8, 0, 1)),
            "shadow": float(np.clip(1.0 - np.quantile(vals, 0.1) / max(np.quantile(vals, 0.9), 1e-6), 0, 1)),
            "stain": float(np.clip(np.std(x[bg], axis=0).mean() * 4, 0, 1)),
        }

    @staticmethod
    def _blur(x):
        pad = np.pad(x, 1, mode="reflect")
        return sum(pad[i:i+x.shape[0], j:j+x.shape[1]] for i in range(3) for j in range(3)) / 9.0

    @staticmethod
    def _entropy(vals):
        hist, _ = np.histogram(vals, bins=16, range=(0, 1), density=False)
        p = hist.astype(np.float64) / max(hist.sum(), 1)
        return float(np.clip(-(p[p > 0] * np.log2(p[p > 0])).sum() / 4.0, 0, 1))


class CalibrationModel:
    def __init__(self, stats):
        self.stats = stats

    @classmethod
    def fit(cls, records):
        if not records:
            raise ValueError("cannot calibrate from empty records")
        keys = sorted({k for r in records for k in r})
        out = {}
        for key in keys:
            v = np.asarray([r[key] for r in records if key in r], np.float64)
            med = float(np.median(v)); mad = float(np.median(np.abs(v - med)))
            out[key] = {"q01": float(np.quantile(v, .01)), "q05": float(np.quantile(v, .05)),
                        "median": med, "q95": float(np.quantile(v, .95)), "q99": float(np.quantile(v, .99)),
                        "mean": float(v.mean()), "std": float(v.std()), "mad": mad}
        matrix = np.asarray([[r.get(k, 0.0) for k in keys] for r in records], np.float64)
        if len(keys) > 1:
            centered = matrix - matrix.mean(axis=0, keepdims=True)
            denom = np.sqrt((centered * centered).sum(axis=0))
            correlation = (centered.T @ centered) / np.maximum(denom[:, None] * denom[None, :], 1e-12)
            correlation = np.nan_to_num(correlation, nan=0.0, posinf=0.0, neginf=0.0)
            np.fill_diagonal(correlation, 1.0)
        else:
            correlation = np.ones((1, 1), dtype=np.float64)
        out["_meta"] = {"factors": keys, "count": len(records), "correlation": correlation.tolist()}
        return cls(out)

    def sample(self, rng=None):
        rng = rng or np.random.default_rng()
        result = {}
        for key in self.stats.get("_meta", {}).get("factors", []):
            s = self.stats[key]
            result[key] = float(np.clip(rng.normal(s["median"], max(s["mad"] * 1.4826, s["std"] * .25, .02)), s["q05"], s["q95"]))
        return result

    def save(self, path):
        Path(path).write_text(json.dumps(self.stats, indent=2, sort_keys=True) + "\n")

    @classmethod
    def load(cls, path):
        return cls(json.loads(Path(path).read_text()))
