"""Data loading shared by training and RGB-only evaluation.

This module intentionally contains no generator, discriminator, critic or AOSK
imports, so deployment code can reuse the manifest loader safely.
"""
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class ManifestDataset(Dataset):
    """Manifest-backed RGB crack-segmentation dataset.

    Manifest ``is_normal=true`` rows are external/explicit true-normal RGB and
    must use ``mask=null``. Crack rows must provide a real mask.

    A crack-source row may also carry
    ``empty_target_status="verified_no_crack"`` after a row-level GT-only
    certification. Such a row still loads its real (empty) mask, but when
    ``return_is_normal=True`` it is exposed to training as a true-negative
    target. This prevents certified N0 rows from being treated as crack-positive
    relational negatives merely because their source row has ``is_normal=false``.
    """

    def __init__(self, manifest, split, size, return_is_normal=False):
        self.rows = [
            json.loads(line)
            for line in Path(manifest).read_text().splitlines()
            if line.strip()
        ]
        self.rows = [row for row in self.rows if row.get("split") == split]
        if not self.rows:
            raise ValueError(f"manifest has no rows for split={split!r}")
        self.size = int(size)
        self.return_is_normal = bool(return_is_normal)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        image_path = Path(row["image"])
        image = Image.open(image_path).convert("RGB").resize(
            (self.size, self.size), resample=Image.Resampling.BILINEAR
        )

        if not isinstance(row.get("is_normal"), bool):
            raise ValueError(
                f"is_normal must be a JSON boolean: image={image_path}"
            )

        manifest_is_normal = row.get("is_normal") is True
        certified_n0 = row.get("empty_target_status") == "verified_no_crack"
        mask_path = row.get("mask")

        if manifest_is_normal:
            if mask_path not in (None, ""):
                raise ValueError(
                    f"true-normal row must use mask=null: image={image_path}"
                )
            y = np.zeros((1, self.size, self.size), dtype=np.float32)
        else:
            if not mask_path:
                raise ValueError(
                    f"crack-source row is missing mask: image={image_path}"
                )
            mask = Image.open(mask_path).convert("L").resize(
                (self.size, self.size), resample=Image.Resampling.NEAREST
            )
            y = (np.asarray(mask, dtype=np.uint8) > 127).astype(np.float32)[None]
            if certified_n0 and float(y.sum()) != 0.0:
                raise ValueError(
                    "empty_target_status=verified_no_crack conflicts with a "
                    f"non-empty target: image={image_path}"
                )

        x = np.asarray(image, dtype=np.float32).transpose(2, 0, 1) / 127.5 - 1.0
        x_t = torch.from_numpy(x)
        y_t = torch.from_numpy(y)

        target_is_normal = manifest_is_normal or certified_n0
        if self.return_is_normal:
            return x_t, y_t, torch.tensor(target_is_normal, dtype=torch.bool)
        return x_t, y_t
