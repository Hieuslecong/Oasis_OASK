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

    True-normal RGB rows are represented explicitly with ``is_normal=true`` and
    may omit ``mask``.  Their segmentation target is a virtual all-zero mask.
    Crack-positive rows MUST provide a real mask; a missing crack mask is an
    error rather than an implicit conversion to a normal example.
    """

    def __init__(self, manifest, split, size):
        self.rows = [
            json.loads(line)
            for line in Path(manifest).read_text().splitlines()
            if line.strip()
        ]
        self.rows = [row for row in self.rows if row.get("split") == split]
        if not self.rows:
            raise ValueError(f"manifest has no rows for split={split!r}")
        self.size = int(size)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        image_path = Path(row["image"])
        image = Image.open(image_path).convert("RGB").resize(
            (self.size, self.size), resample=Image.Resampling.BILINEAR
        )

        is_normal = bool(row.get("is_normal", False))
        mask_path = row.get("mask")
        if is_normal:
            # True normal RGB has no crack anywhere in the frame.  Keep the
            # zero target virtual instead of materialising thousands of black
            # PNG files on disk.
            y = np.zeros((1, self.size, self.size), dtype=np.float32)
        else:
            if not mask_path:
                raise ValueError(
                    f"crack-positive row is missing mask: image={image_path}"
                )
            mask = Image.open(mask_path).convert("L").resize(
                (self.size, self.size), resample=Image.Resampling.NEAREST
            )
            y = (np.asarray(mask, dtype=np.uint8) > 127).astype(np.float32)[None]

        x = np.asarray(image, dtype=np.float32).transpose(2, 0, 1) / 127.5 - 1.0
        return torch.from_numpy(x), torch.from_numpy(y)
