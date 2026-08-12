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
    def __init__(self, manifest, split, size):
        self.rows = [json.loads(line) for line in Path(manifest).read_text().splitlines() if line.strip()]
        self.rows = [row for row in self.rows if row.get("split") == split]
        if not self.rows:
            raise ValueError(f"manifest has no rows for split={split!r}")
        self.size = int(size)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        image = Image.open(row["image"]).convert("RGB").resize(
            (self.size, self.size), resample=Image.Resampling.BILINEAR
        )
        mask_path = row.get("mask")
        if mask_path:
            # Segmentation labels must not be interpolated with a smooth kernel.
            mask = Image.open(mask_path).convert("L").resize(
                (self.size, self.size), resample=Image.Resampling.NEAREST
            )
            y = (np.asarray(mask, dtype=np.uint8) > 127).astype(np.float32)[None]
        else:
            y = np.zeros((1, self.size, self.size), dtype=np.float32)
        x = np.asarray(image, dtype=np.float32).transpose(2, 0, 1) / 127.5 - 1.0
        return torch.from_numpy(x), torch.from_numpy(y)
