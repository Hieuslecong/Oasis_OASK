import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class ManifestDataset(Dataset):
    """Common image/mask interface for explicit JSONL manifests."""

    def __init__(self, manifest, split, size, return_metadata=True):
        self.manifest = str(manifest)
        self.rows = [json.loads(line) for line in Path(manifest).read_text().splitlines() if line.strip()]
        self.rows = [row for row in self.rows if row.get("split") == split]
        if not self.rows:
            raise ValueError(f"no rows for split={split!r} in {manifest}")
        self.size = int(size)
        self.return_metadata = return_metadata

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        image_path = Path(row["image"])
        with Image.open(image_path) as im:
            original_size = tuple(reversed(im.size))
            image = im.convert("RGB").resize((self.size, self.size), Image.Resampling.BILINEAR)
        is_normal = row.get("is_normal", False) is True
        mask_path = row.get("mask")
        if is_normal:
            if mask_path not in (None, ""):
                raise ValueError("true-normal rows must use mask=null")
            mask = np.zeros((self.size, self.size), np.float32)
        else:
            if not mask_path:
                raise ValueError(f"missing mask for crack row {image_path}")
            with Image.open(mask_path) as m:
                mask = (np.asarray(m.convert("L").resize((self.size, self.size), Image.Resampling.NEAREST)) > 127).astype(np.float32)
        x = torch.from_numpy(np.asarray(image, np.float32).transpose(2, 0, 1) / 127.5 - 1.0)
        y = torch.from_numpy(mask[None])
        item = {"image": x, "mask": y, "dataset": row.get("dataset", "unknown"),
                "sample_id": str(row.get("sample_id", index)), "source_id": str(row.get("source_id", row.get("sample_id", index))),
                "original_size": original_size, "metadata": row}
        return item if self.return_metadata else (x, y)


class DatasetRegistry:
    def __init__(self, manifest, size):
        self.manifest, self.size = manifest, int(size)

    @classmethod
    def from_config(cls, config):
        return cls(config["dataset"]["manifest"], config["dataset"].get("size", 128))

    def build(self, split):
        return ManifestDataset(self.manifest, split, self.size)

    def build_loaders(self, config):
        from torch.utils.data import DataLoader

        opts = config.get("dataset", {})
        batch_size = int(opts.get("batch_size", 2))
        return {split: DataLoader(self.build(split), batch_size=batch_size, shuffle=split == "train", num_workers=0)
                for split in opts.get("splits", ["train", "val", "test"]) if split in opts.get("splits", [])}
