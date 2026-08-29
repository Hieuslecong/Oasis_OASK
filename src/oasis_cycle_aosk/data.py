"""Manifest loading and evidence-oriented preflight for OASIS-A2S."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


def _sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_manifest_rows(manifest: str | Path) -> list[dict]:
    return [json.loads(line) for line in Path(manifest).read_text().splitlines() if line.strip()]


def audit_manifest(
    manifest: str | Path,
    splits: tuple[str, ...] | list[str],
    *,
    require_lineage: bool = True,
    allow_size_mismatch: bool = False,
) -> dict:
    """Fail closed on cross-split leakage and bind evidence to dataset bytes."""
    requested = tuple(str(s) for s in splits)
    rows = [r for r in load_manifest_rows(manifest) if r.get("split") in requested]
    counts = {s: sum(r.get("split") == s for r in rows) for s in requested}
    missing = [s for s, n in counts.items() if n == 0]
    if missing:
        raise ValueError(f"manifest has no rows for required split(s): {missing}")

    image_seen: dict[str, str] = {}
    lineage_seen: dict[str, str] = {}
    row_seen: set[str] = set()
    canonical = []
    size_mismatches = 0
    for idx, row in enumerate(rows):
        split = str(row.get("split"))
        image = Path(row.get("image", ""))
        if not image.is_file():
            raise ValueError(f"manifest row {idx}: image not found: {image}")
        image_sha = _sha256_file(image)
        prior_split = image_seen.get(image_sha)
        if prior_split is not None and prior_split != split:
            raise ValueError(f"exact RGB duplicate crosses splits: {prior_split} -> {split}")
        image_seen.setdefault(image_sha, split)

        lineage = row.get("lineage_id")
        if require_lineage and (not isinstance(lineage, str) or not lineage.strip()):
            raise ValueError(f"manifest row {idx}: lineage_id is required for canonical runs")
        if isinstance(lineage, str) and lineage.strip():
            lineage = lineage.strip()
            prior_split = lineage_seen.get(lineage)
            if prior_split is not None and prior_split != split:
                raise ValueError(f"lineage_id={lineage!r} crosses splits: {prior_split} -> {split}")
            lineage_seen.setdefault(lineage, split)

        is_normal = row.get("is_normal") is True
        mask_value = row.get("mask")
        mask_sha = None
        if is_normal:
            if mask_value not in (None, ""):
                raise ValueError(f"manifest row {idx}: true-normal row must use mask=null")
        else:
            if not mask_value:
                raise ValueError(f"manifest row {idx}: crack-source row is missing mask")
            mask = Path(mask_value)
            if not mask.is_file():
                raise ValueError(f"manifest row {idx}: mask not found: {mask}")
            mask_sha = _sha256_file(mask)
            with Image.open(image) as im, Image.open(mask) as mm:
                if im.size != mm.size:
                    size_mismatches += 1
                    if not allow_size_mismatch:
                        raise ValueError(
                            f"manifest row {idx}: image/mask size mismatch {im.size} vs {mm.size}"
                        )

        record = {
            "split": split,
            "image_sha256": image_sha,
            "mask_sha256": mask_sha,
            "source_id": row.get("source_id"),
            "lineage_id": lineage,
            "is_normal": bool(is_normal),
            "empty_target_status": row.get("empty_target_status"),
        }
        row_key = json.dumps(record, sort_keys=True, separators=(",", ":"))
        if row_key in row_seen:
            raise ValueError(f"duplicate manifest evidence row detected at index {idx}")
        row_seen.add(row_key)
        canonical.append(record)

    payload = json.dumps(
        sorted(canonical, key=lambda r: json.dumps(r, sort_keys=True)),
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return {
        "dataset_content_sha256": hashlib.sha256(payload).hexdigest(),
        "split_counts": counts,
        "num_rows": len(rows),
        "num_unique_images": len(image_seen),
        "num_unique_lineages": len(lineage_seen),
        "image_mask_size_mismatch_count": size_mismatches,
        "lineage_required": bool(require_lineage),
    }


class ManifestDataset(Dataset):
    """Manifest-backed RGB crack-segmentation dataset."""

    def __init__(self, manifest, split, size, return_is_normal=False, return_metadata=False):
        self.rows = [row for row in load_manifest_rows(manifest) if row.get("split") == split]
        if not self.rows:
            raise ValueError(f"manifest has no rows for split={split!r}")
        self.size = int(size)
        self.return_is_normal = bool(return_is_normal)
        self.return_metadata = bool(return_metadata)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        image_path = Path(row["image"])
        image = Image.open(image_path).convert("RGB").resize(
            (self.size, self.size), resample=Image.Resampling.BILINEAR
        )

        if not isinstance(row.get("is_normal"), bool):
            raise ValueError(f"is_normal must be a JSON boolean: image={image_path}")
        manifest_is_normal = row.get("is_normal") is True
        certified_n0 = row.get("empty_target_status") == "verified_no_crack"
        mask_path = row.get("mask")

        if manifest_is_normal:
            if mask_path not in (None, ""):
                raise ValueError(f"true-normal row must use mask=null: image={image_path}")
            y = np.zeros((1, self.size, self.size), dtype=np.float32)
        else:
            if not mask_path:
                raise ValueError(f"crack-source row is missing mask: image={image_path}")
            mask = Image.open(mask_path).convert("L").resize(
                (self.size, self.size), resample=Image.Resampling.NEAREST
            )
            y = (np.asarray(mask, dtype=np.uint8) > 127).astype(np.float32)[None]
            if certified_n0 and float(y.sum()) != 0.0:
                raise ValueError(
                    "empty_target_status=verified_no_crack conflicts with a non-empty target: "
                    f"image={image_path}"
                )

        x = np.asarray(image, dtype=np.float32).transpose(2, 0, 1) / 127.5 - 1.0
        x_t, y_t = torch.from_numpy(x), torch.from_numpy(y)
        target_is_normal = manifest_is_normal or certified_n0
        outputs = [x_t, y_t]
        if self.return_is_normal:
            outputs.append(torch.tensor(target_is_normal, dtype=torch.bool))
        if self.return_metadata:
            source_id = str(row.get("source_id") or "unknown")
            lineage_id = str(row.get("lineage_id") or "unknown")
            sample_key = str(row.get("sample_id") or f"{source_id}::{lineage_id}::{image_path.name}")
            outputs.append({"sample_key": sample_key, "source_id": source_id, "lineage_id": lineage_id})
        return tuple(outputs) if len(outputs) > 2 else (x_t, y_t)
