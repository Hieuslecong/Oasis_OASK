import json
import numpy as np
from PIL import Image
import pytest
from oasis_cycle_aosk.data import ManifestDataset


def test_certified_n0_is_true_negative_for_training(tmp_path):
    image, mask = tmp_path/"i.png", tmp_path/"m.png"
    Image.new("RGB", (8,8), color=(50,60,70)).save(image)
    Image.new("L", (8,8), color=0).save(mask)
    manifest = tmp_path/"manifest.jsonl"
    manifest.write_text(json.dumps({
        "image": str(image), "mask": str(mask), "split": "train",
        "source_id": "s", "lineage_id": "p", "is_normal": False,
        "empty_target_status": "verified_no_crack"
    }) + "\n")
    ds = ManifestDataset(manifest, "train", 8, return_is_normal=True)
    _, y, is_normal = ds[0]
    assert float(y.sum()) == 0.0
    assert bool(is_normal) is True


def test_certified_n0_rejects_nonempty_target(tmp_path):
    image, mask = tmp_path/"i.png", tmp_path/"m.png"
    Image.new("RGB", (8,8), color=(50,60,70)).save(image)
    a = np.zeros((8,8), dtype=np.uint8); a[2:4,2:4] = 255
    Image.fromarray(a).save(mask)
    manifest = tmp_path/"manifest.jsonl"
    manifest.write_text(json.dumps({
        "image": str(image), "mask": str(mask), "split": "train",
        "source_id": "s", "lineage_id": "p", "is_normal": False,
        "empty_target_status": "verified_no_crack"
    }) + "\n")
    ds = ManifestDataset(manifest, "train", 8, return_is_normal=True)
    with pytest.raises(ValueError, match="conflicts"):
        _ = ds[0]
