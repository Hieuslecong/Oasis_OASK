import hashlib
import json
import numpy as np
from PIL import Image
from scripts.build_cleaneval_v1 import build


def sha256(path):
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def row(tmp_path, name, split, value, empty=False):
    image = tmp_path/f"{name}.png"
    mask = tmp_path/f"{name}_m.png"
    Image.new("RGB", (16,16), color=(value, value, value)).save(image)
    a = np.zeros((16,16), dtype=np.uint8)
    if not empty:
        a[value % 8:value % 8 + 3, 4:7] = 255
    Image.fromarray(a).save(mask)
    return {"image": str(image), "mask": str(mask), "split": split,
            "source_id": name, "lineage_id": name, "is_normal": False}


def test_builder_excludes_unreviewed_empty_and_freezes_actual_hashes(tmp_path):
    rows = [row(tmp_path, "tr", "train", 20),
            row(tmp_path, "va", "val", 60),
            row(tmp_path, "te", "test", 100),
            row(tmp_path, "empty", "train", 140, True)]
    source = tmp_path/"source.jsonl"
    source.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    out = tmp_path/"out"
    report = build(source, out, certification_csv=None, resize_size=16)
    assert report["status"] == "PASS"
    assert report["empty_target_excluded"] == 1
    freeze = json.loads((out/"benchmark_freeze.json").read_text())
    for name, digest in freeze["hashes"].items():
        assert sha256(out/name) == digest
