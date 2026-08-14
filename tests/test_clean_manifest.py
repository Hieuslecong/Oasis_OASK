import importlib.util
import json
from pathlib import Path

import numpy as np
from PIL import Image


def _load_clean_manifest():
    path = Path(__file__).resolve().parents[1] / "scripts" / "clean_manifest.py"
    spec = importlib.util.spec_from_file_location("clean_manifest_script", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.run, module.strip_lineage


run, strip_lineage = _load_clean_manifest()


def rgb(p, v):
    Image.new("RGB", (16, 16), color=(v, v, v)).save(p)


def mask(p, x=4):
    a = np.zeros((16, 16), dtype=np.uint8)
    a[x:x + 3, x:x + 3] = 255
    Image.fromarray(a).save(p)


def row(i, m, split, lineage):
    return {
        "image": str(i),
        "mask": str(m),
        "split": split,
        "source_id": "s",
        "lineage_id": lineage,
        "is_normal": False,
    }


def write_manifest(p, rows):
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return p


def independent_train(tmp_path):
    i, m = tmp_path / "keep_train.png", tmp_path / "keep_train_m.png"
    rgb(i, 230)
    mask(m, 11)
    return row(i, m, "train", "train:keep-independent")


def test_strip_lineage():
    assert strip_lineage("train:x") == "x"
    assert strip_lineage("val::x") == "x"


def test_lineage_priority_preserves_test(tmp_path):
    ti, vi, ei = tmp_path / "t.png", tmp_path / "v.png", tmp_path / "e.png"
    tm, vm, em = tmp_path / "tm.png", tmp_path / "vm.png", tmp_path / "em.png"
    for p, v in ((ti, 10), (vi, 20), (ei, 30)):
        rgb(p, v)
    for p, x in ((tm, 2), (vm, 6), (em, 9)):
        mask(p, x)
    rows = [
        independent_train(tmp_path),
        row(ti, tm, "train", "train:parent"),
        row(vi, vm, "val", "val:other"),
        row(ei, em, "test", "test:parent"),
    ]
    rep = run(write_manifest(tmp_path / "m.jsonl", rows), tmp_path / "out", 16)
    assert rep["status"] == "PASS"
    assert rep["rows_removed"]["train"] == 1
    assert rep["rows_removed"]["test"] == 0


def test_nonempty_mask_reuse_drops_lower_priority(tmp_path):
    ti, vi, ei = tmp_path / "t.png", tmp_path / "v.png", tmp_path / "e.png"
    shared, em = tmp_path / "shared.png", tmp_path / "em.png"
    for p, v in ((ti, 10), (vi, 20), (ei, 30)):
        rgb(p, v)
    mask(shared, 3)
    mask(em, 9)
    rows = [
        independent_train(tmp_path),
        row(ti, shared, "train", "train:a"),
        row(vi, shared, "val", "val:b"),
        row(ei, em, "test", "test:c"),
    ]
    rep = run(write_manifest(tmp_path / "m.jsonl", rows), tmp_path / "out", 16)
    assert rep["status"] == "PASS"
    assert rep["rows_removed"]["train"] == 1
    assert rep["rows_removed"]["val"] == 0
