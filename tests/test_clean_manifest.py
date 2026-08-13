import json
from pathlib import Path

import pytest

from scripts.clean_manifest import run, strip_lineage  # type: ignore


def _save_rgb(path, value=127, size=(32, 32)):
    from PIL import Image
    Image.new("RGB", size, color=(value, value, value)).save(path)


def _save_mask(path, size=(32, 32), block=True):
    # non-thin crack region so the resized-positive gate (>=1 fg pixel) passes
    from PIL import Image
    import numpy as np
    a = np.zeros(size, dtype=np.uint8)
    if block:
        a[4:10, 4:10] = 255
    Image.fromarray(a).save(path)


def _manifest(tmp_path, rows):
    p = tmp_path / "m.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return p


def _split_rows(tmp_path, train, val, test):
    """Wrap a list of (image,mask,lineage,is_normal) triples into 3 split rows."""
    out = []
    for i, (img, mask, lin, nrm) in enumerate(train):
        out.append({"image": str(img), "mask": str(mask) if mask else None,
                    "split": "train", "source_id": "s", "lineage_id": lin, "is_normal": nrm})
    for i, (img, mask, lin, nrm) in enumerate(val):
        out.append({"image": str(img), "mask": str(mask) if mask else None,
                    "split": "val", "source_id": "s", "lineage_id": lin, "is_normal": nrm})
    for i, (img, mask, lin, nrm) in enumerate(test):
        out.append({"image": str(img), "mask": str(mask) if mask else None,
                    "split": "test", "source_id": "s", "lineage_id": lin, "is_normal": nrm})
    return out


def test_strip_lineage():
    assert strip_lineage("train:BCL_c1.png") == "BCL_c1.png"
    assert strip_lineage("val:foo::bar") == "foo::bar"
    assert strip_lineage("BCL_c1.png") == "BCL_c1.png"


def test_fix_lineage_format_only(tmp_path):
    img = tmp_path / "a.png"; _save_rgb(img, value=40)
    mk = tmp_path / "a_mask.png"; _save_mask(mk)
    val_i = tmp_path / "v.png"; _save_rgb(val_i, value=99)
    val_m = tmp_path / "vm.png"; _save_mask(val_m)
    test_i = tmp_path / "t.png"; _save_rgb(test_i, value=150)
    test_m = tmp_path / "tm.png"; _save_mask(test_m)
    rows = _split_rows(
        tmp_path,
        [(img, mk, "train:foo.png", False)],
        [(val_i, val_m, "val:bar.png", False)],
        [(test_i, test_m, "test:baz.png", False)],
    )
    rep = run(_manifest(tmp_path, rows), tmp_path / "out", resize_size=16)
    assert rep["status"] == "PASS"
    assert rep["decisions_summary"]["FIX_LINEAGE_FORMAT"] == 3
    assert rep["rows_removed"]["train"] == 0


def test_drop_train_exact_rgb_duplicate_keeps_test(tmp_path):
    img_t = tmp_path / "t.png"; _save_rgb(img_t, value=50)
    img_e = tmp_path / "e.png"; _save_rgb(img_e, value=50)  # identical pixels
    mask_t = tmp_path / "mt.png"; _save_mask(mask_t)
    mask_e = tmp_path / "me.png"; _save_mask(mask_e)
    val_i = tmp_path / "v.png"; _save_rgb(val_i, value=99)
    val_m = tmp_path / "vm.png"; _save_mask(val_m)
    # a second, distinct train row so the train split is never vacated after the drop
    tr2_i = tmp_path / "tr2.png"; _save_rgb(tr2_i, value=12)
    tr2_m = tmp_path / "tr2m.png"; _save_mask(tr2_m)
    rows = _split_rows(
        tmp_path,
        [(img_t, mask_t, "train:same.png", False), (tr2_i, tr2_m, "train:keep.png", False)],
        [(val_i, val_m, "val:other.png", False)],
        [(img_e, mask_e, "test:same.png", False)],
    )
    rep = run(_manifest(tmp_path, rows), tmp_path / "out", resize_size=16)
    assert rep["status"] == "PASS"
    assert rep["rows_removed"]["train"] == 1
    assert rep["rows_removed"]["test"] == 0
    dec = (tmp_path / "out" / "manifest_clean_decisions.csv").read_text().splitlines()[1:]
    assert any("DROP_TRAIN_EXACT_RGB_DUPLICATE" in line or "DROP_TRAIN_EXACT_PAIR_DUPLICATE" in line for line in dec)


def test_val_test_exact_rgb_blocks(tmp_path):
    img_v = tmp_path / "v.png"; _save_rgb(img_v, value=80)
    img_e = tmp_path / "e.png"; _save_rgb(img_e, value=80)
    mask_v = tmp_path / "mv.png"; _save_mask(mask_v)
    mask_e = tmp_path / "me.png"; _save_mask(mask_e)
    tr_i = tmp_path / "tr.png"; _save_rgb(tr_i, value=12)
    tr_m = tmp_path / "trm.png"; _save_mask(tr_m)
    tr2_i = tmp_path / "tr2.png"; _save_rgb(tr2_i, value=33)
    tr2_m = tmp_path / "tr2m.png"; _save_mask(tr2_m)
    rows = _split_rows(
        tmp_path,
        [(tr_i, tr_m, "train:other.png", False), (tr2_i, tr2_m, "train:other2.png", False)],
        [(img_v, mask_v, "val:same.png", False)],
        [(img_e, mask_e, "test:same.png", False)],
    )
    rep = run(_manifest(tmp_path, rows), tmp_path / "out", resize_size=16)
    assert rep["status"] == "BLOCKED"
    assert rep["block_val_test_groups"]


def test_class_c_mask_only_different_rgb_kept(tmp_path):
    img_a = tmp_path / "a.png"; _save_rgb(img_a, value=10)
    img_b = tmp_path / "b.png"; _save_rgb(img_b, value=200)  # different RGB
    mask = tmp_path / "m.png"; _save_mask(mask)  # reused mask
    test_i = tmp_path / "ti.png"; _save_rgb(test_i)
    test_m = tmp_path / "tm.png"; _save_mask(test_m)
    rows = _split_rows(
        tmp_path,
        [(img_a, mask, "train:a.png", False)],
        [(img_b, mask, "val:b.png", False)],
        [(test_i, test_m, "test:t.png", False)],
    )
    rep = run(_manifest(tmp_path, rows), tmp_path / "out", resize_size=16)
    # class C is kept (not leakage): reported as repeated_label_geometry, no drop
    assert rep["repeated_label_geometry_count"] >= 1
    assert rep["rows_removed"]["train"] == 0
    assert rep["rows_removed"]["val"] == 0
    # pass gate (spatial/alignment/rgb all fine; mask reuse is tolerated per B+)
    assert rep["status"] == "PASS"
