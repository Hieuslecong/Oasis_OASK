import json

from PIL import Image

from oasis_cycle_aosk.audit import audit


def _rgb(path, size=(8, 8), value=80):
    Image.new("RGB", size, color=(value, value, value)).save(path)


def _mask(path, size=(8, 8), point=(3, 3)):
    im = Image.new("L", size, color=0)
    px = im.load()
    px[point] = 255
    im.save(path)


def test_audit_requires_metadata(tmp_path):
    p = tmp_path / "manifest.jsonl"
    p.write_text('{"image":"x"}\n')
    assert audit(p)


def test_audit_can_block_source_leakage(tmp_path):
    for i, name in enumerate(("x.png", "y.png", "z.png")):
        _rgb(tmp_path / name, value=20 + i * 30)
    rows = [
        {"image": str(tmp_path / "x.png"), "mask": None, "split": "train", "source_id": "s", "lineage_id": "a", "is_normal": True},
        {"image": str(tmp_path / "y.png"), "mask": None, "split": "val", "source_id": "s", "lineage_id": "b", "is_normal": True},
        {"image": str(tmp_path / "z.png"), "mask": None, "split": "test", "source_id": "t", "lineage_id": "c", "is_normal": True},
    ]
    p = tmp_path / "manifest.jsonl"
    p.write_text("\n".join(json.dumps(row) for row in rows))
    assert any("source leakage" in error for error in audit(p, require_source_disjoint=True))


def test_audit_rejects_native_resolution_mismatch_without_alignment_certificate(tmp_path):
    rows = []
    for idx, split in enumerate(("train", "val", "test")):
        image = tmp_path / f"i{idx}.png"
        mask = tmp_path / f"m{idx}.png"
        _rgb(image, size=(16, 16), value=40 + idx * 20)
        _mask(mask, size=(8, 8), point=(2 + idx, 2))
        rows.append({
            "image": str(image), "mask": str(mask), "split": split,
            "source_id": f"s{idx}", "lineage_id": f"p{idx}", "is_normal": False,
        })
    p = tmp_path / "m.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows))
    errors = audit(p, resize_size=8)
    assert any("requires alignment_verified=true" in e for e in errors)


def test_audit_accepts_verified_resolution_mismatch_when_crack_survives_resize(tmp_path):
    rows = []
    for idx, split in enumerate(("train", "val", "test")):
        image = tmp_path / f"i{idx}.png"
        mask = tmp_path / f"m{idx}.png"
        _rgb(image, size=(16, 16), value=40 + idx * 20)
        im = Image.new("L", (16, 16), color=0)
        for y in range(6, 10):
            for x in range(6, 10):
                im.putpixel((x, y), 255)
        im = im.resize((8, 8), resample=Image.Resampling.NEAREST)
        im.save(mask)
        rows.append({
            "image": str(image), "mask": str(mask), "split": split,
            "source_id": f"s{idx}", "lineage_id": f"p{idx}", "is_normal": False,
            "alignment_verified": True,
        })
    p = tmp_path / "m.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows))
    errors = audit(p, resize_size=8)
    assert not any("native-resolution mismatch" in e for e in errors)


def test_audit_detects_crack_that_disappears_after_resize(tmp_path):
    rows = []
    for idx, split in enumerate(("train", "val", "test")):
        image = tmp_path / f"i{idx}.png"
        mask = tmp_path / f"m{idx}.png"
        _rgb(image, size=(8, 8), value=30 + idx * 20)
        _mask(mask, size=(8, 8), point=(0, 0))
        rows.append({
            "image": str(image), "mask": str(mask), "split": split,
            "source_id": f"s{idx}", "lineage_id": f"p{idx}", "is_normal": False,
        })
    p = tmp_path / "m.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows))
    errors = audit(p, resize_size=1)
    assert any("becomes empty after resize" in e for e in errors)


def test_certified_n0_empty_passes_uncertified_empty_fails(tmp_path):
    rows = []
    for i, cert in enumerate(("verified_no_crack", None)):
        img = tmp_path / f"i{i}.png"
        msk = tmp_path / f"m{i}.png"
        _rgb(img, size=(8, 8), value=60 + i)
        Image.new("L", (8, 8), color=0).save(msk)
        r = {
            "image": str(img), "mask": str(msk), "split": "test",
            "source_id": "S", "lineage_id": f"q{i}", "is_normal": False,
        }
        if cert:
            r["empty_target_status"] = cert
        rows.append(r)
    mfp = tmp_path / "m.jsonl"
    mfp.write_text("\n".join(json.dumps(r) for r in rows))
    errors = audit(mfp, resize_size=256)
    msgs = "\n".join(errors)
    assert "row 0:" not in msgs
    assert "row 1:" in msgs


def test_certified_empty_across_splits_no_reuse_error(tmp_path):
    rows = []
    for i, sp in enumerate(("train", "val", "test")):
        img = tmp_path / f"img{i}.png"
        msk = tmp_path / f"msk{i}.png"
        _rgb(img, size=(8, 8), value=20 + 40 * i)
        Image.new("L", (8, 8), color=0).save(msk)
        rows.append({
            "image": str(img), "mask": str(msk), "split": sp,
            "source_id": f"BCL{i}", "lineage_id": f"b{i}", "is_normal": False,
            "empty_target_status": "verified_no_crack",
        })
    mfp = tmp_path / "m.jsonl"
    mfp.write_text("\n".join(json.dumps(r) for r in rows))
    errors = audit(mfp, resize_size=256)
    assert errors == []


def test_audit_detects_perceptual_near_duplicate_across_splits(tmp_path):
    rows = []
    for index, (split, value) in enumerate((("train", 80), ("val", 82))):
        image = tmp_path / f"near_{index}.png"
        _rgb(image, size=(16, 16), value=value)
        rows.append({
            "image": str(image), "mask": None, "split": split,
            "source_id": f"s{index}", "lineage_id": f"l{index}", "is_normal": True,
        })
    manifest = tmp_path / "near.jsonl"
    manifest.write_text("\n".join(json.dumps(row) for row in rows))
    errors = audit(manifest, required_splits=("train", "val"))
    assert any("perceptual-rgb near-duplicate" in error for error in errors)
