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
    assert any(
        "source leakage" in error
        for error in audit(p, require_source_disjoint=True)
    )


def test_audit_rejects_native_resolution_mismatch_without_alignment_certificate(tmp_path):
    rows = []
    for idx, split in enumerate(("train", "val", "test")):
        image = tmp_path / f"i{idx}.png"
        mask = tmp_path / f"m{idx}.png"
        _rgb(image, size=(16, 16), value=40 + idx * 20)
        _mask(mask, size=(8, 8), point=(2 + idx, 2))
        rows.append(
            {
                "image": str(image),
                "mask": str(mask),
                "split": split,
                "source_id": f"s{idx}",
                "lineage_id": f"p{idx}",
                "is_normal": False,
            }
        )
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
        # Thick enough to survive 8x8 resize.
        im = Image.new("L", (16, 16), color=0)
        for y in range(6, 10):
            for x in range(6, 10):
                im.putpixel((x, y), 255)
        im = im.resize((8, 8), resample=Image.Resampling.NEAREST)
        im.save(mask)
        rows.append(
            {
                "image": str(image),
                "mask": str(mask),
                "split": split,
                "source_id": f"s{idx}",
                "lineage_id": f"p{idx}",
                "is_normal": False,
                "alignment_verified": True,
            }
        )
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
        # Corner pixel disappears when reduced to 1x1 with NEAREST.
        _mask(mask, size=(8, 8), point=(0, 0))
        rows.append(
            {
                "image": str(image),
                "mask": str(mask),
                "split": split,
                "source_id": f"s{idx}",
                "lineage_id": f"p{idx}",
                "is_normal": False,
            }
        )
    p = tmp_path / "m.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows))
    errors = audit(p, resize_size=1)
    assert any("becomes empty after resize" in e for e in errors)


def test_certified_n0_empty_passes_uncertified_empty_fails(tmp_path):
    # one certified N0 empty-target row, one uncertified empty-target row
    rows = []
    for i, cert in enumerate(("verified_no_crack", None)):
        img = tmp_path / f"i{i}.png"
        msk = tmp_path / f"m{i}.png"
        _rgb(img, size=(8, 8), value=60 + i)
        # empty mask: no crack pixel
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
    # certified N0 (row 0) must NOT be flagged; uncertified (row 1) must be
    assert "row 0:" not in msgs
    assert "row 1:" in msgs


def test_train_manifest_is_crack_only_and_counts(tmp_path):
    # structural sanity: manifest_clean_train has no is_normal rows, count matches
    import pathlib
    out = pathlib.Path("/hdd1/hieulc/Oasis_AOSK/experiments/local_hy3_validation_20260813_002205/data/cleaneval_v1/manifest_clean_train.jsonl")
    lines = [l for l in out.read_text().splitlines() if l.strip()]
    assert len(lines) == 19187
    normals = [json.loads(l) for l in lines if json.loads(l).get("is_normal")]
    assert normals == []

def test_certified_empty_across_splits_no_reuse_error(tmp_path):
    # two certified N0 empty masks in DIFFERENT splits: identical all-zero
    # content must NOT be reported as cross-split mask reuse (false positive)
    rows = []
    for i, sp in enumerate(("test", "val")):
        img = tmp_path / f"img{i}.png"
        msk = tmp_path / f"msk{i}.png"
        _rgb(img, size=(8, 8))
        Image.new("L", (8, 8), color=0).save(msk)
        rows.append({
            "image": str(img), "mask": str(msk), "split": sp,
            "source_id": "BCL", "lineage_id": f"b{i}", "is_normal": False,
            "empty_target_status": "verified_no_crack",
        })
    mfp = tmp_path / "m.jsonl"
    mfp.write_text("\n".join(json.dumps(r) for r in rows))
    errors = audit(mfp, resize_size=256)
    msgs = "\n".join(errors)
    assert "reused across splits" not in msgs
    assert "native-empty" not in msgs


def test_cross_split_certified_empties_not_mask_reuse(tmp_path):
    # two certified N0 empty masks split across val+test share identical all-zero
    # digest; must NOT be reported as cross-split mask reuse (all it[3] True).
    import shutil, numpy as np
    from PIL import Image
    im1 = tmp_path/"img1.png"; mk1 = tmp_path/"mask1.png"
    im2 = tmp_path/"img2.png"; mk2 = tmp_path/"mask2.png"
    for fn,color in ((im1,(200,20,20)),(im2,(30,30,200))):
        Image.new("RGB",(8,8),color).save(fn)
    black = np.zeros((8,8),dtype=np.uint8)
    for m in (mk1,mk2):
        Image.fromarray(black).save(m)

    import shutil
    from pathlib import Path
    tmp = tmp_path/"bench"; tmp.mkdir()
    def stage(img,mask,split,lineage,itemid):
        d=tmp/split; d.mkdir(exist_ok=True)
        ip=d/img.name; mp=d/mask.name
        shutil.copy(img,ip); shutil.copy(mask,mp)
        return {"item_id":itemid,"image":str(ip),"mask":str(mp),
                "split":split,"source_id":lineage,"lineage_id":lineage,
                "empty_target_status":"verified_no_crack","crack_type":"nan"}
    rows=[stage(im1,mk1,"val","KW_A","v1"), stage(im2,mk2,"test","KW_A","t1")]
    mfp=tmp/"manifest.jsonl"
    mfp.write_text("\n".join(json.dumps(r) for r in rows))
    from oasis_cycle_aosk.audit import audit
    errors = audit(mfp, resize_size=None)
    msgs = "\n".join(errors)
    assert "reused across splits" not in msgs, msgs
    assert "native-empty" not in msgs, msgs
