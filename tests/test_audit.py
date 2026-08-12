import json
from PIL import Image
from oasis_cycle_aosk.audit import audit

def test_audit_requires_normals_and_metadata(tmp_path):
    p = tmp_path / "manifest.jsonl"; p.write_text('{"image":"x"}\n')
    assert audit(p)

def test_audit_can_block_source_leakage(tmp_path):
    for name in ("x.png", "y.png", "z.png"):
        Image.new("RGB", (4, 4)).save(tmp_path / name)
    rows = [
        {"image": str(tmp_path / "x.png"), "split": "train", "source_id": "s", "lineage_id": "a", "is_normal": True},
        {"image": str(tmp_path / "y.png"), "split": "val", "source_id": "s", "lineage_id": "b", "is_normal": True},
        {"image": str(tmp_path / "z.png"), "split": "test", "source_id": "t", "lineage_id": "c", "is_normal": True},
    ]
    p = tmp_path / "manifest.jsonl"
    p.write_text("\n".join(json.dumps(row) for row in rows))
    assert any("source leakage" in error for error in audit(p, require_source_disjoint=True))
