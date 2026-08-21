import hashlib
import json
from pathlib import Path

import pytest
from PIL import Image

from oasis_rc_v2.checkpoint import sha256_file
from oasis_rc_v2.protocol import dataset_content_sha256, verify_gate0_certificate
from oasis_rc_v2.final_bundle import validate_final_bundle


def _img(path, value):
    Image.new("RGB",(8,8),(value,value,value)).save(path)


def _mask(path, on=True):
    im=Image.new("L",(8,8),0)
    if on:
        for y in range(2,6): im.putpixel((4,y),255)
    im.save(path)


def _write_manifest(path, rows):
    path.write_text("".join(json.dumps(r)+"\n" for r in rows))


def _inventory(path, rows):
    inv=[]
    for i,r in enumerate(rows):
        inv.append({"row":i,"split":r["split"],"source_id":r["source_id"],"lineage_id":r["lineage_id"],"is_normal":r["is_normal"],"image":str(Path(r["image"]).resolve()),"image_sha256":sha256_file(r["image"]),"mask":str(Path(r["mask"]).resolve()),"mask_sha256":sha256_file(r["mask"])})
    path.write_text("".join(json.dumps(x)+"\n" for x in inv)); return inv


def _cert(path, scope, manifest, inventory, normal_policy="none", parent=None):
    d={"status":"PASS","scope":scope,"manifest":str(manifest),"manifest_sha256":sha256_file(manifest),"dataset_content_sha256":dataset_content_sha256(manifest),"dataset_inventory":str(inventory),"dataset_inventory_sha256":sha256_file(inventory),"resize_size":8,"normal_policy":normal_policy,"gate0_schema":2,"parent_full_gate0_certificate_sha256":sha256_file(parent) if parent else None}
    path.write_text(json.dumps(d)); return d


def test_training_view_must_be_subset_of_parent_full_inventory(tmp_path):
    i1=tmp_path/"a.png"; m1=tmp_path/"a_m.png"; i2=tmp_path/"b.png"; m2=tmp_path/"b_m.png"; ix=tmp_path/"x.png"; mx=tmp_path/"x_m.png"
    _img(i1,30);_mask(m1);_img(i2,80);_mask(m2);_img(ix,140);_mask(mx)
    full_rows=[{"image":str(i1),"mask":str(m1),"split":"train","source_id":"s","lineage_id":"a","is_normal":False},{"image":str(i2),"mask":str(m2),"split":"val","source_id":"s","lineage_id":"b","is_normal":False}]
    full=tmp_path/"full.jsonl"; _write_manifest(full,full_rows); full_inv=tmp_path/"full.inv"; _inventory(full_inv,full_rows); full_cert=tmp_path/"full.cert"; _cert(full_cert,"full_benchmark",full,full_inv)
    bad_rows=[{"image":str(ix),"mask":str(mx),"split":"train","source_id":"s","lineage_id":"x","is_normal":False}]
    bad=tmp_path/"bad.jsonl"; _write_manifest(bad,bad_rows); bad_inv=tmp_path/"bad.inv"; _inventory(bad_inv,bad_rows); bad_cert=tmp_path/"bad.cert"; _cert(bad_cert,"training_view",bad,bad_inv,parent=full_cert)
    with pytest.raises(ValueError,match="not a subset"):
        verify_gate0_certificate(bad_cert,bad,8,"none",full_cert)


def test_placeholder_lineage_is_rejected(tmp_path):
    i=tmp_path/"a.png"; m=tmp_path/"m.png"; _img(i,10); _mask(m)
    manifest=tmp_path/"m.jsonl"; _write_manifest(manifest,[{"image":str(i),"mask":str(m),"split":"train","source_id":"src","lineage_id":"unknown","is_normal":False}])
    with pytest.raises(ValueError,match="invalid lineage_id"):
        dataset_content_sha256(manifest)


def test_final_bundle_requires_complete_arms_and_bound_full_certificate(tmp_path):
    i=tmp_path/"a.png"; m=tmp_path/"m.png"; _img(i,20); _mask(m)
    rows=[{"image":str(i),"mask":str(m),"split":"test","source_id":"src","lineage_id":"l1","is_normal":False}]
    manifest=tmp_path/"test.jsonl"; _write_manifest(manifest,rows); inv=tmp_path/"inv"; _inventory(inv,rows); full_cert=tmp_path/"full.cert"; _cert(full_cert,"full_benchmark",manifest,inv)
    spec=tmp_path/"spec"; protocol=tmp_path/"protocol"; evaluator=tmp_path/"eval"; spec.write_text("spec"); protocol.write_text("protocol"); evaluator.write_text("eval")
    ckpts=[]
    for arm in ("S0","S1","S2","S3"):
        p=tmp_path/f"{arm}.pt"; p.write_bytes(arm.encode()); ckpts.append((arm,p))
    bundle={"schema":"oasis-rc-v2.1-final-bundle-v1","manifest":str(manifest),"manifest_sha256":sha256_file(manifest),"dataset_content_sha256":dataset_content_sha256(manifest),"full_gate0_certificate":str(full_cert),"full_gate0_certificate_sha256":sha256_file(full_cert),"method_spec":str(spec),"method_spec_sha256":sha256_file(spec),"protocol":str(protocol),"protocol_sha256":sha256_file(protocol),"evaluator":str(evaluator),"evaluator_sha256":sha256_file(evaluator),"metric_spec_sha256":"1"*64,"git_commit_sha":"2"*40,"entries":[{"arm":arm,"seed":2027,"checkpoint":str(p),"checkpoint_sha256":sha256_file(p),"threshold":.5} for arm,p in ckpts]}
    bp=tmp_path/"bundle.json"; bp.write_text(json.dumps(bundle)); out=validate_final_bundle(bp); assert out["seeds"]==[2027]
    bad=dict(bundle); bad["entries"]=bundle["entries"][:-1]; bp.write_text(json.dumps(bad))
    with pytest.raises(ValueError,match="incomplete arms"):
        validate_final_bundle(bp)
