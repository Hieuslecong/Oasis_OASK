"""Immutable multi-checkpoint final-evaluation bundle contract for v2.1."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .checkpoint import sha256_file
from .protocol import dataset_content_sha256

REQUIRED_TOP = {
    "schema", "manifest", "manifest_sha256", "dataset_content_sha256",
    "full_gate0_certificate", "full_gate0_certificate_sha256",
    "method_spec", "method_spec_sha256", "protocol", "protocol_sha256",
    "evaluator", "evaluator_sha256", "metric_spec_sha256", "git_commit_sha",
    "entries"
}
REQUIRED_ENTRY = {"arm","seed","checkpoint","checkpoint_sha256","threshold"}


def canonical_bundle_id(bundle):
    payload={k:v for k,v in bundle.items() if k not in {"bundle_id","opened_utc","completed_utc"}}
    raw=json.dumps(payload,sort_keys=True,separators=(",",":"),ensure_ascii=False).encode()
    return hashlib.sha256(raw).hexdigest()


def validate_final_bundle(bundle_path, expected_arms=("S0","S1","S2","S3")):
    p=Path(bundle_path); b=json.loads(p.read_text())
    missing=sorted(REQUIRED_TOP-set(b));
    if missing: raise ValueError("bundle missing: "+", ".join(missing))
    if b["schema"] != "oasis-rc-v2.1-final-bundle-v1": raise ValueError("invalid bundle schema")
    checks=(("manifest","manifest_sha256"),("full_gate0_certificate","full_gate0_certificate_sha256"),("method_spec","method_spec_sha256"),("protocol","protocol_sha256"),("evaluator","evaluator_sha256"))
    for path_key,sha_key in checks:
        if sha256_file(b[path_key]) != b[sha_key]: raise ValueError(f"{path_key} SHA mismatch")
    if dataset_content_sha256(b["manifest"]) != b["dataset_content_sha256"]: raise ValueError("dataset content SHA mismatch")
    full=json.loads(Path(b["full_gate0_certificate"]).read_text())
    if full.get("status")!="PASS" or full.get("scope")!="full_benchmark": raise ValueError("full Gate0 must be PASS/full_benchmark")
    if full.get("manifest_sha256") != b["manifest_sha256"]: raise ValueError("full Gate0 certificate is not bound to final manifest")
    if full.get("dataset_content_sha256") != b["dataset_content_sha256"]: raise ValueError("full Gate0 certificate is not bound to final dataset bytes")
    entries=b["entries"]
    if not isinstance(entries,list) or not entries: raise ValueError("entries must be non-empty")
    seen=set(); by_seed={}
    for e in entries:
        miss=sorted(REQUIRED_ENTRY-set(e));
        if miss: raise ValueError("entry missing: "+", ".join(miss))
        key=(str(e["arm"]),int(e["seed"]));
        if key in seen: raise ValueError(f"duplicate arm/seed {key}")
        seen.add(key); by_seed.setdefault(int(e["seed"]),set()).add(str(e["arm"]))
        if sha256_file(e["checkpoint"]) != e["checkpoint_sha256"]: raise ValueError(f"checkpoint SHA mismatch {key}")
        t=float(e["threshold"])
        if not 0<t<1: raise ValueError(f"invalid threshold {key}")
    required=set(expected_arms)
    for seed,arms in by_seed.items():
        if arms != required: raise ValueError(f"seed {seed} incomplete arms: {sorted(arms)}")
    actual_id=canonical_bundle_id(b)
    if b.get("bundle_id") not in (None,actual_id): raise ValueError("bundle_id mismatch")
    return {**b,"bundle_id":actual_id,"seeds":sorted(by_seed)}
