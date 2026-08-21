import hashlib
import json
from pathlib import Path

import torch

from .checkpoint import sha256_file

PLACEHOLDER_IDS={"","unknown","none","null","n/a","na","placeholder","unset","missing"}


def _manifest_rows(manifest):
    return [json.loads(line) for line in Path(manifest).read_text().splitlines() if line.strip()]


def _validate_manifest_identity(rows):
    for index,row in enumerate(rows):
        for key in ("source_id","lineage_id"):
            value=row.get(key)
            if value is None or str(value).strip().lower() in PLACEHOLDER_IDS:
                raise ValueError(f"row {index}: invalid {key}={value!r}")


def dataset_content_sha256(manifest):
    """Hash exact image/mask bytes referenced by a manifest."""
    h=hashlib.sha256(); rows=_manifest_rows(manifest); _validate_manifest_identity(rows)
    for index,row in enumerate(rows):
        image=row.get("image")
        if not image or not Path(image).is_file(): raise ValueError(f"dataset-content hash cannot read image at row {index}: {image!r}")
        image_sha=sha256_file(image); mask=row.get("mask")
        if row.get("is_normal") is True:
            if mask not in (None,""): raise ValueError(f"true-normal row {index} must use mask=null")
            mask_sha="VIRTUAL_ZERO_MASK"
        else:
            if not mask or not Path(mask).is_file(): raise ValueError(f"dataset-content hash cannot read mask at row {index}: {mask!r}")
            mask_sha=sha256_file(mask)
        record={"row":index,"image_sha256":image_sha,"mask_sha256":mask_sha}
        h.update(json.dumps(record,sort_keys=True,separators=(",",":")).encode("utf-8")); h.update(b"\n")
    return h.hexdigest()


def _load_inventory(cert):
    path=cert.get("dataset_inventory")
    if not path or not Path(path).is_file(): raise ValueError("Gate0 certificate missing readable dataset_inventory")
    if cert.get("dataset_inventory_sha256") != sha256_file(path): raise ValueError("Gate0 inventory SHA256 mismatch")
    return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]


def _inventory_identity(row):
    return (
        row.get("image_sha256"), row.get("mask_sha256"),
        str(row.get("source_id")), str(row.get("lineage_id")), bool(row.get("is_normal")),
    )


def _prove_training_subset(training_cert, full_cert):
    training=_load_inventory(training_cert); full=_load_inventory(full_cert)
    full_ids={_inventory_identity(r) for r in full}
    missing=[r for r in training if _inventory_identity(r) not in full_ids]
    if missing:
        sample=missing[0]
        raise ValueError("training-view inventory is not a subset of full benchmark; first missing identity="+repr(_inventory_identity(sample)))


def verify_gate0_certificate(certificate_path,training_manifest,image_size,normal_policy,full_certificate_path=None):
    if not certificate_path: raise ValueError("official training requires --gate0-certificate")
    p=Path(certificate_path); cert=json.loads(p.read_text())
    if cert.get("status")!="PASS": raise ValueError("Gate 0 certificate status is not PASS")
    if cert.get("scope")!="training_view": raise ValueError("trainer requires a training_view Gate 0 certificate")
    if cert.get("manifest_sha256")!=sha256_file(training_manifest): raise ValueError("Gate 0 certificate manifest SHA256 mismatch")
    actual_data_sha=dataset_content_sha256(training_manifest)
    if cert.get("dataset_content_sha256")!=actual_data_sha: raise ValueError("Gate 0 certificate dataset-content SHA256 mismatch")
    if int(cert.get("resize_size",-1))!=int(image_size): raise ValueError("Gate 0 certificate resize_size mismatch")
    if cert.get("normal_policy")!=normal_policy: raise ValueError("Gate 0 certificate normal_policy mismatch")
    if not full_certificate_path: raise ValueError("training-view verification requires full Gate 0 certificate")
    full_path=Path(full_certificate_path); full=json.loads(full_path.read_text())
    if full.get("status")!="PASS" or full.get("scope")!="full_benchmark": raise ValueError("full Gate 0 certificate must be PASS/full_benchmark")
    if cert.get("parent_full_gate0_certificate_sha256")!=sha256_file(full_path): raise ValueError("training-view certificate parent full Gate 0 mismatch")
    _prove_training_subset(cert,full)
    return cert


def verify_final_test_authorization(authorization_path,checkpoint,manifest,threshold):
    """Legacy single-checkpoint authorization retained for reconstructed v2.0.4 only.

    OASIS-RC-v2.1 canonical evaluation must use ``run_final_bundle_v21.py``.
    """
    if not authorization_path: raise ValueError("canonical test requires --final-test-authorization")
    auth=json.loads(Path(authorization_path).read_text())
    if auth.get("state") not in {"OPENED","IN_PROGRESS"}: raise ValueError("final-test authorization is not in an opened state")
    if auth.get("checkpoint_sha256")!=sha256_file(checkpoint): raise ValueError("final-test authorization checkpoint SHA256 mismatch")
    if auth.get("manifest_sha256")!=sha256_file(manifest): raise ValueError("final-test authorization manifest SHA256 mismatch")
    if auth.get("dataset_content_sha256")!=dataset_content_sha256(manifest): raise ValueError("final-test authorization dataset-content SHA256 mismatch")
    checkpoint_data=torch.load(checkpoint,map_location="cpu",weights_only=False)
    if checkpoint_data.get("method_version")=="OASIS-RC-v2.1": raise ValueError("v2.1 final test requires immutable multi-checkpoint bundle runner")
    if auth.get("training_view_dataset_sha256")!=checkpoint_data.get("training_view_dataset_sha256"): raise ValueError("final-test authorization training-view provenance mismatch")
    if auth.get("full_gate0_certificate_sha256")!=checkpoint_data.get("full_gate0_certificate_sha256"): raise ValueError("final-test authorization full-benchmark provenance mismatch")
    if abs(float(auth.get("threshold"))-float(threshold))>1e-12: raise ValueError("final-test authorization threshold mismatch")
    return auth
