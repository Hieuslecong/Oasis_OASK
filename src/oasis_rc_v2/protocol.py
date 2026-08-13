import hashlib
import json
from pathlib import Path

from .checkpoint import sha256_file


def _manifest_rows(manifest):
    return [
        json.loads(line)
        for line in Path(manifest).read_text().splitlines()
        if line.strip()
    ]


def dataset_content_sha256(manifest):
    """Hash the exact image/mask bytes referenced by a manifest.

    The manifest SHA binds metadata/order/paths. This second digest binds file bytes,
    so a Gate-0 certificate becomes invalid if an image or mask is modified in place.
    It intentionally reads only rows present in the supplied manifest; a training-view
    verification therefore never opens canonical test files.
    """
    h = hashlib.sha256()
    for index, row in enumerate(_manifest_rows(manifest)):
        image = row.get("image")
        if not image or not Path(image).is_file():
            raise ValueError(f"dataset-content hash cannot read image at row {index}: {image!r}")
        image_sha = sha256_file(image)
        mask = row.get("mask")
        if row.get("is_normal") is True:
            if mask not in (None, ""):
                raise ValueError(f"true-normal row {index} must use mask=null")
            mask_sha = "VIRTUAL_ZERO_MASK"
        else:
            if not mask or not Path(mask).is_file():
                raise ValueError(f"dataset-content hash cannot read mask at row {index}: {mask!r}")
            mask_sha = sha256_file(mask)
        record = {
            "row": index,
            "image_sha256": image_sha,
            "mask_sha256": mask_sha,
        }
        h.update(json.dumps(record, sort_keys=True, separators=(",", ":")).encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def verify_gate0_certificate(certificate_path, training_manifest, image_size, normal_policy):
    if not certificate_path:
        raise ValueError("official training requires --gate0-certificate")
    p = Path(certificate_path)
    cert = json.loads(p.read_text())
    if cert.get("status") != "PASS":
        raise ValueError("Gate 0 certificate status is not PASS")
    if cert.get("scope") != "training_view":
        raise ValueError("trainer requires a training_view Gate 0 certificate")
    if cert.get("manifest_sha256") != sha256_file(training_manifest):
        raise ValueError("Gate 0 certificate manifest SHA256 mismatch")
    actual_data_sha = dataset_content_sha256(training_manifest)
    if cert.get("dataset_content_sha256") != actual_data_sha:
        raise ValueError("Gate 0 certificate dataset-content SHA256 mismatch")
    if int(cert.get("resize_size", -1)) != int(image_size):
        raise ValueError("Gate 0 certificate resize_size mismatch")
    if cert.get("normal_policy") != normal_policy:
        raise ValueError("Gate 0 certificate normal_policy mismatch")
    return cert


def verify_final_test_authorization(
    authorization_path,
    checkpoint,
    manifest,
    threshold,
):
    """Verify an authorization marker created *before* canonical test is opened."""
    if not authorization_path:
        raise ValueError("canonical test requires --final-test-authorization")
    auth = json.loads(Path(authorization_path).read_text())
    if auth.get("state") not in {"OPENED", "IN_PROGRESS"}:
        raise ValueError("final-test authorization is not in an opened state")
    if auth.get("checkpoint_sha256") != sha256_file(checkpoint):
        raise ValueError("final-test authorization checkpoint SHA256 mismatch")
    if auth.get("manifest_sha256") != sha256_file(manifest):
        raise ValueError("final-test authorization manifest SHA256 mismatch")
    if auth.get("dataset_content_sha256") != dataset_content_sha256(manifest):
        raise ValueError("final-test authorization dataset-content SHA256 mismatch")
    if abs(float(auth.get("threshold")) - float(threshold)) > 1e-12:
        raise ValueError("final-test authorization threshold mismatch")
    return auth
