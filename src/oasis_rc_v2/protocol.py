import json
from pathlib import Path
from .checkpoint import sha256_file


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
    if int(cert.get("resize_size", -1)) != int(image_size):
        raise ValueError("Gate 0 certificate resize_size mismatch")
    if cert.get("normal_policy") != normal_policy:
        raise ValueError("Gate 0 certificate normal_policy mismatch")
    return cert
