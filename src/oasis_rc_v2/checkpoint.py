import hashlib
from pathlib import Path

EXPERIMENT_ID = "oasis-rc-v2-relational-hard-negative"
CHECKPOINT_SCHEMA = 2
IMPLEMENTATION_VERSION = "2.0.0"
METHOD_VERSION = "OASIS-RC-v2"


def sha256_file(path):
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        return None
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _require_identity(saved, kind):
    if int(saved.get("checkpoint_schema", -1)) != CHECKPOINT_SCHEMA:
        raise ValueError(f"{kind} checkpoint_schema must be {CHECKPOINT_SCHEMA}; legacy checkpoint rejected")
    if saved.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError(f"{kind} experiment_id mismatch; legacy/foreign checkpoint rejected")
    if saved.get("method_version") != METHOD_VERSION:
        raise ValueError(f"{kind} method_version mismatch")
    if saved.get("implementation_version") != IMPLEMENTATION_VERSION:
        raise ValueError(
            f"{kind} implementation_version mismatch: "
            f"{saved.get('implementation_version')!r} != {IMPLEMENTATION_VERSION!r}"
        )


def validate_critic_checkpoint(saved, manifest, cfg, normal_fraction, normal_critic_weight):
    _require_identity(saved, "critic")
    required = (
        "critic", "config", "manifest_file_sha256", "normal_fraction",
        "normal_critic_weight", "width",
    )
    missing = [k for k in required if k not in saved]
    if missing:
        raise ValueError("critic checkpoint missing: " + ", ".join(missing))
    if saved["manifest_file_sha256"] != sha256_file(manifest):
        raise ValueError("critic checkpoint manifest SHA256 does not match current training manifest")
    saved_cfg = saved.get("config", {})
    if int(saved_cfg.get("image_size", -1)) != int(cfg["image_size"]):
        raise ValueError("critic checkpoint image_size does not match current run")
    if int(saved_cfg.get("seed", -1)) != int(cfg["seed"]):
        raise ValueError("critic checkpoint seed does not match current run")
    if abs(float(saved["normal_fraction"]) - float(normal_fraction)) > 1e-12:
        raise ValueError("critic checkpoint normal_fraction does not match current run")
    if abs(float(saved["normal_critic_weight"]) - float(normal_critic_weight)) > 1e-12:
        raise ValueError("critic checkpoint normal_critic_weight does not match current run")


def validate_student_checkpoint(saved):
    _require_identity(saved, "student")
    forbidden = {"critic", "aosk", "generator", "discriminator"}.intersection(saved)
    if forbidden:
        raise ValueError(f"deployment checkpoint contains training-only state: {sorted(forbidden)}")
    if "student" not in saved:
        raise ValueError("checkpoint does not contain student state")
