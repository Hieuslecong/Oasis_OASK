import hashlib
from pathlib import Path

EXPERIMENT_ID = "oasis-rc-v2-relational-hard-negative"
CHECKPOINT_SCHEMA = 2
IMPLEMENTATION_VERSION = "2.0.3"
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
        raise ValueError(
            f"{kind} checkpoint_schema must be {CHECKPOINT_SCHEMA}; legacy checkpoint rejected"
        )
    if saved.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError(f"{kind} experiment_id mismatch; legacy/foreign checkpoint rejected")
    if saved.get("method_version") != METHOD_VERSION:
        raise ValueError(f"{kind} method_version mismatch")
    if saved.get("implementation_version") != IMPLEMENTATION_VERSION:
        raise ValueError(
            f"{kind} implementation_version mismatch: "
            f"{saved.get('implementation_version')!r} != {IMPLEMENTATION_VERSION!r}"
        )


def _same_number(a, b, tol=1e-12):
    return abs(float(a) - float(b)) <= tol


def validate_critic_checkpoint(
    saved,
    manifest,
    cfg,
    normal_fraction,
    normal_critic_weight,
    dataset_content_sha256_value=None,
    expected_hparams=None,
):
    """Fail closed when a critic was trained under a different data/run contract."""
    _require_identity(saved, "critic")
    required = (
        "critic",
        "config",
        "manifest_file_sha256",
        "dataset_content_sha256",
        "normal_fraction",
        "normal_critic_weight",
        "training_hparams",
        "width",
    )
    missing = [k for k in required if k not in saved]
    if missing:
        raise ValueError("critic checkpoint missing: " + ", ".join(missing))
    if saved["manifest_file_sha256"] != sha256_file(manifest):
        raise ValueError("critic checkpoint manifest SHA256 does not match current training manifest")
    if (
        dataset_content_sha256_value is not None
        and saved["dataset_content_sha256"] != dataset_content_sha256_value
    ):
        raise ValueError("critic checkpoint dataset-content SHA256 does not match current data")

    saved_cfg = saved.get("config", {})
    if int(saved_cfg.get("image_size", -1)) != int(cfg["image_size"]):
        raise ValueError("critic checkpoint image_size does not match current run")
    if int(saved_cfg.get("seed", -1)) != int(cfg["seed"]):
        raise ValueError("critic checkpoint seed does not match current run")
    if not _same_number(saved["normal_fraction"], normal_fraction):
        raise ValueError("critic checkpoint normal_fraction does not match current run")
    if not _same_number(saved["normal_critic_weight"], normal_critic_weight):
        raise ValueError("critic checkpoint normal_critic_weight does not match current run")

    if expected_hparams:
        actual = saved.get("training_hparams", {})
        for key, expected in expected_hparams.items():
            if key not in actual:
                raise ValueError(f"critic checkpoint training_hparams missing {key}")
            got = actual[key]
            if isinstance(expected, (float, int)) and isinstance(got, (float, int)):
                if not _same_number(got, expected):
                    raise ValueError(f"critic checkpoint {key} mismatch")
            elif got != expected:
                raise ValueError(f"critic checkpoint {key} mismatch")


def validate_student_checkpoint(saved):
    _require_identity(saved, "student")
    forbidden = {"critic", "aosk", "generator", "discriminator"}.intersection(saved)
    if forbidden:
        raise ValueError(f"deployment checkpoint contains training-only state: {sorted(forbidden)}")
    required = ("student", "manifest_file_sha256", "dataset_content_sha256")
    missing = [k for k in required if k not in saved]
    if missing:
        raise ValueError("student checkpoint missing: " + ", ".join(missing))
