import hashlib
from pathlib import Path

EXPERIMENT_ID = "oasis-rc-v2.1-gt-anchored-relational-energy-head"
CHECKPOINT_SCHEMA = 5
IMPLEMENTATION_VERSION = "2.1.0-dev2"
METHOD_VERSION = "OASIS-RC-v2.1"
ENERGY_HEAD_CONTRACT = "dedicated-scalar-lower-is-better-v1"
CRITIC_COMPATIBILITY_KEYS = (
    "energy_head_contract",
    "mask_variant_contract",
    "rgb_shuffle_pair_only",
    "mask_flip_training",
    "method_spec",
)


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


def critic_compatibility_contract(saved):
    """Return the scientific compatibility subset recorded by a qualified critic.

    Optimizer history (lr, epochs, Adam settings, etc.) is provenance, not a
    requirement on a later student run. This helper deliberately exposes only
    the contract that a v2.1 critic consumer must opt in to.
    """
    actual = saved.get("training_hparams", {})
    missing = [key for key in CRITIC_COMPATIBILITY_KEYS if key not in actual]
    if missing:
        raise ValueError("critic checkpoint compatibility contract missing: " + ", ".join(missing))
    return {key: actual[key] for key in CRITIC_COMPATIBILITY_KEYS}


def validate_critic_checkpoint(
    saved,
    manifest,
    cfg,
    normal_fraction,
    normal_critic_weight,
    dataset_content_sha256_value=None,
    expected_hparams=None,
    full_gate0_certificate=None,
):
    """Validate a critic for use by a v2.1 consumer."""
    _require_identity(saved, "critic")
    required = (
        "critic", "config", "manifest_file_sha256", "dataset_content_sha256",
        "normal_fraction", "normal_critic_weight", "training_hparams", "width",
        "seed", "full_gate0_certificate_sha256", "energy_head_contract",
        "qualification_v21",
    )
    missing = [k for k in required if k not in saved]
    if missing:
        raise ValueError("critic checkpoint missing: " + ", ".join(missing))
    if saved["energy_head_contract"] != ENERGY_HEAD_CONTRACT:
        raise ValueError("critic energy-head contract mismatch")
    qualification = saved.get("qualification_v21")
    if not isinstance(qualification, dict) or qualification.get("pass") is not True:
        raise ValueError("critic checkpoint is not v2.1-qualified")
    if qualification.get("failures", []):
        raise ValueError("critic checkpoint qualification contains failures")
    if saved["manifest_file_sha256"] != sha256_file(manifest):
        raise ValueError("critic checkpoint manifest SHA256 does not match current training manifest")
    if dataset_content_sha256_value is not None and saved["dataset_content_sha256"] != dataset_content_sha256_value:
        raise ValueError("critic checkpoint dataset-content SHA256 does not match current data")
    if not full_gate0_certificate:
        raise ValueError("critic validation requires full Gate 0 certificate")
    if saved["full_gate0_certificate_sha256"] != sha256_file(full_gate0_certificate):
        raise ValueError("critic checkpoint full Gate 0 certificate mismatch")
    saved_cfg = saved.get("config", {})
    if int(saved_cfg.get("image_size", -1)) != int(cfg["image_size"]):
        raise ValueError("critic checkpoint image_size does not match current run")
    if int(saved_cfg.get("seed", -1)) != int(cfg["seed"]):
        raise ValueError("critic checkpoint seed does not match current run")
    if not _same_number(saved["normal_fraction"], normal_fraction):
        raise ValueError("critic checkpoint normal_fraction does not match current run")
    if not _same_number(saved["normal_critic_weight"], normal_critic_weight):
        raise ValueError("critic checkpoint normal_critic_weight does not match current run")
    if expected_hparams is None:
        raise ValueError("v2.1 critic consumer must declare expected scientific hparams")
    missing_expected = [key for key in CRITIC_COMPATIBILITY_KEYS if key not in expected_hparams]
    if missing_expected:
        raise ValueError("critic consumer does not declare full v2.1 contract: " + ", ".join(missing_expected))
    actual = saved.get("training_hparams", {})
    for key in CRITIC_COMPATIBILITY_KEYS:
        if key not in actual:
            raise ValueError(f"critic checkpoint training_hparams missing {key}")
        expected = expected_hparams[key]
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
    required = (
        "student", "student_kind", "student_width", "seed", "mode",
        "effective_config", "threshold_validation", "manifest_file_sha256",
        "dataset_content_sha256", "training_view_dataset_sha256",
        "gate0_certificate_sha256", "full_gate0_certificate_sha256",
        "student_init_sha256", "inference_contract",
    )
    missing = [k for k in required if k not in saved]
    if missing:
        raise ValueError("student checkpoint missing: " + ", ".join(missing))
    if saved["student_kind"] not in {
        "multiscale", "lightweight", "mobilenetv3", "dsunet", "fastscnn", "bisenet",
    }:
        raise ValueError("student checkpoint has unknown student_kind")
    if saved["student_kind"] == "mobilenetv3" and int(saved["student_width"]) != 16:
        raise ValueError("canonical mobilenetv3 student_width must be 16")
    threshold = float(saved["threshold_validation"])
    if not 0.0 < threshold < 1.0:
        raise ValueError("student checkpoint threshold_validation must be in (0,1)")
    if saved["dataset_content_sha256"] != saved["training_view_dataset_sha256"]:
        raise ValueError("student checkpoint training-view dataset hash mismatch")
    if saved["mode"] not in {
        "control", "connected", "aosk", "aosk_connected", "cldice", "adversarial",
    }:
        raise ValueError("student checkpoint mode is invalid")
    effective = saved["effective_config"]
    if effective.get("student_kind") != saved["student_kind"]:
        raise ValueError("student checkpoint effective student_kind mismatch")
    if int(effective.get("student_width", -1)) != int(saved["student_width"]):
        raise ValueError("student checkpoint effective student_width mismatch")
    if int(effective.get("seed", -1)) != int(saved["seed"]):
        raise ValueError("student checkpoint effective seed mismatch")
    if saved["inference_contract"] != "RGB -> crack logits only":
        raise ValueError("student checkpoint inference contract mismatch")
