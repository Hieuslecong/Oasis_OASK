from types import SimpleNamespace

import pytest

from oasis_cycle_aosk.train_oasis_rc_v2 import validate_loaded_critic
from oasis_rc_v2.checkpoint import (
    CHECKPOINT_SCHEMA,
    EXPERIMENT_ID,
    IMPLEMENTATION_VERSION,
    METHOD_VERSION,
)


def test_legacy_validator_cannot_authorize_v21_dev2_connected_critic(tmp_path):
    """Regression for the audited legacy energy-gate bypass.

    The reconstructed v2.0.4 consumer does not declare the complete v2.1
    scientific compatibility contract and therefore must fail before it could
    produce connected-arm v2.1 evidence.
    """
    manifest = tmp_path / "m.jsonl"
    manifest.write_text("x\n")
    full_certificate = tmp_path / "full_gate0.json"
    full_certificate.write_text('{"status":"PASS","scope":"full_benchmark"}')
    args = SimpleNamespace(
        manifest=str(manifest),
        normal_fraction=0.25,
        normal_critic_weight=1.0,
        crack_dice_weight=1.0,
        mismatch_weight=1.0,
        pair_weight=0.25,
        rgb_mask_weight=1.0,
        _dataset_content_sha256="d" * 64,
        full_gate0_certificate=str(full_certificate),
    )
    cfg = {"seed": 1337, "image_size": 256}
    saved = {
        "checkpoint_schema": CHECKPOINT_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "method_version": METHOD_VERSION,
        "implementation_version": IMPLEMENTATION_VERSION,
        "energy_head_contract": "dedicated-scalar-lower-is-better-v1",
        "critic": {},
        "width": 8,
        "config": cfg,
        "seed": 1337,
        "full_gate0_certificate_sha256": __import__("hashlib").sha256(
            full_certificate.read_bytes()
        ).hexdigest(),
        "manifest_file_sha256": __import__("hashlib").sha256(
            manifest.read_bytes()
        ).hexdigest(),
        "dataset_content_sha256": "d" * 64,
        "normal_fraction": 0.25,
        "normal_critic_weight": 1.0,
        "qualification_v21": {"pass": True, "failures": []},
        "training_hparams": {
            "crack_dice_weight": 1.0,
            "mismatch_weight": 1.0,
            "pair_weight": 0.25,
            "rgb_mask_weight": 1.0,
            "normal_critic_weight": 1.0,
            "normal_fraction": 0.25,
            "rgb_shuffle_pair_only": True,
            "mask_flip_training": False,
            "mask_variant_contract": "operator-preserved-v1",
            "energy_head_contract": "dedicated-scalar-lower-is-better-v1",
            "method_spec": "METHOD_SPEC_V2_1.md",
        },
    }
    with pytest.raises(ValueError, match="full v2.1 contract"):
        validate_loaded_critic(saved, args, cfg)
