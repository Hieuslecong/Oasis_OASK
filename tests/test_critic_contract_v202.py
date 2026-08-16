from types import SimpleNamespace

import pytest

from oasis_cycle_aosk.train_oasis_rc_v2 import validate_loaded_critic
from oasis_rc_v2.checkpoint import (
    CHECKPOINT_SCHEMA, EXPERIMENT_ID, IMPLEMENTATION_VERSION, METHOD_VERSION,
)


def test_strict_validator_rejects_missing_v202_flags(tmp_path):
    manifest = tmp_path / "m.jsonl"; manifest.write_text("x\n")
    full_certificate = tmp_path / "full_gate0.json"
    full_certificate.write_text('{"status":"PASS","scope":"full_benchmark"}')
    args = SimpleNamespace(
        manifest=str(manifest), normal_fraction=0.25, normal_critic_weight=1.0,
        crack_dice_weight=1.0, mismatch_weight=1.0, pair_weight=0.25,
        rgb_mask_weight=1.0, _dataset_content_sha256="d" * 64,
        full_gate0_certificate=str(full_certificate),
    )
    cfg = {"seed": 1337, "image_size": 256}
    saved = {
        "checkpoint_schema": CHECKPOINT_SCHEMA, "experiment_id": EXPERIMENT_ID,
        "method_version": METHOD_VERSION, "implementation_version": IMPLEMENTATION_VERSION,
        "critic": {}, "width": 8, "config": cfg,
        "seed": 1337,
        "full_gate0_certificate_sha256": __import__("hashlib").sha256(
            full_certificate.read_bytes()
        ).hexdigest(),
        "manifest_file_sha256": __import__("hashlib").sha256(manifest.read_bytes()).hexdigest(),
        "dataset_content_sha256": "d" * 64,
        "normal_fraction": 0.25, "normal_critic_weight": 1.0,
        "training_hparams": {
            "crack_dice_weight": 1.0, "mismatch_weight": 1.0,
            "pair_weight": 0.25, "rgb_mask_weight": 1.0,
            "normal_critic_weight": 1.0, "normal_fraction": 0.25,
        },
    }
    with pytest.raises(ValueError, match="rgb_shuffle_pair_only"):
        validate_loaded_critic(saved, args, cfg)
