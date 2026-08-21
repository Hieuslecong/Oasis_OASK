"""Immutable multi-checkpoint final-evaluation bundle contract for v2.1."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch

from .checkpoint import sha256_file, validate_student_checkpoint
from .protocol import dataset_content_sha256

CANONICAL_ARMS = ("B0", "B1", "B2", "S1", "S2", "S3")
CANONICAL_CONFIRMATORY_SEEDS = (2027, 31415, 42421, 51511, 62617)
ARM_TO_MODE = {
    "B0": "control",
    "B1": "cldice",
    "B2": "adversarial",  # frozen pretrained pair-critic implementation token
    "S1": "connected",
    "S2": "aosk",
    "S3": "aosk_connected",
}
REQUIRED_TOP = {
    "schema",
    "manifest",
    "manifest_sha256",
    "dataset_content_sha256",
    "full_gate0_certificate",
    "full_gate0_certificate_sha256",
    "method_spec",
    "method_spec_sha256",
    "protocol",
    "protocol_sha256",
    "evaluator",
    "evaluator_sha256",
    "metric_spec_sha256",
    "git_commit_sha",
    "entries",
}
REQUIRED_ENTRY = {"arm", "seed", "checkpoint", "checkpoint_sha256", "threshold"}


def _content_identity_payload(bundle):
    """Return a relocation-invariant final-test identity."""
    entries = sorted(
        (
            {
                "arm": str(e["arm"]),
                "seed": int(e["seed"]),
                "checkpoint_sha256": str(e["checkpoint_sha256"]),
                "threshold": float(e["threshold"]),
            }
            for e in bundle.get("entries", [])
        ),
        key=lambda x: (x["seed"], x["arm"]),
    )
    return {
        "schema": bundle.get("schema"),
        "dataset_content_sha256": bundle.get("dataset_content_sha256"),
        "manifest_sha256": bundle.get("manifest_sha256"),
        "full_gate0_certificate_sha256": bundle.get("full_gate0_certificate_sha256"),
        "method_spec_sha256": bundle.get("method_spec_sha256"),
        "protocol_sha256": bundle.get("protocol_sha256"),
        "evaluator_sha256": bundle.get("evaluator_sha256"),
        "metric_spec_sha256": bundle.get("metric_spec_sha256"),
        "git_commit_sha": bundle.get("git_commit_sha"),
        "entries": entries,
    }


def canonical_bundle_id(bundle):
    raw = json.dumps(
        _content_identity_payload(bundle),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode()
    return hashlib.sha256(raw).hexdigest()


def _validate_checkpoint_binding(entry, bundle):
    arm = str(entry["arm"])
    seed = int(entry["seed"])
    if arm not in ARM_TO_MODE:
        raise ValueError(f"unknown final-bundle arm {arm!r}")
    ck = torch.load(entry["checkpoint"], map_location="cpu", weights_only=False)
    validate_student_checkpoint(ck)
    if int(ck["seed"]) != seed:
        raise ValueError(f"bundle seed/checkpoint mismatch {(arm, seed)}")
    if ck["mode"] != ARM_TO_MODE[arm]:
        raise ValueError(
            f"bundle arm/checkpoint mode mismatch {(arm, seed)}: "
            f"expected {ARM_TO_MODE[arm]!r}, got {ck['mode']!r}"
        )
    if abs(float(ck["threshold_validation"]) - float(entry["threshold"])) > 1e-12:
        raise ValueError(f"bundle threshold/checkpoint mismatch {(arm, seed)}")
    if ck["full_gate0_certificate_sha256"] != bundle["full_gate0_certificate_sha256"]:
        raise ValueError(f"bundle/checkpoint full Gate0 mismatch {(arm, seed)}")
    return ck


def validate_final_bundle(
    bundle_path,
    expected_arms=CANONICAL_ARMS,
    expected_seeds=CANONICAL_CONFIRMATORY_SEEDS,
):
    """Validate the one-shot confirmatory bundle fail-closed.

    Canonical final evaluation requires exactly the preregistered five seeds,
    all six arms, correct arm-to-checkpoint semantics, and paired provenance.
    Development utilities may override ``expected_seeds`` explicitly; the final
    runner intentionally uses the default contract.
    """
    p = Path(bundle_path)
    b = json.loads(p.read_text())
    missing = sorted(REQUIRED_TOP - set(b))
    if missing:
        raise ValueError("bundle missing: " + ", ".join(missing))
    if b["schema"] != "oasis-rc-v2.1-final-bundle-v1":
        raise ValueError("invalid bundle schema")
    checks = (
        ("manifest", "manifest_sha256"),
        ("full_gate0_certificate", "full_gate0_certificate_sha256"),
        ("method_spec", "method_spec_sha256"),
        ("protocol", "protocol_sha256"),
        ("evaluator", "evaluator_sha256"),
    )
    for path_key, sha_key in checks:
        if sha256_file(b[path_key]) != b[sha_key]:
            raise ValueError(f"{path_key} SHA mismatch")
    if dataset_content_sha256(b["manifest"]) != b["dataset_content_sha256"]:
        raise ValueError("dataset content SHA mismatch")
    full = json.loads(Path(b["full_gate0_certificate"]).read_text())
    if full.get("status") != "PASS" or full.get("scope") != "full_benchmark":
        raise ValueError("full Gate0 must be PASS/full_benchmark")
    if full.get("manifest_sha256") != b["manifest_sha256"]:
        raise ValueError("full Gate0 certificate is not bound to final manifest")
    if full.get("dataset_content_sha256") != b["dataset_content_sha256"]:
        raise ValueError("full Gate0 certificate is not bound to final dataset bytes")

    entries = b["entries"]
    if not isinstance(entries, list) or not entries:
        raise ValueError("entries must be non-empty")
    seen = set()
    by_seed = {}
    paired = {}
    for e in entries:
        miss = sorted(REQUIRED_ENTRY - set(e))
        if miss:
            raise ValueError("entry missing: " + ", ".join(miss))
        key = (str(e["arm"]), int(e["seed"]))
        if key in seen:
            raise ValueError(f"duplicate arm/seed {key}")
        seen.add(key)
        by_seed.setdefault(int(e["seed"]), set()).add(str(e["arm"]))
        if sha256_file(e["checkpoint"]) != e["checkpoint_sha256"]:
            raise ValueError(f"checkpoint SHA mismatch {key}")
        t = float(e["threshold"])
        if not 0 < t < 1:
            raise ValueError(f"invalid threshold {key}")
        ck = _validate_checkpoint_binding(e, b)
        seed = int(e["seed"])
        state = paired.setdefault(
            seed,
            {
                "student_init_sha256": ck["student_init_sha256"],
                "training_view_dataset_sha256": ck["training_view_dataset_sha256"],
                "gate0_certificate_sha256": ck["gate0_certificate_sha256"],
            },
        )
        for field in (
            "student_init_sha256",
            "training_view_dataset_sha256",
            "gate0_certificate_sha256",
        ):
            if ck[field] != state[field]:
                raise ValueError(
                    f"seed {seed} arms are not paired on {field}: "
                    f"expected {state[field]!r}, got {ck[field]!r} for arm {e['arm']}"
                )

    required_arms = set(expected_arms)
    for seed, arms in by_seed.items():
        if arms != required_arms:
            raise ValueError(f"seed {seed} incomplete arms: {sorted(arms)}")

    actual_seeds = tuple(sorted(by_seed))
    if expected_seeds is not None:
        required_seeds = tuple(sorted(int(s) for s in expected_seeds))
        if actual_seeds != required_seeds:
            missing_seeds = sorted(set(required_seeds) - set(actual_seeds))
            extra_seeds = sorted(set(actual_seeds) - set(required_seeds))
            raise ValueError(
                "final bundle confirmatory seed mismatch; "
                f"missing={missing_seeds}, extra={extra_seeds}, "
                f"required={list(required_seeds)}"
            )

    actual_id = canonical_bundle_id(b)
    if b.get("bundle_id") not in (None, actual_id):
        raise ValueError("bundle_id mismatch")
    return {**b, "bundle_id": actual_id, "seeds": list(actual_seeds)}
