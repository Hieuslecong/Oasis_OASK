import importlib.util
import json
from pathlib import Path

import pytest

import oasis_rc_v2.final_bundle as final_bundle


def _load_stats_script():
    path = Path(__file__).resolve().parents[1] / "scripts" / "analyze_v21_paired.py"
    spec = importlib.util.spec_from_file_location("analyze_v21_paired_failclosed", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _bundle_fixture(tmp_path, seeds):
    full = tmp_path / "full.json"
    full.write_text(
        json.dumps(
            {
                "status": "PASS",
                "scope": "full_benchmark",
                "manifest_sha256": "ok",
                "dataset_content_sha256": "data",
            }
        )
    )
    entries = []
    for seed in seeds:
        for arm in final_bundle.CANONICAL_ARMS:
            entries.append(
                {
                    "arm": arm,
                    "seed": seed,
                    "checkpoint": str(tmp_path / f"{arm}_{seed}.pt"),
                    "checkpoint_sha256": "ok",
                    "threshold": 0.5,
                }
            )
    bundle = {
        "schema": "oasis-rc-v2.1-final-bundle-v1",
        "manifest": str(tmp_path / "manifest.jsonl"),
        "manifest_sha256": "ok",
        "dataset_content_sha256": "data",
        "full_gate0_certificate": str(full),
        "full_gate0_certificate_sha256": "ok",
        "method_spec": str(tmp_path / "spec.md"),
        "method_spec_sha256": "ok",
        "protocol": str(tmp_path / "protocol.json"),
        "protocol_sha256": "ok",
        "evaluator": str(tmp_path / "eval.py"),
        "evaluator_sha256": "ok",
        "metric_spec": str(tmp_path / "metric.md"),
        "metric_spec_sha256": "ok",
        "git_commit_sha": "c" * 40,
        "entries": entries,
    }
    path = tmp_path / "bundle.json"
    path.write_text(json.dumps(bundle))
    return path


def _install_bundle_mocks(
    monkeypatch,
    mode_override=None,
    init_override=None,
    git_override=None,
):
    monkeypatch.setattr(final_bundle, "sha256_file", lambda path: "ok")
    monkeypatch.setattr(final_bundle, "dataset_content_sha256", lambda path: "data")
    monkeypatch.setattr(final_bundle, "validate_student_checkpoint", lambda ck: None)

    def fake_load(path, **kwargs):
        stem = Path(path).stem
        arm, seed_text = stem.split("_", 1)
        seed = int(seed_text)
        mode = final_bundle.ARM_TO_MODE[arm]
        if mode_override and arm in mode_override:
            mode = mode_override[arm]
        init = f"init-{seed}"
        if init_override and arm in init_override:
            init = init_override[arm]
        git_sha = "c" * 40
        if git_override and arm in git_override:
            git_sha = git_override[arm]
        return {
            "seed": seed,
            "mode": mode,
            "threshold_validation": 0.5,
            "full_gate0_certificate_sha256": "ok",
            "student_init_sha256": init,
            "training_view_dataset_sha256": f"train-{seed}",
            "gate0_certificate_sha256": f"gate-{seed}",
            "runtime": {"git_sha": git_sha},
        }

    monkeypatch.setattr(final_bundle.torch, "load", fake_load)


def test_final_bundle_rejects_missing_preregistered_seed(tmp_path, monkeypatch):
    _install_bundle_mocks(monkeypatch)
    missing_one = final_bundle.CANONICAL_CONFIRMATORY_SEEDS[:-1]
    path = _bundle_fixture(tmp_path, missing_one)
    with pytest.raises(ValueError, match="confirmatory seed mismatch"):
        final_bundle.validate_final_bundle(path)


def test_final_bundle_accepts_exact_preregistered_seed_set(tmp_path, monkeypatch):
    _install_bundle_mocks(monkeypatch)
    path = _bundle_fixture(tmp_path, final_bundle.CANONICAL_CONFIRMATORY_SEEDS)
    validated = final_bundle.validate_final_bundle(path)
    assert validated["seeds"] == sorted(final_bundle.CANONICAL_CONFIRMATORY_SEEDS)
    assert len(validated["entries"]) == 30


def test_final_bundle_rejects_arm_mode_relabeling(tmp_path, monkeypatch):
    _install_bundle_mocks(monkeypatch, mode_override={"S3": "control"})
    path = _bundle_fixture(tmp_path, final_bundle.CANONICAL_CONFIRMATORY_SEEDS)
    with pytest.raises(ValueError, match="arm/checkpoint mode mismatch"):
        final_bundle.validate_final_bundle(path)


def test_final_bundle_rejects_unpaired_initialization(tmp_path, monkeypatch):
    _install_bundle_mocks(monkeypatch, init_override={"S2": "different-init"})
    path = _bundle_fixture(tmp_path, final_bundle.CANONICAL_CONFIRMATORY_SEEDS)
    with pytest.raises(ValueError, match="not paired on student_init_sha256"):
        final_bundle.validate_final_bundle(path)


def test_final_bundle_rejects_checkpoint_from_different_git_commit(tmp_path, monkeypatch):
    _install_bundle_mocks(monkeypatch, git_override={"B2": "d" * 40})
    path = _bundle_fixture(tmp_path, final_bundle.CANONICAL_CONFIRMATORY_SEEDS)
    with pytest.raises(ValueError, match="git commit mismatch"):
        final_bundle.validate_final_bundle(path)


def _write_eval(path, normal=False, drop_metric=None):
    if normal:
        data = {
            "crack_image_count": 0,
            "normal_image_count": 8,
            "normal_any_fp_rate": 0.1,
            "normal_fp_pixels_mean": 3.0,
            "normal_fp_components_mean": 1.0,
        }
    else:
        data = {
            "crack_image_count": 8,
            "normal_image_count": 0,
            "dice": 0.8,
            "iou": 0.7,
            "cldice": 0.75,
            "mean_component_excess": 1.0,
        }
    if drop_metric:
        data.pop(drop_metric)
    path.write_text(json.dumps(data))


def test_confirmatory_stats_reject_metric_missing_for_one_seed(tmp_path):
    stats = _load_stats_script()
    seeds = [2027, 31415]
    arms = {arm: {} for arm in final_bundle.CANONICAL_ARMS}
    for arm in final_bundle.CANONICAL_ARMS:
        for seed in seeds:
            crack = tmp_path / f"{arm}_{seed}_crack.json"
            normal = tmp_path / f"{arm}_{seed}_normal.json"
            drop = "dice" if arm == "S1" and seed == 31415 else None
            _write_eval(crack, normal=False, drop_metric=drop)
            _write_eval(normal, normal=True)
            arms[arm][str(seed)] = {"crack": str(crack), "normal": str(normal)}
    with pytest.raises(ValueError, match="missing required confirmatory metric 'dice'"):
        stats.analyze({"seeds": seeds, "arms": arms}, bootstrap_reps=1000)
