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
        "metric_spec_sha256": "metric",
        "git_commit_sha": "c" * 40,
        "entries": entries,
    }
    path = tmp_path / "bundle.json"
    path.write_text(json.dumps(bundle))
    return path


def test_final_bundle_rejects_missing_preregistered_seed(tmp_path, monkeypatch):
    monkeypatch.setattr(final_bundle, "sha256_file", lambda path: "ok")
    monkeypatch.setattr(final_bundle, "dataset_content_sha256", lambda path: "data")
    missing_one = final_bundle.CANONICAL_CONFIRMATORY_SEEDS[:-1]
    path = _bundle_fixture(tmp_path, missing_one)
    with pytest.raises(ValueError, match="confirmatory seed mismatch"):
        final_bundle.validate_final_bundle(path)


def test_final_bundle_accepts_exact_preregistered_seed_set(tmp_path, monkeypatch):
    monkeypatch.setattr(final_bundle, "sha256_file", lambda path: "ok")
    monkeypatch.setattr(final_bundle, "dataset_content_sha256", lambda path: "data")
    path = _bundle_fixture(tmp_path, final_bundle.CANONICAL_CONFIRMATORY_SEEDS)
    validated = final_bundle.validate_final_bundle(path)
    assert validated["seeds"] == sorted(final_bundle.CANONICAL_CONFIRMATORY_SEEDS)
    assert len(validated["entries"]) == 30


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
