"""Calibrated checkpoint evaluation for OASIS-A2S v0.1.3.

Development evaluation is bound to both manifest bytes and dataset-content bytes.
Final/external evaluation must opt in explicitly and uses the threshold frozen on
source CAL; no threshold search is permitted here.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .a2s import METHOD_VERSION, OASISA2SDiscriminator
from .data import ManifestDataset, audit_manifest
from .metrics_a2s import evaluate_model
from .train_a2s import _assert_dev_split, _sha256


def _verify_manifest_provenance(checkpoint_manifest_sha: str, evaluation_manifest_sha: str, *, is_dev_split: bool, allow_manifest_mismatch: bool) -> bool:
    if not isinstance(checkpoint_manifest_sha, str) or len(checkpoint_manifest_sha) != 64:
        raise ValueError("checkpoint is missing a valid manifest_sha256")
    if not isinstance(evaluation_manifest_sha, str) or len(evaluation_manifest_sha) != 64:
        raise ValueError("evaluation manifest SHA256 is invalid")
    match = evaluation_manifest_sha == checkpoint_manifest_sha
    if not match and not allow_manifest_mismatch:
        raise ValueError("evaluation manifest SHA256 does not match checkpoint provenance; use --allow-manifest-mismatch only for explicit external/final evaluation")
    if is_dev_split and not match:
        raise ValueError("development evaluation may not override manifest provenance")
    return match


def _verify_dataset_content(ck: dict, manifest: str, split: str, *, is_dev_split: bool, allow_missing_lineage: bool, allow_size_mismatch: bool) -> tuple[dict, bool | None]:
    expected = ck.get("dataset_content_sha256")
    if not isinstance(expected, str) or len(expected) != 64:
        raise ValueError("checkpoint is missing dataset_content_sha256")
    if is_dev_split:
        splits = (ck.get("fit_split"), ck.get("cal_split"), ck.get("val_split"))
        if any(not isinstance(s, str) or not s for s in splits):
            raise ValueError("checkpoint is missing FIT/CAL/VAL split provenance")
        audit = audit_manifest(manifest, splits, require_lineage=not allow_missing_lineage, allow_size_mismatch=allow_size_mismatch)
        match = audit["dataset_content_sha256"] == expected
        if not match: raise ValueError("development dataset bytes do not match checkpoint provenance")
        return audit, True
    return audit_manifest(manifest, (split,), require_lineage=not allow_missing_lineage, allow_size_mismatch=allow_size_mismatch), None


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True); p.add_argument("--manifest", required=True); p.add_argument("--split", required=True)
    p.add_argument("--batch", type=int, default=8); p.add_argument("--workers", type=int, default=0); p.add_argument("--device", default="cpu")
    p.add_argument("--allow-final-test", action="store_true"); p.add_argument("--allow-manifest-mismatch", action="store_true")
    p.add_argument("--allow-missing-lineage", action="store_true"); p.add_argument("--allow-size-mismatch", action="store_true")
    p.add_argument("--per-image-out", default=None); p.add_argument("--out", required=True); a = p.parse_args()

    is_dev_split = True
    try: _assert_dev_split(a.split)
    except ValueError:
        is_dev_split = False
        if not a.allow_final_test: raise ValueError("final/test evaluation is sealed; use --allow-final-test only after protocol freeze")

    ck = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
    if ck.get("method") != METHOD_VERSION: raise ValueError(f"not an {METHOD_VERSION} checkpoint")
    forbidden = {"generator", "optimizer_d", "optimizer_g", "critic", "aosk"}.intersection(ck)
    if forbidden: raise ValueError(f"deployment checkpoint contains training-only state: {sorted(forbidden)}")
    threshold = ck.get("calibrated_threshold")
    if not isinstance(threshold, (int, float)) or not 0.0 < float(threshold) < 1.0:
        raise ValueError("deployment checkpoint must contain a valid CAL-frozen threshold")

    checkpoint_manifest_sha = ck.get("manifest_sha256"); evaluation_manifest_sha = _sha256(a.manifest)
    manifest_match = _verify_manifest_provenance(checkpoint_manifest_sha, evaluation_manifest_sha, is_dev_split=is_dev_split, allow_manifest_mismatch=a.allow_manifest_mismatch)
    audit, content_match = _verify_dataset_content(ck, a.manifest, a.split, is_dev_split=is_dev_split, allow_missing_lineage=a.allow_missing_lineage, allow_size_mismatch=a.allow_size_mismatch)

    device = torch.device(a.device)
    if device.type == "cuda" and not torch.cuda.is_available(): raise RuntimeError("CUDA requested but unavailable")
    width, size = int(ck["width"]), int(ck["image_size"]); arm = ck.get("arm"); stage1 = "stage1_discriminator" in ck
    if stage1:
        model = OASISA2SDiscriminator(width, 3).to(device); model.load_state_dict(ck["stage1_discriminator"])
    elif "segmenter" in ck:
        model = OASISA2SDiscriminator(width, 2).to(device); model.load_state_dict(ck["segmenter"])
    else: raise ValueError("checkpoint contains neither deployable segmenter nor A1 discriminator")

    loader = DataLoader(ManifestDataset(a.manifest, a.split, size, return_metadata=True), batch_size=a.batch, shuffle=False, num_workers=a.workers)
    metrics, rows = evaluate_model(model, loader, device, float(threshold), stage1=stage1)
    result = {**metrics, "method": METHOD_VERSION, "implementation_revision": ck.get("implementation_revision"), "arm": arm, "split": a.split,
        "threshold_source": ck.get("threshold_source"), "checkpoint_sha256": _sha256(a.checkpoint), "checkpoint_manifest_sha256": checkpoint_manifest_sha,
        "evaluation_manifest_sha256": evaluation_manifest_sha, "manifest_provenance_match": manifest_match,
        "checkpoint_dataset_content_sha256": ck.get("dataset_content_sha256"), "evaluation_dataset_content_sha256": audit["dataset_content_sha256"],
        "dataset_content_provenance_match": content_match, "manifest_mismatch_override": bool(a.allow_manifest_mismatch),
        "final_test_override": bool(a.allow_final_test), "image_size": size, "width": width, "inference_contract": ck.get("inference_contract")}
    Path(a.out).write_text(json.dumps(result, indent=2))
    if a.per_image_out:
        with Path(a.per_image_out).open("w") as f:
            for row in rows: f.write(json.dumps(row, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__": main()
