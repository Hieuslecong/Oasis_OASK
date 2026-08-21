#!/usr/bin/env python3
"""Open the canonical v2.1 final benchmark exactly once.

Both ``test`` and ``normal_test`` are evaluated under one content-addressed
ledger marker.  A caller must provide one stable ledger root outside movable
dataset/checkpoint directories; relocating identical files therefore cannot
create a fresh opening namespace.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from oasis_rc_v2.checkpoint import validate_student_checkpoint, sha256_file
from oasis_rc_v2.final_bundle import validate_final_bundle
from oasis_cycle_aosk.data import ManifestDataset
from oasis_cycle_aosk.evaluate_rc import build, manifest_splits
from oasis_cycle_aosk.evaluate_v21 import evaluate

FINAL_SPLITS = ("test", "normal_test")


def _atomic_create(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    fd = os.open(path, flags, 0o600)
    with os.fdopen(fd, "w") as f:
        json.dump(payload, f, indent=2)
        f.flush()
        os.fsync(f.fileno())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bundle", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument(
        "--ledger-root",
        required=True,
        help="Stable, append-only location shared by every canonical-test attempt.",
    )
    p.add_argument("--device", default="cuda")
    a = p.parse_args()

    bundle = validate_final_bundle(a.bundle)
    manifest = Path(bundle["manifest"]).resolve()
    splits = manifest_splits(manifest)
    missing_final = set(FINAL_SPLITS) - splits
    if missing_final:
        raise ValueError(
            "final benchmark missing canonical splits: " + ", ".join(sorted(missing_final))
        )

    ledger_dir = Path(a.ledger_root).expanduser().resolve()
    marker = ledger_dir / f"{bundle['bundle_id']}.json"
    opened = {
        "state": "OPENED",
        "bundle_id": bundle["bundle_id"],
        "bundle": str(Path(a.bundle).resolve()),
        "bundle_sha256": sha256_file(a.bundle),
        "manifest": str(manifest),
        "dataset_content_sha256": bundle["dataset_content_sha256"],
        "opened_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "entries": len(bundle["entries"]),
        "canonical_splits": list(FINAL_SPLITS),
    }
    try:
        _atomic_create(marker, opened)
    except FileExistsError:
        raise SystemExit(f"REFUSE: bundle already opened: {marker}")

    # From this exact point both canonical splits are considered opened even if
    # a later model/split evaluation fails.
    device = torch.device(a.device)
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results = []
    try:
        for entry in bundle["entries"]:
            ck = torch.load(entry["checkpoint"], map_location="cpu", weights_only=False)
            validate_student_checkpoint(ck)
            if int(ck["seed"]) != int(entry["seed"]):
                raise ValueError("bundle seed/checkpoint mismatch")
            if abs(float(ck["threshold_validation"]) - float(entry["threshold"])) > 1e-12:
                raise ValueError("bundle threshold/checkpoint mismatch")
            model = build(ck["student_kind"], int(ck["student_width"])).to(device)
            model.load_state_dict(ck["student"])
            size = int(ck["effective_config"]["image_size"])

            for split in FINAL_SPLITS:
                loader = DataLoader(
                    ManifestDataset(manifest, split, size),
                    batch_size=4,
                    shuffle=False,
                    num_workers=0,
                )
                pred_dir = out_dir / f"pred_{split}_{entry['arm']}_seed{entry['seed']}"
                r = evaluate(
                    model,
                    loader,
                    float(entry["threshold"]),
                    device,
                    pred_dir,
                )
                r.update(
                    {
                        "split": split,
                        "arm": entry["arm"],
                        "seed": int(entry["seed"]),
                        "checkpoint_sha256": entry["checkpoint_sha256"],
                        "method_version": ck["method_version"],
                        "implementation_version": ck["implementation_version"],
                    }
                )
                path = out_dir / f"{split}_{entry['arm']}_seed{entry['seed']}.json"
                path.write_text(json.dumps(r, indent=2))
                results.append(
                    {
                        "split": split,
                        "arm": entry["arm"],
                        "seed": int(entry["seed"]),
                        "result": str(path.resolve()),
                        "result_sha256": sha256_file(path),
                    }
                )

        summary = {
            "bundle_id": bundle["bundle_id"],
            "state": "DONE",
            "completed_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "canonical_splits": list(FINAL_SPLITS),
            "results": results,
        }
        (out_dir / "bundle_results.json").write_text(json.dumps(summary, indent=2))
        tmp = marker.with_suffix(".tmp")
        tmp.write_text(json.dumps({**opened, **summary}, indent=2))
        os.replace(tmp, marker)
        print(json.dumps(summary, indent=2))
    except Exception as exc:
        failed = {
            **opened,
            "state": "FAILED_AFTER_OPEN",
            "failed_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "error": repr(exc),
        }
        tmp = marker.with_suffix(".tmp")
        tmp.write_text(json.dumps(failed, indent=2))
        os.replace(tmp, marker)
        raise


if __name__ == "__main__":
    main()
