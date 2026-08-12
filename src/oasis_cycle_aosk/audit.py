import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

REQUIRED = {"image", "split", "source_id", "lineage_id", "is_normal"}


def _sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _decoded_rgb_sha256(path):
    arr = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
    h = hashlib.sha256()
    h.update(str(arr.shape).encode("ascii"))
    h.update(arr.tobytes(order="C"))
    return h.hexdigest()


def audit(
    path,
    allow_debug_no_test_normals=False,
    test_split="test",
    require_source_disjoint=False,
    require_normal=False,
    normal_policy=None,
):
    """Audit manifest integrity without opening model predictions.

    ``normal_policy`` is one of:
      - ``none``: no external true-normal split is required;
      - ``train``: require ``normal_train``;
      - ``train_and_aux_val``: require ``normal_train`` and ``normal_val``.

    ``require_normal`` is retained only for backward compatibility and maps to
    ``normal_policy='train'``.  Canonical ``val``/``test`` are never required to
    contain normal images merely because normal supervision is used in train.
    """
    rows = [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]
    errors = []
    lineage_splits = defaultdict(set)
    source_splits = defaultdict(set)
    image_seen = set()
    raw_sha_rows = defaultdict(list)
    decoded_sha_rows = defaultdict(list)

    if normal_policy is None:
        normal_policy = "train" if require_normal else "none"
    if normal_policy not in {"none", "train", "train_and_aux_val"}:
        errors.append(f"invalid normal_policy={normal_policy!r}")
        return errors

    for i, r in enumerate(rows):
        missing = REQUIRED - r.keys()
        if missing:
            errors.append(f"row {i}: missing {sorted(missing)}")
            continue

        split = str(r["split"])
        lineage = str(r["lineage_id"])
        source = str(r["source_id"])
        lineage_splits[lineage].add(split)
        source_splits[source].add(split)

        # A split-qualified lineage is invalid by construction because it can
        # hide the same parent/source across train/val/test.
        if lineage.startswith(f"{split}:") or lineage.startswith(f"{split}::"):
            errors.append(f"row {i}: lineage_id is split-qualified: {lineage}")

        image_path = Path(r["image"])
        image_key = str(image_path.resolve())
        if image_key in image_seen:
            errors.append(f"duplicate image path: {image_key}")
        image_seen.add(image_key)
        if not image_path.exists():
            errors.append(f"row {i}: missing image file")
            continue

        try:
            raw_sha_rows[_sha256_file(image_path)].append((i, split, bool(r["is_normal"]), image_key))
            decoded_sha_rows[_decoded_rgb_sha256(image_path)].append((i, split, bool(r["is_normal"]), image_key))
        except Exception as exc:
            errors.append(f"row {i}: cannot decode/hash image: {exc}")
            continue

        is_normal = bool(r["is_normal"])
        mask_path = r.get("mask")
        if is_normal:
            # True-normal RGB uses a virtual zero mask.  A provided mask is not
            # required and is deliberately not used by the deployment loader.
            continue

        if not mask_path or not Path(mask_path).exists():
            errors.append(f"row {i}: missing cracked mask")
            continue

        try:
            iw, ih = Image.open(image_path).size
            mw, mh = Image.open(mask_path).size
            # Different native resolutions are acceptable when aspect/FOV
            # geometry is compatible; the loader resizes masks with NEAREST.
            if abs((iw / max(ih, 1)) - (mw / max(mh, 1))) > 1e-6:
                errors.append(
                    f"row {i}: image/mask aspect-ratio mismatch "
                    f"image={iw}x{ih} mask={mw}x{mh}"
                )
        except Exception as exc:
            errors.append(f"row {i}: cannot inspect mask geometry: {exc}")

    for lineage, splits in lineage_splits.items():
        if len(splits) > 1:
            errors.append(f"lineage leakage: {lineage} in {sorted(splits)}")

    if require_source_disjoint:
        for source, splits in source_splits.items():
            if len(splits) > 1:
                errors.append(f"source leakage: {source} in {sorted(splits)}")

    def check_hash_groups(kind, groups):
        for digest, items in groups.items():
            splits = {item[1] for item in items}
            labels = {item[2] for item in items}
            paths = {item[3] for item in items}
            if len(paths) <= 1:
                continue
            if len(splits) > 1:
                errors.append(
                    f"{kind} duplicate across splits: {digest} in {sorted(splits)}"
                )
            if len(labels) > 1:
                errors.append(
                    f"{kind} cross-label duplicate (normal vs crack): {digest}"
                )

    check_hash_groups("raw-image", raw_sha_rows)
    check_hash_groups("decoded-rgb", decoded_sha_rows)

    required_splits = ("train", "val", test_split)
    for split in required_splits:
        if not any(r.get("split") == split for r in rows):
            errors.append(f"missing required split: {split}")

    if normal_policy in {"train", "train_and_aux_val"}:
        if not any(r.get("split") == "normal_train" and bool(r.get("is_normal")) for r in rows):
            errors.append("normal_train: no true-normal sample")
    if normal_policy == "train_and_aux_val":
        if not any(r.get("split") == "normal_val" and bool(r.get("is_normal")) for r in rows):
            errors.append("normal_val: no true-normal sample")

    # Retained for API compatibility.  It must not weaken the canonical test
    # firewall or force normals into the canonical test set.
    _ = allow_debug_no_test_normals
    return errors


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    p.add_argument("--test-split", default="test")
    p.add_argument("--require-source-disjoint", action="store_true")
    p.add_argument(
        "--normal-policy",
        choices=("none", "train", "train_and_aux_val"),
        default="none",
    )
    a = p.parse_args()
    errors = audit(
        a.manifest,
        test_split=a.test_split,
        require_source_disjoint=a.require_source_disjoint,
        normal_policy=a.normal_policy,
    )
    if errors:
        print("G0 FAIL")
        print("\n".join(errors))
        raise SystemExit(2)
    print("G0 PASS")


if __name__ == "__main__":
    main()
