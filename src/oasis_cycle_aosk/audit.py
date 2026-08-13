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


def _array_sha256(arr, tag):
    arr = np.ascontiguousarray(arr)
    h = hashlib.sha256()
    h.update(str(tag).encode("utf-8"))
    h.update(str(arr.shape).encode("ascii"))
    h.update(str(arr.dtype).encode("ascii"))
    h.update(arr.tobytes(order="C"))
    return h.hexdigest()


def _decoded_rgb_sha256(path):
    arr = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
    return _array_sha256(arr, "rgb")


def _decoded_binary_mask(path):
    arr = np.asarray(Image.open(path).convert("L"), dtype=np.uint8)
    return (arr > 127).astype(np.uint8)


def _pair_sha256(rgb_digest, mask_digest):
    h = hashlib.sha256()
    h.update(rgb_digest.encode("ascii"))
    h.update(b"\0")
    h.update(mask_digest.encode("ascii"))
    return h.hexdigest()


def _resized_foreground_count(mask_path, size):
    mask = Image.open(mask_path).convert("L").resize(
        (int(size), int(size)), resample=Image.Resampling.NEAREST
    )
    return int((np.asarray(mask, dtype=np.uint8) > 127).sum())


def audit(
    path,
    allow_debug_no_test_normals=False,
    test_split="test",
    require_source_disjoint=False,
    require_normal=False,
    normal_policy=None,
    resize_size=None,
):
    """Audit manifest/data integrity without opening model predictions.

    ``resize_size`` should be the effective train/eval resolution. When supplied,
    any crack-positive native mask that becomes empty after NEAREST resize is a
    hard failure rather than being silently reinterpreted as a normal sample.

    Native image/mask resolution mismatches are not rejected solely because the
    dimensions differ, but they require both compatible aspect ratio and an
    explicit ``alignment_verified=true`` manifest certification.
    """
    rows = [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]
    errors = []
    lineage_splits = defaultdict(set)
    source_splits = defaultdict(set)
    image_seen = set()
    raw_rgb_rows = defaultdict(list)
    decoded_rgb_rows = defaultdict(list)
    raw_mask_rows = defaultdict(list)
    decoded_mask_rows = defaultdict(list)
    pair_rows = defaultdict(list)

    if normal_policy is None:
        normal_policy = "train" if require_normal else "none"
    if normal_policy not in {"none", "train", "train_and_aux_val"}:
        return [f"invalid normal_policy={normal_policy!r}"]

    for i, r in enumerate(rows):
        missing = REQUIRED - r.keys()
        if missing:
            errors.append(f"row {i}: missing {sorted(missing)}")
            continue
        if not isinstance(r.get("is_normal"), bool):
            errors.append(f"row {i}: is_normal must be a JSON boolean")
            continue

        split = str(r["split"])
        lineage = str(r["lineage_id"])
        source = str(r["source_id"])
        lineage_splits[lineage].add(split)
        source_splits[source].add(split)

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
            raw_rgb = _sha256_file(image_path)
            decoded_rgb = _decoded_rgb_sha256(image_path)
            item = (i, split, r["is_normal"], image_key)
            raw_rgb_rows[raw_rgb].append(item)
            decoded_rgb_rows[decoded_rgb].append(item)
        except Exception as exc:
            errors.append(f"row {i}: cannot decode/hash image: {exc}")
            continue

        is_normal = r["is_normal"] is True
        mask_path = r.get("mask")
        if is_normal:
            if mask_path not in (None, ""):
                errors.append(
                    f"row {i}: true-normal row must use virtual zero mask (mask=null)"
                )
            virtual_mask_digest = hashlib.sha256(b"virtual-zero-mask").hexdigest()
            pair_rows[_pair_sha256(decoded_rgb, virtual_mask_digest)].append(item)
            continue

        if not mask_path or not Path(mask_path).exists():
            errors.append(f"row {i}: missing cracked mask")
            continue
        mask_path = Path(mask_path)

        try:
            binary = _decoded_binary_mask(mask_path)
            native_fg = int(binary.sum())
            if native_fg == 0:
                if r.get("empty_target_status") == "verified_no_crack":
                    pass
                else:
                    errors.append(
                        f"row {i}: crack-positive row has native-empty mask; classify it "
                        "explicitly as true normal or repair the annotation"
                    )

            raw_mask = _sha256_file(mask_path)
            decoded_mask = _array_sha256(binary, "binary-mask")
            cert_empty = r.get("empty_target_status") == "verified_no_crack"
            mask_item = (i, split, str(mask_path.resolve()), cert_empty)
            raw_mask_rows[raw_mask].append(mask_item)
            decoded_mask_rows[decoded_mask].append(mask_item)
            pair_rows[_pair_sha256(decoded_rgb, decoded_mask)].append(item)

            iw, ih = Image.open(image_path).size
            mw, mh = Image.open(mask_path).size
            if abs((iw / max(ih, 1)) - (mw / max(mh, 1))) > 1e-6:
                errors.append(
                    f"row {i}: image/mask aspect-ratio mismatch "
                    f"image={iw}x{ih} mask={mw}x{mh}"
                )
            elif (iw, ih) != (mw, mh) and r.get("alignment_verified") is not True:
                errors.append(
                    f"row {i}: native-resolution mismatch image={iw}x{ih} "
                    f"mask={mw}x{mh} requires alignment_verified=true after GT-only audit"
                )

            if resize_size is not None and native_fg > 0:
                resized_fg = _resized_foreground_count(mask_path, resize_size)
                if resized_fg == 0:
                    errors.append(
                        f"row {i}: crack mask becomes empty after resize to "
                        f"{resize_size}x{resize_size}; do not treat as normal"
                    )
        except Exception as exc:
            errors.append(f"row {i}: cannot inspect/hash cracked mask: {exc}")

    for lineage, splits in lineage_splits.items():
        if len(splits) > 1:
            errors.append(f"lineage leakage: {lineage} in {sorted(splits)}")

    if require_source_disjoint:
        for source, splits in source_splits.items():
            if len(splits) > 1:
                errors.append(f"source leakage: {source} in {sorted(splits)}")

    def check_rgb_groups(kind, groups):
        for digest, items in groups.items():
            paths = {item[3] for item in items}
            if len(paths) <= 1:
                continue
            splits = {item[1] for item in items}
            labels = {item[2] for item in items}
            if len(splits) > 1:
                errors.append(
                    f"{kind} duplicate across splits: {digest} in {sorted(splits)}"
                )
            if len(labels) > 1:
                errors.append(
                    f"{kind} cross-label duplicate (normal vs crack): {digest}"
                )

    def check_mask_groups(kind, groups):
        for digest, items in groups.items():
            if items and all(it[3] for it in items):
                continue
            splits = {item[1] for item in items}
            if len(splits) > 1:
                paths = sorted({item[2] for item in items})
                errors.append(
                    f"{kind} reused across splits: {digest} in {sorted(splits)} "
                    f"paths={paths[:5]}"
                )

    def check_pair_groups(groups):
        for digest, items in groups.items():
            paths = {item[3] for item in items}
            splits = {item[1] for item in items}
            if len(paths) > 1 and len(splits) > 1:
                errors.append(
                    f"decoded image-mask pair duplicate across splits: "
                    f"{digest} in {sorted(splits)}"
                )

    check_rgb_groups("raw-image", raw_rgb_rows)
    check_rgb_groups("decoded-rgb", decoded_rgb_rows)
    check_mask_groups("raw-mask", raw_mask_rows)
    check_mask_groups("decoded-binary-mask", decoded_mask_rows)
    check_pair_groups(pair_rows)

    for split in ("train", "val", test_split):
        if not any(r.get("split") == split for r in rows):
            errors.append(f"missing required split: {split}")

    if normal_policy in {"train", "train_and_aux_val"}:
        if not any(
            r.get("split") == "normal_train" and r.get("is_normal") is True
            for r in rows
        ):
            errors.append("normal_train: no true-normal sample")
    if normal_policy == "train_and_aux_val":
        if not any(
            r.get("split") == "normal_val" and r.get("is_normal") is True
            for r in rows
        ):
            errors.append("normal_val: no true-normal sample")

    _ = allow_debug_no_test_normals
    return errors


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    p.add_argument("--test-split", default="test")
    p.add_argument("--require-source-disjoint", action="store_true")
    p.add_argument("--resize-size", type=int, default=None)
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
        resize_size=a.resize_size,
    )
    if errors:
        print("G0 FAIL")
        print("\n".join(errors))
        raise SystemExit(2)
    print("G0 PASS")


if __name__ == "__main__":
    main()
