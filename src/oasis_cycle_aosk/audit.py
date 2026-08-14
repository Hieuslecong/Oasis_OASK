import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from oasis_rc_v2.protocol import dataset_content_sha256

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
    h.update(str(tag).encode())
    h.update(str(arr.shape).encode("ascii"))
    h.update(str(arr.dtype).encode("ascii"))
    h.update(arr.tobytes(order="C"))
    return h.hexdigest()


def _decoded_rgb_sha256(path):
    return _array_sha256(np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8), "rgb")


def _perceptual_features(path):
    """Return dHash plus RGB mean to avoid grayscale-uniform false matches."""
    rgb = Image.open(path).convert("RGB")
    image = rgb.convert("L").resize((9, 8), Image.Resampling.LANCZOS)
    values = np.asarray(image, dtype=np.int16)
    bits = (values[:, 1:] > values[:, :-1]).reshape(-1)
    result = 0
    for bit in bits:
        result = (result << 1) | int(bit)
    mean_rgb = tuple(
        float(value)
        for value in np.asarray(rgb.resize((1, 1), Image.Resampling.BOX))[0, 0]
    )
    return result, mean_rgb


def _decoded_binary_mask(path):
    return (np.asarray(Image.open(path).convert("L"), dtype=np.uint8) > 127).astype(np.uint8)


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


def dataset_inventory(path):
    rows = [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]
    inventory = []
    for i, row in enumerate(rows):
        image = Path(row["image"])
        mask = row.get("mask")
        inventory.append(
            {
                "row": i,
                "split": row.get("split"),
                "source_id": row.get("source_id"),
                "lineage_id": row.get("lineage_id"),
                "is_normal": row.get("is_normal"),
                "image": str(image.resolve()),
                "image_sha256": _sha256_file(image),
                "mask": None if mask in (None, "") else str(Path(mask).resolve()),
                "mask_sha256": (
                    "VIRTUAL_ZERO_MASK"
                    if row.get("is_normal") is True
                    else _sha256_file(mask)
                ),
            }
        )
    return inventory


def audit(
    path,
    allow_debug_no_test_normals=False,
    test_split="test",
    require_source_disjoint=False,
    require_normal=False,
    normal_policy=None,
    resize_size=None,
    required_splits=None,
):
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
    perceptual_rows = []
    if normal_policy is None:
        normal_policy = "train" if require_normal else "none"
    if normal_policy not in {"none", "train", "train_and_aux_val"}:
        return [f"invalid normal_policy={normal_policy!r}"]
    if required_splits is None:
        required_splits = ("train", "val", test_split)

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
            dhash, mean_rgb = _perceptual_features(image_path)
            perceptual_rows.append((i, split, image_key, dhash, mean_rgb))
        except Exception as exc:
            errors.append(f"row {i}: cannot decode/hash image: {exc}")
            continue

        is_normal = r["is_normal"] is True
        mask_path = r.get("mask")
        if is_normal:
            if mask_path not in (None, ""):
                errors.append(f"row {i}: true-normal row must use virtual zero mask (mask=null)")
            virtual = hashlib.sha256(b"virtual-zero-mask").hexdigest()
            pair_rows[_pair_sha256(decoded_rgb, virtual)].append(item)
            continue
        if not mask_path or not Path(mask_path).exists():
            errors.append(f"row {i}: missing cracked mask")
            continue
        mask_path = Path(mask_path)
        try:
            binary = _decoded_binary_mask(mask_path)
            native_fg = int(binary.sum())
            cert_empty = r.get("empty_target_status") == "verified_no_crack"
            if native_fg == 0 and not cert_empty:
                errors.append(
                    f"row {i}: crack-positive row has native-empty mask; classify it explicitly as true normal or repair the annotation"
                )
            raw_mask = _sha256_file(mask_path)
            decoded_mask = _array_sha256(binary, "binary-mask")
            mask_item = (i, split, str(mask_path.resolve()), cert_empty)
            raw_mask_rows[raw_mask].append(mask_item)
            decoded_mask_rows[decoded_mask].append(mask_item)
            pair_rows[_pair_sha256(decoded_rgb, decoded_mask)].append(item)
            with Image.open(image_path) as im:
                iw, ih = im.size
            with Image.open(mask_path) as mm:
                mw, mh = mm.size
            if abs((iw / max(ih, 1)) - (mw / max(mh, 1))) > 1e-6:
                errors.append(
                    f"row {i}: image/mask aspect-ratio mismatch image={iw}x{ih} mask={mw}x{mh}"
                )
            elif (iw, ih) != (mw, mh) and r.get("alignment_verified") is not True:
                errors.append(
                    f"row {i}: native-resolution mismatch image={iw}x{ih} mask={mw}x{mh} requires alignment_verified=true after GT-only audit"
                )
            if (
                resize_size is not None
                and native_fg > 0
                and _resized_foreground_count(mask_path, resize_size) == 0
            ):
                errors.append(
                    f"row {i}: crack mask becomes empty after resize to {resize_size}x{resize_size}; do not treat as normal"
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

    def check_rgb(kind, groups):
        for digest, items in groups.items():
            paths = {x[3] for x in items}
            if len(paths) <= 1:
                continue
            splits = {x[1] for x in items}
            labels = {x[2] for x in items}
            if len(splits) > 1:
                errors.append(f"{kind} duplicate across splits: {digest} in {sorted(splits)}")
            if len(labels) > 1:
                errors.append(f"{kind} cross-label duplicate (normal vs crack): {digest}")
            if len(splits) == 1:
                errors.append(f"{kind} duplicate within split {next(iter(splits))}: {digest}")

    def check_mask(kind, groups):
        for digest, items in groups.items():
            if items and all(x[3] for x in items):
                continue
            splits = {x[1] for x in items}
            if len(splits) > 1:
                errors.append(
                    f"{kind} reused across splits: {digest} in {sorted(splits)} paths={sorted({x[2] for x in items})[:5]}"
                )

    def check_pair(groups):
        for digest, items in groups.items():
            paths = {x[3] for x in items}
            splits = {x[1] for x in items}
            if len(paths) > 1 and len(splits) > 1:
                errors.append(
                    f"decoded image-mask pair duplicate across splits: {digest} in {sorted(splits)}"
                )
            elif len(paths) > 1 and len(splits) == 1:
                errors.append(
                    f"decoded image-mask pair duplicate within split {next(iter(splits))}: {digest}"
                )

    check_rgb("raw-image", raw_rgb_rows)
    check_rgb("decoded-rgb", decoded_rgb_rows)
    check_mask("raw-mask", raw_mask_rows)
    check_mask("decoded-binary-mask", decoded_mask_rows)
    check_pair(pair_rows)

    # Five exact bands guarantee that every pair with Hamming distance <= 4
    # shares at least one band, avoiding an O(n^2) scan on large datasets.
    band_ranges = ((0, 13), (13, 26), (26, 39), (39, 52), (52, 64))
    buckets = defaultdict(list)
    for item in perceptual_rows:
        value = item[3]
        for band, (start, stop) in enumerate(band_ranges):
            width = stop - start
            key = (value >> (64 - stop)) & ((1 << width) - 1)
            buckets[(band, key)].append(item)
    seen_pairs = set()
    for candidates in buckets.values():
        for left_index in range(len(candidates)):
            left = candidates[left_index]
            for right in candidates[left_index + 1 :]:
                pair = tuple(sorted((left[0], right[0])))
                if pair in seen_pairs:
                    continue
                seen_pairs.add(pair)
                if left[1] == right[1] or left[2] == right[2]:
                    continue
                distance = (left[3] ^ right[3]).bit_count()
                color_distance = sum(
                    (a - b) ** 2 for a, b in zip(left[4], right[4])
                ) ** 0.5
                if distance <= 4 and color_distance <= 12.0:
                    errors.append(
                        "perceptual-rgb near-duplicate across splits: "
                        f"rows={pair} splits={sorted((left[1], right[1]))} "
                        f"dhash_distance={distance} mean_rgb_distance={color_distance:.3f}"
                    )

    for split in required_splits:
        if not any(r.get("split") == split for r in rows):
            errors.append(f"missing required split: {split}")
    if normal_policy in {"train", "train_and_aux_val"} and not any(
        r.get("split") == "normal_train" and r.get("is_normal") is True for r in rows
    ):
        errors.append("normal_train: no true-normal sample")
    if normal_policy == "train_and_aux_val" and not any(
        r.get("split") == "normal_val" and r.get("is_normal") is True for r in rows
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
    p.add_argument("--required-splits", nargs="+", default=None)
    p.add_argument("--certificate-out", default=None)
    p.add_argument(
        "--certificate-scope",
        choices=("full_benchmark", "training_view"),
        default=None,
    )
    a = p.parse_args()
    errors = audit(
        a.manifest,
        test_split=a.test_split,
        require_source_disjoint=a.require_source_disjoint,
        normal_policy=a.normal_policy,
        resize_size=a.resize_size,
        required_splits=a.required_splits,
    )
    if errors:
        print("G0 FAIL")
        print("\n".join(errors))
        raise SystemExit(2)
    print("G0 PASS")
    if a.certificate_out:
        if not a.certificate_scope:
            raise SystemExit("--certificate-scope is required with --certificate-out")
        out = Path(a.certificate_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        inventory_path = out.with_suffix(out.suffix + ".inventory.jsonl")
        inventory = dataset_inventory(a.manifest)
        inventory_path.write_text(
            "\n".join(json.dumps(r, ensure_ascii=False, sort_keys=True) for r in inventory)
            + ("\n" if inventory else "")
        )
        cert = {
            "status": "PASS",
            "scope": a.certificate_scope,
            "manifest": str(Path(a.manifest).resolve()),
            "manifest_sha256": _sha256_file(a.manifest),
            "dataset_content_sha256": dataset_content_sha256(a.manifest),
            "dataset_inventory": str(inventory_path.resolve()),
            "dataset_inventory_sha256": _sha256_file(inventory_path),
            "resize_size": a.resize_size,
            "normal_policy": a.normal_policy,
            "required_splits": list(
                a.required_splits or ("train", "val", a.test_split)
            ),
            "gate0_schema": 2,
            "perceptual_duplicate_contract": "dhash64<=4-and-mean-rgb-distance<=12-cross-split-v1",
        }
        out.write_text(json.dumps(cert, indent=2))
        print(json.dumps(cert, indent=2))


if __name__ == "__main__":
    main()
