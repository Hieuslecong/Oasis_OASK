import json, csv, collections, sys, traceback
from pathlib import Path
from PIL import Image
import numpy as np

clean = sys.argv[1]
RESIZE = 256
rows = [json.loads(l) for l in open(clean) if l.strip()]

def safe_open(p):
    return Image.open(p)

audit_rows = []          # per-row record for empty/resize-empty rows
e_counts = collections.Counter()
e_by_split = collections.Counter()
src_total = collections.Counter()
src_empty = collections.Counter()
src_empty_split = collections.defaultdict(collections.Counter)

def resized_fg(p):
    m = Image.open(p).convert("L").resize((RESIZE, RESIZE), resample=Image.Resampling.NEAREST)
    return int((np.asarray(m, dtype=np.uint8) > 127).sum())

for i, r in enumerate(rows):
    split = r.get("split")
    src = r.get("source_id")
    src_total[src] += 1
    is_normal = (r.get("is_normal") is True)
    if is_normal:
        continue
    mp = r.get("mask")
    if not mp or not Path(mp).exists():
        cat = "E3_UNRESOLVED"   # missing mask for crack row
        e_counts[cat] += 1
        e_by_split[split] += 1
        src_empty[src] += 1
        src_empty_split[src][split] += 1
        audit_rows.append(dict(row_id=i, image=r["image"], mask=mp, split=split,
            source_id=src, lineage_id=r.get("lineage_id"),
            native_image_size="", native_mask_size="", native_foreground_pixels=-1,
            foreground_pixels_at_256=-1, category=cat))
        continue
    try:
        im = Image.open(r["image"]); m = Image.open(mp)
        iw, ih = im.size; mw, mh = m.size
        binmask = np.asarray(m.convert("L"), dtype=np.uint8)
        native_fg = int((binmask > 127).sum())
        if native_fg == 0:
            cat = "E0_TRUE_NATIVE_EMPTY"
        else:
            rf = resized_fg(mp)
            if rf == 0:
                cat = "E1_POSITIVE_NATIVE_BUT_EMPTY_AFTER_RESIZE"
            else:
                cat = None  # positive native, fine -> not an empty-mask issue
        if cat is None:
            continue
        e_counts[cat] += 1
        e_by_split[split] += 1
        src_empty[src] += 1
        src_empty_split[src][split] += 1
        audit_rows.append(dict(
            row_id=i, image=r["image"], mask=mp, split=split,
            source_id=src, lineage_id=r.get("lineage_id"),
            native_image_size=f"{iw}x{ih}", native_mask_size=f"{mw}x{mh}",
            native_foreground_pixels=native_fg,
            foreground_pixels_at_256=(resized_fg(mp) if native_fg > 0 else 0),
            category=cat))
    except Exception as e:
        cat = "E2_MASK_READ_PARSE_ERROR"
        e_counts[cat] += 1
        e_by_split[split] += 1
        src_empty[src] += 1
        src_empty_split[src][split] += 1
        audit_rows.append(dict(row_id=i, image=r.get("image"), mask=mp, split=split,
            source_id=src, lineage_id=r.get("lineage_id"),
            native_image_size="", native_mask_size="", native_foreground_pixels=-1,
            foreground_pixels_at_256=-1, category=f"{cat}:{type(e).__name__}"))

# write per-row audit (only empty/resize-empty rows)
out_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(clean).parent
out_dir.mkdir(parents=True, exist_ok=True)
with open(out_dir / "empty_mask_audit.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["row_id","image","mask","split","source_id",
        "lineage_id","native_image_size","native_mask_size","native_foreground_pixels",
        "foreground_pixels_at_256","category"])
    w.writeheader()
    w.writerows(audit_rows)

# source table
with open(out_dir / "empty_mask_by_source.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["source_id","total_rows","empty_native_count","empty_fraction",
                "train_empty","val_empty","test_empty"])
    for s in sorted(src_total):
        tot = src_total[s]; em = src_empty[s]
        sp = src_empty_split[s]
        w.writerow([s, tot, em, f"{em/max(tot,1):.4f}", sp.get("train",0),
                    sp.get("val",0), sp.get("test",0)])

summary = {
    "total_crack_rows": sum(1 for r in rows if r.get("is_normal") is not True),
    "E_counts": dict(e_counts),
    "E_by_split": dict(e_by_split),
    "empty_rows_written": len(audit_rows),
}
print(json.dumps(summary, indent=2))
print("--- source table ---")
for s in sorted(src_total):
    sp = src_empty_split[s]
    print(f"  {s:10s} total={src_total[s]:6d} empty={src_empty[s]:5d} "
          f"tr={sp.get('train',0):4d} va={sp.get('val',0):4d} te={sp.get('test',0):4d}")
