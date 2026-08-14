import json, csv, sys, random
from pathlib import Path
from PIL import Image
import numpy as np

random.seed(1337)
clean = Path(sys.argv[1])
out_dir = Path(sys.argv[2])
out_dir.mkdir(parents=True, exist_ok=True)

rows = [json.loads(l) for l in open(clean) if l.strip()]
empty = list(csv.DictReader(open(out_dir.parent / "empty_mask_audit.csv")))

# index empty rows by source -> split -> list
by_src_split = {}
for e in empty:
    by_src_split.setdefault(e["source_id"], {}).setdefault(e["split"], []).append(e)

TILE = 256
COLS = 6

def sheet(images, title):
    n = len(images)
    rows_n = (n + COLS - 1) // COLS
    sheet_img = Image.new("RGB", (COLS * TILE, rows_n * TILE), (255, 255, 255))
    for idx, im in enumerate(images):
        r, c = divmod(idx, COLS)
        try:
            im2 = im.convert("RGB").resize((TILE, TILE), Image.BILINEAR)
        except Exception:
            im2 = Image.new("RGB", (TILE, TILE), (200, 200, 200))
        sheet_img.paste(im2, (c * TILE, r * TILE))
    return sheet_img

for src, sp in by_src_split.items():
    for split, items in sp.items():
        # sample up to 24 representative images per source/split
        sample = items if len(items) <= 24 else random.sample(items, 24)
        imgs = []
        paths = []
        for e in sample:
            try:
                imgs.append(Image.open(e["image"]))
                paths.append(Path(e["image"]).name)
            except Exception:
                pass
        if not imgs:
            continue
        sh = sheet(imgs, f"{src}/{split}")
        fp = out_dir / f"sheet_{src}__{split}.png"
        sh.save(fp)
        # write companion list of filenames
        (out_dir / f"sheet_{src}__{split}.txt").write_text("\n".join(paths))

print("sheets written to", out_dir)
print("count:", sum(1 for _ in out_dir.glob("sheet_*.png")))
