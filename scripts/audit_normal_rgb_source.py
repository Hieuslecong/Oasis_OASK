#!/usr/bin/env python3
"""Audit and prepare true-normal RGB for derived training only."""
import argparse,csv,hashlib,json,random
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw

EXT={".jpg",".jpeg",".png",".bmp",".tif",".tiff",".webp"}
def list_images(root): return sorted(p for p in Path(root).rglob("*") if p.is_file() and p.suffix.lower() in EXT)
def sha(path):
 h=hashlib.sha256()
 with open(path,"rb") as f:
  for c in iter(lambda:f.read(1024*1024),b""): h.update(c)
 return h.hexdigest()
def decoded(path):
 a=np.asarray(Image.open(path).convert("RGB"),dtype=np.uint8); h=hashlib.sha256(); h.update(str(a.shape).encode()); h.update(a.tobytes()); return h.hexdigest()
def inspect(paths,label):
 out=[]
 for p in paths:
  r={"path":str(Path(p).resolve()),"label":label}
  try:
   with Image.open(p) as im: im.verify()
   with Image.open(p) as im:
    rgb=im.convert("RGB"); r.update(width=rgb.width,height=rgb.height,mode=str(im.mode),raw_sha256=sha(p),decoded_rgb_sha256=decoded(p),status="ok")
  except Exception as e: r.update(status="corrupt",error=str(e))
  out.append(r)
 return out
def manifest_images(path):
 seen=set(); out=[]
 for line in Path(path).read_text().splitlines():
  if not line.strip(): continue
  r=json.loads(line)
  if r.get("is_normal") is True: continue
  p=Path(r["image"]).resolve()
  if p not in seen: seen.add(p); out.append(p)
 return sorted(out)
def dup_groups(rows,key):
 d=defaultdict(list)
 for r in rows:
  if r.get("status")=="ok": d[r[key]].append(r)
 return {k:v for k,v in d.items() if len({x["path"] for x in v})>1}
def cross_groups(rows,key):
 d=defaultdict(list)
 for r in rows:
  if r.get("status")=="ok": d[r[key]].append(r)
 return {k:v for k,v in d.items() if len({x["label"] for x in v})>1}
def sheet(paths,out,seed,count=100,thumb=160,cols=10):
 if not paths:return None
 paths=sorted(paths); rng=random.Random(seed)
 if len(paths)>count: paths=rng.sample(paths,count)
 rows=(len(paths)+cols-1)//cols; canvas=Image.new("RGB",(cols*thumb,rows*(thumb+24)),"white"); draw=ImageDraw.Draw(canvas)
 for i,p in enumerate(paths):
  with Image.open(p) as im: tile=im.convert("RGB"); tile.thumbnail((thumb,thumb),Image.Resampling.LANCZOS)
  x=(i%cols)*thumb;y=(i//cols)*(thumb+24);canvas.paste(tile,(x,y));draw.text((x+2,y+thumb+2),p.name[:24],fill="black")
 out.parent.mkdir(parents=True,exist_ok=True);canvas.save(out);return str(out)

def main():
 ap=argparse.ArgumentParser();ap.add_argument("--normal-root",required=True);ap.add_argument("--cracked-manifest",default=None);ap.add_argument("--cracked-reference-root",default=None);ap.add_argument("--out-dir",required=True);ap.add_argument("--repair-exclusions-out",default=None);ap.add_argument("--seed",type=int,default=1337);ap.add_argument("--contact-sheet-count",type=int,default=100);a=ap.parse_args()
 if a.cracked_manifest and a.cracked_reference_root: raise ValueError("choose one cracked reference source")
 normal=list_images(Path(a.normal_root).resolve())
 if not normal: raise RuntimeError("no normal images found")
 cracked=manifest_images(a.cracked_manifest) if a.cracked_manifest else (list_images(Path(a.cracked_reference_root).resolve()) if a.cracked_reference_root else [])
 rows=inspect(normal,"normal")+inspect(cracked,"cracked_reference"); nr=[r for r in rows if r["label"]=="normal"]
 raw=dup_groups(nr,"raw_sha256"); dec=dup_groups(nr,"decoded_rgb_sha256"); cr=cross_groups(rows,"raw_sha256"); cd=cross_groups(rows,"decoded_rgb_sha256"); corrupt=[r for r in nr if r.get("status")!="ok"]
 excluded={}
 def ex(path,reason): excluded.setdefault(str(Path(path).resolve()),set()).add(reason)
 if a.repair_exclusions_out:
  for r in corrupt: ex(r["path"],"corrupt")
  for digest,items in dec.items():
   for p in sorted({x["path"] for x in items})[1:]: ex(p,f"normal_decoded_duplicate:{digest}")
  for digest,items in raw.items():
   left=sorted({x["path"] for x in items if x["path"] not in excluded})
   for p in left[1:]: ex(p,f"normal_raw_duplicate:{digest}")
  for digest,items in list(cr.items())+list(cd.items()):
   for r in items:
    if r["label"]=="normal": ex(r["path"],f"cross_label_duplicate:{digest}")
 accepted=[r for r in nr if r.get("status")=="ok" and r["path"] not in excluded]
 if not accepted: raise RuntimeError("normal-source audit leaves zero usable normal images")
 out=Path(a.out_dir);out.mkdir(parents=True,exist_ok=True);fields=["path","label","status","error","width","height","mode","raw_sha256","decoded_rgb_sha256"]
 with (out/"inventory.csv").open("w",newline="") as f:
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
  for r in rows:w.writerow({k:r.get(k) for k in fields})
 if a.repair_exclusions_out:
  p=Path(a.repair_exclusions_out);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps({"excluded_normal_candidates":[{"path":k,"reasons":sorted(v)} for k,v in sorted(excluded.items())]},indent=2));status="PASS"
 else: status="PASS" if not(corrupt or raw or dec or cr or cd) else "FAIL"
 dims=Counter(f"{r['width']}x{r['height']}" for r in accepted); cs=sheet([Path(r["path"]) for r in accepted],out/"normal_contact_sheet_seed1337.jpg",a.seed,a.contact_sheet_count)
 summary={"normal_root":str(Path(a.normal_root).resolve()),"cracked_manifest":str(Path(a.cracked_manifest).resolve()) if a.cracked_manifest else None,"normal_candidates":len(normal),"cracked_reference_candidates":len(cracked),"corrupt_normal_count":len(corrupt),"normal_raw_duplicate_groups":len(raw),"normal_decoded_duplicate_groups":len(dec),"cross_label_raw_duplicate_groups":len(cr),"cross_label_decoded_duplicate_groups":len(cd),"derived_exclusions":len(excluded),"normal_accepted_after_exclusions":len(accepted),"normal_dimensions":dict(dims),"contact_sheet":cs,"status":status}
 (out/"summary.json").write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
 if status!="PASS":raise SystemExit(2)
if __name__=="__main__":main()
