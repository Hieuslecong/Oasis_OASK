#!/usr/bin/env python3
"""Create the train/validation-only manifest consumed by official trainers."""
import argparse,hashlib,json
from pathlib import Path

def sha(path):
 h=hashlib.sha256()
 with open(path,"rb") as f:
  for c in iter(lambda:f.read(1024*1024),b""):h.update(c)
 return h.hexdigest()

def main():
 p=argparse.ArgumentParser();p.add_argument("--input",required=True);p.add_argument("--out",required=True);a=p.parse_args()
 rows=[json.loads(x) for x in Path(a.input).read_text().splitlines() if x.strip()];allowed={"train","val","normal_train","normal_val"};kept=[r for r in rows if r.get("split") in allowed]
 if not any(r.get("split")=="train" for r in kept) or not any(r.get("split")=="val" for r in kept):raise RuntimeError("training view must contain train and val")
 if any(r.get("split")=="test" for r in kept):raise RuntimeError("training view unexpectedly contains test")
 out=Path(a.out);out.parent.mkdir(parents=True,exist_ok=True);out.write_text("\n".join(json.dumps(r,ensure_ascii=False) for r in kept)+"\n")
 print(json.dumps({"input":str(Path(a.input).resolve()),"output":str(out.resolve()),"rows":len(kept),"splits":sorted({r.get('split') for r in kept}),"sha256":sha(out)},indent=2))
if __name__=="__main__":main()
