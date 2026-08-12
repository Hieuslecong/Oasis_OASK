#!/usr/bin/env python3
"""Create one canonical student initialization shared by all experimental arms."""
import argparse
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch

from oasis_cycle_aosk.train_oasis_rc_v2 import make_student


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--student-kind", default="multiscale")
    p.add_argument("--student-width", type=int, default=16)
    args = p.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    student = make_student(args.student_kind, args.student_width).cpu()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "student": student.state_dict(),
            "student_kind": args.student_kind,
            "student_width": args.student_width,
            "seed": args.seed,
        },
        out,
    )
    summary = {
        "path": str(out.resolve()),
        "sha256": sha256_file(out),
        "seed": args.seed,
        "student_kind": args.student_kind,
        "student_width": args.student_width,
        "parameter_count": sum(p.numel() for p in student.parameters()),
    }
    out.with_suffix(out.suffix + ".json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
