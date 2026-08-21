#!/usr/bin/env python3
"""Create one canonical v2.1 student initialization shared by all arms."""
import argparse
import hashlib
import json
import random
import subprocess
from pathlib import Path

import numpy as np
import torch

from oasis_cycle_aosk.train_oasis_rc_v2 import make_student
from oasis_rc_v2.checkpoint import IMPLEMENTATION_VERSION, METHOD_VERSION


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def git_provenance(repo_root):
    """Return HEAD while failing closed on tracked source mutations.

    Experiment runners legitimately create untracked outputs (datasets,
    checkpoints, metrics) inside the checkout. Those files must not invalidate
    source provenance. Modified/staged/deleted *tracked* files still fail the
    contract, and a readable 40-character Git HEAD remains mandatory.
    """
    try:
        head = subprocess.check_output(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
        tracked_status = subprocess.check_output(
            [
                "git",
                "-C",
                str(repo_root),
                "status",
                "--porcelain",
                "--untracked-files=no",
            ],
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except Exception as exc:
        raise RuntimeError("shared initialization requires readable git provenance") from exc
    if len(head) != 40:
        raise RuntimeError(f"invalid git HEAD {head!r}")
    if tracked_status:
        raise RuntimeError(
            "shared initialization requires clean tracked source files; commit/freeze changes first"
        )
    return head


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--student-kind", default="mobilenetv3")
    p.add_argument("--student-width", type=int, default=16)
    args = p.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    head = git_provenance(repo_root)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    student = make_student(args.student_kind, args.student_width).cpu()
    metadata = {
        "method_version": METHOD_VERSION,
        "implementation_version": IMPLEMENTATION_VERSION,
        "student_kind": args.student_kind,
        "student_width": args.student_width,
        "seed": args.seed,
        "torch_version": torch.__version__,
        "git_head": head,
        "git_tracked_dirty": False,
        "untracked_experiment_outputs_allowed": True,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"student": student.state_dict(), **metadata}, out)
    summary = {
        "path": str(out.resolve()),
        "sha256": sha256_file(out),
        "parameter_count": sum(p.numel() for p in student.parameters()),
        **metadata,
    }
    out.with_suffix(out.suffix + ".json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
