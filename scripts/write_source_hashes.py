#!/usr/bin/env python3
"""Generate SHA256 inventory for reproducibility-critical source files."""
import argparse
import hashlib
from pathlib import Path

DEFAULTS = [
    "src/oasis_cycle_aosk/models.py",
    "src/oasis_cycle_aosk/data.py",
    "src/oasis_cycle_aosk/audit.py",
    "src/oasis_cycle_aosk/aosk.py",
    "src/oasis_rc_v2/losses.py",
    "src/oasis_cycle_aosk/samplers.py",
    "src/oasis_cycle_aosk/train_oasis_rc_v2.py",
    "src/oasis_cycle_aosk/evaluate_rc.py",
    "scripts/audit_normal_rgb_source.py",
    "scripts/add_normal_rgb_to_manifest.py",
    "scripts/diagnose_aux_gradients.py",
    "scripts/create_student_init.py",
    "scripts/run_smoke.sh",
    "scripts/run_all_seeds.sh",
    "scripts/preflight_real_host.py",
]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default=str(Path(__file__).resolve().parents[1]))
    p.add_argument("--out", required=True)
    args = p.parse_args()
    root = Path(args.root).resolve()
    lines = []
    for rel in DEFAULTS:
        path = root / rel
        if not path.exists():
            raise FileNotFoundError(path)
        lines.append(f"{sha256(path)}  {rel}")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    print(out)


if __name__ == "__main__":
    main()
