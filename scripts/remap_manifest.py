#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--old-prefix", required=True)
    parser.add_argument("--new-prefix", required=True)
    args = parser.parse_args()

    rows = []
    for line in Path(args.input).read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        for key in ("image", "mask"):
            if row.get(key):
                row[key] = str(row[key]).replace(args.old_prefix, args.new_prefix, 1)
        rows.append(row)
    Path(args.output).write_text("\n".join(json.dumps(row) for row in rows) + "\n")


if __name__ == "__main__":
    main()
