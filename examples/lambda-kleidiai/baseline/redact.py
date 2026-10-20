"""Copy a run directory into the repo with deployment identifiers redacted. Standard library only.

  python redact.py <run-dir> <dest-dir> --bucket <bucket-name> [--account <id>]

Rewrites every text and .jsonl.gz file, replacing the bucket name with `<bucket>` and the account id with
`<account>`. Fails if either string survives anywhere in the destination. The repository is public; the deployment
state (build/deployment.json) stays out of it.
"""

from __future__ import annotations

import argparse
import gzip
import os
import shutil
import sys


def scrub(text: str, subs: dict[str, str]) -> str:
    for k, v in subs.items():
        text = text.replace(k, v)
    return text


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dest")
    ap.add_argument("--bucket", required=True)
    ap.add_argument("--account")
    a = ap.parse_args()
    subs = {a.bucket: "<bucket>"}
    if a.account:
        subs[a.account] = "<account>"
    if os.path.isfile(a.src):  # one file (reference/outputs.json)
        os.makedirs(os.path.dirname(os.path.abspath(a.dest)), exist_ok=True)
        with open(a.src, encoding="utf-8") as i:
            data = scrub(i.read(), subs)
        if any(k in data for k in subs):
            sys.exit(f"{a.dest} would still contain a redacted value")
        with open(a.dest, "w", encoding="utf-8") as o:
            o.write(data)
        print("redacted copy at", a.dest)
        return
    if os.path.exists(a.dest):
        shutil.rmtree(a.dest)
    for root, _, files in os.walk(a.src):
        out_root = os.path.join(a.dest, os.path.relpath(root, a.src))
        os.makedirs(out_root, exist_ok=True)
        for f in files:
            src, dst = os.path.join(root, f), os.path.join(out_root, f)
            if f.endswith(".gz"):
                with gzip.open(src, "rt") as i, gzip.open(dst, "wt") as o:
                    for line in i:
                        o.write(scrub(line, subs))
            else:
                with open(src, encoding="utf-8") as i, open(dst, "w", encoding="utf-8") as o:
                    o.write(scrub(i.read(), subs))
    for root, _, files in os.walk(a.dest):
        for f in files:
            p = os.path.join(root, f)
            data = (
                gzip.open(p, "rt").read() if f.endswith(".gz") else open(p, encoding="utf-8").read()
            )
            for k in subs:
                if k in data:
                    sys.exit(f"{p} still contains a redacted value")
    print("redacted copy at", a.dest)


if __name__ == "__main__":
    main()
