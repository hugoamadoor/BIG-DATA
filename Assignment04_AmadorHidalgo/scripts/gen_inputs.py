#!/usr/bin/env python3
"""Generate the dense inputs of Assignment 3 and record them in data/manifest.csv (SHA-256).

data/A_<n>.bin, data/B_<n>.bin : n*n float64, little-endian, row-major, uniform [0, 1),
numpy.random.default_rng(seed) (PCG64), seed = 20261012 + n (A) and 20261013 + n (B).
Same generator and seeds as Assignments 1 and 2, so the files are byte-identical to theirs for
the sizes they share; the SHA-256 values can be compared with
Assignment02_AmadorHidalgo/data/manifest.csv.

Usage: python3 scripts/gen_inputs.py [--quick]
"""
import csv
import hashlib
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
SIZES = [16, 64, 100, 256, 512, 1024, 2048, 3072]
QUICK = [16, 64, 100, 256]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    DATA.mkdir(exist_ok=True)
    rows = []
    for n in (QUICK if "--quick" in sys.argv else SIZES):
        for which, base in (("A", 20261012), ("B", 20261013)):
            p = DATA / f"{which}_{n}.bin"
            seed = base + n
            if not p.exists():
                np.random.default_rng(seed).random((n, n), dtype=np.float64).astype("<f8").tofile(p)
            rows.append(dict(file=p.name, n=n, seed=seed, generator="numpy.random.default_rng (PCG64)",
                             distribution="uniform [0, 1)", dtype="float64 little-endian row-major",
                             bytes=p.stat().st_size, sha256=sha256(p)))
    with open(DATA / "manifest.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"manifest written: {DATA / 'manifest.csv'} ({len(rows)} files)")


if __name__ == "__main__":
    main()
