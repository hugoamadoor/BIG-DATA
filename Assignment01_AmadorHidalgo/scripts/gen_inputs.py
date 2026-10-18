#!/usr/bin/env python3
"""Generate the shared input matrices and verify them against data/manifest.csv.

Every language reads the SAME files, so all comparisons use identical logical inputs.

  A_<n>.bin, B_<n>.bin : n*n float64 values, little-endian, row-major,
                         uniform in [0, 1), NumPy PCG64 generator.
  Seeds                : A uses SEED_A, B uses SEED_B, both mixed with n so each
                         size has its own (but reproducible) pair.

Usage:
  python3 scripts/gen_inputs.py            # generate missing files, then verify
  python3 scripts/gen_inputs.py --write-manifest   # (re)create manifest.csv
"""
import csv
import hashlib
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
MANIFEST = DATA / "manifest.csv"

SEED_A = 20261012
SEED_B = 20261013
SIZES = [1, 16, 32, 64, 128, 256, 512, 768, 1000, 1024, 1025, 1536, 2048]   # 1000/1025: supplementary S2


def make(n, which):
    seed = (SEED_A if which == "A" else SEED_B) + n
    rng = np.random.default_rng(seed)
    return rng.random((n, n), dtype=np.float64)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    DATA.mkdir(exist_ok=True)
    rows = []
    for n in SIZES:
        for which in "AB":
            p = DATA / f"{which}_{n}.bin"
            if not p.exists():
                make(n, which).astype("<f8").tofile(p)
            rows.append((n, which, p.name, p.stat().st_size, sha256(p)))

    if "--write-manifest" in sys.argv or not MANIFEST.exists():
        with open(MANIFEST, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["n", "matrix", "file", "bytes", "sha256", "generator", "seed"])
            for n, which, name, size, digest in rows:
                seed = (SEED_A if which == "A" else SEED_B) + n
                w.writerow([n, which, name, size, digest, "numpy.random.default_rng (PCG64) .random()", seed])
        print(f"manifest written: {MANIFEST}")
        return 0

    ok = True
    with open(MANIFEST) as f:
        for r in csv.DictReader(f):
            p = DATA / r["file"]
            if not p.exists() or sha256(p) != r["sha256"]:
                print("MISMATCH:", r["file"])
                ok = False
    print("inputs verified against manifest" if ok else
          "inputs differ from the manifest (different NumPy version?). Results remain valid "
          "but are not bit-identical to the published ones.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
