#!/usr/bin/env python3
"""Generate every input of Assignment 2 and record it in data/manifest.csv (SHA-256).

Dense inputs (Part A)
  data/dense/A_<n>.bin, B_<n>.bin : n*n float64, little-endian, row-major, uniform [0, 1),
  numpy.random.default_rng(seed) (PCG64), seed = 20261012 + n (A) and 20261013 + n (B).
  These are the SAME generator and seeds as Assignment 1, so for the sizes both studies use
  the files are byte-identical to the Assignment 1 inputs.

Sparse inputs (Part B)
  data/sparse/<tag>_A.csr, <tag>_B.csr : CSR (int64 n, int64 nnz, int64 rowptr[n+1],
  int32 col[nnz], float64 val[nnz]); tag = <structure>_n<n>_d<density %>.
  For n <= 2048 (and n = 4096, 8192 of the large series) the same logical matrices are also written as dense files
  data/sparse/<tag>_A.bin, <tag>_B.bin so the dense methods and the conversion step can use them.
  Nonzero values: uniform in (0, 1] (1 - rng.random()), so no stored value is zero and,
  because all values are positive, no cancellation can create zeros in the product.
  Structures:
    uniform : every entry is nonzero independently with probability d (Bernoulli). Row counts
              are therefore Binomial(n, d); columns uniform without replacement.
    banded  : all entries with |i - j| <= w, w chosen so that the density is as close to d as
              possible. Same density as uniform, very different placement.
    skewed  : row nonzero counts follow a power law (count_r proportional to 1 / r for rank r,
              ranks randomly assigned to rows), scaled to the total d*n^2; rows that would
              exceed n are capped at n and the excess is spread over the other rows;
              columns uniform within each row.
  Seeds: 20261026 + 7*n + round(d*1e4) + structure offset, +0 for A and +1 for B.

Usage: python3 scripts/gen_inputs.py [--quick]   (--quick: only what the tests need)
"""
import csv
import hashlib
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
DENSE_MAX = 2048
DENSE_MAX_LARGE = 8192   # experiment L: dense files also for 4096 and 8192

DENSE_SIZES = [16, 64, 256, 512, 768, 1024, 1536, 2048, 3072, 4096]
# (structure, n, density in %)
MATCHED = [("uniform", n, d) for n in (1024, 2048)
           for d in (100, 85, 70, 50, 20, 10, 5, 2, 1, 0.5, 0.2, 0.1)]
STRUCTURE = [(s, 2048, d) for s in ("banded", "skewed") for d in (5, 1)]
SCALING_NNZ_PER_ROW = 16
SCALING = [("uniform", n, 100.0 * SCALING_NNZ_PER_ROW / n) for n in (4096, 8192, 16384, 32768, 65536, 131072)]
TEST = [(s, n, d) for s in ("uniform", "banded", "skewed") for n in (64, 200) for d in (30, 5)]
STRUCT_OFFSET = {"uniform": 0, "banded": 100003, "skewed": 200003}


def dense_matrix(n, which):
    seed = (20261012 if which == "A" else 20261013) + n
    return np.random.default_rng(seed).random((n, n), dtype=np.float64), seed


def tag(structure, n, d):
    return f"{structure}_n{n}_d{d:g}"


def sparse_seed(structure, n, d, which):
    return 20261026 + 7 * n + int(round(d * 1e4)) + STRUCT_OFFSET[structure] + (0 if which == "A" else 1)


def sparse_matrix(structure, n, d, seed):
    """Return (rowptr int64, col int32, val float64), columns sorted within each row."""
    rng = np.random.default_rng(seed)
    p = d / 100.0
    if structure == "banded":
        # nnz of a band of half-width w in an n x n matrix: n(2w+1) - w(w+1)
        target = p * n * n
        w = min(range(n), key=lambda w: abs(n * (2 * w + 1) - w * (w + 1) - target))
        rows, cols = [], []
        counts = np.empty(n, dtype=np.int64)
        for i in range(n):
            lo, hi = max(0, i - w), min(n - 1, i + w)
            counts[i] = hi - lo + 1
            cols.append(np.arange(lo, hi + 1, dtype=np.int32))
        col = np.concatenate(cols)
    else:
        if structure == "uniform":
            counts = rng.binomial(n, p, size=n).astype(np.int64)
        else:  # skewed
            weights = 1.0 / np.arange(1, n + 1)
            rng.shuffle(weights)
            # water-filling: rows that would exceed n are capped at n and their excess is
            # redistributed over the other rows in proportion to their weights
            target, capped = p * n * n, np.zeros(n, dtype=bool)
            while True:
                free = target - n * capped.sum()
                share = np.where(capped, n, weights * free / weights[~capped].sum())
                newly = (share > n) & ~capped
                if not newly.any():
                    break
                capped |= newly
            counts = np.minimum(np.round(share), n).astype(np.int64)
        cols = []
        for i in range(n):
            c = counts[i]
            if c == 0:
                cols.append(np.empty(0, dtype=np.int32))
            elif c > n // 4:
                cols.append(np.sort(rng.permutation(n)[:c]).astype(np.int32))
            else:  # sample without replacement cheaply for small rows
                s = np.unique(rng.integers(0, n, size=int(c * 1.2) + 8))
                while s.size < c:
                    s = np.unique(np.concatenate([s, rng.integers(0, n, size=c)]))
                cols.append(np.sort(rng.choice(s, size=c, replace=False)).astype(np.int32))
        col = np.concatenate(cols)
    rowptr = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(counts, out=rowptr[1:])
    val = 1.0 - rng.random(col.size)
    return rowptr, col, val


def write_csr(path, n, rowptr, col, val):
    with open(path, "wb") as f:
        np.array([n, col.size], dtype="<i8").tofile(f)
        rowptr.astype("<i8").tofile(f)
        col.astype("<i4").tofile(f)
        val.astype("<f8").tofile(f)


def read_csr(path):
    with open(path, "rb") as f:
        n, nnz = np.fromfile(f, dtype="<i8", count=2)
        rp = np.fromfile(f, dtype="<i8", count=n + 1)
        ci = np.fromfile(f, dtype="<i4", count=nnz)
        v = np.fromfile(f, dtype="<f8", count=nnz)
    return int(n), rp, ci, v


def to_dense(n, rowptr, col, val):
    D = np.zeros((n, n), dtype=np.float64)
    rows = np.repeat(np.arange(n), np.diff(rowptr))
    D[rows, col] = val
    return D


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    quick = "--quick" in sys.argv
    (DATA / "dense").mkdir(parents=True, exist_ok=True)
    (DATA / "sparse").mkdir(parents=True, exist_ok=True)
    rows = []
    for n in ([16, 64] if quick else DENSE_SIZES):
        for which in "AB":
            p = DATA / "dense" / f"{which}_{n}.bin"
            M, seed = dense_matrix(n, which)
            if not p.exists():
                M.astype("<f8").tofile(p)
            rows.append(dict(file=str(p.relative_to(DATA)), kind="dense", structure="dense", n=n,
                             density_pct=100, nnz=n * n, seed=seed))
            del M
    configs = TEST if quick else TEST + MATCHED + STRUCTURE + SCALING
    for structure, n, d in configs:
        for which in "AB":
            seed = sparse_seed(structure, n, d, which)
            base = f"{tag(structure, n, d)}_{which}"      # (no Path.with_suffix: tags contain dots)
            pc = DATA / "sparse" / (base + ".csr")
            if not pc.exists():
                rp, ci, v = sparse_matrix(structure, n, d, seed)
                write_csr(pc, n, rp, ci, v)
            _, rp, ci, v = read_csr(pc)
            if n <= DENSE_MAX or (n <= DENSE_MAX_LARGE and (structure, n, d) in SCALING):
                pb = DATA / "sparse" / (base + ".bin")
                if not pb.exists():
                    to_dense(n, rp, ci, v).astype("<f8").tofile(pb)
            rows.append(dict(file=str(pc.relative_to(DATA)), kind="csr", structure=structure, n=n,
                             density_pct=d, nnz=int(ci.size), seed=seed))
        print(f"  {tag(structure, n, d)}: nnz(A) = {rows[-2]['nnz']}, nnz(B) = {rows[-1]['nnz']}", flush=True)

    with open(DATA / "manifest.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["file", "kind", "structure", "n", "density_pct", "nnz",
                                          "achieved_density_pct", "seed", "generator", "bytes", "sha256"])
        w.writeheader()
        for r in rows:
            p = DATA / r["file"]
            r["achieved_density_pct"] = f"{100.0 * r['nnz'] / r['n'] ** 2:.6g}"
            r["generator"] = "numpy.random.default_rng (PCG64)"
            r["bytes"] = p.stat().st_size
            r["sha256"] = sha256(p)
            w.writerow(r)
    print(f"manifest written: {DATA / 'manifest.csv'} ({len(rows)} files)")


if __name__ == "__main__":
    main()
