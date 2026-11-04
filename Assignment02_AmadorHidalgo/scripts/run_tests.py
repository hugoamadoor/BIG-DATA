#!/usr/bin/env python3
"""Correctness tests for every method of Assignment 2. Must pass before benchmarking.

Reference: NumPy (A @ B, BLAS) for dense values, SciPy CSR (A @ B) for sparse structure;
both are used only for checking and are not benchmarked as student implementations.

Tolerance: the componentwise forward-error bound of recursive summation of n products
  |C_hat - C| <= gamma_n |A| |B|,  gamma_n = n u / (1 - n u),  u = 2^-53  [Higham, Thm 3.5]
applies to our kernels AND to the BLAS reference, so the two may differ by up to 2 gamma_n |A||B|.
In addition every optimised dense method and the sparse method must reproduce the baseline
BIT FOR BIT: they add the same products in the same order (see the source headers).

Writes results/correctness.csv. Exit code 0 only if every check passes.
"""
import csv
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gen_inputs import TEST, read_csr, tag, to_dense, write_csr  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
BUILD, DATA, RES = ROOT / "build", ROOT / "data", ROOT / "results"
U = 2.0 ** -53
METHODS = ["ijk", "ijk_bt", "ikj", "blocked"]
BLOCKS = [1, 7, 64, 256]
PROTO = ["0", "1", "1", "0", "0", "60"]          # no warm-up, one sample, no batching, 60 s limit
rows, failures = [], 0


def gamma(k):
    return k * U / (1 - k * U)


def record(name, ok, detail=""):
    global failures
    rows.append(dict(test=name, passed=ok, detail=detail))
    if not ok:
        failures += 1
        print("FAIL", name, detail)


def run_dense(method, n, block, a, b, out):
    r = subprocess.run([str(BUILD / "dense"), method, str(n), str(block), str(a), str(b), *PROTO, str(out)],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(r.stderr)
    return np.fromfile(out, dtype="<f8").reshape(n, n)


def run_sparse(mode, n, a, b, out):
    r = subprocess.run([str(BUILD / "sparse"), mode, str(n), str(a), str(b), *PROTO, str(out)],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(r.stderr)
    nnz = next(ln.split() for ln in r.stdout.splitlines() if ln.startswith("NNZ"))
    return np.fromfile(out, dtype="<f8").reshape(n, n), int(nnz[3])


def within_bound(C, A, B):
    n = A.shape[0]
    ref = A @ B
    bound = 2 * gamma(n) * (np.abs(A) @ np.abs(B))
    err = np.abs(C - ref)
    return bool(np.all(err <= bound)), float(err.max(initial=0.0))


def main():
    tmp = Path(tempfile.mkdtemp(prefix="a2tests_"))

    def save(M, name):
        p = tmp / name
        np.ascontiguousarray(M, dtype="<f8").tofile(p)
        return p

    # ---------------- dense: known answers ----------------
    A3 = np.array([[1, 2, 0], [0, 1, 0], [2, 0, 1]], dtype=float)
    B3 = np.array([[1, 0, 2], [3, 1, 0], [0, 4, 1]], dtype=float)
    C3 = np.array([[7, 2, 2], [3, 1, 0], [2, 4, 5]], dtype=float)   # computed by hand
    cases = {"hand 3x3": (A3, B3, C3)}
    rng = np.random.default_rng(1)
    for n in (1, 5, 70):
        R = rng.random((n, n))
        cases[f"identity * R (n={n})"] = (np.eye(n), R, R)
        cases[f"R * identity (n={n})"] = (R, np.eye(n), R)
        cases[f"zero * R (n={n})"] = (np.zeros((n, n)), R, np.zeros((n, n)))
    I = np.arange(1, 37, dtype=float).reshape(6, 6)
    cases["small integers 6x6 (exact)"] = (I, I.T.copy(), I @ I.T)
    for name, (A, B, C) in cases.items():
        n = A.shape[0]
        pa, pb = save(A, "a.bin"), save(B, "b.bin")
        for m in METHODS:
            for blk in (BLOCKS if m == "blocked" else [64]):
                out = run_dense(m, n, blk, pa, pb, tmp / "c.bin")
                record(f"dense {m} block={blk}: {name}", np.array_equal(out, C))
    # ---------------- dense: random inputs vs NumPy + bitwise vs baseline ----------------
    for n in (16, 64, 100, 200):
        if (DATA / "dense" / f"A_{n}.bin").exists():
            pa, pb = DATA / "dense" / f"A_{n}.bin", DATA / "dense" / f"B_{n}.bin"
        else:
            pa = save(np.random.default_rng(n).random((n, n)), "ra.bin")
            pb = save(np.random.default_rng(n + 1).random((n, n)), "rb.bin")
        A = np.fromfile(pa, dtype="<f8").reshape(n, n)
        B = np.fromfile(pb, dtype="<f8").reshape(n, n)
        base = run_dense("ijk", n, 64, pa, pb, tmp / "c0.bin")
        ok, e = within_bound(base, A, B)
        record(f"dense ijk n={n}: within 2*gamma_n*|A||B| of NumPy", ok, f"max abs err {e:.3e}")
        for m in METHODS[1:]:
            for blk in (BLOCKS if m == "blocked" else [64]):
                out = run_dense(m, n, blk, pa, pb, tmp / "c.bin")
                ok, e = within_bound(out, A, B)
                record(f"dense {m} block={blk} n={n}: within bound of NumPy", ok, f"max abs err {e:.3e}")
                record(f"dense {m} block={blk} n={n}: bit-identical to baseline",
                       base.tobytes() == out.tobytes())

    # ---------------- sparse: edge cases ----------------
    n = 50
    R = sp.random(n, n, density=0.1, random_state=3, format="csr", data_rvs=lambda k: 1 - np.random.default_rng(4).random(k))
    edge = {"identity * R": (sp.identity(n, format="csr"), R),
            "R * identity": (R, sp.identity(n, format="csr")),
            "zero * R": (sp.csr_matrix((n, n)), R),
            "R with empty rows * R": (sp.csr_matrix(R.multiply(np.arange(n)[:, None] % 3 != 0)), R)}
    edge["R with empty rows * R"][0].eliminate_zeros()      # drop the explicit zeros multiply() keeps
    for name, (SA, SB) in edge.items():
        for SM, nm in ((SA, "a"), (SB, "b")):
            SM = sp.csr_matrix(SM)
            SM.sort_indices()
            write_csr(tmp / f"{nm}.csr", n, SM.indptr.astype(np.int64), SM.indices.astype(np.int32),
                      SM.data.astype(np.float64))
        out, nnzc = run_sparse("csr", n, tmp / "a.csr", tmp / "b.csr", tmp / "c.bin")
        ref = (SA @ SB).toarray()
        record(f"sparse csr: {name}", np.array_equal(out, ref) and nnzc == sp.csr_matrix(SA @ SB).nnz)

    # ---------------- sparse: generated test inputs (all three structures) ----------------
    for structure, n, d in TEST:
        t = tag(structure, n, d)
        pa, pb = DATA / "sparse" / f"{t}_A.csr", DATA / "sparse" / f"{t}_B.csr"
        _, rpa, cia, va = read_csr(pa)
        _, rpb, cib, vb = read_csr(pb)
        A, B = to_dense(n, rpa, cia, va), to_dense(n, rpb, cib, vb)
        SA = sp.csr_matrix((va, cia, rpa), shape=(n, n))
        SB = sp.csr_matrix((vb, cib, rpb), shape=(n, n))
        ref_nnz = sp.csr_matrix(SA @ SB).nnz
        base = run_dense("ijk", n, 64, DATA / "sparse" / f"{t}_A.bin", DATA / "sparse" / f"{t}_B.bin", tmp / "c0.bin")
        for mode, a, b in (("csr", pa, pb),
                           ("dense", DATA / "sparse" / f"{t}_A.bin", DATA / "sparse" / f"{t}_B.bin")):
            out, nnzc = run_sparse(mode, n, a, b, tmp / "c.bin")
            ok, e = within_bound(out, A, B)
            record(f"sparse {mode}-input {t}: within bound of NumPy", ok, f"max abs err {e:.3e}")
            record(f"sparse {mode}-input {t}: bit-identical to dense baseline", out.tobytes() == base.tobytes())
            record(f"sparse {mode}-input {t}: nnz(C) = SciPy nnz ({ref_nnz})", nnzc == ref_nnz, f"got {nnzc}")

    RES.mkdir(exist_ok=True)
    with open(RES / "correctness.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["test", "passed", "detail"])
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows) - failures}/{len(rows)} checks passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
