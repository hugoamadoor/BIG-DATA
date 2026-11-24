#!/usr/bin/env python3
"""Correctness and thread-safety tests for Assignment 3. Must pass before benchmarking.

1. Known answers (hand-computed 3x3, identity, zero, exact integer product) for every binary,
   method and thread count.
2. Random inputs: within 2 gamma_n |A||B| of NumPy (gamma_n = n u / (1 - n u), u = 2^-53,
   Higham Thm 3.5; factor 2 = our error + the BLAS reference's error).
3. Bit-identical to the sequential baseline for every method (seq, tiles, omp) and binary (-O2, -O2 without OpenMP, -O3 scalar,
   -O3 AVX2), thread counts 1, 2, 3, 4, 8 and sizes whose tile count is smaller than, equal to or
   not a multiple of the thread count (n = 1, 33, 64, 100, 256 with tile 32).
4. Thread safety: the parallel method writes disjoint row tiles of C. A data race on C would make
   results depend on thread timing; 20 repeated runs with 8 threads must all give the same bytes.
5. The program reports the number of threads that executed the parallel region; it must equal
   OMP_NUM_THREADS.

Writes results/correctness.csv. Exit code 0 only if every check passes.
"""
import csv
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
BUILD, DATA, RES = ROOT / "build", ROOT / "data", ROOT / "results"
U = 2.0 ** -53
BINARIES = ["mm", "mm_nomp", "mm_scalar", "mm_simd"]
THREADS = [1, 2, 3, 4, 8]
PROTO = ["0", "1", "1", "0", "0", "60"]       # no warm-up, one sample, no batching, 60 s limit
rows, failures = [], 0


def record(name, ok, detail=""):
    global failures
    rows.append(dict(test=name, passed=bool(ok), detail=detail))
    if not ok:
        failures += 1
        print("FAIL", name, detail)


def run(binary, method, n, block, a, b, out, threads=1):
    env = dict(os.environ, OMP_NUM_THREADS=str(threads), OMP_DYNAMIC="false")
    r = subprocess.run([str(BUILD / binary), method, str(n), str(block), str(a), str(b), *PROTO, str(out)],
                       capture_output=True, text=True, env=env)
    if r.returncode != 0:
        raise RuntimeError(r.stderr)
    used = int(next(ln.split()[1] for ln in r.stdout.splitlines() if ln.startswith("THREADS")))
    return np.fromfile(out, dtype="<f8").reshape(n, n), used


def configs():
    for b in BINARIES:
        yield b, "seq", 1
        yield b, "tiles", 1
        if b != "mm_nomp":
            for p in THREADS:
                yield b, "omp", p


def main():
    tmp = Path(tempfile.mkdtemp(prefix="a3tests_"))

    def save(M, name):
        p = tmp / name
        np.ascontiguousarray(M, dtype="<f8").tofile(p)
        return p

    # 1. known answers
    A3 = np.array([[1, 2, 0], [0, 1, 0], [2, 0, 1]], dtype=float)
    B3 = np.array([[1, 0, 2], [3, 1, 0], [0, 4, 1]], dtype=float)
    C3 = np.array([[7, 2, 2], [3, 1, 0], [2, 4, 5]], dtype=float)     # computed by hand
    R = np.random.default_rng(1).random((70, 70))
    I6 = np.arange(1, 37, dtype=float).reshape(6, 6)
    cases = {"hand 3x3": (A3, B3, C3), "identity * R (n=70)": (np.eye(70), R, R),
             "R * identity (n=70)": (R, np.eye(70), R), "zero * R (n=70)": (np.zeros((70, 70)), R, np.zeros((70, 70))),
             "small integers 6x6 (exact)": (I6, I6.T.copy(), I6 @ I6.T)}
    for name, (A, B, C) in cases.items():
        n = A.shape[0]
        pa, pb = save(A, "a.bin"), save(B, "b.bin")
        for b, m, p in configs():
            out, used = run(b, m, n, 32, pa, pb, tmp / "c.bin", p)
            record(f"{b} {m} p={p}: {name}", np.array_equal(out, C))

    # 2-3. random inputs: bound vs NumPy and bitwise vs sequential baseline
    for n in (1, 33, 64, 100, 256):
        if (DATA / f"A_{n}.bin").exists():
            pa, pb = DATA / f"A_{n}.bin", DATA / f"B_{n}.bin"
        else:
            pa = save(np.random.default_rng(n).random((n, n)), "ra.bin")
            pb = save(np.random.default_rng(n + 1).random((n, n)), "rb.bin")
        A = np.fromfile(pa, dtype="<f8").reshape(n, n)
        B = np.fromfile(pb, dtype="<f8").reshape(n, n)
        ref = A @ B
        bound = 2 * (n * U / (1 - n * U)) * (np.abs(A) @ np.abs(B))
        base, _ = run("mm", "seq", n, 32, pa, pb, tmp / "c0.bin")
        for b, m, p in configs():
            out, used = run(b, m, n, 32, pa, pb, tmp / "c.bin", p)
            err = float(np.abs(out - ref).max())
            record(f"{b} {m} p={p} n={n}: within 2*gamma_n*|A||B| of NumPy", np.all(np.abs(out - ref) <= bound),
                   f"max abs err {err:.3e}")
            record(f"{b} {m} p={p} n={n}: bit-identical to sequential baseline", out.tobytes() == base.tobytes())
            if m == "omp":
                record(f"{b} {m} p={p} n={n}: reported threads = OMP_NUM_THREADS", used == p, f"reported {used}")

    # 4. repeated runs with 8 threads (oversubscribed) must be identical
    pa, pb = DATA / "A_256.bin", DATA / "B_256.bin"
    first, _ = run("mm", "omp", 256, 32, pa, pb, tmp / "r.bin", 8)
    same = all(run("mm", "omp", 256, 32, pa, pb, tmp / "r.bin", 8)[0].tobytes() == first.tobytes()
               for _ in range(20))
    record("mm omp p=8 n=256: 20 repeated runs bit-identical (no race on C)", same)

    RES.mkdir(exist_ok=True)
    with open(RES / "correctness.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["test", "passed", "detail"])
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows) - failures}/{len(rows)} checks passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
