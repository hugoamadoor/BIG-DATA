#!/usr/bin/env python3
"""Correctness tests. Run BEFORE benchmarking (run_benchmarks.py refuses to start otherwise).

1. Known answers (exact, integer-valued doubles, so equality must be exact):
   identity * M, M * identity, zero * M, 2x2 and 3x3 hand-computed products, n = 1.
2. Random matrices in [0,1) of several sizes, FULL matrix compared with a trusted
   reference (NumPy/BLAS) using a tolerance derived below, and compared bit-for-bit
   between the three implementations.
3. Checksum / spot values reported by the programs on the shared benchmark inputs
   are checked in run_benchmarks.py with the same tolerance.

Tolerance. For a dot product of length n of non-negative terms, the standard bound
for recursive summation (Higham 2002, Thm 3.1 / Eq. 3.4) is |c_hat - c| <= gamma_n * c with
gamma_n = n*u / (1 - n*u), u = 2^-53. Two independent evaluation orders (ours vs BLAS)
can therefore differ by at most 2*gamma_n relative. That is the relative tolerance used
(rtol = 2*gamma_n, atol = 0) for random inputs in [0,1). Integer test data is exact.
"""
import csv
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
U = 2.0 ** -53


def rtol_for(n):
    g = n * U / (1.0 - n * U)
    return 2.0 * g


def commands(out_bin):
    """name -> function(n, A, B, out) -> argv. Quick settings: no warm-up, one sample."""
    tail = ["0", "1", "1", "1000", "0", "1000"]      # warmup_s reps min_reps budget batch limit
    return {
        "C": lambda n, A, B, o: [str(ROOT / "build/c/matmul_O2"), str(n), str(A), str(B), *tail, str(o)],
        "Java": lambda n, A, B, o: ["java", "-XX:+UseSerialGC", "-cp", str(ROOT / "build/java"), "MatMul",
                                    str(n), str(A), str(B), *tail, str(o)],
        "Python": lambda n, A, B, o: [sys.executable, str(ROOT / "src/python/matmul.py"),
                                      str(n), str(A), str(B), *tail, str(o)],
    }


def write(path, m):
    np.ascontiguousarray(m, dtype="<f8").tofile(path)


def run_all(tmp, n, A, B):
    """Run the three programs on A,B (n x n) and return their output matrices."""
    pa, pb = tmp / "A.bin", tmp / "B.bin"
    write(pa, A)
    write(pb, B)
    outs = {}
    for name, mk in commands(tmp).items():
        po = tmp / f"C_{name}.bin"
        if po.exists():
            po.unlink()
        r = subprocess.run(mk(n, pa, pb, po), capture_output=True, text=True)
        if r.returncode != 0 or not po.exists():
            raise RuntimeError(f"{name} failed for n={n}: {r.stderr[-500:]}")
        outs[name] = np.fromfile(po, dtype="<f8").reshape(n, n)
    return outs


def main():
    results = []   # (test, n, language, max_abs, max_rel, passed)
    failures = 0
    rng = np.random.default_rng(7)
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)

        # ---- 1. known answers (exact) ----
        known = []
        M = rng.integers(-5, 6, size=(7, 7)).astype(float)
        I, Z = np.eye(7), np.zeros((7, 7))
        known += [("identity*M", 7, I, M, M), ("M*identity", 7, M, I, M), ("zero*M", 7, Z, M, Z),
                  ("M*zero", 7, M, Z, Z)]
        A2 = np.array([[1., 2.], [3., 4.]]); B2 = np.array([[5., 6.], [7., 8.]])
        known.append(("2x2 by hand", 2, A2, B2, np.array([[19., 22.], [43., 50.]])))
        A3 = np.array([[2., 0., 1.], [1., 3., 2.], [0., 1., 4.]])
        B3 = np.array([[1., 2., 0.], [0., 1., 3.], [5., 0., 1.]])
        known.append(("3x3 by hand", 3, A3, B3, np.array([[7., 4., 1.], [11., 5., 11.], [20., 1., 7.]])))
        known.append(("1x1", 1, np.array([[3.]]), np.array([[-4.]]), np.array([[-12.]])))
        for name, n, A, B, expect in known:
            assert np.array_equal(A @ B, expect), f"reference check of '{name}' is wrong"
            outs = run_all(tmp, n, A, B)
            for lang, C in outs.items():
                ok = np.array_equal(C, expect)
                err = float(np.max(np.abs(C - expect)))
                results.append((name, n, lang, err, 0.0, ok))
                failures += (not ok)

        # ---- 2. random inputs: full-matrix comparison vs NumPy + bitwise cross-language ----
        for n in (2, 3, 5, 16, 33, 64, 100, 128):
            A, B = rng.random((n, n)), rng.random((n, n))
            ref = A @ B
            outs = run_all(tmp, n, A, B)
            tol = rtol_for(n)
            for lang, C in outs.items():
                abs_err = float(np.max(np.abs(C - ref)))
                rel_err = float(np.max(np.abs(C - ref) / np.abs(ref)))
                ok = rel_err <= tol
                results.append(("random vs NumPy", n, lang, abs_err, rel_err, ok))
                failures += (not ok)
            same = (outs["C"].tobytes() == outs["Java"].tobytes() == outs["Python"].tobytes())
            results.append(("bitwise C==Java==Python", n, "all", 0.0, 0.0, same))
            failures += (not same)

    out = ROOT / "results" / "correctness.csv"
    out.parent.mkdir(exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["test", "n", "language", "max_abs_error", "max_rel_error", "rtol_used", "passed"])
        for t, n, lang, a, r, ok in results:
            w.writerow([t, n, lang, f"{a:.3e}", f"{r:.3e}",
                        f"{rtol_for(n):.3e}" if t == "random vs NumPy" else "exact", int(ok)])
    total = len(results)
    print(f"{total - failures}/{total} correctness checks passed  ->  {out}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
