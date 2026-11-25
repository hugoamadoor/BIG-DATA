#!/usr/bin/env python3
"""Correctness tests for Assignment 4. Must pass before benchmarking.

1. The block kernel (build/libmmkernel.so) called directly: known answers, and mm_rows on the whole
   matrix bit-identical to the Assignment 3 sequential program (build/mm seq) - so the kernel inside
   the Spark tasks is the validated sequential kernel.
2. The reference programs (build/mm seq/omp, build/dense ijk) within 2 gamma_n |A||B| of NumPy.
3. On a running 2-worker Spark cluster: rowblock with P = 1, 3, 8 and more partitions than row tiles,
   for n = 1, 33, 64, 100, 256 and known-answer matrices, bit-identical to the sequential kernel;
   block2d with g = 1, 2, 3 within 2 gamma_n |A||B| of NumPy.
Writes results/correctness.csv. Exit code 0 only if every check passes.
"""
import csv
import ctypes
import os
import subprocess
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src" / "spark"))
BUILD, DATA, RES = ROOT / "build", ROOT / "data", ROOT / "results"
U = 2.0 ** -53
rows, failures = [], 0


def record(name, ok, detail=""):
    global failures
    rows.append(dict(test=name, passed=bool(ok), detail=detail))
    if not ok:
        failures += 1
        print("FAIL", name, detail)


def bound_ok(C, A, B):
    n = A.shape[0]
    return bool(np.all(np.abs(C - A @ B) <= 2 * (n * U / (1 - n * U)) * (np.abs(A) @ np.abs(B))))


lib = ctypes.CDLL(str(BUILD / "libmmkernel.so"))
P = ctypes.POINTER(ctypes.c_double)
lib.mm_rows.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, P, P, P]
lib.mm_acc.argtypes = [ctypes.c_int] * 4 + [P, P, P]
ptr = lambda a: a.ctypes.data_as(P)


def seq_kernel(A, B):
    n = A.shape[0]
    C = np.empty((n, n))
    lib.mm_rows(n, n, 32, ptr(np.ascontiguousarray(A)), ptr(np.ascontiguousarray(B)), ptr(C))
    return C


def main():
    tmp = Path(tempfile.mkdtemp(prefix="a4tests_"))
    A3 = np.array([[1, 2, 0], [0, 1, 0], [2, 0, 1]], dtype=float)
    B3 = np.array([[1, 0, 2], [3, 1, 0], [0, 4, 1]], dtype=float)
    C3 = np.array([[7, 2, 2], [3, 1, 0], [2, 4, 5]], dtype=float)      # computed by hand
    R = np.random.default_rng(1).random((70, 70))
    known = {"hand 3x3": (A3, B3, C3), "identity * R": (np.eye(70), R, R), "zero * R": (np.zeros((70, 70)), R, np.zeros((70, 70)))}

    # 1. kernel
    for name, (A, B, C) in known.items():
        record(f"kernel mm_rows: {name}", np.array_equal(seq_kernel(A, B), C))
        Cacc = np.zeros_like(C)
        n = A.shape[0]
        lib.mm_acc(n, n, n, 32, ptr(np.ascontiguousarray(A)), ptr(np.ascontiguousarray(B)), ptr(Cacc))
        record(f"kernel mm_acc: {name}", np.array_equal(Cacc, C))
    rand = {}
    for n in (1, 33, 64, 100, 256):
        if (DATA / f"A_{n}.bin").exists():
            A = np.fromfile(DATA / f"A_{n}.bin").reshape(n, n)
            B = np.fromfile(DATA / f"B_{n}.bin").reshape(n, n)
        else:
            A, B = np.random.default_rng(n).random((n, n)), np.random.default_rng(n + 1).random((n, n))
        rand[n] = (A, B)
        pa, pb, pc = tmp / "a.bin", tmp / "b.bin", tmp / "c.bin"
        A.tofile(pa)
        B.tofile(pb)
        Ck = seq_kernel(A, B)
        r = subprocess.run([str(BUILD / "mm"), "seq", str(n), "32", str(pa), str(pb), "0", "1", "1", "0", "0", "60", str(pc)],
                           capture_output=True, text=True, env=dict(os.environ, OMP_NUM_THREADS="1"))
        Cs = np.fromfile(pc).reshape(n, n)
        record(f"kernel n={n}: bit-identical to Assignment 3 sequential program", r.returncode == 0 and Ck.tobytes() == Cs.tobytes())
        record(f"kernel n={n}: within bound of NumPy", bound_ok(Ck, A, B))
        # 2. references
        for prog, meth, thr in (("mm", "omp", 2), ("dense", "ijk", 1)):
            arg = "32" if prog == "mm" else "0"
            r = subprocess.run([str(BUILD / prog), meth, str(n), arg, str(pa), str(pb), "0", "1", "1", "0", "0", "60", str(pc)],
                               capture_output=True, text=True, env=dict(os.environ, OMP_NUM_THREADS=str(thr)))
            record(f"reference {prog}:{meth} n={n}: within bound of NumPy",
                   r.returncode == 0 and bound_ok(np.fromfile(pc).reshape(n, n), A, B))

    # 3. Spark
    from cluster import LocalCluster
    from session import start_app
    import spark_matmul as sm
    cl = LocalCluster(2, 1, "1600m", BUILD / "cluster_tests")
    try:
        cl.start()
        spark, _, _ = start_app(2, 1)
        sc = spark.sparkContext
        cases = {f"known {k}": v[:2] for k, v in known.items()}
        cases.update({f"random n={n}": v for n, v in rand.items()})
        for name, (A, B) in cases.items():
            n = A.shape[0]
            ref = seq_kernel(A, B)
            for parts in (1, 3, 8, 64):
                C, m = sm.rowblock(sc, A, B, parts)
                record(f"spark rowblock P={parts} {name}: bit-identical to sequential kernel", C.tobytes() == ref.tobytes())
            for g in (1, 2, 3):
                C, m = sm.block2d(sc, A, B, g)
                record(f"spark block2d g={g} {name}: within bound of NumPy", bound_ok(C, A, B))
                if name.startswith("known"):
                    record(f"spark block2d g={g} {name}: exact", np.array_equal(C, known[name[6:]][2]))
        spark.stop()
    finally:
        cl.stop()

    RES.mkdir(exist_ok=True)
    with open(RES / "correctness.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["test", "passed", "detail"])
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows) - failures}/{len(rows)} checks passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
