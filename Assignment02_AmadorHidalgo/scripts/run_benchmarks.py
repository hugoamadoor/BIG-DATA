#!/usr/bin/env python3
"""Benchmark harness for Assignment 2 (optimised dense methods and CSR sparse multiplication).

Experiments (each one is a separate, labelled comparison):
  T  block-size tuning   blocked, n in TUNE_SIZES, block in TUNE_BLOCKS. The block with the lowest
                         median at the largest tuning size is used for every later blocked run.
  D  dense comparison    ijk (baseline), ijk_bt, ikj, blocked; random dense inputs; DENSE_SIZES.
  M  matched sparse      uniform sparse inputs, n in {1024, 2048}, nine densities. Sparse CSR
                         SpGEMM (csr-input run: multiply time and memory; dense-input run:
                         dense->CSR conversion time) against dense blocked on the same logical
                         matrices, and against the baseline at n = 1024.
  S  structure           n = 2048, densities 5 % and 1 %: uniform vs banded vs skewed rows.
  X  cross-check        the unchanged Assignment 1 C binary (../Assignment01_AmadorHidalgo/build/c/
                         matmul_O2) against the baseline here, n = 1024, same machine and session.
  L  large sparse        uniform, 16 nonzeros per row on average, n = 4096 ... 131072; dense
                         blocked attempted on the same logical matrices while the budget allows.

One process per configuration. Timing happens inside the process around the kernel only.
The harness records peak RSS of that process (wait4 via build/rss_run), validates every result
and appends raw samples to CSV (nothing is discarded).

Usage:
  python3 scripts/run_benchmarks.py                 # everything
  python3 scripts/run_benchmarks.py --only T,D      # selected experiments
  python3 scripts/run_benchmarks.py --quick         # small smoke test into results/quick/
"""
import argparse
import csv
import json
import math
import os
import struct
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parent))
from env_info import collect  # noqa: E402
from gen_inputs import MATCHED, SCALING, STRUCTURE, read_csr, tag  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DATA, BUILD = ROOT / "data", ROOT / "build"
U = 2.0 ** -53

# ----------------------------- declared protocol ---------------------------------------
TUNE_SIZES = [1024, 2048]
TUNE_BLOCKS = [16, 32, 64, 128, 256]
DENSE_SIZES = [256, 512, 768, 1024, 1536, 2048, 3072, 4096]
DENSE_METHODS = ["ijk", "ijk_bt", "ikj", "blocked"]
LIMIT_S = 180.0            # one multiplication may take at most 180 s (enforced in-process)
WALL_LIMIT_S = 900.0       # one configuration (warm-up + samples) may take at most 15 min
MEM_BUDGET_B = 4 * 2 ** 30 # planned storage of one process must stay below 4 GiB
BUDGET_S = 300.0           # stop sampling once measured time exceeds this (after MIN_REPS)
MIN_REPS = 3
BATCH_S = 0.1              # samples shorter than this are batched


def reps_for(n_or_time):
    return 10 if n_or_time <= 512 else 5


def gamma(k):
    return k * U / (1.0 - k * U)


def hexf(h):
    return struct.unpack("<d", bytes.fromhex(h)[::-1])[0]


# ----------------------------- process runner ------------------------------------------
def run_process(argv, timeout):
    t0 = time.monotonic()
    p = subprocess.Popen([str(BUILD / "rss_run")] + argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                         text=True, start_new_session=True)
    out, err = [], []
    th = [threading.Thread(target=lambda: out.extend(p.stdout), daemon=True),
          threading.Thread(target=lambda: err.extend(p.stderr), daemon=True)]
    for t in th:
        t.start()
    timed_out = False
    while True:
        pid, status = os.waitpid(p.pid, os.WNOHANG)
        if pid:
            break
        if time.monotonic() - t0 > timeout:
            os.killpg(p.pid, 9)
            pid, status = os.waitpid(p.pid, 0)
            timed_out = True
            break
        time.sleep(0.05)
    for t in th:
        t.join(5)
    rc = os.waitstatus_to_exitcode(status)
    rss_kb, lines = float("nan"), []
    for ln in out:
        if ln.startswith("RUSAGE_MAXRSS_KB"):
            rss_kb = float(ln.split()[1])
        else:
            lines.append(ln.strip())
    return lines, rc, rss_kb, time.monotonic() - t0, timed_out, "".join(err)


def parse(lines):
    phase, samples, warm, limit, nnz, check = "multiply", [], {}, None, None, None
    for ln in lines:
        f = ln.split()
        if not f:
            continue
        if f[0] == "PHASE":
            phase = f[1]
        elif f[0] == "WARMUP":
            warm[phase] = int(f[1])
        elif f[0] == "RESULT":
            samples.append((phase, int(f[1]), int(f[2]), int(f[3])))
        elif f[0] == "LIMIT":
            limit = (phase, f[1], int(f[2]))
        elif f[0] == "NNZ":
            nnz = tuple(int(x) for x in f[1:4])
        elif f[0] == "CHECK":
            check = f[1:5]
    return phase, samples, warm, limit, nnz, check


# ----------------------------- references for validation -------------------------------
class Refs:
    """Spot values (C[0,0], C[n-1,n-1], C[n/2,n/3]), sum and nnz from NumPy / SciPy."""
    def __init__(self):
        self.cache = {}

    def dense(self, n, pa, pb):
        key = ("dense", str(pa))
        if key not in self.cache:
            A = np.fromfile(pa, dtype="<f8").reshape(n, n)
            B = np.fromfile(pb, dtype="<f8").reshape(n, n)
            C = A @ B
            absC = np.abs(A) @ np.abs(B)
            idx = [(0, 0), (n - 1, n - 1), (n // 2, n // 3)]
            self.cache = {key: dict(spots=[C[i] for i in idx], abs=[absC[i] for i in idx],
                                    sum=math.fsum(C.ravel()), abssum=float(absC.sum()),
                                    nnz=int(np.count_nonzero(C)))}
            del A, B, C, absC
        return self.cache[key]

    def sparse(self, n, pa, pb):
        key = ("sparse", str(pa))
        if key not in self.cache:
            def load(p):
                _, rp, ci, v = read_csr(p)
                return sp.csr_matrix((v, ci, rp), shape=(n, n))
            A, B = load(pa), load(pb)
            C = sp.csr_matrix(A @ B)
            Aa, Ba = abs(A), abs(B)
            idx = [(0, 0), (n - 1, n - 1), (n // 2, n // 3)]
            absC = sp.csr_matrix(Aa @ Ba)
            self.cache = {key: dict(spots=[C[i] for i in idx], abs=[absC[i] for i in idx],
                                    sum=math.fsum(C.data), abssum=float(absC.data.sum()), nnz=int(C.nnz))}
        return self.cache[key]


def validate(check, ref, n):
    """Spot values within 2 gamma_n |A||B| of the reference; sum within 2 gamma_(n^2+n) sum|A||B|."""
    if check is None:
        return False, "no CHECK record"
    spots = [hexf(h) for h in check[1:]]
    for v, r, a in zip(spots, ref["spots"], ref["abs"]):
        if abs(v - r) > 2 * gamma(n) * a:
            return False, f"spot value {v!r} vs reference {r!r}"
    s = hexf(check[0])
    if abs(s - ref["sum"]) > 2 * gamma(n * n + n) * ref["abssum"]:
        return False, f"sum {s!r} vs reference {ref['sum']!r}"
    return True, ""


# ----------------------------- one configuration ---------------------------------------
class Bench:
    def __init__(self, out):
        self.out = out
        out.mkdir(parents=True, exist_ok=True)
        self.raw = open(out / "raw_measurements.csv", "a", newline="")
        self.runs = open(out / "runs.csv", "a", newline="")
        self.raw_w = csv.writer(self.raw)
        self.runs_w = csv.writer(self.runs)
        if self.raw.tell() == 0:
            self.raw_w.writerow(["experiment", "method", "input", "structure", "n", "density_pct", "block",
                                 "phase", "rep", "inner", "batch_ns", "ns_per_mult", "numeric_type", "threads"])
        if self.runs.tell() == 0:
            self.runs_w.writerow(["timestamp", "experiment", "method", "input", "structure", "n", "density_pct",
                                  "block", "status", "detail", "warmup_runs", "samples", "peak_rss_kb",
                                  "planned_bytes", "nnz_A", "nnz_B", "nnz_C", "check_sum", "check_c00",
                                  "check_clast", "check_cmid", "valid", "wall_s"])
        self.refs = Refs()
        self.log = open(out / "run.log", "a")

    def say(self, msg):
        line = f"[{datetime.now():%H:%M:%S}] {msg}"
        print(line, flush=True)
        self.log.write(line + "\n")
        self.log.flush()

    def row(self, exp, method, inp, structure, n, d, block, status, detail="", warm=None, samples=0,
            rss=float("nan"), planned=0, nnz=None, check=None, valid="", wall=0.0):
        nnz = nnz or ("", "", "")
        check = check or ("", "", "", "")
        self.runs_w.writerow([datetime.now().isoformat(timespec="seconds"), exp, method, inp, structure, n, d,
                              block, status, detail, warm if warm is not None else "", samples, rss, planned,
                              *nnz, *check, valid, f"{wall:.2f}"])
        self.runs.flush()

    def run(self, exp, method, inp, structure, n, d, block, argv, planned, ref, warmup_s, reps):
        if planned > MEM_BUDGET_B:
            self.row(exp, method, inp, structure, n, d, block, "memory_budget",
                     f"planned storage {planned / 2**30:.2f} GiB > 4 GiB, not run", planned=planned)
            self.say(f"{exp} {method} n={n} d={d}: NOT RUN (planned {planned / 2**30:.2f} GiB > budget)")
            return "memory_budget"
        proto = [str(warmup_s), str(reps), str(MIN_REPS), str(BUDGET_S), str(BATCH_S), str(LIMIT_S)]
        lines, rc, rss, wall, timed_out, err = run_process(argv + proto, WALL_LIMIT_S)
        phase, samples, warm, limit, nnz, check = parse(lines)
        for ph, rep, inner, ns in samples:
            self.raw_w.writerow([exp, method, inp, structure, n, d, block, ph, rep, inner, ns,
                                 f"{ns / inner:.1f}", "float64", 1])
        self.raw.flush()
        mult = [s for s in samples if s[0] == "multiply"]
        if timed_out:
            status, detail = "wall_limit", f"killed after {WALL_LIMIT_S:.0f} s"
        elif limit is not None:
            status, detail = "time_limit", f"a multiplication exceeded {LIMIT_S:.0f} s ({limit[1]})"
        elif rc != 0:
            status, detail = "error", f"exit code {rc}: {err.strip()[:200]}"
        else:
            status, detail = "ok", ""
        valid = ""
        if status == "ok":
            ok, why = validate(check, ref, n) if ref else (True, "")
            if ref and nnz is not None and nnz[2] != ref["nnz"]:
                ok, why = False, f"nnz(C) {nnz[2]} vs reference {ref['nnz']}"
            valid = "yes" if ok else "NO"
            if not ok:
                status, detail = "invalid", why
        self.row(exp, method, inp, structure, n, d, block, status, detail, warm.get("multiply"), len(mult),
                 rss, planned, nnz, check, valid, wall)
        med = np.median([s[3] / s[2] for s in mult]) * 1e-9 if mult else float("nan")
        self.say(f"{exp} {method:10s} {inp:5s} {structure:8s} n={n:<6d} d={d!s:<6} b={block!s:<4} {status:10s} "
                 f"median {med:10.4f} s  samples {len(mult)}  rss {rss / 1024:8.1f} MiB  {detail}")
        return status


def dense_planned(method, n):
    return (4 if method == "ijk_bt" else 3) * 8 * n * n


def csr_bytes(n, nnz):
    return 8 * (n + 1) + 12 * nnz


def warm_for(n):
    return 1.0 if n <= 1536 else 0.0   # >= 2048: one run takes seconds to minutes; warm-up omitted


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="T,D,M,S,X,L")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--m-densities", default=None, help="only these densities in experiment M, e.g. 100,85,70")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    only = set(args.only.split(","))
    out = Path(args.out) if args.out else ROOT / ("results/quick" if args.quick else "results")
    b = Bench(out)

    t = subprocess.run([sys.executable, str(ROOT / "scripts/run_tests.py")], capture_output=True, text=True)
    b.say("tests: " + t.stdout.strip().splitlines()[-1])
    if t.returncode != 0:
        b.say("correctness tests failed - refusing to benchmark")
        return 1
    env = collect()
    env["build_flags"] = "-O2 -std=c11 -ffp-contract=off (gcc), no -march: x86-64 baseline ISA"
    env["protocol"] = dict(limit_s=LIMIT_S, wall_limit_s=WALL_LIMIT_S, mem_budget_bytes=MEM_BUDGET_B,
                           budget_s=BUDGET_S, min_reps=MIN_REPS, batch_s=BATCH_S)
    (out / "environment.json").write_text(json.dumps(env, indent=2))

    # baseline RSS of the two programs (n = 16, trivial work)
    base = {}
    for prog, argv in (("dense", [str(BUILD / "dense"), "ijk", "16", "64", str(DATA / "dense/A_16.bin"),
                                  str(DATA / "dense/B_16.bin")]),
                       ("sparse", [str(BUILD / "sparse"), "dense", "16", str(DATA / "dense/A_16.bin"),
                                   str(DATA / "dense/B_16.bin")])):
        lines, rc, rss, *_ = run_process(argv + ["0", "1", "1", "0", "0", "10"], 60)
        base[prog] = rss
    (out / "baseline_rss.json").write_text(json.dumps(base, indent=2))

    def dense_run(exp, method, n, block, pa, pb, structure="dense", d=100, reps=None):
        ref = b.refs.dense(n, pa, pb)
        return b.run(exp, method, "dense", structure, n, d, block,
                     [str(BUILD / "dense"), method, str(n), str(block), str(pa), str(pb)],
                     dense_planned(method, n), ref, warm_for(n), reps or reps_for(n))

    # ---------------- T: block-size tuning ----------------
    block = 64
    if "T" in only:
        sizes, blocks = ([256], [16, 64]) if args.quick else (TUNE_SIZES, TUNE_BLOCKS)
        for n in sizes:
            for bs in blocks:
                dense_run("T", "blocked", n, bs, DATA / f"dense/A_{n}.bin", DATA / f"dense/B_{n}.bin")
        rows = list(csv.DictReader(open(out / "raw_measurements.csv")))
        top = max(sizes)
        med = {bs: np.median([float(r["ns_per_mult"]) for r in rows
                              if r["experiment"] == "T" and int(r["n"]) == top and int(r["block"]) == bs])
               for bs in blocks}
        block = min(med, key=med.get)
        (out / "block_choice.json").write_text(json.dumps(dict(chosen_block=block, tuned_at_n=top,
                                                               median_s={k: v * 1e-9 for k, v in med.items()}), indent=2))
        b.say(f"chosen block size: {block}")
    elif (out / "block_choice.json").exists():
        block = json.loads((out / "block_choice.json").read_text())["chosen_block"]

    # ---------------- D: dense comparison ----------------
    if "D" in only:
        stopped = {}
        for n in ([256, 512] if args.quick else DENSE_SIZES):
            for m in DENSE_METHODS:
                if m in stopped:
                    b.row("D", m, "dense", "dense", n, 100, block if m == "blocked" else "", "not_attempted",
                          f"smaller size n={stopped[m]} already exceeded a limit")
                    continue
                st = dense_run("D", m, n, block if m == "blocked" else 0, DATA / f"dense/A_{n}.bin",
                               DATA / f"dense/B_{n}.bin")
                if st in ("time_limit", "wall_limit"):
                    stopped[m] = n

    def sparse_run(exp, mode, structure, n, d, nnzA, nnzB, nnzC_est, reps=None):
        t = tag(structure, n, d)
        pa, pb = DATA / "sparse" / f"{t}_A.{'csr' if mode == 'csr' else 'bin'}", \
            DATA / "sparse" / f"{t}_B.{'csr' if mode == 'csr' else 'bin'}"
        ref = b.refs.sparse(n, DATA / "sparse" / f"{t}_A.csr", DATA / "sparse" / f"{t}_B.csr")
        planned = csr_bytes(n, nnzA) + csr_bytes(n, nnzB) + csr_bytes(n, ref["nnz"]) + 20 * n
        if mode == "dense":
            planned += 2 * 8 * n * n
        return b.run(exp, "csr_spgemm", mode, structure, n, d, "",
                     [str(BUILD / "sparse"), mode, str(n), str(pa), str(pb)], planned, ref,
                     warm_for(n) if n <= 2048 else 1.0, reps or 5)

    def nnz_of(structure, n, d):
        t = tag(structure, n, d)
        return [int(np.fromfile(DATA / "sparse" / f"{t}_{w}.csr", dtype="<i8", count=2)[1]) for w in "AB"]

    def dense_on_sparse(exp, method, structure, n, d):
        t = tag(structure, n, d)
        pa, pb = DATA / "sparse" / f"{t}_A.bin", DATA / "sparse" / f"{t}_B.bin"
        if not pa.exists():
            planned = dense_planned(method, n)
            if planned > MEM_BUDGET_B:
                return b.run(exp, method, "dense", structure, n, d, block, [], planned, None, 0, 0)
            b.row(exp, method, "dense", structure, n, d, block, "no_input", "dense file not generated")
            return "no_input"
        ref = b.refs.sparse(n, DATA / "sparse" / f"{t}_A.csr", DATA / "sparse" / f"{t}_B.csr")
        return b.run(exp, method, "dense", structure, n, d, block if method == "blocked" else 0,
                     [str(BUILD / "dense"), method, str(n), str(block if method == "blocked" else 0),
                      str(pa), str(pb)], dense_planned(method, n), ref, warm_for(n), reps_for(n))

    # ---------------- M: matched sparse vs dense ----------------
    if "M" in only:
        configs = [("uniform", 200, 30), ("uniform", 200, 5)] if args.quick else MATCHED
        if args.m_densities:
            keep = [float(x) for x in args.m_densities.split(",")]
            configs = [c for c in configs if any(math.isclose(c[2], k) for k in keep)]
        for structure, n, d in configs:
            na, nb = nnz_of(structure, n, d)
            sparse_run("M", "csr", structure, n, d, na, nb, None)
            sparse_run("M", "dense", structure, n, d, na, nb, None)
            dense_on_sparse("M", "blocked", structure, n, d)
            if n == 1024 or args.quick:
                dense_on_sparse("M", "ijk", structure, n, d)

    # ---------------- S: structure ----------------
    if "S" in only and not args.quick:
        for structure, n, d in [("uniform", 2048, 5), ("uniform", 2048, 1)] + STRUCTURE:
            na, nb = nnz_of(structure, n, d)
            sparse_run("S", "csr", structure, n, d, na, nb, None)
            if d == 1:
                dense_on_sparse("S", "blocked", structure, n, d)

    # ---------------- X: Assignment 1 binary on this machine ----------------
    a1 = ROOT.parent / "Assignment01_AmadorHidalgo" / "build" / "c" / "matmul_O2"
    if "X" in only and a1.exists():
        for n in [512, 1024]:
            pa, pb = DATA / f"dense/A_{n}.bin", DATA / f"dense/B_{n}.bin"
            b.run("X", "a1_matmul_O2", "dense", "dense", n, 100, 0, [str(a1), str(n), str(pa), str(pb)],
                  dense_planned("ijk", n), b.refs.dense(n, pa, pb), warm_for(n), reps_for(n))
            dense_run("X", "ijk", n, 0, pa, pb)

    # ---------------- L: large sparse, dense attempted while feasible ----------------
    if "L" in only and not args.quick:
        dense_stopped = None
        for structure, n, d in SCALING:
            na, nb = nnz_of(structure, n, d)
            sparse_run("L", "csr", structure, n, d, na, nb, None)
            if dense_stopped:
                b.row("L", "blocked", "dense", structure, n, d, block, "not_attempted",
                      f"smaller size n={dense_stopped} already exceeded a limit")
                continue
            st = dense_on_sparse("L", "blocked", structure, n, d)
            if st in ("time_limit", "wall_limit"):
                dense_stopped = n
    b.say("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
