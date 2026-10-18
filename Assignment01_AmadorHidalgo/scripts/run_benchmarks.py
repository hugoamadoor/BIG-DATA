#!/usr/bin/env python3
"""Benchmark harness for Assignment 1 (basic triple-loop matrix multiplication).

Main study  : C (-O2), Java (JIT, SerialGC) and CPython, sizes from SIZES, same input files.
Supplementary S1 (kept separate): C compiled with -O0 / -O2 / -O3.

Every process runs one (variant, n): warm-up, then the measured samples (timing happens
inside the process, around the kernel only). The harness measures the peak resident set
size of the process with wait4(), validates the result against NumPy and across
languages, and appends everything to CSV files (raw samples are never discarded).

Usage:
  python3 scripts/run_benchmarks.py                  # main study + supplementary S1
  python3 scripts/run_benchmarks.py --main-only
  python3 scripts/run_benchmarks.py --sizes 16,32,64 --quick
"""
import argparse
import csv
import json
import math
import os
import statistics
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from env_info import collect  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DATA, RES = ROOT / "data", ROOT / "results"
U = 2.0 ** -53

# ----------------------------- declared protocol ---------------------------------------
SIZES = [16, 32, 64, 128, 256, 512, 768, 1024, 1536, 2048]
S1_SIZES = [256, 512, 1024]
S2_SIZES = [1024, 1025, 1000]      # S2: neighbouring sizes with very different row strides
LIMIT_S = 180.0          # time budget: one multiplication may take at most 180 s
WALL_LIMIT_S = 900.0     # a whole configuration (warm-up + samples) may take at most 15 min
MEM_BUDGET_KB = 4 * 1024 * 1024   # memory budget: 4 GiB peak RSS per process
BUDGET_S = 300.0         # stop sampling once the measured time exceeds this (after MIN_REPS)
MIN_REPS = 3
BATCH_S = 0.1            # target duration of one sample; shorter kernels are batched
JAVA = ["java", "-XX:+UseSerialGC", "-Xmx3g", "-cp", str(ROOT / "build/java"), "MatMul"]

VARIANTS = {   # name -> (language, argv prefix)
    "C-O2":   ("C",      [str(ROOT / "build/c/matmul_O2")]),
    "Java":   ("Java",   JAVA),
    "Python": ("Python", [sys.executable, str(ROOT / "src/python/matmul.py")]),
    "C-O0":   ("C",      [str(ROOT / "build/c/matmul_O0")]),
    "C-O3":   ("C",      [str(ROOT / "build/c/matmul_O3")]),
}


def gamma(k):
    return k * U / (1.0 - k * U)


def settings(variant, n):
    reps = 10 if n <= 256 else 5
    # CPython has no JIT: once a multiplication takes >10 s a warm-up run only costs time.
    warm = 0.0 if (variant == "Python" and n >= 512) else 1.0
    return warm, reps


def run_process(argv, timeout):
    """Run argv under the rss_run wrapper. Returns (lines, rc, rss_kb, wall_s, timed_out, stderr)."""
    t0 = time.monotonic()
    p = subprocess.Popen([str(ROOT / "build/rss_run")] + argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
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
            os.killpg(p.pid, 9)                     # wrapper and benchmark process
            pid, status = os.waitpid(p.pid, 0)
            timed_out = True
            break
        time.sleep(0.05)
    for t in th:
        t.join(5)
    p.returncode = os.waitstatus_to_exitcode(status)
    rss_kb, lines = float("nan"), []
    for ln in out:
        if ln.startswith("RUSAGE_MAXRSS_KB"):
            rss_kb = float(ln.split()[1])
        else:
            lines.append(ln)
    return lines, p.returncode, rss_kb, time.monotonic() - t0, timed_out, "".join(err)


class Reference:
    """NumPy/BLAS reference product per size (trusted implementation, used only for checking)."""
    def __init__(self):
        self.n, self.vals = None, None

    def get(self, n):
        if self.n != n:
            A = np.fromfile(DATA / f"A_{n}.bin", dtype="<f8").reshape(n, n)
            B = np.fromfile(DATA / f"B_{n}.bin", dtype="<f8").reshape(n, n)
            C = A @ B
            self.n = n
            self.vals = (math.fsum(C.ravel()), C[0, 0], C[n - 1, n - 1], C[n // 2, n // 3])
        return self.vals


def bits_to_float(h):
    import struct
    return struct.unpack("<d", struct.pack("<Q", int(h, 16)))[0]


def validate(n, check, ref):
    """check = four hex words (sum, c00, clast, cmid). Returns (ok, max_rel_dev_elements, rel_dev_sum)."""
    got = [bits_to_float(h) for h in check]
    rs, r00, rl, rm = ref
    tol_el = 2.0 * gamma(n)
    tol_sum = 2.0 * (gamma(n) + gamma(n * n))
    dev_sum = abs(got[0] - rs) / abs(rs)
    dev_el = max(abs(g - r) / abs(r) for g, r in zip(got[1:], (r00, rl, rm)))
    return (dev_sum <= tol_sum and dev_el <= tol_el), dev_el, dev_sum


def parse(lines):
    d = {"warm": (0, 0), "results": [], "limit": None, "check": None}
    for ln in lines:
        f = ln.split()
        if not f:
            continue
        if f[0] == "WARMUP":
            d["warm"] = (int(f[1]), int(f[2]))
        elif f[0] == "RESULT":
            d["results"].append((int(f[1]), int(f[2]), int(f[3])))
        elif f[0] == "LIMIT":
            d["limit"] = (f[1], int(f[2]))
        elif f[0] == "CHECK":
            d["check"] = f[1:5]
    return d


def baseline(variant, writer_runs):
    """Peak RSS of the bare runtime: same program, n = 1, five repetitions."""
    lang, prefix = VARIANTS[variant]
    vals = []
    for _ in range(5):
        argv = prefix + ["1", str(DATA / "A_1.bin"), str(DATA / "B_1.bin"), "0", "1", "1", "1", "0", str(LIMIT_S)]
        _, rc, rss, wall, _, _ = run_process(argv, 120)
        vals.append(rss)
    return statistics.median(vals), vals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", help="comma separated list overriding SIZES")
    ap.add_argument("--quick", action="store_true", help="3 samples, no budget waits (smoke test)")
    ap.add_argument("--main-only", action="store_true")
    ap.add_argument("--only-s2", action="store_true", help="run only supplementary S2 and APPEND to the existing CSV files")
    ap.add_argument("--out", default=str(RES))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(exist_ok=True, parents=True)
    sizes = [int(s) for s in args.sizes.split(",")] if args.sizes else SIZES

    # refuse to benchmark an unvalidated build
    if subprocess.run([sys.executable, str(ROOT / "scripts/run_tests.py")], capture_output=True).returncode:
        sys.exit("correctness tests failed - not benchmarking")

    env = collect(java_flags=" ".join(JAVA[1:3]))
    env["protocol"] = {"sizes": sizes, "supplementary_sizes": S1_SIZES, "time_limit_per_multiplication_s": LIMIT_S,
                       "wall_limit_per_configuration_s": WALL_LIMIT_S, "memory_budget_GiB": MEM_BUDGET_KB / 2**20,
                       "min_reps": MIN_REPS, "sample_budget_s": BUDGET_S, "batch_target_s": BATCH_S,
                       "started": datetime.now().astimezone().isoformat(timespec="seconds")}
    (out / "environment.json").write_text(json.dumps(env, indent=2))

    mode = "a" if args.only_s2 else "w"
    f_s = open(out / "raw_measurements.csv", mode, newline="")
    f_r = open(out / "runs.csv", mode, newline="")
    ws, wr = csv.writer(f_s), csv.writer(f_r)
    if not args.only_s2: ws.writerow(["study", "variant", "language", "n", "dtype", "rep", "inner_multiplications", "batch_ns",
                 "time_s_per_multiplication", "warmup_multiplications", "peak_rss_kb"])
    if not args.only_s2: wr.writerow(["study", "variant", "language", "n", "status", "samples", "warmup_multiplications", "warmup_s",
                 "peak_rss_kb", "wall_s", "validated", "max_rel_dev_elements", "rel_dev_sum", "checksum_bits",
                 "note"])
    ref = Reference()
    checks = {}            # (study, n) -> {variant: checksum words} for bitwise cross-check
    dead = set()           # (study, variant) that exceeded a limit: larger sizes are not attempted

    def one(study, variant, n):
        lang, prefix = VARIANTS[variant]
        warm_s, reps = settings(variant, n)
        if args.quick:
            reps = 3
        if (study, variant) in dead:
            wr.writerow([study, variant, lang, n, "not_attempted", 0, 0, 0, "", "", "", "", "", "",
                         "previous size exceeded the declared limit"])
            f_r.flush()
            print(f"  {variant:7s} n={n:5d} not attempted")
            return
        argv = prefix + [str(n), str(DATA / f"A_{n}.bin"), str(DATA / f"B_{n}.bin"), str(warm_s), str(reps),
                         str(MIN_REPS), str(BUDGET_S), str(BATCH_S), str(LIMIT_S)]
        lines, rc, rss, wall, timed_out, err = run_process(argv, WALL_LIMIT_S)
        d = parse(lines)
        status, note = "ok", ""
        if timed_out:
            status, note = "timeout", f"killed after {WALL_LIMIT_S:.0f} s"
        elif rc != 0:
            status, note = "error", err[-300:].replace("\n", " ")
        elif d["limit"]:
            status = "over_limit"
            note = f"a single multiplication took {d['limit'][1] / 1e9:.1f} s > {LIMIT_S:.0f} s ({d['limit'][0]})"
        elif rss > MEM_BUDGET_KB:
            status, note = "over_memory", f"peak RSS {rss / 2**20:.2f} GiB"
        valid, dev_el, dev_sum = ("", "", "")
        if d["check"] and status in ("ok", "over_limit"):
            ok, dev_el, dev_sum = validate(n, d["check"], ref.get(n))
            valid = int(ok)
            if not ok:
                print(f"  !! VALIDATION FAILED {variant} n={n}")
            if status == "ok":
                checks.setdefault((study, n), {})[variant] = d["check"]
        for rep, inner, ns in d["results"]:
            ws.writerow([study, variant, lang, n, "float64", rep, inner, ns, f"{ns / inner / 1e9:.9f}",
                         d["warm"][0], f"{rss:.0f}"])
        wr.writerow([study, variant, lang, n, status, len(d["results"]), d["warm"][0], f"{d['warm'][1] / 1e9:.3f}",
                     f"{rss:.0f}", f"{wall:.1f}", valid, dev_el, dev_sum, d["check"][0] if d["check"] else "", note])
        f_s.flush()
        f_r.flush()
        if status in ("over_limit", "timeout", "over_memory", "error"):
            dead.add((study, variant))
        med = statistics.median([ns / inner / 1e9 for _, inner, ns in d["results"]]) if d["results"] else float("nan")
        print(f"  {variant:7s} n={n:5d} {status:10s} samples={len(d['results']):2d} median={med:10.5f} s "
              f"rss={rss / 1024:7.1f} MB wall={wall:6.1f} s {note}")
        sys.stdout.flush()

    if not args.only_s2:
        print("== runtime baselines (n=1)")
        wr.writerow(["baseline"] + [""] * 14)
        base = {}
        for v in ["C-O2", "Java", "Python"]:
            med, vals = baseline(v, wr)
            base[v] = med
            print(f"  {v:7s} baseline peak RSS median {med / 1024:.1f} MB")
        (out / "baseline_rss.json").write_text(json.dumps({k: v for k, v in base.items()}, indent=2))

        print("== main study")
        for n in sizes:
            for v in ["C-O2", "Java", "Python"]:
                one("main", v, n)

    if args.only_s2 or not args.main_only:
        print("== supplementary S2: neighbouring sizes, C and Java (row stride experiment)")
        dead.clear()
        for n in S2_SIZES:
            for v in ["C-O2", "Java"]:
                one("S2", v, n)

    if not args.main_only and not args.only_s2:
        print("== supplementary S1: C optimisation levels (separate from the main comparison)")
        dead.clear()
        for n in S1_SIZES:
            for v in ["C-O0", "C-O2", "C-O3"]:
                one("S1", v, n)

    # bitwise cross-check of checksums between all variants that completed the same size
    bad = [(k, v) for k, v in checks.items() if len({tuple(w) for w in v.values()}) != 1]
    (out / ("crosscheck_s2.json" if args.only_s2 else "crosscheck.json")).write_text(json.dumps({"mismatches": [str(b) for b in bad],
                                                      "compared_sizes": len(checks)}, indent=2))
    print("cross-language checksum comparison:", "ALL IDENTICAL" if not bad else f"MISMATCH {bad}")
    if not args.only_s2:
        env["protocol"]["finished"] = datetime.now().astimezone().isoformat(timespec="seconds")
        (out / "environment.json").write_text(json.dumps(env, indent=2))
    f_s.close()
    f_r.close()


if __name__ == "__main__":
    main()
