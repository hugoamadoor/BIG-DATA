#!/usr/bin/env python3
"""Benchmark harness for Assignment 3 (thread scaling of the blocked kernel; optional vectorisation).

Experiments
  P  thread scaling   build/mm (-O2, scalar, the Assignment 2 flags + -fopenmp).
                      Methods: seq (baseline), tiles (parallel code path, no OpenMP), omp with
                      p = 1, 2 (= hardware contexts), 3, 4, 8 (oversubscribed).
                      n in SIZES.
  N  control          build/mm_nomp seq (no -fopenmp at all), n <= 2048.
  A  code placement   build/mm_noalign (build/mm without -falign-loops=32): seq, tiles, omp p = 1,
                      n in ALIGN_SIZES. Documents the alignment effect found during development.
  V  vectorisation    build/mm_scalar (-O3 -mavx2 -fno-tree-vectorize) and build/mm_simd (-O3 -mavx2):
                      seq (1 thread) and omp p = 2, n in VEC_SIZES. A 2 x 2 design, so the effect of
                      vectorisation (at fixed p) and of threads (at fixed build) are measured separately.
                      Plus tiles and omp p = 1 of both builds, to check that the -O3 sequential and
                      parallel code paths are equally fast at one thread.

The whole set is run in ROUNDS rounds. Inside a round, sizes go in increasing order and the
configurations of one size run in a random order (fixed seed per round), so slow drift of the
machine cannot line up with one method. One process per configuration; the timer is inside the
process around the kernel only. Peak RSS and CPU time come from wait4() (build/rss_run); the
process also reports CPU and wall time of its kernel phase (KCPU). Every result is validated.
Raw samples are appended to CSV; nothing is discarded.

Usage:
  python3 scripts/run_benchmarks.py            # everything (about 1 hour on the 2-vCPU VM)
  python3 scripts/run_benchmarks.py --quick    # smoke test into results/quick/
"""
import argparse
import csv
import json
import math
import os
import random
import struct
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")    # the harness's own NumPy (validation only)
os.environ.setdefault("MKL_NUM_THREADS", "1")
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from env_info import collect  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DATA, BUILD = ROOT / "data", ROOT / "build"
U = 2.0 ** -53

# ----------------------------- declared protocol ---------------------------------------
SIZES = [256, 512, 1024, 2048, 3072, 4096]
VEC_SIZES = [512, 1024, 2048, 4096]
CONTROL_MAX_N = 2048
ALIGN_SIZES = [512, 1024]
THREADS = [1, 2, 3, 4, 8]
BLOCK = 32                 # tile side chosen in Assignment 2 (experiment T); not re-tuned
ROUNDS = 2
LIMIT_S = 180.0            # one multiplication may take at most 180 s (enforced in-process)
WALL_LIMIT_S = 900.0       # one configuration may take at most 15 min
MEM_BUDGET_B = 4 * 2 ** 30
BUDGET_S = 300.0           # stop sampling once measured time exceeds this (after MIN_REPS)
MIN_REPS = 3
BATCH_S = 0.1              # samples shorter than this are batched


def reps_for(n):
    return 10 if n <= 512 else (5 if n <= 3072 else 3)


def warm_for(n):
    return 1.0 if n <= 1024 else 0.0      # larger n: the untimed calibration call is the warm-up


def gamma(k):
    return k * U / (1.0 - k * U)


def hexf(h):
    return struct.unpack("<d", bytes.fromhex(h)[::-1])[0]


def run_process(argv, env, timeout):
    t0 = time.monotonic()
    p = subprocess.Popen([str(BUILD / "rss_run")] + argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                         text=True, start_new_session=True, env=env)
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
    return [ln.strip() for ln in out], os.waitstatus_to_exitcode(status), time.monotonic() - t0, timed_out, "".join(err)


def parse(lines):
    d = dict(samples=[], warm=None, limit=None, check=None, threads=None, env="", kcpu=None,
             rss_kb=float("nan"), cpu=(float("nan"), float("nan")))
    for ln in lines:
        f = ln.split()
        if not f:
            continue
        if f[0] == "RESULT":
            d["samples"].append((int(f[1]), int(f[2]), int(f[3])))
        elif f[0] == "WARMUP":
            d["warm"] = int(f[1])
        elif f[0] == "LIMIT":
            d["limit"] = f[1]
        elif f[0] == "CHECK":
            d["check"] = f[1:5]
        elif f[0] == "THREADS":
            d["threads"] = int(f[1])
        elif f[0] == "ENV":
            d["env"] = " ".join(f[1:])
        elif f[0] == "KCPU":
            d["kcpu"] = tuple(float(x) for x in f[1:4])
        elif f[0] == "RUSAGE_MAXRSS_KB":
            d["rss_kb"] = float(f[1])
        elif f[0] == "RUSAGE_CPU_S":
            d["cpu"] = (float(f[1]), float(f[2]))
    return d


class Ref:
    def __init__(self, n):
        A = np.fromfile(DATA / f"A_{n}.bin", dtype="<f8").reshape(n, n)
        B = np.fromfile(DATA / f"B_{n}.bin", dtype="<f8").reshape(n, n)
        C = A @ B
        absC = np.abs(A) @ np.abs(B)
        idx = [(0, 0), (n - 1, n - 1), (n // 2, n // 3)]
        self.n, self.spots, self.abs = n, [C[i] for i in idx], [absC[i] for i in idx]
        self.sum, self.abssum = math.fsum(C.ravel()), float(absC.sum())

    def ok(self, check):
        if check is None:
            return False, "no CHECK record"
        for v, r, a in zip([hexf(h) for h in check[1:]], self.spots, self.abs):
            if abs(v - r) > 2 * gamma(self.n) * a:
                return False, f"spot {v!r} vs {r!r}"
        s = hexf(check[0])
        if abs(s - self.sum) > 2 * gamma(self.n * self.n + self.n) * self.abssum:
            return False, f"sum {s!r} vs {self.sum!r}"
        return True, ""


class Bench:
    RAW = ["round", "experiment", "binary", "method", "threads", "n", "block", "rep", "inner", "batch_ns",
           "ns_per_mult", "numeric_type"]
    RUNS = ["timestamp", "round", "order", "experiment", "binary", "method", "threads", "threads_reported", "n",
            "block", "status", "detail", "warmup_runs", "samples", "peak_rss_kb", "cpu_user_s", "cpu_sys_s",
            "kernel_wall_s", "kernel_user_s", "kernel_sys_s", "omp_env", "check_sum", "valid", "wall_s"]

    def __init__(self, out):
        out.mkdir(parents=True, exist_ok=True)
        self.out = out
        self.raw = open(out / "raw_measurements.csv", "a", newline="")
        self.runs = open(out / "runs.csv", "a", newline="")
        self.rw, self.uw = csv.writer(self.raw), csv.writer(self.runs)
        if self.raw.tell() == 0:
            self.rw.writerow(self.RAW)
        if self.runs.tell() == 0:
            self.uw.writerow(self.RUNS)
        self.log = open(out / "run.log", "a")

    def say(self, msg):
        line = f"[{datetime.now():%H:%M:%S}] {msg}"
        print(line, flush=True)
        self.log.write(line + "\n")
        self.log.flush()

    def run(self, rnd, order, exp, binary, method, p, n, ref):
        if 3 * 8 * n * n > MEM_BUDGET_B:
            self.say(f"{exp} {binary} {method} p={p} n={n}: NOT RUN (memory budget)")
            return "memory_budget"
        proto = [str(warm_for(n)), str(reps_for(n)), str(MIN_REPS), str(BUDGET_S), str(BATCH_S), str(LIMIT_S)]
        argv = [str(BUILD / binary), method, str(n), str(BLOCK), str(DATA / f"A_{n}.bin"), str(DATA / f"B_{n}.bin")] + proto
        env = dict(os.environ, OMP_NUM_THREADS=str(p), OMP_DYNAMIC="false")
        lines, rc, wall, timed_out, err = run_process(argv, env, WALL_LIMIT_S)
        d = parse(lines)
        for rep, inner, ns in d["samples"]:
            self.rw.writerow([rnd, exp, binary, method, p, n, BLOCK, rep, inner, ns, f"{ns / inner:.1f}", "float64"])
        self.raw.flush()
        if timed_out:
            status, detail = "wall_limit", f"killed after {WALL_LIMIT_S:.0f} s"
        elif d["limit"]:
            status, detail = "time_limit", f"a multiplication exceeded {LIMIT_S:.0f} s"
        elif rc != 0:
            status, detail = "error", f"exit {rc}: {err.strip()[:200]}"
        else:
            status, detail = "ok", ""
        valid = ""
        if status == "ok":
            ok, why = ref.ok(d["check"])
            if method == "omp" and d["threads"] != p:
                ok, why = False, f"reported {d['threads']} threads, expected {p}"
            valid = "yes" if ok else "NO"
            if not ok:
                status, detail = "invalid", why
        k = d["kcpu"] or (float("nan"),) * 3
        self.uw.writerow([datetime.now().isoformat(timespec="seconds"), rnd, order, exp, binary, method, p,
                          d["threads"], n, BLOCK, status, detail, d["warm"], len(d["samples"]), d["rss_kb"],
                          *d["cpu"], *k, d["env"], d["check"][0] if d["check"] else "", valid, f"{wall:.2f}"])
        self.runs.flush()
        med = np.median([ns / inner for _, inner, ns in d["samples"]]) * 1e-9 if d["samples"] else float("nan")
        busy = (k[1] + k[2]) / k[0] if k[0] == k[0] and k[0] > 0 else float("nan")
        self.say(f"r{rnd} {exp} {binary:9s} {method:5s} p={p} n={n:<5d} {status:10s} median {med:9.4f} s "
                 f"cores busy {busy:4.2f}  rss {d['rss_kb'] / 1024:7.1f} MiB {detail}")
        return status


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default=None)
    ap.add_argument("--only", default=None, help="comma-separated experiments to run, e.g. V")
    ap.add_argument("--skip-existing", action="store_true",
                    help="skip configurations that already have an ok row for the same round in runs.csv")
    args = ap.parse_args()
    sizes, vsizes, rounds = (([256, 512], [512], 1) if args.quick else (SIZES, VEC_SIZES, ROUNDS))
    out = Path(args.out) if args.out else ROOT / ("results/quick" if args.quick else "results")
    b = Bench(out)

    t = subprocess.run([sys.executable, str(ROOT / "scripts/run_tests.py")], capture_output=True, text=True)
    b.say("tests: " + (t.stdout.strip().splitlines() or ["(no output)"])[-1])
    if t.returncode != 0:
        b.say("correctness tests failed - refusing to benchmark")
        return 1
    env = collect("-O2 -std=c11 -ffp-contract=off -fopenmp (build/mm); see build.sh for the other builds")
    env["threads_note"] = ("Kernels: sequential (seq, tiles) or OpenMP with OMP_NUM_THREADS = p. No BLAS or other "
                           "library threads inside a benchmark process. NumPy runs only in the harness, between "
                           "benchmark processes, with OPENBLAS_NUM_THREADS=1.")
    env["omp_settings"] = dict(OMP_NUM_THREADS="p (per configuration)", OMP_DYNAMIC="false",
                               OMP_PROC_BIND=os.environ.get("OMP_PROC_BIND", "unset"),
                               OMP_PLACES=os.environ.get("OMP_PLACES", "unset"),
                               OMP_WAIT_POLICY=os.environ.get("OMP_WAIT_POLICY", "unset"),
                               schedule="static (in the source)", libgomp="GCC's OpenMP runtime")
    env["protocol"] = dict(limit_s=LIMIT_S, wall_limit_s=WALL_LIMIT_S, mem_budget_bytes=MEM_BUDGET_B,
                           budget_s=BUDGET_S, min_reps=MIN_REPS, batch_s=BATCH_S, block=BLOCK, rounds=rounds,
                           threads=THREADS, sizes=sizes, vec_sizes=vsizes)
    env["invocation"] = " ".join(sys.argv[1:]) or "(full run)"
    name = "environment_followup.json" if (args.only or args.skip_existing) else "environment.json"
    (out / name).write_text(json.dumps(env, indent=2))
    # runtime baseline of the program (n = 16)
    ln, *_ = run_process([str(BUILD / "mm"), "omp", "16", str(BLOCK), str(DATA / "A_16.bin"), str(DATA / "B_16.bin"),
                          "0", "1", "1", "0", "0", "10"], dict(os.environ, OMP_NUM_THREADS="2"), 60)
    if name == "environment.json":
        (out / "baseline_rss.json").write_text(json.dumps({"mm_omp_p2_n16_kb": parse(ln)["rss_kb"]}))

    done = set()
    if args.skip_existing and (out / "runs.csv").exists():
        with open(out / "runs.csv") as f:
            done = {(int(r["round"]), r["experiment"], r["binary"], r["method"], int(r["threads"]), int(r["n"]))
                    for r in csv.DictReader(f) if r["status"] == "ok"}
    for rnd in range(1, rounds + 1):
        rng = random.Random(20261109 + rnd)
        for n in sizes:
            ref = Ref(n)
            cfg = [("P", "mm", "seq", 1), ("P", "mm", "tiles", 1)] + [("P", "mm", "omp", p) for p in THREADS]
            if n <= CONTROL_MAX_N:
                cfg.append(("N", "mm_nomp", "seq", 1))
            if n in ALIGN_SIZES or args.quick:
                cfg += [("A", "mm_noalign", m, 1) for m in ("seq", "tiles", "omp")]
            if n in vsizes:
                cfg += [("V", bn, m, p) for bn in ("mm_scalar", "mm_simd")
                        for m, p in (("seq", 1), ("omp", 2), ("tiles", 1), ("omp", 1))]
            rng.shuffle(cfg)
            if args.only:
                cfg = [c for c in cfg if c[0] in args.only.split(",")]
            if args.skip_existing:
                cfg = [c for c in cfg if (rnd, *c, n) not in done]
            if not cfg:
                continue
            for order, (exp, bn, m, p) in enumerate(cfg):
                b.run(rnd, order, exp, bn, m, p, n, ref)
    b.say("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
