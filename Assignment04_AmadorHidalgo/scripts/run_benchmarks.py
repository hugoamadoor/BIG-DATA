#!/usr/bin/env python3
"""Benchmark harness for Assignment 4 (distributed matrix multiplication with Apache Spark).

Environment: ONE physical machine (2-vCPU cloud VM). The Spark standalone cluster runs as several
processes on that machine and communicates over the loopback interface; this is a LOCAL multi-worker
execution and measures framework and process overhead, not inter-machine network performance.

Experiments
  R  references       Assignment 3 program (build/mm): seq = blocked sequential baseline,
                      omp p = 2 = OpenMP version on both cores; Assignment 2 program (build/dense):
                      ijk = basic triple loop (n <= 2048). Same machine, same inputs, same session.
  D  distributed      Spark cluster configurations CONFIGS (workers x cores per worker); strategy
                      rowblock (P = 8 row blocks, B broadcast) in every configuration, strategy block2d
                      (2 x 2 grid, shuffles) in BLOCK2D_CONFIGS. Sizes SIZES (strong scaling: n fixed
                      while resources change).
  G  granularity      configuration 2 x 1, n = 2048, rowblock with P in GRAIN_PARTS.

Protocol
  * Each configuration: start a fresh cluster (timed), start the application (timed), then for each n:
    one warm-up multiplication (recorded as rep 0, excluded from the statistics) and REPS(n) measured ones.
  * End-to-end boundary: from A and B held as NumPy arrays in driver memory to C held as a NumPy array in
    driver memory. Cluster start-up, application start-up and reading the input files are outside it and
    are reported separately.
  * ROUNDS rounds; the second runs the configurations in reverse order.
  * Every result is validated (rowblock: SHA-256 of C equal to the sequential baseline's C; block2d and
    references: |C - C_ref| <= 2 gamma_n |A||B| element-wise, with the NumPy product as C_ref).

Usage: python3 scripts/run_benchmarks.py [--quick]
"""
import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src" / "spark"))
sys.path.insert(0, str(ROOT / "scripts"))
from cluster import LocalCluster, MemorySampler  # noqa: E402
from env_info import collect  # noqa: E402
from session import WORKER_MEMORY_PER_CORE_MB, job_metrics, spark_conf, start_app  # noqa: E402
import spark_matmul as sm  # noqa: E402

DATA, BUILD = ROOT / "data", ROOT / "build"
U = 2.0 ** -53

SIZES = [1024, 2048, 3072]
CONFIGS = [(1, 1), (2, 1), (4, 1), (1, 2)]         # (worker processes, cores per worker)
# Memory budget: the benchmark runs inside a 6 GiB memory limit (cgroup). Preliminary runs at n = 4096
# were stopped by the OOM killer (results/oom_events.txt): every executor JVM and every Python worker
# holds its own copy of B, so the largest size is 3072. With 4 executors even n = 3072 exceeded the limit
# in a preliminary attempt (OOM after two multiplications), so 4 x 1 stops at 2048. A configuration that
# still fails is recorded in not_run.csv and the harness continues.
MAX_N = {(4, 1): 2048}
BLOCK2D_MAX_N = 3072
BLOCK2D_CONFIGS = [(1, 1), (2, 1)]
PARTS = 8
GRID = 2
GRAIN_N, GRAIN_PARTS = 2048, [2, 4, 8, 32, 64]
ROUNDS = 2
REF_IJK_MAX_N = 2048
LIMIT_S = 180.0


def reps(n):
    return 3 if n >= 3072 else 5


def gamma(k):
    return k * U / (1 - k * U)


class Out:
    def __init__(self, d):
        d.mkdir(parents=True, exist_ok=True)
        self.d, self.files = d, {}
        self.log = open(d / "run.log", "a")

    def row(self, name, rec):
        new = name not in self.files
        if new:
            p = self.d / f"{name}.csv"
            exists = p.exists() and p.stat().st_size > 0
            f = open(p, "a", newline="")
            w = csv.DictWriter(f, fieldnames=list(rec), extrasaction="ignore")
            if not exists:
                w.writeheader()
            self.files[name] = (f, w)
        f, w = self.files[name]
        w.writerow(rec)
        f.flush()

    def say(self, msg):
        line = f"[{datetime.now():%H:%M:%S}] {msg}"
        print(line, flush=True)
        self.log.write(line + "\n")
        self.log.flush()


def load(n):
    return (np.fromfile(DATA / f"A_{n}.bin", dtype="<f8").reshape(n, n),
            np.fromfile(DATA / f"B_{n}.bin", dtype="<f8").reshape(n, n))


def check_bound(C, ref, n):
    """Inputs are non-negative, so |A||B| = A B = ref."""
    err = np.abs(C - ref)
    return bool(np.all(err <= 2 * gamma(n) * ref)), float(err.max())


# ------------------------------------------------------------------ references ------------------
def run_reference(out, rnd, prog, method, threads, n, write_c=None):
    proto = ["1" if n <= 1024 else "0", str(reps(n)), "3", "300", "0.1", str(LIMIT_S)]
    if prog == "mm":
        argv = [str(BUILD / "mm"), method, str(n), "32"]
    else:
        argv = [str(BUILD / "dense"), method, str(n), "0"]
    argv += [str(DATA / f"A_{n}.bin"), str(DATA / f"B_{n}.bin")] + proto + ([str(write_c)] if write_c else [])
    env = dict(os.environ, OMP_NUM_THREADS=str(threads), OMP_DYNAMIC="false")
    t0 = time.monotonic()
    r = subprocess.run([str(BUILD / "rss_run")] + argv, capture_output=True, text=True, env=env,
                       timeout=LIMIT_S * 8)
    wall = time.monotonic() - t0
    samples, rss, limit = [], float("nan"), False
    for ln in r.stdout.splitlines():
        f = ln.split()
        if f and f[0] == "RESULT":
            samples.append(int(f[3]) / int(f[2]) * 1e-9)
            out.row("reference_raw", dict(round=rnd, program=prog, method=method, threads=threads, n=n,
                                          rep=int(f[1]), seconds=samples[-1]))
        elif f and f[0] == "RUSAGE_MAXRSS_KB":
            rss = float(f[1])
        elif f and f[0] == "LIMIT":
            limit = True
    status = "time_limit" if limit else ("ok" if r.returncode == 0 else f"error {r.returncode}")
    out.row("reference_runs", dict(round=rnd, program=prog, method=method, threads=threads, n=n, status=status,
                                   samples=len(samples), median_s=float(np.median(samples)) if samples else "",
                                   peak_rss_kb=rss, wall_s=round(wall, 2)))
    out.say(f"r{rnd} R {prog}:{method} p={threads} n={n} {status} median "
            f"{np.median(samples) if samples else float('nan'):.4f} s rss {rss / 1024:.1f} MiB")
    return status


# ------------------------------------------------------------------ one Spark multiplication ----
def spark_mult(out, ctx, rnd, cfg, strategy, parts, n, rep, A, B, ref, ref_hash, exp="D"):
    sc, cl, ms = ctx["sc"], ctx["cluster"], ctx["sampler"]
    W, c = cfg
    group = f"r{rnd}-{W}x{c}-{strategy}-{parts}-{n}-{rep}-{exp}"
    sc.setJobGroup(group, group)
    ms.reset()
    if strategy == "rowblock":
        C, m = sm.rowblock(sc, A, B, parts)
    else:
        C, m = sm.block2d(sc, A, B, parts)
    peak_exec, peak_driver, nsamp = ms.peaks()
    jm = job_metrics(sc, group)
    # outside the timed region: let both the Python and the driver JVM release the data of this run
    import gc
    gc.collect()
    sc._jvm.System.gc()
    # map Python-worker pids to worker index
    pid2w = {}
    for w, procs in cl.executor_trees().items():
        for p in procs:
            try:
                for ch in [p] + p.children(recursive=True):
                    pid2w[ch.pid] = w
            except Exception:
                pass
    tasks = m["tasks"]
    lanes = {}
    for t in tasks:
        lanes.setdefault(t["pid"], 0.0)
        lanes[t["pid"]] += t["kernel_s"]
        out.row("spark_tasks", dict(round=rnd, experiment=exp, config=f"{W}x{c}", strategy=strategy, parts=parts, n=n,
                                    rep=rep, worker=pid2w.get(t["pid"], -1), pid=t["pid"], partition=t["partition"],
                                    stage=t["stage"], rows=t["rows"], kernel_s=t["kernel_s"],
                                    start_rel_s=t["t_start"] - m["job_wall_start"],
                                    got_b_rel_s=(t.get("t_got_b", t["t_start"]) - m["job_wall_start"]),
                                    end_rel_s=t["t_end"] - m["job_wall_start"],
                                    job_end_rel_s=m["job_wall_end"] - m["job_wall_start"]))
    if strategy == "rowblock":
        h = hashlib.sha256(np.ascontiguousarray(C).tobytes()).hexdigest()
        valid, detail = h == ref_hash, "" if h == ref_hash else "C differs from the sequential baseline"
        if not valid:
            valid, err = check_bound(C, ref, n)
            detail += f"; within bound: {valid}"
            valid = False
    else:
        valid, err = check_bound(C, ref, n)
        detail = f"max abs err {err:.3e}"
    rec = dict(round=rnd, experiment=exp, config=f"{W}x{c}", workers=W, cores_per_worker=c, total_cores=W * c,
               strategy=strategy, parts=parts, n=n, rep=rep, warmup=rep == 0,
               end_to_end_s=m["end_to_end_s"], broadcast_s=m["broadcast_s"], job_s=m["job_s"], assemble_s=m["assemble_s"],
               crit_kernel_s=max(lanes.values()), sum_kernel_s=sum(lanes.values()),
               first_task_start_s=min(t["t_start"] for t in tasks) - m["job_wall_start"],
               b_fetch_max_s=max(t.get("t_got_b", t["t_start"]) - t["t_start"] for t in tasks),
               after_last_task_s=m["job_wall_end"] - max(t["t_end"] for t in tasks),
               n_tasks=m["n_tasks"], python_workers=len(lanes), executors_used=len({pid2w.get(p, -1) for p in lanes}),
               run_time_ms=jm["executorRunTime"], cpu_time_ms=jm["executorCpuTime"] / 1e6,
               deserialize_ms=jm["executorDeserializeTime"], gc_ms=jm["jvmGcTime"], result_bytes=jm["resultSize"],
               shuffle_read_bytes=jm["shuffleReadBytes"], shuffle_write_bytes=jm["shuffleWriteBytes"],
               spilled_bytes=jm["memoryBytesSpilled"] + jm["diskBytesSpilled"], stages=jm["stages"],
               failed_tasks=jm["numFailedTasks"], bytes_A=m["bytes_A"], bytes_B_broadcast=m["bytes_B_broadcast"],
               bytes_C=m["bytes_C"], peak_rss_exec_max_mib=max(peak_exec.values()) / 2 ** 20 if peak_exec else "",
               peak_rss_exec_sum_mib=sum(peak_exec.values()) / 2 ** 20, peak_rss_driver_mib=peak_driver / 2 ** 20,
               rss_samples=nsamp, valid="yes" if valid else "NO", detail=detail)
    out.row("spark_runs", rec)
    out.say(f"r{rnd} {exp} {W}x{c} {strategy:8s} P={parts:<2d} n={n:<5d} rep={rep} e2e {m['end_to_end_s']:8.3f} s "
            f"job {m['job_s']:8.3f} crit-kernel {rec['crit_kernel_s']:8.3f} valid={rec['valid']}")
    return valid


def stop_driver_jvm():
    """Shut down the driver's JVM (py4j gateway) so every configuration starts a fresh driver and the
    driver memory of one configuration is not carried into the next."""
    import psutil
    from pyspark import SparkContext
    gw = SparkContext._gateway
    if gw is not None:
        try:
            gw.shutdown()
        except Exception:
            pass
    from cluster import driver_jvms
    from pyspark.sql import SparkSession
    for ch in driver_jvms(psutil.Process()):
        try:
            ch.kill()
            ch.wait(10)
        except psutil.NoSuchProcess:
            pass
    SparkContext._gateway = None
    SparkContext._jvm = None
    SparkContext._active_spark_context = None
    SparkSession._instantiatedSession = None
    SparkSession._activeSession = None


def run_config(out, rnd, cfg, sizes, refs, hashes, quick):
    """One fresh cluster and application per (configuration, n), so that memory left over from one size
    cannot affect the next; start-up is therefore measured once per (configuration, n)."""
    W, c = cfg
    for n in sizes:
        if n > MAX_N.get(cfg, 10 ** 9):
            out.row("not_run", dict(round=rnd, config=f"{W}x{c}", n=n, reason="memory budget (6 GiB container)"))
            out.say(f"r{rnd} {W}x{c} n={n}: NOT RUN (memory budget)")
            continue
        try:
            run_config_n(out, rnd, cfg, n, refs, hashes, quick)
        except Exception as e:                              # record and go on
            out.row("not_run", dict(round=rnd, config=f"{W}x{c}", n=n, reason=f"failed: {e!r}"[:300]))
            out.say(f"r{rnd} {W}x{c} n={n}: FAILED {e!r}"[:300])
            stop_driver_jvm()


def run_config_n(out, rnd, cfg, n, refs, hashes, quick):
    W, c = cfg
    cl = LocalCluster(W, c, f"{WORKER_MEMORY_PER_CORE_MB * c}m", ROOT / "build" / "cluster")
    try:
        ct = cl.start()
        spark, at, conf = start_app(W, c)
        (out.d / f"spark_config_{W}x{c}.json").write_text(json.dumps(conf, indent=2))
        sc = spark.sparkContext
        ms = MemorySampler(cl)
        ms.start()
        out.row("startup", dict(round=rnd, config=f"{W}x{c}", n=n, workers=W, cores_per_worker=c, **ct, **at,
                                processes=f"1 master + {W} worker daemons + {W} executor JVMs + Python workers "
                                          f"+ driver (Python + JVM)", physical_machines=1))
        out.say(f"r{rnd} {W}x{c} n={n}: cluster {ct['cluster_start_s']:.1f} s, application {at['app_start_s']:.1f} s")
        ctx = dict(sc=sc, cluster=cl, sampler=ms)
        A, B = load(n)
        strategies = [("rowblock", PARTS)] + ([("block2d", GRID)] if cfg in BLOCK2D_CONFIGS else [])
        if cfg in BLOCK2D_CONFIGS and n > BLOCK2D_MAX_N:
            strategies = strategies[:1]
            out.row("not_run", dict(round=rnd, config=f"{W}x{c}", n=n, reason="block2d: memory budget (6 GiB container)"))
        for strat, parts in strategies:
            for rep in range(0, reps(n) + 1):
                spark_mult(out, ctx, rnd, cfg, strat, parts, n, rep, A, B, refs[n], hashes[n])
        if cfg == (2, 1) and n == GRAIN_N and not quick:
            for parts in GRAIN_PARTS:
                if parts == PARTS:
                    continue
                for rep in range(0, reps(n) + 1):
                    spark_mult(out, ctx, rnd, cfg, "rowblock", parts, n, rep, A, B, refs[n], hashes[n], exp="G")
        del A, B
        ms.stop()
        spark.stop()
        stop_driver_jvm()
    finally:
        cl.stop()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    sizes = [256, 512] if args.quick else SIZES
    rounds = 1 if args.quick else ROUNDS
    out = Out(ROOT / ("results/quick" if args.quick else "results"))

    t = subprocess.run([sys.executable, str(ROOT / "scripts/run_tests.py")], capture_output=True, text=True)
    out.say("tests: " + (t.stdout.strip().splitlines() or ["(no output)"])[-1])
    if t.returncode != 0:
        out.say("correctness tests failed - refusing to benchmark")
        return 1
    env = collect("-O2 -std=c11 -ffp-contract=off -falign-loops=32 (kernel, references)")
    import pyspark
    env.update(spark_version=pyspark.__version__, java=subprocess.run(["java", "-version"], capture_output=True,
                                                                      text=True).stderr.splitlines()[-3:],
               environment_label="LOCAL multi-worker execution on one physical machine (2 vCPUs); loopback network",
               configs=[f"{w}x{c}" for w, c in CONFIGS], sizes=sizes, parts=PARTS, grid=GRID, rounds=rounds)
    (out.d / "environment.json").write_text(json.dumps(env, indent=2))

    refs, hashes = {}, {}
    for n in sizes:
        A, B = load(n)
        refs[n] = A @ B
        del A, B
    for rnd in range(1, rounds + 1):
        for n in sizes:
            cpath = BUILD / f"C_seq_{n}.bin"
            run_reference(out, rnd, "mm", "seq", 1, n, write_c=cpath)
            hashes[n] = hashlib.sha256(cpath.read_bytes()).hexdigest()
            cpath.unlink()
            run_reference(out, rnd, "mm", "omp", 2, n)
            if n <= REF_IJK_MAX_N and rnd == 1:      # the basic version is slow (~80 s at n = 2048): round 1 only
                run_reference(out, rnd, "dense", "ijk", 1, n)
        (out.d / "reference_hashes.json").write_text(json.dumps(hashes, indent=2))
        for cfg in (CONFIGS if rnd % 2 else list(reversed(CONFIGS))):
            run_config(out, rnd, cfg, sizes, refs, hashes, args.quick)
    out.say("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
