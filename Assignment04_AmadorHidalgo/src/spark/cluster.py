"""Start and stop a Spark standalone cluster on this machine, and sample its memory.

Topology (all on ONE physical machine, loopback network 127.0.0.1):
  * 1 master process (JVM, control plane only, does no computation),
  * W worker daemons (one JVM each), each offering `cores` cores and `memory` to the master,
  * per worker, one executor JVM launched by the worker for the application, which runs `cores`
    tasks at a time; every PySpark task runs in a Python worker process forked by that executor,
  * the driver: this Python process plus its own JVM (py4j gateway).
So "workers" = worker daemons = executors here (one executor per worker), "processes" are listed in
the run records, and there is exactly one physical machine.

The memory sampler records, every `interval` seconds, the resident set size (RSS) of each worker's
executor process tree (executor JVM + its Python workers) and of the driver tree, and keeps the peaks.
"""
import json
import os
import shutil
import subprocess
import threading
import time
import urllib.request
from pathlib import Path

import psutil

SPARK_HOME = os.environ.get("SPARK_HOME") or str(Path(__import__("pyspark").__file__).parent)
MASTER_URL = "spark://127.0.0.1:7077"
MASTER_UI = "http://127.0.0.1:8080/json/"


def _get_json(url, timeout=2):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read())


class LocalCluster:
    def __init__(self, workers, cores, memory, workdir):
        self.workers, self.cores, self.memory = workers, cores, memory
        self.workdir = Path(workdir)
        self.procs, self.timings = [], {}

    def _spawn(self, args, log):
        env = dict(os.environ, SPARK_HOME=SPARK_HOME, SPARK_LOCAL_IP="127.0.0.1", OPENBLAS_NUM_THREADS="1")
        f = open(log, "w")
        p = subprocess.Popen([f"{SPARK_HOME}/bin/spark-class"] + args, stdout=f, stderr=subprocess.STDOUT,
                             env=env, start_new_session=True)
        self.procs.append(p)
        return p

    def start(self, timeout=120):
        if self.workdir.exists():
            shutil.rmtree(self.workdir)
        self.workdir.mkdir(parents=True)
        t0 = time.monotonic()
        self.master = self._spawn(["org.apache.spark.deploy.master.Master", "--host", "127.0.0.1", "--port", "7077",
                                   "--webui-port", "8080"], self.workdir / "master.log")
        while True:
            try:
                if _get_json(MASTER_UI)["status"] == "ALIVE":
                    break
            except Exception:
                pass
            if time.monotonic() - t0 > timeout:
                raise RuntimeError("master did not start")
            time.sleep(0.2)
        t1 = time.monotonic()
        self.worker_procs = []
        for w in range(self.workers):
            wd = self.workdir / f"worker{w}"
            wd.mkdir()
            self.worker_procs.append(self._spawn(
                ["org.apache.spark.deploy.worker.Worker", "--host", "127.0.0.1", "--cores", str(self.cores),
                 "--memory", self.memory, "--webui-port", str(8090 + w), "--work-dir", str(wd), MASTER_URL],
                self.workdir / f"worker{w}.log"))
        while len([x for x in _get_json(MASTER_UI)["workers"] if x["state"] == "ALIVE"]) < self.workers:
            if time.monotonic() - t0 > timeout:
                raise RuntimeError("workers did not register")
            time.sleep(0.2)
        t2 = time.monotonic()
        self.timings = dict(master_start_s=t1 - t0, workers_register_s=t2 - t1, cluster_start_s=t2 - t0)
        return self.timings

    def stop(self):
        for p in reversed(self.procs):
            try:
                for c in psutil.Process(p.pid).children(recursive=True):
                    c.kill()
                p.kill()
                p.wait(10)
            except (psutil.NoSuchProcess, ProcessLookupError):
                pass
        self.procs = []
        time.sleep(1)

    def executor_trees(self):
        """{worker index: [executor JVM process]} - children of each worker daemon."""
        out = {}
        for w, p in enumerate(self.worker_procs):
            try:
                kids = [c for c in psutil.Process(p.pid).children(recursive=True)
                        if "java" in c.name() and "CoarseGrainedExecutorBackend" in " ".join(c.cmdline())]
                out[w] = kids
            except psutil.NoSuchProcess:
                out[w] = []
        return out


def tree_rss(proc):
    try:
        procs = [proc] + proc.children(recursive=True)
    except psutil.NoSuchProcess:
        return 0
    total = 0
    for p in procs:
        try:
            total += p.memory_info().rss
        except psutil.NoSuchProcess:
            pass
    return total


def driver_jvms(proc):
    """The driver's own JVM(s): children running SparkSubmit (the py4j gateway), not the cluster daemons
    that the harness also starts as child processes."""
    out = []
    for c in proc.children(recursive=False):
        try:
            if "java" in c.name() and "org.apache.spark.deploy.SparkSubmit" in " ".join(c.cmdline()):
                out.append(c)
        except psutil.NoSuchProcess:
            pass
    return out


def driver_rss(proc):
    """RSS of the driver: this Python process plus its SparkSubmit JVM tree."""
    try:
        return proc.memory_info().rss + sum(tree_rss(j) for j in driver_jvms(proc))
    except psutil.NoSuchProcess:
        return 0


class MemorySampler(threading.Thread):
    """Samples RSS of each executor tree and of the driver tree; peaks since the last reset()."""

    def __init__(self, cluster, interval=0.1):
        super().__init__(daemon=True)
        self.cluster, self.interval = cluster, interval
        self.driver = psutil.Process(os.getpid())
        self.lock = threading.Lock()
        self.running = True
        self.reset()

    def reset(self):
        with self.lock:
            self.peak_exec, self.peak_driver, self.samples = {}, 0, 0

    def peaks(self):
        with self.lock:
            return dict(self.peak_exec), self.peak_driver, self.samples

    def run(self):
        while self.running:
            trees = self.cluster.executor_trees()
            d = driver_rss(self.driver)
            with self.lock:
                for w, procs in trees.items():
                    r = sum(tree_rss(p) for p in procs)
                    self.peak_exec[w] = max(self.peak_exec.get(w, 0), r)
                self.peak_driver = max(self.peak_driver, d)
                self.samples += 1
            time.sleep(self.interval)

    def stop(self):
        self.running = False
