"""Create the Spark application (driver) for a given cluster configuration and read its metrics.

Framework configuration used in every run (also written to results/spark_config_<config>.json):
  spark.executor.cores        = cores per worker (one executor per worker)
  spark.executor.memory       = 800 MiB per core (JVM heap of each executor)
  spark.cores.max             = workers * cores
  spark.driver.memory         = 1600m (the driver JVM holds the serialised A blocks, B and the returned C blocks)
  spark.python.worker.reuse   = true (default; Python workers and their copy of B survive between tasks)
  spark.task.maxFailures      = 1 (a failure is reported, not silently retried)
  spark.executorEnv.OPENBLAS_NUM_THREADS = 1 (no hidden library threads; NumPy only moves data)
"""
import json
import os
import time
import urllib.request
from pathlib import Path

from pyspark.sql import SparkSession

from cluster import MASTER_URL

ROOT = Path(__file__).resolve().parents[2]
EXECUTOR_MEMORY_PER_CORE_MB = 800
WORKER_MEMORY_PER_CORE_MB = 1000


def spark_conf(workers, cores):
    return {
        "spark.master": MASTER_URL,
        "spark.app.name": f"matmul-{workers}x{cores}",
        "spark.executor.cores": str(cores),
        "spark.executor.memory": f"{EXECUTOR_MEMORY_PER_CORE_MB * cores}m",
        "spark.cores.max": str(workers * cores),
        "spark.driver.memory": "1600m",
        "spark.driver.host": "127.0.0.1",
        "spark.driver.bindAddress": "127.0.0.1",
        "spark.python.worker.reuse": "true",
        "spark.task.maxFailures": "1",
        "spark.ui.showConsoleProgress": "false",
        "spark.executorEnv.OPENBLAS_NUM_THREADS": "1",
        "spark.ui.port": "4040",
        "spark.scheduler.mode": "FIFO",
    }


def start_app(workers, cores, timeout=120):
    conf = spark_conf(workers, cores)
    t0 = time.monotonic()
    b = SparkSession.builder
    for k, v in conf.items():
        b = b.config(k, v)
    spark = b.getOrCreate()
    sc = spark.sparkContext
    sc.setLogLevel("ERROR")
    sc.addFile(str(ROOT / "build" / "libmmkernel.so"))
    sc.addPyFile(str(ROOT / "src" / "spark" / "spark_matmul.py"))
    t1 = time.monotonic()
    while len(executors(sc)) < workers:          # wait until every executor has registered
        if time.monotonic() - t0 > timeout:
            raise RuntimeError("executors did not register")
        time.sleep(0.2)
    t2 = time.monotonic()
    return spark, dict(driver_start_s=t1 - t0, executors_register_s=t2 - t1, app_start_s=t2 - t0), conf


def api(sc, path):
    url = f"{sc.uiWebUrl}/api/v1/applications/{sc.applicationId}/{path}"
    with urllib.request.urlopen(url, timeout=10) as r:
        return json.loads(r.read())


def executors(sc):
    return [e for e in api(sc, "executors") if e["id"] != "driver"]


def job_metrics(sc, group, timeout=10):
    """Aggregate stage metrics of all jobs of one job group (one multiplication)."""
    t0 = time.monotonic()
    while True:
        jobs = [j for j in api(sc, "jobs") if j.get("jobGroup") == group]
        if jobs and all(j["status"] in ("SUCCEEDED", "FAILED") for j in jobs):
            stage_ids = sorted({s for j in jobs for s in j["stageIds"]})
            stages = []
            for sid in stage_ids:
                for st in api(sc, f"stages/{sid}"):
                    if st["status"] in ("COMPLETE", "FAILED"):
                        stages.append(st)
            if len(stages) >= len([s for j in jobs for s in j["stageIds"]]) - sum(j.get("numSkippedStages", 0) for j in jobs):
                break
        if time.monotonic() - t0 > timeout:
            break
        time.sleep(0.2)
    keys = ["executorRunTime", "executorCpuTime", "executorDeserializeTime", "resultSize", "jvmGcTime",
            "resultSerializationTime", "shuffleReadBytes", "shuffleWriteBytes", "inputBytes", "outputBytes",
            "numCompleteTasks", "numFailedTasks", "memoryBytesSpilled", "diskBytesSpilled"]
    agg = {k: sum(st.get(k, 0) or 0 for st in stages) for k in keys}
    agg["stages"] = len(stages)
    agg["jobs_failed"] = sum(j["status"] == "FAILED" for j in jobs) if jobs else -1
    return agg
