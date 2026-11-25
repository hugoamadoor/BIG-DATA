# Assignment 4: Distributed Matrix Multiplication with Apache Spark

Big Data, Grado en Ciencia e Ingeniería de Datos, ULPGC, 2026-2027.
Author: Hugo Amador Hidalgo.
Report: `report/Assignment04_AmadorHidalgo_Report.pdf`

## Research question

How does a distributed matrix multiplication on Apache Spark scale with the number of workers, and how is
its time divided between computation, communication and framework overhead? How does it compare with
the sequential and parallel versions of the earlier assignments?

## Environment

**This is a local multi-worker execution.** All Spark processes run on one physical machine: a 2-vCPU
cloud VM with 7.8 GiB of RAM, talking over the loopback interface (127.0.0.1). The results describe
framework and process overhead. They do not measure inter-machine network performance or the combined
memory of several machines. The assignment allows local execution only with the instructor's
permission.

| term | here |
|---|---|
| physical machines | 1 |
| worker | a Spark standalone worker daemon (one JVM). Each worker runs one executor (JVM) for the application |
| core | a task slot of an executor. Each PySpark task runs in its own Python worker process forked by the executor |
| driver | this Python process and its JVM. It holds A, B and C |

Configurations (workers × cores per worker): `1x1`, `2x1`, `4x1` and `1x2`. `4x1` runs 4 executors on 2
vCPUs, so it is oversubscribed. `1x2` runs one executor with two task slots.

The benchmark process runs inside a 6 GiB memory limit (cgroup), and every executor JVM and every
Python worker keeps its own copy of B. Preliminary attempts at n = 4096, and with `4x1` at n = 3072,
were stopped by the OOM killer (`results/oom_events.txt`). The largest size is therefore n = 3072, and
`4x1` stops at n = 2048. Skipped or failed configurations are recorded in `results/not_run.csv`.

## Strategies (`src/spark/spark_matmul.py`)

| strategy | partitioning | communication | result |
|---|---|---|---|
| `rowblock` (main) | A split into P = 8 row blocks (one per task), B broadcast to every executor | A once, B once per executor, C back to the driver | bit-identical to the sequential baseline |
| `block2d` (comparison) | 2×2 grid of A and B blocks. A join pairs blocks (I,K) and (K,J), and `reduceByKey` adds the partial products | two shuffles | within rounding of the sequential result |

Inside every task, the computation is the blocked C kernel of Assignments 2 and 3
(`src/kernel/mmkernel.c`, built as `build/libmmkernel.so` and called through `ctypes`). The arithmetic is
therefore the same as in the single-process versions, and only distribution and framework costs differ.

## References on the same machine

The reference programs are copied unchanged into `src/reference/`:

- `build/dense ijk`: the basic triple loop (Assignment 1 kernel, via Assignment 2).
- `build/mm seq`: the blocked sequential baseline (Assignment 3).
- `build/mm omp` with 2 threads: the parallel version (Assignment 3).

## Requirements

- Linux or macOS.
- gcc.
- Java 17 or 21 (tested with OpenJDK 21).
- Python 3.10+ with `pip install -r requirements.txt` (includes `pyspark==4.0.1`, which bundles Spark).

## Reproduce

```bash
./build.sh
python3 scripts/gen_inputs.py        # inputs + data/manifest.csv (SHA-256)
python3 scripts/run_tests.py         # 91 checks, including on a 2-worker Spark cluster
python3 scripts/run_benchmarks.py    # references + Spark experiments, 2 rounds (~1 h on 2 vCPUs)
python3 scripts/analyze.py
python3 scripts/build_report.py
```

`./run_all.sh` runs all of the above. `python3 scripts/run_benchmarks.py --quick` is a 5-minute smoke test
that writes to `results/quick/`.

The harness starts and stops the cluster itself (`src/spark/cluster.py`):

- one master on `spark://127.0.0.1:7077`
- W workers, each launched with `spark-class org.apache.spark.deploy.worker.Worker --cores c --memory ...`

The Spark settings used for every run are in `src/spark/session.py` and in
`results/spark_config_<config>.json`.

## Measurement boundary

**End-to-end** runs from A and B as NumPy arrays in driver memory to C as a NumPy array in driver memory.
It equals broadcast + job + assemble.

Cluster start-up, application start-up and reading the input files are reported separately
(`results/startup.csv`).

Inside the job, the tasks report their own timestamps and the time spent in the C kernel. Activities of
different tasks overlap, so only a critical-path split (kernel versus everything else) is given, never a
sum of overlapping parts.

## Results files

| file | content |
|---|---|
| `results/spark_runs.csv` | one row per multiplication. Columns include phase times, kernel critical path, Spark stage metrics (run time, CPU, GC, deserialisation, result bytes, shuffle bytes), bytes of A, B and C, peak RSS per executor tree and of the driver, and validation |
| `results/spark_tasks.csv` | one row per task: worker, Python process, timestamps, kernel time |
| `results/reference_raw.csv`, `reference_runs.csv` | references |
| `results/startup.csv` | cluster and application start-up per configuration and round |
| `results/summary_*.csv`, `key_numbers.json`, `figures/` | analysis |
| `results/environment.json`, `spark_config_*.json`, `correctness.csv` | environment, framework configuration, tests |
