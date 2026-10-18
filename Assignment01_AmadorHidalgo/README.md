# Assignment 1 - Basic Matrix Multiplication in Python, Java and C

Big Data, Grado en Ciencia e Ingeniería de Datos, ULPGC - academic year 2026-2027.
Author: Hugo Amador Hidalgo.

Research question: *how does the matrix size n affect the runtime and the memory use of the
straightforward triple-loop multiplication of two n x n float64 matrices in Python, Java and C,
under one documented set of execution conditions?* The full study is in
`report/Assignment01_AmadorHidalgo_Report.pdf`.

## What is in this repository

| Path | Content |
|---|---|
| `src/c/matmul.c`, `src/java/MatMul.java`, `src/python/matmul.py` | The three implementations (i-j-k triple loop, no library multiplication) |
| `scripts/gen_inputs.py` | Generates the **shared input files** `data/A_<n>.bin`, `data/B_<n>.bin` and checks them against `data/manifest.csv` (SHA-256) |
| `scripts/run_tests.py` | Correctness tests (identity, zero, hand-computed products, random vs NumPy, bitwise C = Java = Python) |
| `scripts/run_benchmarks.py` | Benchmark harness (warm-up, repetitions, peak memory, validation, raw CSV) |
| `scripts/rss_run.c` | Tiny wrapper that reports the peak resident memory of the benchmarked process |
| `scripts/analyze.py` | Tables (`summary*.csv`) and figures from the raw measurements |
| `scripts/build_report.py` | Builds the PDF report from the results |
| `scripts/env_info.py` | Records hardware / software versions into `results/environment.json` |
| `results/` | `raw_measurements.csv` (every sample), `runs.csv` (one row per process), `summary.csv`, `summary_s1.csv`, `correctness.csv`, `environment.json`, `baseline_rss.json`, `figures/` |
| `report/` | PDF report |

## Requirements

* Python 3.10+ with the packages in `requirements.txt` (`pip install -r requirements.txt`).
  NumPy is used **only** to generate inputs and as the trusted reference for validation.
* A C compiler: `gcc` on Linux; on macOS `gcc` is Apple clang, which also works (`CC=clang ./build.sh`).
* JDK 17 or newer (`javac`, `java`).

Versions used for the published results are in `results/environment.json`.

## Exact commands

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
./build.sh                       # C at -O0/-O2/-O3, Java class, peak-memory wrapper
python3 scripts/gen_inputs.py    # creates data/*.bin (~165 MB) and verifies them
python3 scripts/run_tests.py     # must print "53/53 correctness checks passed"
python3 scripts/run_benchmarks.py
python3 scripts/analyze.py
python3 scripts/build_report.py
```

`./run_all.sh` runs the whole chain. A quick trial: `python3 scripts/run_benchmarks.py --sizes 16,64,256 --main-only --out /tmp/trial`.

Run the benchmarks on an otherwise idle machine, plugged in, without other heavy programs: the
harness runs one configuration at a time, but it cannot protect the measurements from other load.

## Command-line interface of the three programs

```
matmul  n A.bin B.bin warmup_s reps min_reps budget_s batch_s limit_s [C_out.bin]
```

`warmup_s` minimum seconds of warm-up (0 = none), `reps` maximum samples, `min_reps` samples always
collected, `budget_s` stop sampling after this much measured time, `batch_s` target length of one
sample (very short kernels are repeated inside a sample and the time per multiplication is
reported), `limit_s` a single multiplication longer than this stops the program.
Records on stdout: `WARMUP`, `RESULT rep inner batch_ns`, `LIMIT`, `CHECK` (checksum + 3 sample
values as IEEE-754 hex). Run directly, for example:

```
build/c/matmul_O2 512 data/A_512.bin data/B_512.bin 1 5 3 300 0.1 180
java -XX:+UseSerialGC -cp build/java MatMul 512 data/A_512.bin data/B_512.bin 1 5 3 300 0.1 180
python3 src/python/matmul.py 512 data/A_512.bin data/B_512.bin 0 3 3 300 0.1 180
```

## Experimental protocol (summary)

* **Question / factor / outcome:** factor = n (16 ... 2048); outcome = time per multiplication and peak memory.
* **Inputs:** uniform [0,1) float64, NumPy PCG64, seed 20261012 + n (A) and 20261013 + n (B); stored as
  little-endian binary files that all three programs read, so the logical inputs are identical.
* **Timer:** monotonic clock inside each process, around the multiplication kernel only. Inputs are read
  and C is allocated before timing. The result is consumed afterwards (checksum printed and validated).
* **Warm-up:** until >= 1 s of warm-up multiplications (at least one) for C and Java and for CPython at n <= 256.
  For CPython at n >= 512 no explicit warm-up is requested (no JIT, each run > 10 s), but the program then runs one
  untimed *calibration* multiplication to size the sample batch, which acts as a warm-up. The programs' `WARMUP`
  counter shows 0 for those runs; `summary.csv` counts it as 1 (see the report, Section 3.4).
* **Samples:** 10 for n <= 256, 5 for n >= 512; at least 3 and at most ~300 s of measured time for costly runs.
  Sizes where a multiplication takes < 0.1 s use batches, and the *time per multiplication* is stored.
  Median and IQR are reported; every raw sample is kept.
* **Memory:** peak resident set size of the process (kernel statistic `ru_maxrss` through `rss_run`),
  next to the runtime baseline (n = 1) and the ideal array storage 24 n^2 bytes.
* **Supplementary experiments (kept apart from the main comparison):** S1 = C at -O0/-O2/-O3 (n = 256, 512, 1024);
  S2 = C and Java at n = 1000, 1024, 1025. `run_benchmarks.py --only-s2` appends S2 to the existing CSV files.
* **Budgets:** one multiplication <= 180 s; one configuration <= 15 min wall clock; peak RSS <= 4 GiB.
  Over-budget runs are reported as such, without invented times.
* **Correctness:** known-answer and random tests, tolerance `2*gamma_n` (derivation in `scripts/run_tests.py`),
  bitwise equality between the three languages.

## Regenerating the inputs

`python3 scripts/gen_inputs.py` recreates every `data/*.bin` and compares SHA-256 digests with
`data/manifest.csv`. The binaries are not stored in git (165 MB); the manifest is.

## Use of generative AI

An AI assistant (Claude, Anthropic) was used for drafting code, the experimental scripts and the first
draft of the report text. All results come from the measurements stored in `results/`. See the
disclosure section of the report.
