# Assignment 3: Parallel Matrix Multiplication and Optional Vectorisation

Big Data, Grado en Ciencia e Ingeniería de Datos, ULPGC, 2026-2027.
Author: Hugo Amador Hidalgo.
Report: `report/Assignment03_AmadorHidalgo_Report.pdf`

## Research question

How does the number of OpenMP threads change the runtime, speed-up, efficiency and memory of the
blocked matrix-multiplication kernel, as a function of n, on a 2-vCPU machine? When does parallel
execution help, and what does running more threads than hardware contexts do?

The optional vectorisation extension asks a second question: how much does AVX2 auto-vectorisation add,
measured separately from the effect of threads?

## What is reused

- **Sequential baseline.** `mm_blocked` from `Assignment02_AmadorHidalgo/src/dense.c` (commit `1d0726d` of
  this repository), copied unchanged into `src/matmul_par.c`. It is blocked i-k-j with 32×32 tiles, the
  fastest sequential method of Assignment 2. The tile size of 32 is reused and not re-tuned.
- **Timing harness.** `src/harness.h`, the same as Assignment 2.
- **Peak-RSS wrapper.** `scripts/rss_run.c`, now also reporting CPU time.
- **Inputs.** The same generator and seeds as Assignments 1 and 2. The SHA-256 values in
  `data/manifest.csv` match the Assignment 2 manifest for all shared sizes.

## Methods (all in `src/matmul_par.c`)

| method | description |
|---|---|
| `seq` | sequential baseline (Assignment 2 kernel, verbatim) |
| `omp` | same kernel; `#pragma omp for schedule(static)` over row tiles. Each tile is a disjoint block of rows of C, so there are no shared writes |
| `tiles` | diagnostic: the parallel code path (`row_tile` called in a plain loop) without OpenMP |

All three methods give bit-identical results for every thread count, because every C[i,j] receives its
products in the same order.

## Builds (`build.sh`)

| binary | flags | used in |
|---|---|---|
| `build/mm` | `-O2 -std=c11 -ffp-contract=off -falign-loops=32 -fopenmp` | main study (scalar code, as in Assignment 2) |
| `build/mm_nomp` | as `mm`, without `-fopenmp` | control: OpenMP linking does not change `seq` |
| `build/mm_noalign` | as `mm`, without `-falign-loops=32` | control: code placement effect |
| `build/mm_scalar` | `-O3 -mavx2 -fno-tree-vectorize -fno-tree-slp-vectorize ...` | vectorisation off |
| `build/mm_simd` | `-O3 -mavx2 ...` | vectorisation on (256-bit AVX2) |

Vectorisation evidence:

- `build/vec_report_*.txt` holds GCC's `-fopt-info-vec` report.
- `results/vec_evidence.csv` holds packed and scalar instruction counts in the kernel functions, produced
  by `scripts/vec_evidence.py`.

## Requirements

- Linux with GCC and OpenMP (tested with gcc 13.3 and libgomp).
- An x86-64 CPU with AVX2, for `mm_scalar` and `mm_simd` only.
- Python 3.10+ with the packages in `requirements.txt`. NumPy is used only to generate inputs and validate
  results.

On macOS, Apple clang has no OpenMP by default. Use `brew install gcc` and `CC=gcc-14 ./build.sh`, or install
`libomp`. Timings from another machine are not comparable with the published ones.

## Reproduce

```bash
./build.sh
python3 scripts/gen_inputs.py            # inputs (~800 MB) + data/manifest.csv
python3 scripts/run_tests.py             # correctness and thread-safety checks -> results/correctness.csv
python3 scripts/vec_evidence.py          # vectorisation evidence -> results/vec_evidence.csv
python3 scripts/run_benchmarks.py        # experiments P, N, A, V in 2 rounds (~1 h on 2 vCPUs)
python3 scripts/analyze.py               # summaries, key_numbers.json, figures
python3 scripts/build_report.py          # the PDF
```

`./run_all.sh` runs all of the above in order. `python3 scripts/run_benchmarks.py --quick` is a 1-minute
smoke test that writes to `results/quick/`.

## Program

```
build/mm seq|omp|tiles n block A.bin B.bin warmup_s reps min_reps budget_s batch_s limit_s [C_out.bin]
```

- **Thread count.** Set with `OMP_NUM_THREADS`. The harness sets it for every configuration, together with
  `OMP_DYNAMIC=false`.
- **Thread placement.** `OMP_PROC_BIND` and `OMP_PLACES` are not set, so the OS scheduler places the threads.
- **Thread check.** The program prints the number of threads that actually ran the parallel region
  (`THREADS`). The harness checks it against the requested count.

## Thread configurations and protocol

- **Hardware.** 2 vCPUs (KVM guest), 1 thread per core.
- **Thread counts.** p = 1 and 2 are within the hardware. p = 3, 4 and 8 are oversubscribed.
- **Sizes.** n = 256, 512, 1024, 2048, 3072 and 4096.
- **Repetitions.** 10 measured repetitions for n ≤ 512, 5 up to 3072, and 3 at 4096, where one sequential
  multiplication takes about 40 s.
- **Warm-up.** 1 s for n ≤ 1024. Larger sizes use one untimed calibration call instead.
- **Rounds.** Two rounds. Within a round, the configurations of each size run in random order (fixed seed).
- **Budgets.** 180 s per multiplication (enforced in the process), 15 min per configuration, 4 GiB of storage.
- **Statistic.** Median and IQR of all samples, with S(p) = median Tseq / median Tp and E(p) = S(p)/p.
- **Memory.** Peak RSS from `wait4()`.
- **CPU use.** CPU time divided by wall time over the kernel phase.

## Results files

| file | content |
|---|---|
| `results/raw_measurements.csv` | every sample: round, experiment, binary, method, threads, n, block, rep, batch |
| `results/runs.csv` | one row per process: status, threads reported, peak RSS, CPU and kernel times, OpenMP settings, validation |
| `results/summary_scaling.csv` | median, IQR, speed-up, efficiency, per-round range, RSS, busy cores |
| `results/summary_vectorisation.csv` | 2×2 vectorisation × threads results |
| `results/summary_controls.csv` | no-OpenMP and code-placement controls |
| `results/key_numbers.json`, `environment.json`, `correctness.csv`, `vec_evidence.csv` | as named |
| `results/figures/` | figures of the report |
