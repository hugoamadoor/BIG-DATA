# Assignment 2: Optimised and Sparse Matrix Multiplication

Big Data, Grado en Ciencia e Ingeniería de Datos, ULPGC, 2026-2027.
Author: Hugo Amador Hidalgo.
Report: `report/Assignment02_AmadorHidalgo_Report.pdf`

## Research questions

- **Part A.** How much do loop reordering, B-transposition and cache blocking speed up the sequential dense
  triple-loop baseline from Assignment 1, as a function of n?
- **Part B.** Under which tested densities and structures does a CSR sparse-sparse product (Gustavson SpGEMM)
  beat the best dense method, once conversion and fill-in are taken into account?

## What is reused from Assignment 1

- **Baseline.** The C kernel from `Assignment01_AmadorHidalgo/src/c/matmul.c` (repository
  `github.com/hugoamadoor/BIG-DATA`, commit `ca7622a`), copied unchanged as `mm_ijk` in `src/dense.c`.
  The only addition is `restrict` qualifiers.
- **Compiler flags.** The same flags: `gcc -O2 -std=c11 -ffp-contract=off`.
- **Timing harness and peak-RSS wrapper.** Both are reused. The harness now lives in `src/harness.h` and
  gains a hard per-multiplication time limit. The wrapper is `scripts/rss_run.c`.
- **Dense inputs.** The same generator and seeds, so the files are byte-identical to the Assignment 1
  inputs for the sizes both studies share.

## Methods

| name | file | change relative to the baseline |
|---|---|---|
| `ijk` | src/dense.c | baseline: i-j-k, inner loop walks B down a column |
| `ijk_bt` | src/dense.c | B transposed first (inside the timed region), then unit-stride dot products |
| `ikj` | src/dense.c | loop reordering: `C[i,:] += A[i,k] * B[k,:]` |
| `blocked` | src/dense.c | ikj with square tiles (ii, kk, jj); tile side chosen by experiment T |
| `csr_spgemm` | src/sparse.c | CSR x CSR, row-wise Gustavson with a dense accumulator; own code |

All methods add the products of each `C[i,j]` in increasing k. The optimised dense methods and the sparse
method are therefore **bit-identical** to the baseline, and the tests check this.

## Requirements

- Linux or macOS
- gcc (tested 13.3)
- Python 3.10 or later with NumPy, SciPy, pandas, matplotlib and reportlab (`pip install -r requirements.txt`)

NumPy and SciPy are used only to generate inputs and to validate results. They are never benchmarked as one
of the methods. On macOS, `gcc` is usually Apple clang. It builds and runs, but its timings are not
comparable with the published ones.

## Reproduce

```bash
./build.sh                               # build/dense, build/sparse, build/rss_run, vectorisation reports
python3 scripts/gen_inputs.py            # all inputs (~3 GB) + data/manifest.csv with SHA-256
python3 scripts/run_tests.py             # 205 correctness checks -> results/correctness.csv
python3 scripts/run_benchmarks.py        # experiments T, D, M, S, X, L (about 1 h 15 min here)
python3 scripts/analyze.py               # summaries, key_numbers.json, figures
python3 scripts/build_report.py          # the PDF
```

Or run everything with `./run_all.sh`. Some useful variants:

- `python3 scripts/run_benchmarks.py --quick` runs a 1-minute smoke test into `results/quick/`.
- `--only T,D` runs only the selected experiments.

The input files are not in the repository because of their size. `gen_inputs.py` regenerates them and
rewrites `data/manifest.csv` with their SHA-256 hashes. Run `git diff data/manifest.csv` afterwards: no
difference means the regenerated inputs are byte-identical to the published ones.

## Programs

```
build/dense  ijk|ijk_bt|ikj|blocked  n block A.bin B.bin  warmup_s reps min_reps budget_s batch_s limit_s [C_out.bin]
build/sparse dense|csr               n A B                warmup_s reps min_reps budget_s batch_s limit_s [C_out.bin]
```

- **Dense input files.** n×n float64, little-endian, row-major.
- **CSR files.** int64 n, int64 nnz, int64 rowptr[n+1], int32 col[nnz], float64 val[nnz].
- **`sparse dense ...`** converts dense files to CSR first and reports that conversion as a separate,
  separately timed phase.

## Experiments and protocol

| id | question | configuration |
|---|---|---|
| T | which tile side? | blocked, n = 1024 and 2048, tile side 16 to 256 |
| D | dense optimisations against n | 4 dense methods, n = 256 to 4096, random dense inputs |
| M | sparse against dense, by density | uniform sparsity, n = 1024 and 2048, density 50 % to 0.1 % |
| S | does structure matter at equal density? | n = 2048, 5 % and 1 %: uniform, banded, power-law rows |
| L | sizes beyond the dense budget | 16 nonzeros per row, n = 4096 to 131072 |
| X | cross-check | unchanged Assignment 1 binary against the baseline here, n = 512 and 1024 |

- **Budget.** 180 s per multiplication, enforced inside the process. 15 min per configuration. 4 GiB planned
  storage.
- **Repetitions.** 10 for n ≤ 512, otherwise 5. Samples shorter than 0.1 s are batched.
- **Statistic.** Median and IQR.
- **Memory.** Peak RSS from `wait4()`, plus analytic storage.

The published data were produced by one full run and one follow-up run. The follow-up was
`--only M,X --m-densities 100,85,70`, which added the high densities and the Assignment 1 cross-check.
One process from an aborted first launch is kept in the CSV files with the label `T_aborted`. It is
excluded from all summaries.

## Results files

| file | content |
|---|---|
| `results/raw_measurements.csv` | every sample (experiment, method, input, structure, n, density, block, phase, rep) |
| `results/runs.csv` | one row per process: status, peak RSS, nnz of A, B and C, checksums, validation |
| `results/summary_*.csv` | median and IQR per configuration, GFLOP/s, speed-ups, fill-in |
| `results/key_numbers.json` | fitted exponents, crossover densities, limits |
| `results/environment.json` | hardware, compiler and protocol settings |
| `results/correctness.csv` | all test cases |
| `results/figures/` | the figures in the report |
| `build/vec_report_*.txt` | GCC vectorisation report (evidence that the kernels run scalar) |
