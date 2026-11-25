# Big Data - Matrix Multiplication Performance Study

Individual assignments for **Big Data**, Grado en Ciencia e Ingeniería de Datos,
Universidad de Las Palmas de Gran Canaria (ULPGC), academic year 2026-2027.

**Author:** Hugo Amador Hidalgo

This repository holds a series of experimental studies on one problem: multiplying two n × n matrices.
Each stage builds on the previous one. It starts with basic implementations in three languages, then
optimises the sequential code, adds sparse matrices, parallelises the fastest kernel, and finally
distributes it with Apache Spark. Every stage has its own code, inputs, raw measurements, analysis scripts and a written report.

## Assignments

| # | Topic | Main question | Report |
|---|---|---|---|
| 1 | [Basic matrix multiplication](Assignment01_AmadorHidalgo/) | How do Python, Java and C compare on the same naive triple-loop algorithm? | [PDF](Assignment01_AmadorHidalgo/report/Assignment01_AmadorHidalgo_Report.pdf) |
| 2 | [Optimised and sparse multiplication](Assignment02_AmadorHidalgo/) | How much do loop reordering and cache blocking help, and when does a sparse (CSR) format beat dense? | [PDF](Assignment02_AmadorHidalgo/report/Assignment02_AmadorHidalgo_Report.pdf) |
| 3 | [Parallel multiplication and vectorisation](Assignment03_AmadorHidalgo/) | How does the best kernel scale with OpenMP threads, and what does AVX2 vectorisation add? | [PDF](Assignment03_AmadorHidalgo/report/Assignment03_AmadorHidalgo_Report.pdf) |
| 4 | [Distributed multiplication](Assignment04_AmadorHidalgo/) | How does a Spark implementation scale with workers, and where does its time go? | [PDF](Assignment04_AmadorHidalgo/report/Assignment04_AmadorHidalgo_Report.pdf) |

## Key findings

- **Assignment 1.** On the same naive algorithm, Python (CPython) was 26 to 78 times slower than C.
  Java ranged from 0.8 to 2.2 times C's time, depending on n. The results matched bit for bit across the
  three languages.
- **Assignment 2.** Removing the column-stride access to B made the C kernel 11 to 17 times faster for
  n = 1024 to 2048. Cache blocking only paid off for n of 3072 and above. With uniform random sparsity, a
  CSR sparse product beat the best dense method up to 50 % density, but conversion cost and fill-in
  reduced that advantage.
- **Assignment 3.** OpenMP with two threads gave a speed-up of 1.84 to 2.02 on a 2-core machine, and the
  results stayed bit-identical. Running more threads than cores added nothing. AVX2 vectorisation added a
  further 1.4 to 1.7 times, independently of the number of threads.
- **Assignment 4.** In a local Spark cluster, run on one 2-core machine, two workers were 1.61 times faster
  than one at n = 3072 and 1.42 times faster than the sequential program. They were still 1.38 times
  slower than OpenMP on the same cores. Framework overhead and replicating B in every worker explain the
  gap, and memory limited the largest size.

All timings come from cloud virtual machines, and each assignment ran on a different instance. Absolute
times are therefore comparable within one report, not across reports.

## Structure of each assignment

```
AssignmentNN_AmadorHidalgo/
├── README.md          exact build and run instructions
├── src/               implementations
├── scripts/           input generation, tests, benchmark harness, analysis, report
├── data/manifest.csv  seeds and SHA-256 of every input (inputs are regenerated, not stored)
├── results/           raw measurements, summaries, environment, figures
└── report/            the PDF report
```

## Reproducing a study

Each assignment's README has the full instructions. In short:

```bash
cd Assignment03_AmadorHidalgo
./run_all.sh        # build, generate inputs, test, benchmark, analyse, build the report
```

The requirements are GCC, Python 3 with the packages in each `requirements.txt`, and a JDK for
Assignments 1 and 4.

## Use of generative AI

The AI assistant Claude (Anthropic) was used during these assignments, as disclosed in each report. All
results come from measurements produced by the code in this repository.
