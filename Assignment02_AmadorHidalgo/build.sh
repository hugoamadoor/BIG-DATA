#!/usr/bin/env bash
# Builds the Assignment 2 binaries.
#   build/dense        all dense methods, -O2 (main study)
#   build/sparse       CSR SpGEMM, -O2
#   build/rss_run      peak-RSS wrapper (from Assignment 1)
# The compiler's vectorisation report is saved to build/vec_report_*.txt. With GCC 13 at -O2 it
# lists no vectorised loop in any kernel (only straight-line code), so every kernel runs scalar.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p build
CC="${CC:-gcc}"
FLAGS="-O2 -std=c11 -ffp-contract=off"   # same flags as the Assignment 1 C baseline
$CC $FLAGS -fopt-info-vec-optimized=build/vec_report_dense.txt src/dense.c -o build/dense
$CC $FLAGS -fopt-info-vec-optimized=build/vec_report_sparse.txt src/sparse.c -o build/sparse
$CC -O2 scripts/rss_run.c -o build/rss_run
echo "build OK"
