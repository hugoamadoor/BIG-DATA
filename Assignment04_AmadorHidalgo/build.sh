#!/usr/bin/env bash
# Builds everything Assignment 4 runs. Same compiler flags as Assignment 3's main study.
#   build/libmmkernel.so  block kernel called inside every Spark task (src/kernel/mmkernel.c)
#   build/mm              Assignment 3 program (src/reference/matmul_par.c, unchanged):
#                         seq = blocked sequential baseline, omp = OpenMP parallel version
#   build/dense           Assignment 2 program (src/reference/dense.c, unchanged):
#                         ijk = basic sequential triple loop (Assignment 1 kernel)
#   build/rss_run         peak-RSS / CPU-time wrapper
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p build
CC="${CC:-gcc}"
FLAGS="-O2 -std=c11 -ffp-contract=off -falign-loops=32"
$CC $FLAGS -fPIC -shared src/kernel/mmkernel.c -o build/libmmkernel.so
$CC $FLAGS -fopenmp src/reference/matmul_par.c -o build/mm
$CC $FLAGS src/reference/dense.c -o build/dense
$CC -O2 scripts/rss_run.c -o build/rss_run
echo "build OK"
