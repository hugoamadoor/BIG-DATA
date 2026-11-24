#!/usr/bin/env bash
# Builds the Assignment 3 binaries from one source file (src/matmul_par.c).
#   build/mm          main study: Assignment 2 flags + OpenMP          (-O2, scalar code)
#   build/mm_nomp     control: same flags without -fopenmp (seq only; checks that linking OpenMP
#                     does not change the sequential baseline)
#   build/mm_scalar   vectorisation study, auto-vectorisation OFF       (-O3 -mavx2 -fno-tree-vectorize)
#   build/mm_simd     vectorisation study, auto-vectorisation ON        (-O3 -mavx2)
#   build/mm_noalign  build/mm without -falign-loops=32 (experiment A: code-placement control)
#   build/rss_run     peak-RSS / CPU-time wrapper
# -falign-loops=32 everywhere except mm_noalign: during development the same inner loop ran up to
# 1.6x slower or faster depending only on where the linker placed it (see the report, section 4.1),
# so loop starts are aligned to 32 bytes in every build that is compared.
# -ffp-contract=off everywhere: no fused multiply-add, so every build adds the same products in the
# same order and all results are bit-identical. Vectorisation evidence: build/vec_report_*.txt
# (GCC -fopt-info-vec) and build/disasm_*.txt (instruction counts, see scripts/vec_evidence.py).
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p build
CC="${CC:-gcc}"
BASE="-std=c11 -ffp-contract=off -falign-loops=32"
$CC -O2 $BASE -fopenmp -fopt-info-vec-optimized=build/vec_report_mm.txt src/matmul_par.c -o build/mm
$CC -O2 $BASE src/matmul_par.c -o build/mm_nomp
$CC -O3 -mavx2 $BASE -fopenmp -fno-tree-vectorize -fno-tree-slp-vectorize \
    -fopt-info-vec-optimized=build/vec_report_scalar.txt src/matmul_par.c -o build/mm_scalar
$CC -O3 -mavx2 $BASE -fopenmp -fopt-info-vec-optimized=build/vec_report_simd.txt src/matmul_par.c -o build/mm_simd
$CC -O2 -std=c11 -ffp-contract=off -fopenmp src/matmul_par.c -o build/mm_noalign
$CC -O2 scripts/rss_run.c -o build/rss_run
echo "build OK"
