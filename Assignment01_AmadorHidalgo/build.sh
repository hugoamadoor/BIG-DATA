#!/usr/bin/env bash
# Builds the C binaries (three optimisation levels) and the Java class.
# The main study uses c-O2; -O0 and -O3 are only used by supplementary experiment S1.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p build/c build/java
CC="${CC:-gcc}"
CFLAGS_COMMON="-std=c11 -ffp-contract=off"   # no fused multiply-add: same arithmetic on every CPU
for opt in O0 O2 O3; do
  $CC -$opt $CFLAGS_COMMON src/c/matmul.c -o build/c/matmul_$opt
done
javac -d build/java src/java/MatMul.java
$CC -O2 scripts/rss_run.c -o build/rss_run   # peak-RSS wrapper used by the harness
echo "build OK"
