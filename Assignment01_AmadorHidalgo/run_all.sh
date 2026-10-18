#!/usr/bin/env bash
# Full pipeline: build -> inputs -> correctness tests -> benchmarks -> figures/tables -> PDF report.
# Expect roughly 40 minutes on a 2-core machine (pure-Python runs dominate).
set -euo pipefail
cd "$(dirname "$0")"
./build.sh
python3 scripts/gen_inputs.py
python3 scripts/run_tests.py
python3 scripts/run_benchmarks.py          # add --sizes 16,64,256 --main-only for a quick trial
python3 scripts/analyze.py
python3 scripts/build_report.py
