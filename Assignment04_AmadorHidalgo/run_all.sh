#!/usr/bin/env bash
# Full pipeline: build, inputs, tests, benchmarks, analysis, report.
set -euo pipefail
cd "$(dirname "$0")"
./build.sh
python3 scripts/gen_inputs.py
python3 scripts/run_tests.py
python3 scripts/run_benchmarks.py
python3 scripts/analyze.py
python3 scripts/build_report.py
