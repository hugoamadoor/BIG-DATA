#!/usr/bin/env python3
"""Collect hardware / software / runtime details for reproducibility (Linux and macOS)."""
import json
import os
import platform
import re
import shutil
import subprocess
import sys


def sh(cmd):
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=20)
        return (r.stdout + r.stderr).strip()
    except Exception as e:  # noqa: BLE001
        return f"unavailable ({e})"


def first_line(text, skip="Picked up"):
    for line in text.splitlines():
        if line.strip() and not line.startswith(skip):
            return line.strip()
    return ""


def collect(c_flags="-O2 -std=c11 -ffp-contract=off", java_flags="-XX:+UseSerialGC"):
    info = {
        "os": platform.platform(),
        "kernel": platform.release(),
        "machine": platform.machine(),
        "logical_processors": os.cpu_count(),
    }
    if sys.platform.startswith("linux"):
        cpu = sh("lscpu")
        def field(name):
            m = re.search(rf"^{name}:\s*(.+)$", cpu, re.M)
            return m.group(1).strip() if m else None
        cores = field(r"Core\(s\) per socket")
        sockets = field(r"Socket\(s\)")
        info.update({
            "cpu_model": field("Model name"),
            "physical_cores": int(cores) * int(sockets) if cores and sockets else None,
            "threads_per_core": field(r"Thread\(s\) per core"),
            "l1d_cache": field("L1d cache"),
            "l2_cache": field("L2 cache"),
            "l3_cache": field("L3 cache"),
            "hypervisor": field("Hypervisor vendor"),
            "ram": first_line(sh("grep MemTotal /proc/meminfo")),
            "cpu_governor": sh("cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null") or "n/a (not exposed)",
        })
    elif sys.platform == "darwin":
        info.update({
            "cpu_model": sh("sysctl -n machdep.cpu.brand_string"),
            "physical_cores": int(sh("sysctl -n hw.physicalcpu") or 0),
            "l1d_cache": sh("sysctl -n hw.l1dcachesize"),
            "l2_cache": sh("sysctl -n hw.l2cachesize"),
            "ram_bytes": sh("sysctl -n hw.memsize"),
        })
    try:
        import numpy
        numpy_v = numpy.__version__
    except Exception:  # noqa: BLE001
        numpy_v = None
    info.update({
        "python": sys.version.replace("\n", " "),
        "python_implementation": platform.python_implementation(),
        "numpy_used_only_for_input_generation_and_validation": numpy_v,
        "gcc": first_line(sh("gcc --version")) if shutil.which("gcc") else None,
        "java": first_line(sh("java -version")) if shutil.which("java") else None,
        "javac": first_line(sh("javac -version")) if shutil.which("javac") else None,
        "c_flags": c_flags,
        "java_flags": java_flags + " (JIT: default tiered C1/C2)",
        "JAVA_TOOL_OPTIONS": ("set by the environment (trust-store / proxy options, unrelated to performance; value not recorded)" if os.environ.get("JAVA_TOOL_OPTIONS") else ""),
        "threads_note": ("All three kernels are single-threaded. The JVM additionally runs background "
                         "JIT-compiler and GC threads; SerialGC avoids parallel GC worker threads."),
        "memory_metric": "peak resident set size (ru_maxrss) of the benchmark process, from wait4()",
        "timer": {"C": "clock_gettime(CLOCK_MONOTONIC)", "Java": "System.nanoTime()",
                  "Python": "time.perf_counter_ns()"},
        "load_average_at_start": os.getloadavg() if hasattr(os, "getloadavg") else None,
    })
    return info


if __name__ == "__main__":
    print(json.dumps(collect(), indent=2))
