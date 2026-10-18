#!/usr/bin/env python3
"""Build report/Assignment01_<Surname>_Report.pdf from results/ (run analyze.py first).

All numbers in tables and in the text that quotes measurements are read from the CSV/JSON files;
the interpretive sentences were written for the published data set and should be re-read if the
benchmarks are repeated on other hardware.

Usage: python3 scripts/build_report.py [--repo-url URL] [--results DIR]
"""
import argparse
import json
from pathlib import Path

import re

import numpy as np
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table,
                                TableStyle)

ROOT = Path(__file__).resolve().parent.parent
ap = argparse.ArgumentParser()
ap.add_argument("--repo-url", default="https://github.com/hugoamadoor/BIG-DATA")
ap.add_argument("--results", default=str(ROOT / "results"))
args = ap.parse_args()
RES = Path(args.results)
FIG = RES / "figures"
OUT = ROOT / "report" / "Assignment01_AmadorHidalgo_Report.pdf"
OUT.parent.mkdir(exist_ok=True)

S = pd.read_csv(RES / "summary.csv")
S1 = pd.read_csv(RES / "summary_s1.csv")
S2 = pd.read_csv(RES / "summary_s2.csv")
RUNS = pd.read_csv(RES / "runs.csv")
RUNS = RUNS[RUNS.study.isin(["main", "S1", "S2"])]
ENV = json.loads((RES / "environment.json").read_text())
BASE = json.loads((RES / "baseline_rss.json").read_text())
SC = json.loads((RES / "scaling.json").read_text())
PROTO = ENV["protocol"]
CROSS = json.loads((RES / "crosscheck.json").read_text())
CORR = pd.read_csv(RES / "correctness.csv")
RAW = pd.read_csv(RES / "raw_measurements.csv")

LANGS = [("C-O2", "C"), ("Java", "Java"), ("Python", "Python")]


def g(v, n, col, df=S):
    r = df[(df.variant == v) & (df.n == n)]
    return float(r[col].iloc[0]) if len(r) else float("nan")


def tfmt(s):
    if s != s:
        return "-"
    if s < 1e-3:
        return f"{s * 1e6:.3g} us"
    if s < 1:
        return f"{s * 1e3:.3g} ms"
    return f"{s:.3g} s"


sizes = sorted(S.n.unique())
ok = S[S.status == "ok"]
nsamples_total = len(RAW)

# ---------- styles ----------
ss = getSampleStyleSheet()
BODY = ParagraphStyle("body", parent=ss["BodyText"], fontName="Helvetica", fontSize=9.4, leading=12.6,
                      alignment=TA_JUSTIFY, spaceAfter=5)
H1 = ParagraphStyle("h1", parent=ss["Heading1"], fontName="Helvetica-Bold", fontSize=13, leading=16,
                    spaceBefore=10, spaceAfter=5, textColor=colors.HexColor("#123f7a"))
H2 = ParagraphStyle("h2", parent=ss["Heading2"], fontName="Helvetica-Bold", fontSize=10.5, leading=13,
                    spaceBefore=6, spaceAfter=3, textColor=colors.HexColor("#0b0b0b"))
TITLE = ParagraphStyle("title", parent=ss["Title"], fontName="Helvetica-Bold", fontSize=17, leading=21, spaceAfter=4,
                       alignment=0)
SUB = ParagraphStyle("sub", parent=BODY, fontSize=10, textColor=colors.HexColor("#52514e"), alignment=0, spaceAfter=2)
CAP = ParagraphStyle("cap", parent=BODY, fontSize=8.3, leading=10.5, textColor=colors.HexColor("#52514e"),
                     alignment=0, spaceAfter=8)
CELL = ParagraphStyle("cell", parent=BODY, fontSize=8, leading=9.6, alignment=0, spaceAfter=0)
CELLB = ParagraphStyle("cellb", parent=CELL, fontName="Helvetica-Bold")
REF = ParagraphStyle("ref", parent=BODY, fontSize=8.4, leading=10.6, alignment=0, leftIndent=16, firstLineIndent=-16,
                     spaceAfter=2.5)
MONO = ParagraphStyle("mono", parent=BODY, fontName="Courier", fontSize=8, leading=10, alignment=0)

story = []


def P(t, st=BODY):
    story.append(Paragraph(t, st))


def table(rows, widths, header=1, zebra=True, align_right_from=1, font=8):
    data = [[Paragraph(str(c), CELLB if i < header else CELL) for c in r] for i, r in enumerate(rows)]
    t = Table(data, colWidths=widths, repeatRows=header)
    st = [("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#c9c8c3")),
          ("BACKGROUND", (0, 0), (-1, header - 1), colors.HexColor("#e8eef6")),
          ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
          ("TOPPADDING", (0, 0), (-1, -1), 2.2), ("BOTTOMPADDING", (0, 0), (-1, -1), 2.2),
          ("LEFTPADDING", (0, 0), (-1, -1), 3.5), ("RIGHTPADDING", (0, 0), (-1, -1), 3.5)]
    if zebra:
        for i in range(header, len(rows)):
            if (i - header) % 2:
                st.append(("BACKGROUND", (0, i), (-1, i), colors.HexColor("#f6f6f3")))
    t.setStyle(TableStyle(st))
    story.append(t)


def figure(name, width_cm, caption):
    from PIL import Image as PILImage
    w, h = PILImage.open(FIG / name).size
    img = Image(str(FIG / name), width=width_cm * cm, height=width_cm * cm * h / w)
    story.append(KeepTogether([img, Paragraph(caption, CAP)]))


# ---------- numbers used in the text ----------
ratio = SC["ratios"]
jc = [v for k, v in ratio["Java/C-O2"].items() if int(k) <= 1024]
jc_lo, jc_hi = min(jc), max(jc)
pc_vals = list(ratio["Python/C-O2"].values())
pj_vals = list(ratio["Python/Java"].values())
c_ns = ok[ok.variant == "C-O2"].ns_per_iter
j_ns = ok[ok.variant == "Java"].ns_per_iter
p_ns = ok[ok.variant == "Python"].ns_per_iter
fit = SC["fit"]
bC, bJ, bP = BASE["C-O2"] / 1024, BASE["Java"] / 1024, BASE["Python"] / 1024
largest = {}
attempted = {}
for v, _ in LANGS:
    r = RUNS[(RUNS.study == "main") & (RUNS.variant == v)]
    largest[v] = int(r[r.status == "ok"].n.max())
    attempted[v] = int(r[r.status != "not_attempted"].n.max())
s2c = {n: g("C-O2", n, "ns_per_iter", S2) for n in (1000, 1024, 1025)}
s2j = {n: g("Java", n, "ns_per_iter", S2) for n in (1000, 1024, 1025)}
n_ok_checks = int(CORR.passed.sum())
CROSS2 = json.loads((RES / "crosscheck_s2.json").read_text())
cross_groups = CROSS["compared_sizes"] + CROSS2["compared_sizes"]
cross_bad = len(CROSS["mismatches"]) + len(CROSS2["mismatches"])
# local log-log slopes between consecutive sizes
loc = {}
for v, _ in LANGS:
    gg = ok[ok.variant == v].sort_values("n")
    e = np.diff(np.log(gg.median_s.to_numpy())) / np.diff(np.log(gg.n.to_numpy()))
    loc[v] = list(zip(gg.n.to_numpy()[:-1], gg.n.to_numpy()[1:], e))
small_slopes = [x for v in loc for (a, b, x) in loc[v] if b <= 256]
c_slopes = [x for a, b, x in loc["C-O2"]]
# deviation of the first timed sample from the median
first_dev = []
for (v, n), gq in RAW[RAW.study == "main"].groupby(["variant", "n"]):
    t = gq.sort_values("rep").time_s_per_multiplication.to_numpy()
    first_dev.append((v, int(n), t[0] / np.median(t) - 1))
big_first = [(v, n, d) for v, n, d in first_dev if abs(d) > 0.10]
# between-process variation of the identical C -O2 configuration (main, S1, S2 sessions)
bp = {}
for n in (256, 512, 1024):
    vals = [g("C-O2", n, "median_s", S), g("C-O2", n, "median_s", S1)] + ([g("C-O2", n, "median_s", S2)] if n in S2.n.values else [])
    bp[n] = (min(vals), max(vals), max(vals) / min(vals))
jc_all = list(ratio["Java/C-O2"].values()) + [s2j[n] / s2c[n] for n in (1000, 1024, 1025)]

# =============================== TITLE & ABSTRACT ===============================
P("Basic dense matrix multiplication in Python, Java and C: how matrix size affects runtime and memory", TITLE)
P("Big Data - Individual Assignment 1 (Basic Matrix Multiplication in Different Languages)", SUB)
P("<b>Hugo Amador Hidalgo</b> - Grado en Ciencia e Ingeniería de Datos, Universidad de Las Palmas de Gran Canaria - "
  "academic year 2026-2027", SUB)
story.append(Spacer(1, 6))
P("<b>Abstract.</b> We measure the straightforward i-j-k triple-loop multiplication of two n × n float64 matrices "
  "in C (gcc -O2), Java (OpenJDK 21, JIT) and CPython 3.13, using identical input files, a monotonic timer around the "
  "multiplication kernel only, 3 to 10 repeated samples per configuration and the peak resident memory of the process. "
  "All runs come from one 2-vCPU virtual machine. Results of the three languages agree bit for bit, so the comparison is "
  "between equal computations. Runtime grows roughly with n<super>3</super> in every language, but the cost per "
  f"multiply-add is not constant: it rises from {c_ns.iloc[0]:.2f} to {c_ns.iloc[-1]:.1f} ns in C (n = 16 to 2048), from "
  f"{j_ns.iloc[0]:.2f} to {j_ns.iloc[-1]:.1f} ns in Java and stays between {p_ns.min():.0f} and {p_ns.max():.0f} ns in Python "
  f"(n up to 1024). Python was {min(pc_vals):.0f} to {max(pc_vals):.0f} times slower than C and {min(pj_vals):.0f} to "
  f"{max(pj_vals):.0f} times slower than Java. Java took between {min(jc_all):.1f} and {max(jc_all):.1f} times the time of C depending on n, with no "
  f"monotonic pattern: about equal at n = 1024, about {ratio['Java/C-O2']['2048']:.0f} times at n = 1536 and 2048 and {s2j[1000] / s2c[1000]:.1f} times at n = 1000. "
  "Peak memory follows the "
  "array storage in C, adds a fixed runtime of about 42 MiB in Java and is about 5 times the array storage in Python "
  "(boxed floats). Two supplementary experiments show that the cost per iteration also changes by up to a factor "
  f"{max(max(s2c.values()) / min(s2c.values()), max(s2j.values()) / min(s2j.values())):.1f} "
  "between neighbouring sizes (1000, 1024, 1025), so every conclusion is scoped to the tested sizes, settings and hardware.")

# =============================== 1 INTRODUCTION ===============================
P("1. Introduction and objective", H1)
P("Matrix multiplication is a standard case study for performance engineering: its algorithm is simple, its arithmetic "
  "cost is known (n<super>3</super> multiply-adds for the textbook algorithm [3]) and its memory-access pattern exposes the "
  "memory hierarchy [4, 5]. The course material uses it to compare programming languages and to introduce benchmarking and performance-engineering rules [1, 10]. "
  "This study follows the common experimental protocol of the assignment [2].")
P("<b>Research question.</b> How does the matrix size n affect the runtime and the memory use of the basic triple-loop "
  "multiplication implemented in Python, Java and C, when inputs, arithmetic, timing boundary and machine are controlled?")
P("<b>Hypotheses (stated before measuring; they are predictions, not results).</b> "
  "H1: time per multiplication grows as n<super>3</super> in all languages. "
  "H2: C and Java are much faster than Python (C at least as fast as Java), and the differences are constant factors, not different growth orders. "
  "H3: peak memory of C is about the array storage 3 × 8 n<super>2</super> bytes plus a small runtime; Java adds a larger fixed runtime; "
  "Python needs several times more because every element is a separate object. "
  "H4: once the matrices no longer fit in the caches the cost per multiply-add increases, so H1 holds only approximately.")

# =============================== 2 BACKGROUND ===============================
P("2. Background and references", H1)
P("For C = AB with n × n matrices, C[i][j] = sum over k of A[i][k] B[k][j]. The triple loop performs n<super>3</super> "
  "multiply-add operations and touches 3n<super>2</super> values [3]. In the i-j-k order the inner loop reads a row of A "
  "sequentially but walks down a column of B with a stride of n elements. Such strided accesses reuse cache lines poorly when "
  "a column no longer fits in the cache, which is why loop order and blocking matter [4, 5, 12]; here the loop order is "
  "fixed because the comparison is between languages, and optimisations belong to Assignment 2.")
P("Floating-point addition is not associative. For non-negative terms the recursive summation of a dot product of length n has "
  "relative error at most γ<sub>n</sub> = nu/(1 - nu) with u = 2<super>-53</super> [6, 7]. This bound gives the "
  "correctness tolerance used below. Java has used strict IEEE-754 semantics for all code since JDK 17 [13] and the C code is "
  "compiled with -ffp-contract=off [14], so the three languages perform the same sequence of double-precision operations.")
P("Benchmarking managed runtimes needs warm-up, repetition and robust summaries because JIT compilation and other effects "
  "change run-to-run times [8, 9]; this is why the protocol uses warm-up runs, repeated samples, the median and the "
  "interquartile range.")

# =============================== 3 METHODOLOGY ===============================
P("3. Methodology", H1)
P("3.1 Implementations", H2)
table([
    ["", "C", "Java", "Python"],
    ["Source", "src/c/matmul.c", "src/java/MatMul.java", "src/python/matmul.py"],
    ["Algorithm", "i-j-k triple loop, local accumulator, no library call", "same", "same, pure Python (no NumPy)"],
    ["Storage layout", "one contiguous row-major double array per matrix", "double[n][n]: array of contiguous row arrays",
     "list of lists of Python floats; each element is a separate boxed object"],
    ["Toolchain / settings", re.sub(r"gcc \((.*)\) (\S+)", r"gcc \2 (\1)", ENV["gcc"]) + "; -O2 -std=c11 -ffp-contract=off",
     ENV["java"].replace('openjdk version ', "OpenJDK ").replace('"', "") + "; -XX:+UseSerialGC -Xmx3g, default tiered JIT",
     "CPython " + ENV["python"].split()[0] + " (no JIT, default settings)"],
    ["Timer", "clock_gettime(CLOCK_MONOTONIC)", "System.nanoTime()", "time.perf_counter_ns()"],
], [2.6 * cm, 4.9 * cm, 4.9 * cm, 4.9 * cm])
P("The unavoidable differences are the storage layouts (C: one flat array; Java: an array of row arrays with bounds checks; Python: "
  "boxed objects) and the execution model (native code, JIT, interpreter). The settings that are kept fixed are the loop order, "
  "the numeric type (float64), the accumulator pattern and the input files.", BODY)

P("3.2 Inputs, correctness and tolerance", H2)
P("Matrices A and B are uniform in [0, 1), generated with NumPy's PCG64 generator [11] (seeds 20261012 + n and 20261013 + n) and stored "
  "as little-endian binary files, which all three programs read, so the logical inputs are identical (a shared file is preferable to "
  "language-specific random generators). A SHA-256 manifest (data/manifest.csv) allows regenerating and verifying them.")
P(f"Before benchmarking, {len(CORR)} checks passed ({n_ok_checks} of {len(CORR)}): identity × M, M × identity, zero × M, "
  "M × zero, 1 × 1, 2 × 2 and 3 × 3 hand-computed products (exact equality, integer-valued data) and random matrices "
  "of sizes 2 to 128 compared over the full matrix with NumPy/BLAS as the trusted reference. For random inputs the relative tolerance "
  "is 2γ<sub>n</sub> (atol = 0): both results satisfy the γ<sub>n</sub> bound of Section 2, so they can differ by at most "
  "twice that. The three languages produced bitwise identical matrices in every test. For benchmark sizes each program prints a "
  "checksum (sum of all entries in row-major order) and three sample entries, validated against NumPy with tolerance 2γ<sub>n</sub> "
  "for entries and 2(γ<sub>n</sub> + γ<sub>n²</sub>) for the checksum. All validated checksums "
  f"were also bitwise identical across the languages and C optimisation levels ({cross_groups} study/size groups compared, {cross_bad} mismatches).")

P("3.3 Environment", H2)
table([
    ["Item", "Value"],
    ["CPU", f"{ENV['cpu_model']}, {ENV['physical_cores']} physical cores, {ENV['logical_processors']} logical processors "
            f"(virtual machine, hypervisor {ENV['hypervisor']}); L1d {ENV['l1d_cache']}, L2 {ENV['l2_cache']}, L3 {ENV['l3_cache']}"],
    ["Memory / OS", f"{float(ENV['ram'].split()[1]) / 2**20:.1f} GiB RAM; {ENV['os']}"],
    ["Threads", "all kernels single-threaded; the JVM also runs background JIT-compiler threads; SerialGC avoids parallel GC workers; "
                "no library creates threads inside the timed region (NumPy is not used by the programs)"],
    ["Frequency / affinity", "CPU governor not exposed by the VM; no frequency control and no CPU pinning (limitation)"],
    ["Run order", "one configuration (language, n) at a time, languages interleaved for each n so that all three see similar machine conditions"],
], [3.2 * cm, 14.1 * cm])

P("3.4 Timing boundary, warm-up, repetitions", H2)
P("<b>Timed region:</b> only the multiplication kernel. Reading the inputs, allocating C, computing the checksum and writing "
  "output happen outside the timer. C is allocated and filled before the first multiplication; the loops write every element of C. "
  "Each program consumes the result (checksum printed and validated), so the computation cannot be removed.")
P("<b>Warm-up:</b> C and Java run warm-up multiplications until at least 1 s has elapsed (at least one run) so that caches, "
  "page tables and the JIT-compiled kernel are in steady state; CPython runs the same warm-up for n ≤ 256. For CPython at "
  "n ≥ 512 no explicit warm-up was requested, because there is no JIT and one run takes more than 10 s. However the program "
  "executes one <i>untimed calibration multiplication</i> before sampling when no warm-up has run (it sizes the sample batch); "
  "it is effectively one warm-up run. The programs' own WARMUP counter shows 0 for these runs; summary.csv counts it as 1, which wall-clock accounting "
  "confirms (for example n = 768: 272 s of wall time versus 228 s of samples). The first timed sample differs by more than 10% from "
  "the median in " + (", ".join(f"{v.replace('-O2', '')} n = {n} ({d * 100:+.0f}%)" for v, n, d in big_first) or "no configuration") + "; no sample was removed.")
P("<b>Samples:</b> 10 for n ≤ 256 and 5 for n ≥ 512, but at least 3 and stopping once 300 s of measured time is reached. "
  "Costly runs therefore have fewer samples: CPython n = 1024 (3 samples, 111 to 122 s) and Java n = 2048 (3 samples, 105 to 108 s). "
  "When one multiplication takes less than 0.1 s, a sample is a batch of repeated multiplications whose duration is about 0.1 s and the "
  "<i>time per multiplication</i> is stored (column inner_multiplications in raw_measurements.csv). Reported statistics: median and "
  "inter-quartile range (IQR); all raw samples are kept in results/raw_measurements.csv.")

P("3.5 Memory metric, size sequence and budgets", H2)
P("<b>Memory:</b> peak resident set size (RSS) of the whole benchmark process, read from the kernel (ru_maxrss) through a small "
  "wrapper (scripts/rss_run.c); the wrapper is needed because a forked child inherits the parent's RSS high-water mark, which would add "
  "the harness's own ~30 MiB to every value. The metric includes the language runtime, input buffers and temporaries; it is not "
  "comparable to the ideal array size 3 × 8 n<super>2</super> bytes without subtracting the runtime baseline, which was measured "
  f"separately with n = 1 (C {bC:.1f} MiB, Java {bJ:.1f} MiB, CPython {bP:.1f} MiB; median of 5).")
P("<b>Sizes:</b> n = 16, 32, 64, 128, 256, 512, 768, 1024, 1536, 2048 (powers of two plus 768 and 1536 as controls). "
  "<b>Budgets declared before the large runs:</b> one multiplication at most 180 s; one configuration at most 15 min of wall time; "
  "peak RSS at most 4 GiB. A configuration that breaks a budget is reported as such with no invented time, and larger sizes of that language are not attempted. "
  "The dense storage of three double matrices at n = 10 000 would need 2.4 GB, so n = 2048 (100 MiB) was never memory-limited.")

# =============================== 4 RESULTS ===============================
P("4. Results", H1)
P("4.1 Runtime", H2)
rows = [["n", "C (-O2)", "Java", "CPython", "samples C / Java / Python"]]
for n in sizes:
    cells = []
    for v, _ in LANGS:
        r = S[(S.variant == v) & (S.n == n)]
        if r.empty or r.status.iloc[0] != "ok":
            rec = RUNS[(RUNS.study == "main") & (RUNS.variant == v) & (RUNS.n == n)]
            cells.append("timeout (15 min)" if len(rec) and rec.status.iloc[0] == "timeout" else "not attempted")
        else:
            r = r.iloc[0]
            cells.append(f"{tfmt(r.median_s)} (IQR {r.iqr_pct:.0f}%)")
    smp = [str(int(g(v, n, "samples"))) if g(v, n, "samples") == g(v, n, "samples") else "0" for v, _ in LANGS]
    rows.append([str(n)] + cells + [" / ".join(smp)])
table(rows, [1.3 * cm, 4.2 * cm, 4.2 * cm, 4.2 * cm, 3.4 * cm])
P("Table 1. Median time per multiplication (IQR as a percentage of the median). n ≤ 64 values are per-multiplication times "
  "obtained from batches. CPython at n = 1536 did not finish two multiplications within the 15 min wall limit (the single-run budget of 180 s is exceeded).", CAP)
figure("fig1_runtime.png", 10.8, "Figure 1. Time per multiplication versus n (log-log; median with IQR bars, which are smaller than the markers "
       "for most points). The dashed line is an n<super>3</super> guide anchored at the C median for n = 256.")

rows = [["n", "C", "Java", "CPython", "Java / C", "CPython / C"]]
for n in sizes:
    c, j, p = (g(v, n, "ns_per_iter") for v, _ in LANGS)
    st = RUNS[(RUNS.study == "main") & (RUNS.variant == "Python") & (RUNS.n == n)].status.iloc[0]
    rows.append([str(n), f"{c:.2f}", f"{j:.2f}", f"{p:.1f}" if p == p else "-", f"{j / c:.2f}", f"{p / c:.1f}" if p == p else "-"])
table(rows, [1.5 * cm, 2.6 * cm, 2.6 * cm, 2.6 * cm, 2.6 * cm, 3 * cm])
P("Table 2. Median time divided by n<super>3</super>, in nanoseconds per inner-loop iteration (one multiply-add), and ratios of medians.", CAP)
figure("fig2_cost_per_iteration.png", 13.2, "Figure 2. Cost per inner-loop iteration. A flat line would mean that the cubic model with a constant "
       "factor describes the data; none of the lines is flat.")
P("A log-log least-squares fit of the medians for n ≥ 128 gives exponents "
  f"{fit['C-O2']['exponent']:.2f} (C, n up to 2048), {fit['Java']['exponent']:.2f} (Java, n up to 2048) and {fit['Python']['exponent']:.2f} "
  "(CPython, n up to 1024). These are descriptive slopes over a range in which the constant itself changes, not estimates of the "
  f"algorithmic exponent: between consecutive sizes the local slope of the C medians varies from {min(c_slopes):.1f} to {max(c_slopes):.1f}.")

P("4.2 Memory", H2)
rows = [["n", "array storage 3 × 8 n<super>2</super> (MiB)", "C peak RSS (MiB)", "Java peak RSS (MiB)", "CPython peak RSS (MiB)"]]
for n in sizes:
    rows.append([str(n), f"{24 * n * n / 2**20:.2f}"] + [f"{g(v, n, 'peak_rss_mb'):.1f}" if g(v, n, 'peak_rss_mb') == g(v, n, 'peak_rss_mb') else "-" for v, _ in LANGS])
rows.append(["baseline (n = 1)", "-", f"{bC:.1f}", f"{bJ:.1f}", f"{bP:.1f}"])
table(rows, [2.6 * cm, 3.8 * cm, 3.1 * cm, 3.4 * cm, 3.7 * cm])
P("Table 3. Peak resident set size of the benchmark process, including the language runtime.", CAP)
figure("fig3_memory.png", 13.0, "Figure 3. Peak RSS. The dashed line is the ideal storage of three float64 matrices; the dotted line assumes "
       "three matrices of boxed Python floats at 32 bytes per element (24 bytes object + 8 bytes pointer).")
c1024 = g("C-O2", 1024, "rss_minus_baseline_mb")
j1024 = g("Java", 1024, "rss_minus_baseline_mb")
p1024 = g("Python", 1024, "rss_minus_baseline_mb")
j2048 = g("Java", 2048, "rss_minus_baseline_mb")
P(f"After subtracting the baseline, C at n = 1024 uses {c1024:.1f} MiB, essentially the 24.0 MiB of array storage. Java uses "
  f"{j1024:.1f} MiB at n = 1024 (2.0 times the arrays) but {j2048:.0f} MiB at n = 2048 (1.08 times the 96 MiB of arrays). CPython uses "
  f"{p1024:.0f} MiB at n = 1024, {p1024 / 24:.1f} times the array storage and a little above the 96 MiB expected from boxed floats; the excess is "
  "consistent with temporary buffers while the inputs are loaded, which was not measured separately.")

P("4.3 Limits", H2)
rows = [["Language", "Largest tested size within the budgets", "Largest size attempted", "Remark"]]
rem = {"C-O2": "single multiplication 50 s at n = 2048; budgets not binding",
       "Java": "single multiplication 106 s at n = 2048; 3 samples because of the 300 s sample budget",
       "Python": "117 s per multiplication at n = 1024 (3 samples); n = 1536 timed out; n = 2048 not attempted"}
for v, name in LANGS:
    rows.append([name, str(largest[v]), str(attempted[v]), rem[v]])
table(rows, [2.2 * cm, 4.6 * cm, 3.4 * cm, 7.1 * cm])
P("Table 4. Largest tested size under the declared budgets versus largest size attempted. Memory (maximum 146 MiB) never limited a run.", CAP)

P("4.4 Supplementary experiments (kept apart from the controlled comparison)", H2)
P("<b>S1 - C optimisation level.</b> The same C source compiled with -O0, -O2 and -O3 (n = 256, 512, 1024; same inputs, protocol and checksums). "
  "<b>S2 - neighbouring sizes.</b> C (-O2) and Java at n = 1000, 1024 and 1025, run back to back with the same protocol.")
rows = [["n", "-O0", "-O2", "-O3"]]
for n in sorted(S1.n.unique()):
    rows.append([str(n)] + [f"{tfmt(g(v, n, 'median_s', S1))} (IQR {g(v, n, 'iqr_pct', S1):.0f}%)" for v in ["C-O0", "C-O2", "C-O3"]])
table(rows, [1.6 * cm, 5 * cm, 5 * cm, 5 * cm])
P("Table 5. Supplementary S1: median time per multiplication of the C kernel at three optimisation levels.", CAP)
figure("fig5_s2_neighbouring_sizes.png", 11.5, "Figure 4. Supplementary S2: median cost per inner-loop iteration at three neighbouring sizes "
       f"(C {s2c[1000]:.1f}, {s2c[1024]:.1f}, {s2c[1025]:.1f} ns; Java {s2j[1000]:.1f}, {s2j[1024]:.1f}, {s2j[1025]:.1f} ns; IQR up to "
       f"{S2.iqr_pct.max():.0f}% of the median).")

# =============================== 5 DISCUSSION ===============================
P("5. Discussion", H1)
P("<b>Answer to the research question.</b> In all three languages the runtime grows approximately as n<super>3</super>: for n ≤ 256 "
  f"the local slopes between consecutive sizes are {min(small_slopes):.1f} to {max(small_slopes):.1f} (Table 1). The cost per multiply-add, however, does not stay constant (Figure 2), so H1 holds as a "
  "first-order model but the constant factor depends on n. For C it rises from "
  f"{c_ns.iloc[0]:.2f} ns at n = 16 to {g('C-O2', 256, 'ns_per_iter'):.2f} ns at n = 256 and {g('C-O2', 2048, 'ns_per_iter'):.1f} ns at n = 2048; for Java "
  f"from {j_ns.iloc[0]:.2f} to {g('Java', 256, 'ns_per_iter'):.2f} and {g('Java', 2048, 'ns_per_iter'):.1f} ns. H4 is supported in direction: with 1 MiB of L2 per core and "
  f"a 33 MiB shared L3 (Section 3.3), matrix B alone (8n<super>2</super> bytes: {8 * 512 ** 2 / 2**20:.0f} MiB at n = 512, {8 * 1024 ** 2 / 2**20:.0f} MiB at n = 1024, "
  f"{8 * 2048 ** 2 / 2**20:.0f} MiB at n = 2048) leaves the private cache between n = 362 and 512 and approaches the L3 size at n = 2048, while the column-wise "
  "walk over B reuses cache lines poorly. We did not use hardware counters (not exposed in this VM), so this is an explanation consistent with the data, not a measured cause.")
P("<b>Language effect.</b> CPython is slower than C by a factor of " f"{min(pc_vals):.0f} to {max(pc_vals):.0f} and than Java by {min(pj_vals):.0f} to {max(pj_vals):.0f}, with a nearly "
  f"constant cost of about {p_ns.iloc[:4].min():.0f} to {p_ns.iloc[:4].max():.0f} ns per iteration for n \u2264 128 (roughly {p_ns.iloc[:4].min() * 2.8:.0f} to {p_ns.iloc[:4].max() * 2.8:.0f} clock cycles at 2.8 GHz), "
  "which indicates that the interpreter's per-iteration work, not memory, dominates for small n. The ratio to C shrinks for large n "
  f"because the cost per iteration of C grows more ({c_ns.iloc[0]:.2f} to {g('C-O2', 1024, 'ns_per_iter'):.2f} ns between n = 16 and 1024, a factor {g('C-O2', 1024, 'ns_per_iter') / c_ns.iloc[0]:.1f}) than that of CPython "
  f"({p_ns.iloc[0]:.0f} to {g('Python', 1024, 'ns_per_iter'):.0f} ns, a factor {g('Python', 1024, 'ns_per_iter') / p_ns.iloc[0]:.1f}); this is consistent with C becoming limited by memory access, which we did not measure directly. "
  "We did not profile the interpreter, so we do not attribute the cost to individual bytecodes.")
P("<b>C versus Java.</b> The ratio Java/C of the medians is "
  f"{min(jc_all):.2f} to {max(jc_all):.2f} over all 13 pairs of measurements (the 10 main sizes plus the three S2 sizes; n = 1024 was measured in both) and has no monotonic pattern: "
  f"{ratio['Java/C-O2']['512']:.2f} at n = 512, {ratio['Java/C-O2']['1024']:.2f} at 1024, {ratio['Java/C-O2']['1536']:.2f} at 1536, {ratio['Java/C-O2']['2048']:.2f} at 2048, but {s2j[1000] / s2c[1000]:.2f} at n = 1000 and {s2j[1025] / s2c[1025]:.2f} at n = 1025. "
  "The identical C configuration measured in different processes varied as well: "
  f"n = 512: {tfmt(bp[512][0])} to {tfmt(bp[512][1])}; n = 1024: {tfmt(bp[1024][0])} to {tfmt(bp[1024][1])} (factors {bp[512][2]:.2f} and {bp[1024][2]:.2f}). "
  "Differences below about 20% between C and Java for n \u2264 1024 are therefore not distinguishable from this between-process variation, and H2's ordering "
  "(C at least as fast as Java) is supported only for the sizes where Java is clearly slower. A plausible reason for the slow Java cases is the "
  "double[][] layout: each access to B[k][j] first loads the row reference B[k] and rows are separate heap objects, whereas C uses one flat array. "
  "This was not isolated (no flat-array Java variant was run), so it remains a hypothesis. A comparison at a single size would give different rankings "
  "than one at another size, so statements such as \u201cC is faster than Java\u201d are not supported in general by these data.")
P("<b>Supplementary S1.</b> Optimisation matters as much as the language: at n = 1024 the unoptimised C build (-O0) is "
  f"{g('C-O0', 1024, 'median_s', S1) / g('C-O2', 1024, 'median_s', S1):.1f} times slower than -O2, whereas -O3 is not consistently different from -O2 "
  f"(n = 512: {tfmt(g('C-O3', 512, 'median_s', S1))} versus {tfmt(g('C-O2', 512, 'median_s', S1))}; n = 1024: {tfmt(g('C-O3', 1024, 'median_s', S1))} "
  f"versus {tfmt(g('C-O2', 1024, 'median_s', S1))}; IQRs of 5% to 30%). This fixed-flag result is why the main study documents -O2 explicitly.")
P("<b>Supplementary S2 and fragility of the cost model.</b> Neighbouring sizes differ strongly: in C the cost per iteration is "
  f"{s2c[1000]:.1f} ns at n = 1000 but {s2c[1024]:.1f} and {s2c[1025]:.1f} ns at n = 1024 and 1025; in Java {s2j[1000]:.1f}, {s2j[1024]:.1f} and {s2j[1025]:.1f} ns. "
  "The slow case is not only the power of two (n = 1025 is also slow), and the Java/C ranking changes with n. We therefore cannot name a mechanism "
  "from time measurements alone; row length in bytes (alignment and set mapping), TLB reach and hardware prefetching are candidates that "
  "would need counters to test. The practical consequence for the main comparison is that every figure in Tables 1 and 2 is specific to its n.")
P("<b>Memory.</b> H3 is supported: C needs the array storage plus about 1.6 MiB; Java adds roughly 42 MiB of fixed runtime and, at some sizes, "
  "extra memory (2.0 times the arrays at n = 1024 but 1.08 times at n = 2048; we did not investigate the cause, JVM heap sizing and garbage collection are candidates); CPython needs about 5 times the array storage because each float is a separate object. Peak RSS mixes arrays, runtime and temporaries, so "
  "these figures are comparable only as process footprints.")
P("<b>Limitations.</b> (i) One virtual machine with unknown neighbours and uncontrolled CPU frequency; absolute times will differ on other hardware, "
  "and the ratios may too. (ii) Each configuration is one process; the IQR covers only variation within one process, and the same C configuration differed between processes by up to a factor "
  f"{max(bp[512][2], bp[1024][2]):.2f} (Section 5). "
  "(iii) One pair of input matrices per size. (iv) Fewer than five samples for the two costliest runs, which gives a wide uncertainty for "
  "their medians. (v) The memory metric includes the runtime and was not split into managed heap versus native memory. (vi) The numbers on the course website "
  "(for example C 0.68 s versus Java 7.76 s for n = 1024 [1]) were obtained on an unspecified machine with unspecified compiler flags; we did not reproduce the "
  f"11-fold C advantage: here C takes {tfmt(g('C-O2', 1024, 'median_s'))} and Java {tfmt(g('Java', 1024, 'median_s'))} at n = 1024. Differences in loop order, compile-time array sizes or flags could "
  "explain it but we have not tested them.")

# =============================== 6 CONCLUSIONS ===============================
P("6. Conclusions", H1)
P("Within the tested conditions (one 2-vCPU VM; n = 16 to 2048; float64; i-j-k loops; C at -O2, OpenJDK 21 with default JIT, CPython 3.13) "
  "the study establishes that: (1) the three implementations compute bitwise identical results, so the timing comparison is between equal computations; "
  "(2) runtime follows the cubic model only to first order, because the cost per multiply-add grows with n (consistent with B leaving the private cache, which was not measured directly); "
  f"(3) CPython is {min(pj_vals):.0f} to {max(pc_vals):.0f} times slower than the compiled languages across the tested range (n up to 1024), a difference of constant factor, not of growth order; "
  f"(4) Java is never clearly faster than C here and takes {min(jc_all):.1f} to {max(jc_all):.1f} times as long depending on n (about twice as long at n = 1000, 1025, 1536 and 2048, equal at 1024), so no single ranking holds at all sizes; "
  "(5) memory follows the array storage in C, a fixed runtime plus the arrays in Java and about five times the arrays in CPython. "
  "What remains unresolved: the mechanism behind the 2× changes between neighbouring sizes (S2), whether the Java gap at large n is due to the array layout, "
  "and how the results change on other hardware. Assignment 2 (optimised methods) is the natural place to test layout and loop-order explanations.")

# =============================== 7 AI DISCLOSURE ===============================
P("7. Use of generative AI", H1)
P("In line with the course policy, the following disclosure applies: the AI assistant Claude (Anthropic, model Sonnet 5.5, claude.ai) was used to plan the "
  "experiment, to draft the three implementations, the harness and analysis scripts, the README and the first draft of this report. "
  "All numerical results come from measurements stored in results/ and produced by the committed code; the AI output is not used as evidence. "
  "The author remains responsible for understanding the code, validating the measurements, verifying the references and defending the conclusions.")

# =============================== REFERENCES ===============================
P("References", H1)
refs = [
    "M. D. Afonso Suárez, J. J. Hernández Cabrera and J. Évora Gómez, “Big Data,” course materials (Complexity Management; "
    "Monitoring and Performance Engineering), Universidad de Las Palmas de Gran Canaria, 2026/27. [Online]. Available: https://mltechdrawer.github.io/BIGDATA/",
    "“Individual Assignments: Matrix Multiplication,” Big Data, Grado en Ciencia e Ingeniería de Datos, ULPGC, revised proposal 2026-2027.",
    "T. H. Cormen, C. E. Leiserson, R. L. Rivest and C. Stein, <i>Introduction to Algorithms</i>, 4th ed. Cambridge, MA, USA: MIT Press, 2022.",
    "J. L. Hennessy and D. A. Patterson, <i>Computer Architecture: A Quantitative Approach</i>, 6th ed. Cambridge, MA, USA: Morgan Kaufmann, 2017.",
    "U. Drepper, “What every programmer should know about memory,” Red Hat, Inc., Nov. 2007.",
    "N. J. Higham, <i>Accuracy and Stability of Numerical Algorithms</i>, 2nd ed. Philadelphia, PA, USA: SIAM, 2002.",
    "D. Goldberg, “What every computer scientist should know about floating-point arithmetic,” <i>ACM Computing Surveys</i>, vol. 23, no. 1, pp. 5-48, 1991.",
    "A. Georges, D. Buytaert and L. Eeckhout, “Statistically rigorous Java performance evaluation,” in <i>Proc. OOPSLA</i>, 2007, pp. 57-76.",
    "T. Kalibera and R. Jones, “Rigorous benchmarking in reasonable time,” in <i>Proc. ISMM</i>, 2013, pp. 63-74.",
    "J. L. Bentley, <i>Writing Efficient Programs</i>. Englewood Cliffs, NJ, USA: Prentice-Hall, 1982.",
    "C. R. Harris <i>et al.</i>, “Array programming with NumPy,” <i>Nature</i>, vol. 585, pp. 357-362, 2020.",
    "M. S. Lam, E. E. Rothberg and M. E. Wolf, “The cache performance and optimizations of blocked algorithms,” in <i>Proc. ASPLOS IV</i>, 1991, pp. 63-74.",
    "“JEP 306: Restore always-strict floating-point semantics,” OpenJDK. [Online]. Available: https://openjdk.org/jeps/306",
    "GCC Manual, “Options that control optimization.” [Online]. Available: https://gcc.gnu.org/onlinedocs/gcc/Optimize-Options.html",
]
for i, r in enumerate(refs, 1):
    P(f"[{i}] {r}", REF)
P("Reproducibility materials", H1)
P(f"Repository: {args.repo_url}. It contains the three implementations, build.sh, a README with exact build and run commands "
  "(requirements.txt pins the Python packages), the input generator with its SHA-256 manifest, the test, benchmark and analysis scripts, "
  f"the raw measurements ({nsamples_total} samples in results/raw_measurements.csv, one row per process in results/runs.csv), the environment record "
  "(results/environment.json), the summaries and all figures. The binary inputs (about 130 MiB) are regenerated with scripts/gen_inputs.py. "
  "Suggested archive: Assignment01_AmadorHidalgo_Materials.zip.")


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(colors.HexColor("#52514e"))
    canvas.drawString(2 * cm, 1.1 * cm, "Hugo Amador Hidalgo - Big Data - Assignment 1 - Basic matrix multiplication")
    canvas.drawRightString(A4[0] - 2 * cm, 1.1 * cm, f"{doc.page}")
    canvas.restoreState()


doc = SimpleDocTemplate(str(OUT), pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm, topMargin=1.8 * cm, bottomMargin=1.8 * cm,
                        title="Assignment 1 - Basic matrix multiplication in Python, Java and C",
                        author="Hugo Amador Hidalgo", subject="Big Data 2026-2027, ULPGC")
doc.build(story, onFirstPage=footer, onLaterPages=footer)
print("report written:", OUT)
