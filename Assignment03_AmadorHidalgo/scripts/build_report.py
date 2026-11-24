#!/usr/bin/env python3
"""Build report/Assignment03_AmadorHidalgo_Report.pdf from results/ (run analyze.py first).

Numbers in tables and in sentences that quote measurements are read from the CSV/JSON files.
The interpretive sentences were written for the published data set; re-read them if the
benchmarks are repeated on other hardware.

Usage: python3 scripts/build_report.py [--repo-url URL] [--results DIR]
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Image, KeepTogether, Paragraph, SimpleDocTemplate, Table, TableStyle

ROOT = Path(__file__).resolve().parent.parent
ap = argparse.ArgumentParser()
ap.add_argument("--repo-url", default="https://github.com/hugoamadoor/BIG-DATA")
ap.add_argument("--results", default=str(ROOT / "results"))
args = ap.parse_args()
RES = Path(args.results)
FIG = RES / "figures"
OUT = ROOT / "report" / "Assignment03_AmadorHidalgo_Report.pdf"
OUT.parent.mkdir(exist_ok=True)

SC = pd.read_csv(RES / "summary_scaling.csv")
VEC = pd.read_csv(RES / "summary_vectorisation.csv")
VC = pd.read_csv(RES / "summary_vectorisation_configs.csv")
CTL = pd.read_csv(RES / "summary_controls.csv")
RUNS = pd.read_csv(RES / "runs.csv")
RAW = pd.read_csv(RES / "raw_measurements.csv")
ENV = json.loads((RES / "environment.json").read_text())
CORR = pd.read_csv(RES / "correctness.csv")
VEV = pd.read_csv(RES / "vec_evidence.csv")
BASE = json.loads((RES / "baseline_rss.json").read_text())
PROTO = ENV["protocol"]


def tfmt(s):
    if s != s:
        return "-"
    if s < 1:
        return f"{s * 1e3:.3g} ms"
    return f"{s:.3g} s"


def g(n, m, p, col):
    r = SC[(SC.n == n) & (SC.method == m) & (SC.threads == p)]
    return float(r[col].iloc[0])


def vrow(n, col):
    return float(VEC[VEC.n == n][col].iloc[0])


def vc(n, b, m, p, col="median_s"):
    return float(VC[(VC.n == n) & (VC.binary == b) & (VC.method == m) & (VC.threads == p)][col].iloc[0])


def ctl(exp, n, m, col):
    return float(CTL[(CTL.experiment == exp) & (CTL.n == n) & (CTL.method == m)][col].iloc[0])


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
story = []


def P(t, st=BODY):
    story.append(Paragraph(t, st))


def table(rows, widths, header=1, caption=None):
    data = [[Paragraph(str(c), CELLB if i < header else CELL) for c in r] for i, r in enumerate(rows)]
    t = Table(data, colWidths=widths, repeatRows=header)
    st = [("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#c9c8c3")),
          ("BACKGROUND", (0, 0), (-1, header - 1), colors.HexColor("#e8eef6")),
          ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
          ("TOPPADDING", (0, 0), (-1, -1), 2.2), ("BOTTOMPADDING", (0, 0), (-1, -1), 2.2),
          ("LEFTPADDING", (0, 0), (-1, -1), 3.5), ("RIGHTPADDING", (0, 0), (-1, -1), 3.5)]
    for i in range(header, len(rows)):
        if (i - header) % 2:
            st.append(("BACKGROUND", (0, i), (-1, i), colors.HexColor("#f6f6f3")))
    t.setStyle(TableStyle(st))
    story.append(KeepTogether([t] + ([Paragraph(caption, CAP)] if caption else [])))


def figure(name, width_cm, caption):
    from PIL import Image as PILImage
    w, h = PILImage.open(FIG / name).size
    img = Image(str(FIG / name), width=width_cm * cm, height=width_cm * cm * h / w)
    story.append(KeepTogether([img, Paragraph(caption, CAP)]))


# ---------- numbers used in the text ----------
sizes = sorted(SC.n.unique())
threads = sorted(SC[SC.method == "omp"].threads.unique())
omp = SC[SC.method == "omp"]
S2 = omp[omp.threads == 2].set_index("n").speedup
E2 = omp[omp.threads == 2].set_index("n").efficiency
ov1 = {n: 100 * (g(n, "omp", 1, "median_s") / g(n, "seq", 1, "median_s") - 1) for n in sizes}
ovt = {n: 100 * (g(n, "tiles", 1, "median_s") / g(n, "seq", 1, "median_s") - 1) for n in sizes}
over = omp[omp.threads > 2]
n_samples = len(RAW)
n_runs = len(RUNS)
n_checks, n_ok = len(CORR), int(CORR.passed.sum())
big = max(sizes)
s_rel1 = {n: g(n, "omp", 1, "median_s") / g(n, "omp", 2, "median_s") for n in sizes}
mib_base = BASE["mm_omp_p2_n16_kb"] / 1024
rss_spread = max(SC[SC.n == n].peak_rss_mib.max() - SC[SC.n == n].peak_rss_mib.min() for n in sizes)
vmin, vmax = min(VEC.vec_speedup_p1.min(), VEC.vec_speedup_p2.min()), max(VEC.vec_speedup_p1.max(), VEC.vec_speedup_p2.max())
tmin = min(VEC.thread_speedup_scalar.min(), VEC.thread_speedup_simd.min())
tmax = max(VEC.thread_speedup_scalar.max(), VEC.thread_speedup_simd.max())
best_gf = float(VEC.gflops_simd_p2.max())

# =============================== TITLE ===============================
P("Thread Scaling of a Blocked Matrix-Multiplication Kernel with OpenMP on Two Cores, and the Separate Effect "
  "of AVX2 Vectorisation", TITLE)
P("Big Data - Individual Assignment 3 (Parallel multiplication; optional vectorisation)", SUB)
P("Hugo Amador Hidalgo - Grado en Ciencia e Ingeniería de Datos, ULPGC - Academic year 2026-2027", SUB)
P(f"Repository: {args.repo_url} (folder Assignment03_AmadorHidalgo)", SUB)

P("Abstract", H1)
P(f"This study measures how the number of OpenMP threads changes the runtime, speed-up, efficiency and memory of "
  f"the blocked i-k-j matrix-multiplication kernel from Assignment 2, and when parallel execution helps. The parallel "
  f"version shares row tiles of C between threads; it uses the same algorithm, numeric type (float64), storage, tile "
  f"size and input files as the sequential baseline and produces bit-identical results for every thread count. "
  f"On a 2-vCPU cloud VM, two threads gave S(2) = {S2.min():.2f}-{S2.max():.2f} for n = {sizes[0]} to {big} "
  f"(efficiency {E2.min():.2f}-{E2.max():.2f}); the parallel code with one thread was within "
  f"{max(abs(v) for v in ov1.values()):.0f} % of the baseline, so OpenMP overhead was not measurable at these sizes. "
  f"Running 3, 4 or 8 threads on the 2 vCPUs never beat 2 threads by more than the run-to-run spread and, at small n "
  f"with 3 threads, lost up to {100 * (1 - g(256, 'omp', 3, 'speedup') / S2[256]):.0f} % because one core sat idle part of "
  "the time. An apparent 60 % one-thread overhead seen during development was traced to code placement, not to "
  "OpenMP, and is controlled by aligning loops. A slightly superlinear S(2) at n = 4096 is explained by the baseline "
  f"being a few per cent slower than the parallel code path at one thread. In the optional extension, AVX2 auto-vectorisation added a "
  f"{vmin:.2f}-{vmax:.2f}× speed-up, independent of the thread count, and two threads added {tmin:.2f}-{tmax:.2f}× "
  f"independent of vectorisation; together they reached {best_gf:.0f} GFLOP/s. Peak memory did not depend on the thread count.")

# =============================== 1 INTRODUCTION ===============================
P("1. Introduction and objective", H1)
P("Assignments 1 and 2 studied sequential matrix multiplication. The fastest sequential dense method found there was "
  "a cache-blocked i-k-j loop with 32×32 tiles. This study keeps that kernel and asks what shared-memory parallelism adds.")
P("<b>Research question.</b> For n × n float64 matrices, how do runtime, speed-up S(p), efficiency E(p) and peak "
  "memory of the blocked kernel change with the number of OpenMP threads p, including p = 1 and p larger than the "
  "number of hardware execution contexts, and for which n does parallel execution help? "
  "<b>Optional extension.</b> How much does AVX2 auto-vectorisation add, measured separately from the thread count?")
P("<b>Hypotheses (stated before measuring).</b> H1: because threads write disjoint rows of C and the per-thread work is "
  "the same blocked kernel, S(2) approaches the hardware limit of 2 for large n. H2: at small n the start-up and "
  "synchronisation cost of the parallel region and load imbalance between threads lower S(2). H3: with only two hardware "
  "contexts, p > 2 cannot add speed-up; it may cost time through time-slicing. H4: vectorisation and threads act on "
  "different resources (vector width within a core vs. number of cores), so their effects multiply.")

# =============================== 2 BACKGROUND ===============================
P("2. Background", H1)
P("<b>Speed-up and efficiency.</b> With T<sub>seq</sub>(n) the time of the sequential baseline and T<sub>p</sub>(n) the "
  "time with p threads, S(p) = T<sub>seq</sub>/T<sub>p</sub> and E(p) = S(p)/p [1]. Amdahl's law bounds S(p) by "
  "1/((1 - f) + f/p) when a fraction f of the work is parallel [3]; here almost all the work is in the parallel loop, "
  "so the binding limit is the number of cores: on c cores S(p) ≤ min(p, c) unless something other than the core "
  "count changes, for example cache capacity per thread or the code path itself [4]. "
  "<b>OpenMP worksharing.</b> A <i>parallel for</i> with schedule(static) splits the iterations into p contiguous chunks "
  "of nearly equal size, one per thread, and ends with an implicit barrier [5]. Threads are created by the runtime the "
  "first time a parallel region runs and are then reused. If threads outnumber cores (oversubscription), the operating "
  "system time-slices them, and a thread that is descheduled delays the barrier [6]. "
  "<b>Vectorisation.</b> SIMD instructions apply one operation to several values; AVX2 registers hold four float64 "
  "values [7]. GCC vectorises loops automatically at -O3, and from version 12 to a lesser degree at -O2 [8]. "
  "The i-k-j inner loop C[i,j] += A[i,k]·B[k,j] over j has no dependence between iterations and is a standard "
  "vectorisation candidate. Because each C[i,j] still receives its products in the same order, vectorising it, like "
  "parallelising over rows, does not change the result when fused multiply-add is disabled.")

# =============================== 3 METHODOLOGY ===============================
P("3. Methodology", H1)
P("3.1 Implementations", H2)
table([["Method", "Description", "Use"],
       ["seq", "Assignment 2 mm_blocked, copied unchanged: one memset of C, then loops ii, kk, jj (tiles of 32), i, k, j",
        "sequential baseline T<sub>seq</sub>"],
       ["omp", "#pragma omp parallel + omp for schedule(static) over row tiles t; each iteration calls row_tile, which zeroes "
               "its 32 rows of C and runs the kk, jj, i, k, j loops of mm_blocked", "parallel method, p = 1, 2, 3, 4, 8"],
       ["tiles", "row_tile called in a plain sequential loop, no OpenMP", "diagnostic: parallel code path without the runtime"]],
      [1.6 * cm, 11.4 * cm, 4.0 * cm],
      caption="Table 1. Methods (src/matmul_par.c). Only the timed call differs; loading, allocation of C and "
              "validation are identical.")
P("The kernel inside a tile is the Assignment 2 kernel line for line; row_tile is a separate function only because GCC "
  "drops the restrict qualifier of variables captured by an OpenMP region and would otherwise insert a run-time aliasing "
  "check before the vectorised loop of the -O3 builds. Each row tile is a disjoint block of rows of C that only its thread "
  "zeroes and writes; A and B are read-only, so the loop has no shared writes and needs no locks or atomics. Two differences "
  "from the baseline are unavoidable and are measured: C is zeroed tile by tile instead of by one memset, and the tile "
  "loop goes through a function call. The method tiles isolates them from the OpenMP runtime. "
  "Thread count: OMP_NUM_THREADS = p, OMP_DYNAMIC = false. Affinity: OMP_PROC_BIND and OMP_PLACES unset, so the OS "
  "scheduler places threads; with one hardware thread per core and one socket there is no SMT or NUMA choice to make. "
  "Every process prints the number of threads that actually executed the region, and the harness rejects a run if it "
  "differs from p. No numerical library runs inside a benchmark process, so there are no hidden library threads.")
P("3.2 Builds", H2)
table([["Binary", "Flags (all: -std=c11 -ffp-contract=off)", "Role"],
       ["mm", "-O2 -falign-loops=32 -fopenmp", "main study: the Assignment 2 flags plus OpenMP and loop alignment"],
       ["mm_nomp", "-O2 -falign-loops=32 (no OpenMP)", "control N: linking OpenMP does not change seq"],
       ["mm_noalign", "-O2 -fopenmp (default code placement)", "control A: code-placement effect (Section 4.1)"],
       ["mm_scalar", "-O3 -mavx2 -falign-loops=32 -fopenmp -fno-tree-vectorize -fno-tree-slp-vectorize",
        "extension V: vectorisation off"],
       ["mm_simd", "-O3 -mavx2 -falign-loops=32 -fopenmp", "extension V: vectorisation on"]],
      [2.0 * cm, 8.0 * cm, 7.0 * cm], caption="Table 2. Builds of the single source file (build.sh), gcc 13.3, libgomp.")
ev = VEV.set_index(["binary", "function"])
P("The main study uses scalar code, as in Assignment 2: GCC's vectorisation report for build/mm lists no vectorised loop "
  f"in the kernels and their disassembly contains only scalar mulsd/addsd ({int(ev.loc[('mm', 'row_tile')].scalar)} per function). "
  "For mm_simd the report lists the inner j loop of both mm_blocked and row_tile as 'vectorized using 32 byte vectors', and "
  f"the disassembly of each function contains {int(ev.loc[('mm_simd', 'row_tile')].packed_ymm)} packed ymm (256-bit) "
  "multiply or add instructions; mm_scalar contains none (results/vec_evidence.csv, scripts/vec_evidence.py). "
  "-ffp-contract=off forbids fused multiply-add in every build, so all builds compute exactly the same sums.")
P("3.3 Inputs, environment and protocol", H2)
P(f"Inputs are the Assignment 1 and 2 files: uniform [0, 1) float64, NumPy PCG64, seed 20261012 + n for A and "
  f"20261013 + n for B; their SHA-256 values (data/manifest.csv) match the Assignment 2 manifest for all shared sizes. "
  f"Hardware: {ENV['cpu_model']} ({ENV['hypervisor']} guest), {ENV['physical_cores']} physical cores = "
  f"{ENV['logical_processors']} logical processors (1 thread per core), L1d {ENV['l1d_cache']}, L2 {ENV['l2_cache']}, "
  f"L3 reported as {ENV['l3_cache']} by the hypervisor, {int(ENV['ram'].split()[1]) / 2 ** 20:.1f} GiB RAM; "
  f"{ENV['os']}; {ENV['gcc']}. This is a different VM instance from Assignment 2, so absolute times are not comparable "
  "across the two reports; every comparison here is within one session on one machine.")
P(f"Sizes n = {', '.join(map(str, sizes))}; threads p = {', '.join(map(str, threads))}, where p = 3, 4 and 8 are "
  "oversubscribed. Timer: clock_gettime(CLOCK_MONOTONIC) around the method call only. Warm-up: 1 s for n ≤ 1024; for "
  "larger n one untimed calibration call, which also creates the OpenMP thread pool, precedes the samples in every case. "
  "Samples per process: 10 for n ≤ 512, 5 up to 3072, 3 at 4096 (one sequential multiplication takes about 33 s there; "
  "with two rounds this gives 6 samples). Samples under 0.1 s are batched and reported per multiplication. "
  f"The whole set ran in {PROTO['rounds']} rounds; within a round the configurations of each size ran in a random order "
  "(fixed seed), so slow drift of the VM is spread over all methods. "
  f"Budgets declared in advance: {PROTO['limit_s']:.0f} s per multiplication, {PROTO['wall_limit_s'] / 60:.0f} min per "
  "configuration, 4 GiB of storage; no run reached any of them. Statistic: median of all samples of a configuration "
  "(both rounds pooled) and IQR; S(p) and E(p) use those medians, and the error bars of Figure 2 show the range of "
  "S(p) computed round by round. Memory: peak RSS of the process (ru_maxrss via wait4). CPU use: CPU time of all "
  "threads divided by wall time over the kernel phase, i.e. the average number of busy cores. "
  "The vectorisation extension (V) ran mm_scalar and mm_simd with seq, tiles and omp at p = 1 and 2 for n = 512, 1024, "
  "2048 and 4096 in the same rounds; the tiles and omp p = 1 configurations of V were added by a follow-up run "
  "(run_benchmarks.py --only V --skip-existing) after the first analysis, for the reason given in Section 4.7.")
P("3.4 Correctness and thread safety", H2)
P(f"scripts/run_tests.py ran {n_checks} checks ({n_ok} passed) before any benchmark: hand-computed 3×3, identity, zero "
  "and an exact integer product for every build, method and p; random matrices with n = 1, 33, 64, 100 and 256 (fewer tiles "
  "than threads, a partial last tile, tile counts not divisible by p) within 2γ<sub>n</sub>|A||B| of NumPy, "
  "γ<sub>n</sub> = nu/(1 - nu), u = 2<super>-53</super> [9]; and bit-for-bit equality with the sequential baseline for "
  "every build, method and thread count. A data race on C would make the result depend on thread timing, so the tests "
  "also repeat the 8-thread run 20 times and require identical bytes. Every benchmark process prints a checksum and three "
  "elements of C, which the harness checks against NumPy with the same tolerance; all "
  f"{n_runs} benchmark processes passed and reported the requested number of threads.")

# =============================== 4 RESULTS ===============================
P("4. Results", H1)
P(f"The data set has {n_samples} measured samples from {n_runs} processes (results/raw_measurements.csv, runs.csv), "
  "all valid and within budget.")

P("4.1 A confound found first: code placement", H2)
P(f"During development the parallel code path with one thread was about 60 % slower than the baseline, which looked "
  "like OpenMP overhead. The method tiles, which uses no OpenMP at all, was just as slow, and the inner loops of the two "
  "functions are instruction for instruction the same. During development, changing only code-layout options of the "
  "compiler and assembler moved the slowdown from one function to the other, and aligning loop starts to 32 bytes "
  "(-falign-loops=32) removed it. Control A keeps the default placement "
  f"(build/mm_noalign): tiles and omp p = 1 took {ctl('A', 512, 'tiles', 'ratio_to_P_seq'):.2f}× and "
  f"{ctl('A', 512, 'omp', 'ratio_to_P_seq'):.2f}× the aligned baseline at n = 512 and "
  f"{ctl('A', 1024, 'tiles', 'ratio_to_P_seq'):.2f}× and {ctl('A', 1024, 'omp', 'ratio_to_P_seq'):.2f}× at n = 1024, "
  f"while seq was unaffected ({ctl('A', 512, 'seq', 'ratio_to_P_seq'):.2f}×, {ctl('A', 1024, 'seq', 'ratio_to_P_seq'):.2f}×) "
  "(Figure 1). The mechanism was not identified (no hardware counters are available on this VM), but the effect is "
  "reproducible and independent of OpenMP, so every build used for comparisons aligns its loops. Without this control the "
  "p = 1 overhead and every speed-up would have been misattributed by up to 60 %.")
figure("fig6_code_placement.png", 11.0, "Figure 1. Time of the same kernel at one thread relative to the aligned "
       "baseline, with default code placement (grey) and with loops aligned to 32 bytes (blue). Median of 20 samples.")

P("4.2 Runtime, speed-up and efficiency", H2)
rows = [["n", "T<sub>seq</sub>"] + [f"p = {p}" + (" (over)" if p > 2 else "") for p in threads]]
for n in sizes:
    cells = [str(n), f"{tfmt(g(n, 'seq', 1, 'median_s'))} ±{g(n, 'seq', 1, 'iqr_pct'):.0f}%"]
    for p in threads:
        cells.append(f"{tfmt(g(n, 'omp', p, 'median_s'))} ±{g(n, 'omp', p, 'iqr_pct'):.0f}%<br/>"
                     f"S {g(n, 'omp', p, 'speedup'):.2f} · E {g(n, 'omp', p, 'efficiency'):.2f}")
    rows.append(cells)
table(rows, [1.0 * cm, 2.4 * cm] + [2.7 * cm] * len(threads),
      caption="Table 3. Median time per multiplication ± IQR (% of median), speed-up S(p) = Tseq/Tp and efficiency "
              "E(p) = S/p. 20 samples per cell for n ≤ 512, 10 up to 3072, 6 at 4096 (two rounds pooled). "
              "'over' = oversubscribed.")
figure("fig1_speedup.png", 13.0, "Figure 2. Speed-up against threads. Error bars: range of S(p) computed separately for "
       "each round. Dashed: min(p, 2), the limit set by two cores.")
P(f"<b>Two threads.</b> S(2) was {S2.min():.2f}-{S2.max():.2f} across all sizes and at least "
  f"{S2[S2.index >= 512].min():.2f} for n ≥ 512; efficiency was {E2.min():.2f}-{E2.max():.2f}. The lowest value, "
  f"{S2.min():.2f} at n = {int(S2.idxmin())}, has a round range of {g(int(S2.idxmin()), 'omp', 2, 'speedup_round_min'):.2f}-"
  f"{g(int(S2.idxmin()), 'omp', 2, 'speedup_round_max'):.2f} and coincides with a slower omp p = 1 at that size "
  f"({ov1[int(S2.idxmin())]:+.1f} %); relative to the parallel code at one thread the two-thread speed-up there is "
  f"{s_rel1[int(S2.idxmin())]:.2f}. H1 is supported: the kernel scaled almost perfectly to the two available cores. "
  f"H2 is not supported at these sizes: even n = 256, where one multiplication takes {tfmt(g(256, 'omp', 2, 'median_s'))} "
  f"with two threads and there are only 8 row tiles, reached S(2) = {S2[256]:.2f}. The fork/join cost of a parallel region "
  "is of the order of microseconds [6], too small to see against milliseconds of work. Parallel execution therefore "
  "helped for every tested size; it would stop helping only for multiplications lasting microseconds, below the tested range.")
figure("fig2_efficiency.png", 12.0, "Figure 3. Parallel efficiency. Dashed: min(p, 2)/p, the best possible on two cores.")

P("4.3 One thread: the cost of the parallel implementation", H2)
P(f"With p = 1 the OpenMP version took between {min(ov1.values()):+.1f} % and {max(ov1.values()):+.1f} % of the baseline "
  f"time, and the no-OpenMP tiles path between {min(ovt.values()):+.1f} % and {max(ovt.values()):+.1f} %. These differences "
  "have both signs and are of the same order as the IQRs (Table 3), so neither the OpenMP runtime nor the restructuring had a "
  "measurable cost once code placement was controlled. Control N (build/mm_nomp, no -fopenmp) gave "
  + ", ".join(f"{ctl('N', n, 'seq', 'ratio_to_P_seq'):.2f}×" for n in sorted(CTL[CTL.experiment == 'N'].n.unique()))
  + " the baseline for n = " + ", ".join(str(n) for n in sorted(CTL[CTL.experiment == 'N'].n.unique()))
  + ": linking OpenMP does not change the sequential baseline either.")

P("4.4 Oversubscription: more threads than cores", H2)
P(f"With p = 3, 4 and 8 the speed-up ranged from {over.speedup.min():.2f} to {over.speedup.max():.2f}, never clearly above "
  f"S(2) (largest gain over S(2) at the same n: {max(g(n, 'omp', p, 'speedup') - S2[n] for n in sizes for p in threads if p > 2):+.2f}, "
  "within the round-to-round spread). H3 is supported. The clearest loss was p = 3 at small n: "
  f"S = {g(256, 'omp', 3, 'speedup'):.2f} at n = 256 and {g(512, 'omp', 3, 'speedup'):.2f} at n = 512, against "
  f"{S2[256]:.2f} and {S2[512]:.2f} with two threads. Figure 4 shows why: the average number of busy cores fell from "
  f"{g(256, 'omp', 2, 'cores_busy'):.2f} with two threads to {g(256, 'omp', 3, 'cores_busy'):.2f} with three at n = 256. "
  "schedule(static) gives each of three threads a third of the 8 or 16 row tiles; two threads run while the third waits for a "
  "core, and at the end of the loop one core is idle while the last thread finishes. A multiplication of a few milliseconds "
  "is comparable to the scheduler's time slice, so this imbalance is not averaged out. With p = 4 and 8 the work is split "
  "into smaller pieces and the cores stayed busier (≥ "
  f"{min(g(n, 'omp', p, 'cores_busy') for n in sizes for p in (4, 8)):.2f}), and for n ≥ 1024 every oversubscribed "
  f"configuration kept the cores ≥ {min(g(n, 'omp', p, 'cores_busy') for n in sizes if n >= 1024 for p in (3, 4, 8)):.2f} "
  "busy and came within a few per cent of S(2). Efficiency drops as 2/p by construction (Figure 3). The n = 256, p = 4 cell "
  f"also had the largest IQR of the study ({g(256, 'omp', 4, 'iqr_pct'):.0f} %).")
figure("fig4_cores_busy.png", 11.5, "Figure 4. CPU time of all threads divided by wall time over the kernel phase "
       "(warm-up, calibration and samples). 2 = both vCPUs busy all the time.")

P("4.5 Is the superlinear speed-up real?", H2)
sl = [(n, p) for n in sizes for p in threads if g(n, "omp", p, "speedup") > min(p, 2)]
P(f"S(p) exceeded min(p, 2) in {len(sl)} configurations, all by at most "
  f"{max(g(n, 'omp', p, 'speedup') / min(p, 2) - 1 for n, p in sl) * 100:.1f} %: "
  + ", ".join(f"n = {n}, p = {p} ({g(n, 'omp', p, 'speedup'):.2f})" for n, p in sl) + ". "
  f"The largest is at n = {big}, where both rounds gave S(2) > 2 ({g(big, 'omp', 2, 'speedup_round_min'):.3f} and "
  f"{g(big, 'omp', 2, 'speedup_round_max'):.3f}). The evidence points to the baseline, not to extra cache per thread: at "
  f"n = {big} the parallel code path was already {-ovt[big]:.1f} % (tiles) and {-ov1[big]:.1f} % (omp p = 1) faster than "
  "the baseline with a single thread. A plausible reason is that the baseline zeroes all of C "
  f"({8 * big ** 2 / 2 ** 20:.0f} MiB) before computing, so those rows must come back from memory later, while row_tile "
  "zeroes 32 rows (1 MiB) just before using them; this was not isolated by a separate experiment. Measured against the "
  f"parallel code at one thread, the two-thread speed-up at n = {big} is {s_rel1[big]:.2f}, i.e. not superlinear. "
  "The other cases are single values within their IQR of 2.")

P("4.6 Memory", H2)
rows = [["n", "3 × 8n²", "seq", "omp p = 1", "omp p = 2", "omp p = 8"]]
for n in sizes:
    rows.append([str(n), f"{3 * 8 * n * n / 2 ** 20:.1f} MiB"] +
                [f"{g(n, m, p, 'peak_rss_mib'):.1f} MiB" for m, p in (("seq", 1), ("omp", 1), ("omp", 2), ("omp", 8))])
table(rows, [1.2 * cm, 2.3 * cm, 2.3 * cm, 2.3 * cm, 2.3 * cm, 2.3 * cm],
      caption="Table 4. Peak resident set size (largest of the two rounds) against the analytic storage of A, B and C.")
P(f"Peak RSS was the array storage plus about {mib_base:.0f} MiB of program and runtime baseline, and differed by at most "
  f"{rss_spread:.1f} MiB between thread counts at any size. The parallel version needs no extra matrix: each thread writes "
  "straight into its rows of C. Thread stacks are reserved as virtual memory and only the touched pages count in RSS, "
  "which is why 8 threads cost no visible memory.")

P("4.7 Optional extension: vectorisation", H2)
rows = [["n", "scalar p = 1", "AVX2 p = 1", "scalar p = 2", "AVX2 p = 2", "vec. gain p = 1 / p = 2",
         "thread gain scalar / AVX2", "both"]]
for r in VEC.itertuples():
    rows.append([str(r.n), tfmt(r.scalar_p1_s), tfmt(r.simd_p1_s), tfmt(r.scalar_p2_s), tfmt(r.simd_p2_s),
                 f"{r.vec_speedup_p1:.2f} / {r.vec_speedup_p2:.2f}", f"{r.thread_speedup_scalar:.2f} / {r.thread_speedup_simd:.2f}",
                 f"{r.combined_speedup:.2f}"])
table(rows, [1.0 * cm, 1.9 * cm, 1.9 * cm, 1.9 * cm, 1.9 * cm, 2.8 * cm, 2.8 * cm, 1.3 * cm],
      caption="Table 5. Experiment V, OpenMP code path (omp with p = 1 and 2) of the two -O3 -mavx2 builds that differ only in "
              "auto-vectorisation. Medians; 20, 10, 10 and 6 samples. 'both' = scalar p = 1 / AVX2 p = 2.")
figure("fig5_vectorisation.png", 13.0, "Figure 5. GFLOP/s (2n³ / median time) for the 2×2 design.")
P(f"Vectorisation and threads were varied one at a time on the same code path. AVX2 sped up the kernel by "
  f"{vmin:.2f}-{vmax:.2f}× at both thread counts, and two threads sped it up by {tmin:.2f}-{tmax:.2f}× with or "
  "without vectorisation, so the two effects were independent and multiplied (H4 supported): "
  f"{VEC.combined_speedup.min():.1f}-{VEC.combined_speedup.max():.1f}× together, and "
  f"{VEC.combined_vs_o2_seq.min():.1f}-{VEC.combined_vs_o2_seq.max():.1f}× over the -O2 sequential baseline of the main "
  "study. The vector gain is well below the factor of four that 256-bit registers allow and it fell at "
  f"n = 4096 ({vrow(4096, 'vec_speedup_p1'):.2f}×), where the AVX2 version dropped from about 20 to "
  f"{vrow(4096, 'gflops_simd_p2'):.1f} GFLOP/s. The 32-element inner loop does one multiply and one add per loaded and "
  "stored element of C, so once the arithmetic is four times cheaper the loads and stores dominate; the larger working set at "
  "n = 4096 makes them slower. This explanation is consistent with the data but no memory-traffic counters were available "
  "to confirm it. The results stayed bit-identical to the scalar baseline.")
P(f"<b>Why the follow-up run.</b> In the first analysis, the -O3 scalar build's sequential function ran "
  f"{100 * (vrow(1024, 'scalar_seq_over_omp1') - 1):.0f} % slower than its own row_tile path at one thread (n = 1024; "
  f"{VEC.scalar_seq_over_omp1.min():.2f}-{VEC.scalar_seq_over_omp1.max():.2f}× over all n), although their inner loops "
  "are identical in the disassembly; in the AVX2 build the two paths were within "
  f"{100 * max(abs(VEC.simd_seq_over_omp1 - 1)):.1f} %. Comparing scalar seq with AVX2 seq would have mixed this code "
  f"difference into the vectorisation effect, and comparing scalar seq with scalar omp p = 2 would have shown a misleading "
  f"'superlinear' {vc(1024, 'mm_scalar', 'seq', 1) / vc(1024, 'mm_scalar', 'omp', 2):.2f}×. Table 5 therefore uses "
  "the parallel code path at p = 1 as the reference for both effects. In the -O2 main study the two paths were within the "
  "noise (Section 4.3), so this issue does not affect Sections 4.2-4.5.")

# =============================== 5 DISCUSSION ===============================
P("5. Discussion", H1)
P(f"<b>Answer to the research question.</b> On this two-core machine the blocked kernel parallelised by row tiles reached "
  f"S(2) = {S2.min():.2f}-{S2.max():.2f} with efficiency close to 1 for every size from 256 to 4096, with no measurable "
  "OpenMP overhead at one thread and identical memory for every thread count. Parallel execution helped whenever there were "
  "at least as many row tiles as threads and the work lasted milliseconds or more; it stopped helping beyond the number of "
  "hardware contexts, and with an uneven split across time-sliced threads (p = 3, small n) it lost up to "
  f"{100 * (1 - g(256, 'omp', 3, 'speedup') / S2[256]):.0f} % of the two-thread speed-up. The near-perfect scaling is "
  "expected for this kernel: threads share only read-only inputs, write disjoint rows, and each works on tiles that fit in "
  "its own core's L1 and L2, so neither synchronisation nor shared data limits it at p = 2.")
P("<b>What the study shows about measuring parallel speed-up.</b> Two of the most striking raw numbers were artefacts of "
  "the code, not of parallelism: a 60 % 'overhead' at one thread caused by code placement, and a 'superlinear' speed-up "
  "explained by a baseline that was a few per cent slower than the parallel code path. Both were found by keeping a "
  "parallel-code-path-without-OpenMP diagnostic, the parallel version at one thread, and the raw per-round data. "
  "The equivalent sequential baseline remains the reference for S(p), as the assignment asks; Table 3 and Section 4.5 also "
  "give the speed-up relative to the parallel code at one thread so the reader can see both.")
P("<b>Limitations.</b> (1) Two vCPUs are the minimum for a parallel study: the results say nothing about scaling to more "
  "cores, where memory bandwidth, shared caches and NUMA effects appear. (2) The VM's neighbours and the hypervisor are not "
  "under our control; the two rounds and randomised order reduce but do not remove that noise, and the load average at the "
  f"start was {ENV['load_average_at_start'][0]:.1f} because the tests had just run. (3) No hardware performance counters "
  "were available, so the cache and memory explanations in Sections 4.5 and 4.7 are inferences. (4) Only schedule(static), "
  "unbound threads and one tile size were tested; dynamic scheduling might reduce the p = 3 imbalance. (5) The vectorisation "
  "extension relied on GCC's auto-vectoriser with AVX2; explicit intrinsics or AVX-512, which this CPU also supports, were "
  "not tested. (6) Timings come from a cloud VM, not from the author's laptop.")

# =============================== 6 CONCLUSIONS ===============================
P("6. Conclusions", H1)
P(f"Within the tested scope (C, OpenMP, float64, blocked kernel with 32×32 tiles, n = 256-4096, one 2-vCPU VM), sharing "
  f"row tiles between two threads halved the runtime: S(2) = {S2.min():.2f}-{S2.max():.2f}, with bit-identical results and "
  "no extra memory. Threads beyond the two hardware contexts added nothing and, at small n with an uneven split, cost time. "
  f"Auto-vectorisation with AVX2 added a further {vmin:.1f}-{vmax:.1f}× independently of the thread count, for "
  f"{VEC.combined_vs_o2_seq.min():.1f}-{VEC.combined_vs_o2_seq.max():.1f}× over the scalar sequential baseline in total. "
  "Before trusting a speed-up, code placement and the code path of the baseline had to be checked. Open questions: scaling "
  "beyond two cores and the memory-bound limit of the vectorised kernel, which need a larger shared-memory machine, and "
  "the effect of thread placement and dynamic scheduling.")

# =============================== 7 AI ===============================
P("7. Use of generative AI", H1)
P("In line with the course policy, the AI assistant Claude (Anthropic) was used to plan the experiments, draft the C "
  "implementation, the harness and analysis scripts, the README and the first draft of this report. All numerical results "
  "come from measurements stored in results/ and produced by the committed code; the AI output is not used as evidence. "
  "The author remains responsible for understanding the code, validating the measurements, verifying the references and "
  "defending the conclusions.")

# =============================== REFERENCES ===============================
P("References", H1)
refs = [
    "“Individual Assignments: Matrix Multiplication,” Big Data, Grado en Ciencia e Ingeniería de Datos, ULPGC, revised proposal 2026-2027.",
    "M. D. Afonso Suárez, J. J. Hernández Cabrera and J. Évora Gómez, “Big Data,” course materials, Universidad de Las Palmas "
    "de Gran Canaria, 2026/27. [Online]. Available: https://mltechdrawer.github.io/BIGDATA/",
    "G. M. Amdahl, “Validity of the single processor approach to achieving large scale computing capabilities,” in "
    "<i>Proc. AFIPS Spring Joint Computer Conf.</i>, 1967, pp. 483-485.",
    "J. L. Hennessy and D. A. Patterson, <i>Computer Architecture: A Quantitative Approach</i>, 6th ed. Cambridge, MA, USA: Morgan Kaufmann, 2017.",
    "OpenMP Architecture Review Board, “OpenMP Application Programming Interface, Version 4.5,” Nov. 2015. [Online]. Available: https://www.openmp.org/specifications/",
    "B. Chapman, G. Jost and R. van der Pas, <i>Using OpenMP: Portable Shared Memory Parallel Programming</i>. Cambridge, MA, USA: MIT Press, 2008.",
    "Intel Corporation, <i>Intel 64 and IA-32 Architectures Optimization Reference Manual</i>. [Online]. Available: https://www.intel.com/",
    "“GCC 12 Release Series: Changes, New Features, and Fixes,” Free Software Foundation. [Online]. Available: https://gcc.gnu.org/gcc-12/changes.html",
    "N. J. Higham, <i>Accuracy and Stability of Numerical Algorithms</i>, 2nd ed. Philadelphia, PA, USA: SIAM, 2002.",
    "H. Amador Hidalgo, “Optimised dense and sparse matrix multiplication,” Big Data, Individual Assignment 2, ULPGC, 2026.",
]
for i, r in enumerate(refs, 1):
    P(f"[{i}] {r}", REF)
P("Reproducibility materials", H1)
P(f"Repository: {args.repo_url}, folder Assignment03_AmadorHidalgo: src/ (matmul_par.c, harness.h), build.sh, a README with "
  "exact commands and the thread configurations, requirements.txt, scripts/ (input generator, tests, vectorisation evidence, "
  "benchmark harness, analysis, this report), data/manifest.csv, and results/ (raw samples, one row per process with thread "
  "count, OpenMP settings, RSS and CPU time, summaries, key numbers, environment, correctness log and figures). "
  "The sequential baseline is reused from Assignment02_AmadorHidalgo/src/dense.c in the same repository [10].")


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(colors.HexColor("#52514e"))
    canvas.drawString(2 * cm, 1.1 * cm, "Hugo Amador Hidalgo - Big Data - Assignment 3 - Parallel matrix multiplication")
    canvas.drawRightString(A4[0] - 2 * cm, 1.1 * cm, f"{doc.page}")
    canvas.restoreState()


doc = SimpleDocTemplate(str(OUT), pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm, topMargin=1.8 * cm, bottomMargin=1.8 * cm,
                        title="Assignment 3 - Parallel matrix multiplication and optional vectorisation",
                        author="Hugo Amador Hidalgo", subject="Big Data 2026-2027, ULPGC")
doc.build(story, onFirstPage=footer, onLaterPages=footer)
print("report written:", OUT)
