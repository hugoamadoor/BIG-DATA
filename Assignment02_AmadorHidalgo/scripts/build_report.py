#!/usr/bin/env python3
"""Build report/Assignment02_AmadorHidalgo_Report.pdf from results/ (run analyze.py first).

Numbers in tables and in the sentences that quote measurements are read from the CSV/JSON files.
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
OUT = ROOT / "report" / "Assignment02_AmadorHidalgo_Report.pdf"
OUT.parent.mkdir(exist_ok=True)

DEN = pd.read_csv(RES / "summary_dense.csv")
TUN = pd.read_csv(RES / "summary_tuning.csv")
SPA = pd.read_csv(RES / "summary_sparse.csv")
XC = pd.read_csv(RES / "summary_crosscheck.csv")
RUNS = pd.read_csv(RES / "runs.csv")
RAW = pd.read_csv(RES / "raw_measurements.csv")
ENV = json.loads((RES / "environment.json").read_text())
KEY = json.loads((RES / "key_numbers.json").read_text())
CORR = pd.read_csv(RES / "correctness.csv")
BASE = json.loads((RES / "baseline_rss.json").read_text())
MAN = pd.read_csv(ROOT / "data" / "manifest.csv")
PROTO = ENV["protocol"]
B = KEY["block"]


def tfmt(s):
    if s != s:
        return "-"
    if s < 1e-3:
        return f"{s * 1e6:.3g} µs"
    if s < 1:
        return f"{s * 1e3:.3g} ms"
    return f"{s:.3g} s"


def xf(v, d=1):
    return "-" if v != v else f"{v:,.{d}f}×"


def dv(method, n, col):
    r = DEN[(DEN.method == method) & (DEN.n == n)]
    return float(r[col].iloc[0]) if len(r) else float("nan")


def sv(exp, n, d, col, method="csr_spgemm", inp="csr", structure="uniform"):
    r = SPA[(SPA.experiment == exp) & (SPA.n == n) & np.isclose(SPA.density_pct, d) & (SPA.method == method) &
            (SPA.input == inp) & (SPA.structure == structure)]
    return float(r[col].iloc[0]) if len(r) else float("nan")


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
dense_sizes = sorted(DEN.n.unique())
sp = {m: dv(m, 2048, "speedup_vs_ijk") for m in ["ijk_bt", "ikj", "blocked"]}
sp_rng = {m: (DEN[(DEN.method == m) & (DEN.n >= 1024)].speedup_vs_ijk.min(),
              DEN[(DEN.method == m) & (DEN.n >= 1024)].speedup_vs_ijk.max()) for m in ["ijk_bt", "ikj", "blocked"]}
blk_vs_ikj = {n: dv("ikj", n, "median_s") / dv("blocked", n, "median_s") for n in dense_sizes}
fit = KEY["dense_fitted_exponent_n_ge_512"]
ns_blk = KEY["dense_ns_per_mult_add_blocked"]
ijk1024 = RAW[(RAW.method == "ijk") & (RAW.n == 1024) & (RAW.phase == "multiply") &
              RAW.experiment.isin(["D", "M", "X"])].groupby(["experiment", "density_pct"]).ns_per_mult.median() * 1e-9
xc = KEY.get("crosscheck_a1_binary", {})
n_checks = len(CORR)
n_ok = int(CORR.passed.sum())
n_samples = int((~RAW.experiment.str.endswith("aborted")).sum())
n_runs = int((~RUNS.experiment.str.endswith("aborted")).sum())
M = SPA[(SPA.experiment == "M") & (SPA.method == "csr_spgemm") & (SPA.input == "csr")]
cross = KEY["crossover"]
lo_win = {n: max(cross[str(n)]["sparse_faster_at"]) for n in (1024, 2048)}
hi_lose = {n: min(cross[str(n)]["dense_faster_at"]) if cross[str(n)]["dense_faster_at"] else float("nan")
           for n in (1024, 2048)}
c_sp_dense = float(M[M.density_pct >= 50].ns_per_mult_add.median())
c_dense = float(np.median(list(ns_blk.values())))
d_star = 100 * np.sqrt(c_dense / c_sp_dense)
L = SPA[(SPA.experiment == "L") & (SPA.method == "csr_spgemm")].sort_values("n")
L_exp = float(np.polyfit(np.log(L.n), np.log(L.median_s), 1)[0])
gib16384 = 3 * 8 * 16384 ** 2 / 2 ** 30
mib_base = BASE["dense"] / 1024

# =============================== TITLE ===============================
P("Optimised Dense and Sparse Matrix Multiplication: Loop Order, Cache Blocking and CSR SpGEMM", TITLE)
P("Big Data - Individual Assignment 2 (Optimised approaches and sparse matrices)", SUB)
P("Hugo Amador Hidalgo - Grado en Ciencia e Ingeniería de Datos, ULPGC - Academic year 2026-2027", SUB)
P(f"Repository: {args.repo_url} (folder Assignment02_AmadorHidalgo)", SUB)

P("Abstract", H1)
P(f"This study asks two separate questions about sequential float64 matrix multiplication in C. Part A: how much do "
  f"three optimisations - transposing B, reordering the loops to i-k-j, and cache blocking - speed up the Assignment 1 "
  f"i-j-k baseline as n grows? Part B: at which tested densities and structures does a CSR sparse-sparse product "
  f"(Gustavson's algorithm, own code) beat the best dense method? All methods were validated against NumPy/SciPy and are "
  f"bit-identical to the baseline. On a 2-vCPU cloud VM (gcc 13, -O2, scalar code), the three optimisations were "
  f"{sp_rng['ijk_bt'][0]:.0f}-{sp_rng['blocked'][1]:.0f}× faster than the baseline for 1024 ≤ n ≤ 2048, mainly by "
  f"removing the column-stride access to B. Blocking (tile {B}) matched loop reordering up to n = 2048 and was "
  f"{blk_vs_ikj[3072]:.2f}× and {blk_vs_ikj[4096]:.2f}× faster than it at n = 3072 and 4096. The baseline exceeded the "
  f"180 s limit at n = 3072; blocked was the only method above 3 GFLOP/s at n = 4096 (40 s). For uniformly random sparsity "
  f"at n = 1024 and 2048, CSR SpGEMM was faster than dense blocked at every density up to {lo_win[2048]:g} % and slower from "
  f"{hi_lose[2048]:g} %, consistent with its d²n³ work and a cost per useful multiply-add about "
  f"{c_sp_dense / c_dense:.0f}× higher than dense. Conversion from dense storage and fill-in reduced the advantage: "
  f"at 0.1 % density the multiply alone was about 45 000× faster than dense, but only about 300× once the conversion was "
  f"counted. At equal density, banded and power-law matrices ran up to {sv('S', 2048, 5, 'median_s') / sv('S', 2048, 5, 'median_s', structure='banded'):.1f}× "
  f"faster than uniform ones, so density alone did not determine the cost. With 16 nonzeros per row, CSR "
  f"multiplied n = 131 072 matrices in {tfmt(L.median_s.iloc[-1])}, while dense blocked hit the time limit at n = 8192.")

# =============================== 1 INTRODUCTION ===============================
P("1. Introduction and objective", H1)
P("Assignment 1 showed that the textbook i-j-k loop in C spends most of its time waiting for memory once the matrices "
  "no longer fit in cache: its inner loop walks B down a column, touching a new cache line on every iteration. "
  "This study keeps the same language, compiler flags, numeric type and inputs and asks how far that cost can be reduced "
  "without changing the O(n³) arithmetic, and when it is better not to store the zeros at all.")
P("<b>Research question A.</b> For n × n dense float64 matrices, how do B-transposition, i-k-j loop reordering and "
  "cache blocking change runtime and memory relative to the Assignment 1 baseline, as n increases? "
  "<b>Research question B.</b> For matched logical inputs, at which densities and nonzero structures does CSR SpGEMM run faster "
  "than the best dense method, and how much of that advantage survives conversion costs and fill-in?")
P("<b>Hypotheses (stated before measuring).</b> H1: any method that removes the stride-n access to B gives a large speed-up "
  "once B no longer fits in the caches. H2: blocking helps beyond loop reordering only when the data that i-k-j reuses "
  "between iterations of i (all of B) exceeds the effective cache. H3: SpGEMM wins below some density threshold; because "
  "its work is about d²n³ for uniform inputs but its cost per operation is higher, the threshold is well below 100 %, "
  "and conversion and fill-in lower its advantage.")

# =============================== 2 BACKGROUND ===============================
P("2. Background", H1)
P("<b>Locality and loop order.</b> With row-major storage, i-j-k reads A with unit stride but B with stride n. Reordering "
  "to i-k-j turns the inner loop into C[i,:] += A[i,k]·B[k,:], so all three operands are read with unit stride [3], [4]. "
  "Transposing B first gives the same unit-stride access while keeping the i-j-k dot product. "
  "<b>Cache blocking</b> partitions the loops into tiles so that a tile of B is reused for many rows of A while it is still "
  "in cache; the best tile size depends on the cache hierarchy and is usually found empirically [5], [6]. "
  "Algorithmic alternatives such as Strassen's method [7] reduce the operation count but change the order of rounding and "
  "only pay off at large n, so they were not chosen here. Keeping the summation order fixed is what makes the bitwise "
  "comparison in Section 3.4 possible.")
P("<b>Sparse storage and SpGEMM.</b> Compressed sparse row (CSR) stores for each row the column indices and values of its "
  "nonzeros, plus one row pointer per row [8], [9]. Gustavson's row-wise algorithm [10] computes C[i,:] as a combination of "
  "the rows B[k,:] selected by the nonzeros of A[i,:], gathering partial sums in a dense accumulator. Its work is the number "
  "of products a<sub>ik</sub>b<sub>kj</sub> with both factors nonzero; for independent uniform patterns with density d this is "
  "about d²n³, and the expected density of C is 1 - (1 - d²)<super>n</super>, which is close to 1 once d²n is a few "
  "units (fill-in). SciPy's sparse product, used here only as a reference, is based on SMMP [11], [12].")

# =============================== 3 METHODOLOGY ===============================
P("3. Methodology", H1)
P("3.1 Implementations", H2)
table([["Method", "Change relative to the baseline", "Timed region"],
       ["ijk (baseline)", "Assignment 1 C kernel, copied unchanged (commit ca7622a); only restrict qualifiers added",
        "loops"],
       ["ijk_bt", "B transposed into a scratch buffer, then i-j-k with unit stride in both operands", "transpose + loops"],
       ["ikj", "loop reordering: C[i,:] += A[i,k] B[k,:]", "zeroing C + loops"],
       ["blocked", f"i-k-j inside square tiles (loop order ii, kk, jj, i, k, j); tile side {B} from experiment T",
        "zeroing C + loops"],
       ["csr_spgemm", "CSR x CSR, row-wise Gustavson with a dense accumulator (two passes: symbolic, then numeric)",
        "free old C + symbolic + allocation + numeric"]],
      [2.4 * cm, 10.4 * cm, 4.2 * cm],
      caption="Table 1. Methods. All are single-threaded C in src/dense.c and src/sparse.c; no library is called. "
              "Inputs and C (and the transpose buffer and accumulator) are allocated before the timer starts.")
P("All dense methods use one contiguous row-major array per matrix, as in Assignment 1. CSR uses int64 row pointers, int32 "
  "column indices and float64 values (8(n+1) + 12·nnz bytes). Output column indices are left in production order (not "
  "sorted), which is valid CSR for this study but would need a sort for libraries that require ordered indices. "
  "Everything is compiled with gcc 13.3 -O2 -std=c11 -ffp-contract=off, the Assignment 1 flags. GCC enables limited "
  "auto-vectorisation at -O2 since version 12 [13], but the vectorisation report (build/vec_report_dense.txt) lists no "
  "vectorised loop in any kernel and the disassembly of ikj contains only scalar mulsd/addsd, so every result here is for "
  "scalar code. SIMD belongs to Assignment 3.")

P("3.2 Inputs", H2)
nm = MAN[MAN.kind == "csr"]
P("Dense inputs are uniform in [0, 1), produced by NumPy's PCG64 generator with the Assignment 1 seeds, so they are "
  "byte-identical to the Assignment 1 files for the shared sizes. Sparse inputs are generated by scripts/gen_inputs.py "
  "(seed formula in the file); nonzero values are uniform in (0, 1], so no stored value is zero and no cancellation can occur. "
  "Three structures are used. <b>Uniform</b>: each entry is nonzero independently with probability d. <b>Banded</b>: every "
  "entry with |i - j| ≤ w, w chosen to match d. <b>Skewed</b>: row counts follow a 1/rank power law, ranks shuffled, rows "
  "capped at n. Sparsity is the percentage of zero entries and density the percentage of nonzeros; the achieved nonzero count "
  f"of every file is in data/manifest.csv ({len(MAN)} files with SHA-256). For n ≤ 2048, and for n = 4096 and 8192 of "
  "experiment L, the same logical matrices are also written as dense files, so dense and sparse methods see identical inputs.")

P("3.3 Experiments, environment and protocol", H2)
table([["Id", "Question", "Configurations"],
       ["T", "tile side", "blocked; n = 1024, 2048; tile 16, 32, 64, 128, 256. Rule declared in advance: lowest median at n = 2048"],
       ["D", "dense optimisations vs n", "4 dense methods; n = " + ", ".join(str(n) for n in dense_sizes)],
       ["M", "sparse vs dense, by density", "uniform; n = 1024, 2048; d = 100, 85, 70, 50, 20, 10, 5, 2, 1, 0.5, 0.2, 0.1 %. "
                                            "SpGEMM (CSR input), SpGEMM (dense input: conversion timed separately), "
                                            "dense blocked; baseline ijk at n = 1024"],
       ["S", "structure at equal density", "n = 2048, d = 5 % and 1 %; uniform, banded, skewed; dense blocked at 1 %"],
       ["L", "sizes beyond the dense budget", "uniform, 16 nonzeros per row on average; n = 4096 to 131 072; dense blocked "
                                              "attempted until a limit is hit"],
       ["X", "cross-check", "unchanged Assignment 1 binary vs baseline here; n = 512, 1024"]],
      [0.9 * cm, 3.8 * cm, 12.3 * cm], caption="Table 2. Experiments.")
P(f"Hardware: {ENV.get('cpu_model')} ({ENV.get('hypervisor')} guest), {ENV.get('physical_cores')} vCPUs, 1 thread per core, "
  f"L1d {ENV.get('l1d_cache')}, L2 {ENV.get('l2_cache')}, L3 reported as {ENV.get('l3_cache')} (a value exposed by the "
  f"hypervisor, not necessarily what this guest can use), {ENV.get('ram', '').split(':')[-1].strip()} RAM; "
  f"{ENV.get('os')}; {ENV.get('gcc')}. Every kernel is sequential; NumPy and SciPy are used only for inputs and "
  "validation and never run concurrently with a benchmark. One process runs one configuration. "
  "Timing uses clock_gettime(CLOCK_MONOTONIC) around the method only. One warm-up second was used for n ≤ 1536; "
  "for larger n no explicit warm-up was run, but one untimed calibration multiplication always precedes the samples. "
  "Ten measured samples for n ≤ 512, otherwise five (four for the baseline at n = 2048, where the 300 s sampling budget "
  "stopped it after the minimum of three). Samples shorter than 0.1 s are batched and reported per multiplication. "
  f"Budgets declared in advance: {PROTO['limit_s']:.0f} s per multiplication, enforced inside the process with a timer "
  f"signal; {PROTO['wall_limit_s'] / 60:.0f} min per configuration; 4 GiB of planned storage per process. "
  "A method that hits the time limit is not run at larger sizes. Statistic: median and inter-quartile range (IQR). "
  "Memory: peak resident set size (ru_maxrss) of the benchmark process, read with wait4() by a small wrapper, together with "
  f"the analytic storage of the arrays. The runtime baseline of the programs, measured at n = 16, is {mib_base:.1f} MiB.")

P("3.4 Correctness", H2)
P(f"scripts/run_tests.py runs {n_checks} checks ({n_ok} passed) before any benchmark. They cover a hand-computed 3×3 "
  "product, identity and zero matrices, an exact integer product, tile sizes that do not divide n (1, 7, 64, 256), CSR "
  "edge cases (empty rows, zero matrix, identity), and all three sparse structures. Every result must lie within "
  "2γ<sub>n</sub>|A||B| of NumPy, with γ<sub>n</sub> = nu/(1 - nu), u = 2<super>-53</super> [14]; the factor 2 covers "
  "the error of both our kernel and the BLAS reference. Because every method adds the products of C[i,j] in increasing k "
  "(zero products add exactly +0.0), the optimised dense methods and SpGEMM must also reproduce the baseline <b>bit for bit</b>; "
  "the number of nonzeros of C must equal SciPy's. During the benchmarks every process prints a checksum and three spot "
  "values, which the harness checks against NumPy or SciPy with the same tolerance; all completed runs passed.")

# =============================== 4 RESULTS ===============================
P("4. Results", H1)
P(f"The full data set has {n_samples} measured samples from {n_runs} processes (results/raw_measurements.csv, runs.csv). "
  "One process from an aborted first launch of the harness is kept in the files under the label T_aborted and is excluded "
  "from the summaries. It had the same configuration as the first tuning run, and its last seconds overlapped with the "
  "correctness tests of the restarted launch, which is a further reason to exclude it.")

P("4.1 Tile size (experiment T)", H2)
t2 = TUN[TUN.n == 2048].sort_values("block")
t1 = TUN[TUN.n == 1024].sort_values("block")
figure("fig3_block_tuning.png", 11.5,
       f"Figure 1. Blocked i-k-j against tile side. Median of 5 samples, IQR bars. The declared rule picked {B} "
       f"(lowest median at n = 2048).")
P(f"Tile side changed the blocked runtime by at most {100 * (t2.median_s.max() / t2.median_s.min() - 1):.0f} % at n = 2048 "
  f"({tfmt(t2.median_s.min())} to {tfmt(t2.median_s.max())}) and {100 * (t1.median_s.max() / t1.median_s.min() - 1):.0f} % "
  f"at n = 1024, with no monotonic trend: at n = 1024 tiles of 128 and 256 were fastest, at n = 2048 tile 256 was slowest. "
  f"The chosen tile {B} and the runner-up 128 differed by {100 * abs(t2[t2.block == 128].median_s.iloc[0] / t2[t2.block == B].median_s.iloc[0] - 1):.0f} %, "
  "less than the spread between runs observed elsewhere, so the choice is weakly supported and was not tuned further. "
  f"A {B}×{B} tile of each matrix is {3 * 8 * B * B / 1024:.0f} KiB, which fits in the 48 KiB L1d of one core.")

P("4.2 Dense methods (experiment D)", H2)
rows = [["n", "ijk (baseline)", "ijk_bt", "ikj", f"blocked ({B})", "blocked vs ikj", "best GFLOP/s"]]
for n in dense_sizes:
    cells = [str(n)]
    for m in ["ijk", "ijk_bt", "ikj", "blocked"]:
        st = DEN[(DEN.method == m) & (DEN.n == n)].status.iloc[0]
        if st != "ok":
            cells.append({"time_limit": "> 180 s (limit)", "not_attempted": "not attempted"}.get(st, st))
            continue
        med, iqr = dv(m, n, "median_s"), dv(m, n, "iqr_pct")
        spd = dv(m, n, "speedup_vs_ijk")
        cells.append(f"{tfmt(med)} ±{iqr:.0f}%" + (f" ({spd:.1f}×)" if m != "ijk" and spd == spd else ""))
    cells.append(xf(blk_vs_ikj[n], 2))
    cells.append(f"{DEN[DEN.n == n].gflops.max():.2f}")
    rows.append(cells)
table(rows, [1.0 * cm, 2.9 * cm, 2.9 * cm, 2.9 * cm, 2.9 * cm, 2.1 * cm, 2.0 * cm],
      caption="Table 3. Median time per multiplication ± IQR as % of the median; speed-up over the baseline in brackets. "
              "GFLOP/s = 2n³ / median time.")
figure("fig1_dense_runtime.png", 13.5, "Figure 2. Dense runtime against n (log-log). The cross marks the baseline run "
                                       "stopped by the 180 s limit at n = 3072.")
P(f"<b>The column stride is the dominant cost.</b> All three optimisations were {sp_rng['ijk_bt'][0]:.0f}× to "
  f"{sp_rng['blocked'][1]:.0f}× faster than the baseline for 1024 ≤ n ≤ 2048, but only "
  f"{dv('ijk_bt', 256, 'speedup_vs_ijk'):.1f}-{dv('ikj', 256, 'speedup_vs_ijk'):.1f}× at n = 256. The baseline itself "
  f"jumped from {tfmt(dv('ijk', 768, 'median_s'))} at n = 768 to {tfmt(dv('ijk', 1024, 'median_s'))} at n = 1024 "
  f"({dv('ijk', 1024, 'median_s') / dv('ijk', 768, 'median_s'):.0f}× the time for {(1024 / 768) ** 3:.2f}× the work), "
  f"giving a fitted exponent of {fit['ijk']:.1f} for n ≥ 512, against {fit['blocked']:.2f} for blocked. That jump at a "
  "power of two repeats the Assignment 1 observation that a row stride of 8192 bytes is especially costly for the i-j-k "
  "access to B; a cache-set-conflict explanation is plausible but was not measured with hardware counters here. "
  "H1 is supported.")
P(f"<b>Loop order versus transposition.</b> ikj was faster than ijk_bt at every size ({dv('ijk_bt', 2048, 'median_s') / dv('ikj', 2048, 'median_s'):.2f}× "
  "at n = 2048) although both read memory with unit stride and do the same arithmetic. ijk_bt also pays for the transposition, "
  "but that is O(n²) and small at these sizes. A more likely difference is that its inner loop is a reduction into one "
  "accumulator, so each addition waits for the previous one, while the ikj inner loop updates independent elements of C. "
  "This explanation is consistent with the data but was not isolated by a separate experiment.")
P(f"<b>Blocking only mattered at the largest sizes.</b> Up to n = 2048 blocked and ikj were within "
  f"{100 * (max(blk_vs_ikj[n] for n in dense_sizes if n <= 2048) - 1):.0f} % of each other, and ikj was faster at "
  f"n = 256. At n = 3072 and 4096 ikj fell to {dv('ikj', 3072, 'gflops'):.2f} and {dv('ikj', 4096, 'gflops'):.2f} GFLOP/s "
  f"while blocked stayed at {dv('blocked', 3072, 'gflops'):.2f} and {dv('blocked', 4096, 'gflops'):.2f}; blocked's cost per "
  f"multiply-add stayed between {min(ns_blk.values()):.2f} and {max(ns_blk.values()):.2f} ns over the whole range. "
  f"For every row i, ikj streams all of B, {8 * 2048 ** 2 / 2 ** 20:.0f} MiB at n = 2048 and {8 * 3072 ** 2 / 2 ** 20:.0f} MiB "
  "at n = 3072. The change of behaviour between these sizes is consistent with H2, an effective cache capacity "
  "between those two values. The hypervisor does not reveal how much of the reported 260 MiB L3 this guest can actually use, "
  f"so this is an inference. ikj at n = 3072 also had the largest IQR of the study ({dv('ikj', 3072, 'iqr_pct'):.0f} %).")
P(f"<b>Memory.</b> Peak RSS was the array storage plus the {mib_base:.1f} MiB runtime baseline for ijk, ikj and blocked "
  f"(for example {dv('blocked', 4096, 'peak_rss_mib'):.0f} MiB at n = 4096, where 3·8n² = 384 MiB). ijk_bt needs one more "
  f"n×n buffer ({dv('ijk_bt', 4096, 'peak_rss_mib'):.0f} MiB). <b>Limits:</b> the largest size within the budget was "
  f"n = {KEY['largest_ok_dense']['ijk']} for the baseline (n = 3072 hit the 180 s limit; 4096 not attempted) and "
  f"n = 4096, the largest size attempted, for the three optimised methods.")

P("4.3 Sparse versus dense on matched inputs (experiment M)", H2)
figure("fig4_sparse_vs_dense.png", 15.5,
       "Figure 3. Time per multiplication against density (log-log), uniform random sparsity, same logical matrices for "
       "every method. Dense methods ignore the zeros, so their time is flat. The dashed line adds the dense-to-CSR "
       "conversion of A and B, measured separately.")
rows = [["d (%)", "nnz(A)", "nnz(C)", "density C", "SpGEMM", "convert", "dense blocked", "speed-up", "speed-up incl. conv.",
         "ns / mult-add"]]
for d in sorted(M[M.n == 2048].density_pct.unique(), reverse=True):
    g = M[(M.n == 2048) & np.isclose(M.density_pct, d)].iloc[0]
    rows.append([f"{d:g}", f"{int(g.nnz_A):,}", f"{int(g.nnz_C):,}", f"{g.density_C_pct:.3g} %", tfmt(g.median_s),
                 tfmt(g.convert_s_from_dense_run), tfmt(g.dense_blocked_s), xf(g.speedup_vs_dense_blocked, 2 if g.speedup_vs_dense_blocked < 10 else 0),
                 xf(g.speedup_incl_convert, 2 if g.speedup_incl_convert < 10 else 0), f"{g.ns_per_mult_add:.2f}"])
table(rows, [1.0 * cm, 1.75 * cm, 1.75 * cm, 1.55 * cm, 1.6 * cm, 1.65 * cm, 1.75 * cm, 1.65 * cm, 2.0 * cm, 1.6 * cm],
      caption="Table 4. n = 2048, uniform. Medians of 5 samples (IQRs in results/summary_sparse.csv). Speed-up = dense blocked "
              "time / SpGEMM time; above 1 means sparse is faster. ns per mult-add = SpGEMM time divided by its actual "
              "scalar multiply-adds (Σ over nonzeros a_ik of nnz(B[k,:])).")
P(f"<b>Where sparse helps.</b> At both sizes SpGEMM was faster than dense blocked at every density from 0.1 % to "
  f"{lo_win[2048]:g} % and slower at {hi_lose[2048]:g} % and above (n = 1024 showed the same pattern). Its time fell with "
  "density over more than five orders of magnitude, while the dense methods, which do not look at the values, stayed flat. "
  f"That the break-even lies between 50 % and 70 % follows from the work model: SpGEMM performs about d²n³ multiply-adds "
  f"at about {c_sp_dense:.1f} ns each in the dense-like regime (d ≥ 50 %), against n³ at about {c_dense:.2f} ns for "
  f"dense blocked, which predicts a break-even near d = √({c_dense:.2f}/{c_sp_dense:.2f}) ≈ {d_star:.0f} %. "
  "This is a consistency check on the measured costs, not an independent prediction.")
P("<b>Where its costs reduce the advantage.</b> (i) <i>Indexing.</i> The cost per useful multiply-add rose from about 1.6 ns "
  f"at high density to {M[M.n == 2048].ns_per_mult_add.max():.0f} ns at 0.1 % (Table 4), because at low density per-row and "
  "per-nonzero bookkeeping (row pointers, the accumulator marker, writing output indices) is no longer amortised over many "
  "products, and each product needs an indirect access to the accumulator. "
  f"(ii) <i>Conversion.</i> Converting A and B from dense to CSR took {tfmt(M[M.n == 2048].convert_s_from_dense_run.min())} to "
  f"{tfmt(M[M.n == 2048].convert_s_from_dense_run.max())} at n = 2048, because it must scan all 2n² entries whatever the density. "
  f"At 0.1 % it was about {sv('M', 2048, 0.1, 'convert_s_from_dense_run') / sv('M', 2048, 0.1, 'median_s'):.0f}× "
  f"the multiply itself, and it cut the speed-up from {sv('M', 2048, 0.1, 'speedup_vs_dense_blocked'):,.0f}× to "
  f"{sv('M', 2048, 0.1, 'speedup_incl_convert'):,.0f}×. Conversion is therefore worth paying only when the matrices "
  "already exist in sparse form or are reused. (iii) <i>Fill-in.</i> The product was much denser than its factors (Figure 4): "
  f"at n = 2048, inputs with 1 % nonzeros gave {sv('M', 2048, 1, 'density_C_pct'):.1f} % nonzeros in C and inputs with 5 % "
  f"gave {sv('M', 2048, 5, 'density_C_pct'):.1f} %, matching 1 - (1 - d²)<super>n</super> to within "
  f"{100 * (M.density_C_pct / M.pred_density_C_pct - 1).abs().max():.0f} %. The output then costs 12 bytes per entry "
  "in CSR instead of 8 in dense storage.")
figure("fig5_fill_in.png", 11.0, "Figure 4. Density of the product against density of the factors, observed (solid) and "
                                 "the independent-pattern model 1 - (1 - d²)<super>n</super> (dotted, almost hidden by the solid line).")
P(f"<b>Memory.</b> At n = 2048 SpGEMM's peak RSS was {sv('M', 2048, 0.1, 'peak_rss_mib'):.1f} MiB at 0.1 % and "
  f"{sv('M', 2048, 1, 'peak_rss_mib'):.1f} MiB at 1 %, against {dv('blocked', 2048, 'peak_rss_mib'):.1f} MiB for any dense "
  f"method. The two met near 50 % ({sv('M', 2048, 50, 'peak_rss_mib'):.1f} MiB), and at 100 % the sparse run needed "
  f"{sv('M', 2048, 100, 'peak_rss_mib'):.1f} MiB, more than dense (Figure 5). Above about 2 % density, analytic CSR storage "
  "of A, B and C tracks the RSS closely; below it the fixed costs dominate (program baseline, row pointers, the 20n-byte "
  "accumulator). The memory advantage disappears together with the time advantage, at about the same density.")
figure("fig6_memory.png", 11.0, "Figure 5. Peak RSS (solid) and analytic storage (dotted) at n = 2048.")

P("4.4 Structure at equal density (experiment S)", H2)
Srows = [["structure", "d (%)", "nnz(A)", "mult-adds", "nnz(C)", "density C", "SpGEMM", "ns / mult-add", "peak RSS"]]
S = SPA[(SPA.experiment == "S") & (SPA.method == "csr_spgemm")]
for d in (5, 1):
    for s_ in ("uniform", "banded", "skewed"):
        g = S[(S.structure == s_) & np.isclose(S.density_pct, d)].iloc[0]
        Srows.append([s_, f"{d}", f"{int(g.nnz_A):,}", f"{int(g.mult_adds):,}", f"{int(g.nnz_C):,}",
                      f"{g.density_C_pct:.3g} %", tfmt(g.median_s), f"{g.ns_per_mult_add:.2f}", f"{g.peak_rss_mib:.1f} MiB"])
table(Srows, [1.7 * cm, 1.0 * cm, 1.8 * cm, 2.2 * cm, 1.9 * cm, 1.7 * cm, 1.7 * cm, 1.9 * cm, 1.9 * cm],
      caption="Table 5. n = 2048, CSR input, median of 5 samples. Dense blocked on the 1 % matrices: "
              + ", ".join(f"{s_} {tfmt(float(SPA[(SPA.experiment == 'S') & (SPA.method == 'blocked') & (SPA.structure == s_)].median_s.iloc[0]))}"
                          for s_ in ('uniform', 'banded', 'skewed')) + ".")
figure("fig7_structure.png", 14.5, "Figure 6. SpGEMM time and output density for three structures with the same input density.")
P(f"At 5 % density the banded product took {tfmt(sv('S', 2048, 5, 'median_s', structure='banded'))} and the skewed one "
  f"{tfmt(sv('S', 2048, 5, 'median_s', structure='skewed'))}, against {tfmt(sv('S', 2048, 5, 'median_s'))} for uniform. "
  "Two separate effects explain this. The amount of work and fill-in differ: the band of the product is only twice as wide "
  f"as the band of the factors, so banded C had {sv('S', 2048, 5, 'density_C_pct', structure='banded'):.1f} % nonzeros where "
  f"uniform C had {sv('S', 2048, 5, 'density_C_pct'):.1f} %. And the cost per multiply-add differs "
  f"({sv('S', 2048, 5, 'ns_per_mult_add', structure='banded'):.2f} ns banded vs {sv('S', 2048, 5, 'ns_per_mult_add'):.2f} ns "
  "uniform). Neighbouring rows of a banded A select overlapping, neighbouring rows of B and a contiguous range of the "
  "accumulator, which is plausibly cache-friendly, although this was not measured directly. The skewed matrices concentrate "
  "nonzeros in a few long rows, which produced fewer multiply-adds and fewer output entries than uniform at the same density. "
  "The dense method was insensitive to structure (all three within "
  f"{100 * (SPA[(SPA.experiment == 'S') & (SPA.method == 'blocked')].median_s.max() / SPA[(SPA.experiment == 'S') & (SPA.method == 'blocked')].median_s.min() - 1):.0f} %). "
  "Density alone therefore did not determine sparse performance in these tests.")

P("4.5 Sizes beyond the dense budget (experiment L)", H2)
figure("fig8_large_sparse.png", 13.0, "Figure 7. Uniform matrices with 16 nonzeros per row on average, so density falls as n grows.")
Lrows = [["n", "density (%)", "nnz(C)", "SpGEMM", "ns / mult-add", "peak RSS", "dense 3·8n²", "dense blocked"]]
for _, g in L.iterrows():
    db = SPA[(SPA.experiment == "L") & (SPA.method == "blocked") & (SPA.n == g.n)].iloc[0]
    dstat = tfmt(db.median_s) if db.status == "ok" else {"time_limit": "> 180 s (limit)",
                                                            "not_attempted": "not attempted"}.get(db.status, db.status)
    Lrows.append([f"{int(g.n):,}", f"{g.density_pct:.3g}", f"{int(g.nnz_C):,}", tfmt(g.median_s), f"{g.ns_per_mult_add:.1f}",
                  f"{g.peak_rss_mib:.0f} MiB", f"{3 * 8 * g.n ** 2 / 2 ** 30:,.1f} GiB", dstat])
table(Lrows, [1.6 * cm, 1.8 * cm, 2.0 * cm, 1.8 * cm, 2.0 * cm, 1.8 * cm, 2.2 * cm, 2.6 * cm],
      caption="Table 6. Experiment L. Dense blocked hit the 180 s limit at n = 8192 and was not attempted above. "
              f"From n = 16 384 dense storage alone would also exceed the 4 GiB budget ({gib16384:.0f} GiB).")
P(f"With a fixed number of nonzeros per row, the work grows only linearly in n (about 256n multiply-adds), yet the measured "
  f"time grew as n<super>{L_exp:.2f}</super>. The cost per multiply-add rose from {L.ns_per_mult_add.iloc[0]:.1f} to "
  f"{L.ns_per_mult_add.iloc[-1]:.1f} ns, which is consistent with irregular accesses to B rows and to an accumulator that "
  f"grows with n ({20 * 131072 / 2 ** 20:.1f} MiB at n = 131 072) and stops fitting in the faster caches. "
  f"The largest size tested, n = 131 072, completed in {tfmt(L.median_s.iloc[-1])} with {L.peak_rss_mib.iloc[-1]:.0f} MiB. "
  "That was the largest size declared, not a limit reached: no sparse configuration hit a budget. "
  f"Dense storage at that size would need {3 * 8 * 131072 ** 2 / 2 ** 30:,.0f} GiB.")

P("4.6 Cross-check with the Assignment 1 binary (experiment X)", H2)
P(f"The unchanged Assignment 1 binary took {tfmt(xc.get('a1_matmul_O2@1024', float('nan')))} at n = 1024 and the baseline "
  f"here took {tfmt(xc.get('ijk@1024', float('nan')))}, so adding restrict did not change the baseline at that size. "
  f"At n = 512 the two differed by {100 * abs(xc.get('a1_matmul_O2@512', 1) / xc.get('ijk@512', 1) - 1):.0f} %. "
  f"The Assignment 1 report measured {4.55:.2f} s at n = 1024 for the same binary and inputs on a different VM instance. "
  f"Across the {len(ijk1024)} separate processes that ran the baseline at n = 1024 in this study, the median ranged from "
  f"{tfmt(ijk1024.min())} to {tfmt(ijk1024.max())}. That spread between processes ({100 * (ijk1024.max() / ijk1024.min() - 1):.0f} %) "
  f"is larger than the within-run IQRs, so the baseline speed-ups should be read with about ±{50 * (ijk1024.max() / ijk1024.min() - 1):.0f} % uncertainty, and "
  "absolute times should not be compared with Assignment 1.")

# =============================== 5 DISCUSSION ===============================
P("5. Discussion", H1)
P(f"<b>Answer to question A.</b> On this machine the decisive optimisation was removing the stride-n access to B. Any of "
  f"the three methods that do so gave an order-of-magnitude speed-up once n ≥ 1024. Beyond that, loop reordering beat "
  f"transposition by about {100 * (dv('ijk_bt', 2048, 'median_s') / dv('ikj', 2048, 'median_s') - 1):.0f} %, and blocking "
  f"added a further {100 * (blk_vs_ikj[4096] - 1):.0f} % only when B exceeded the data that apparently stayed cached "
  f"(n ≥ 3072). All gains here came from memory behaviour: the arithmetic and its order are identical in all four methods "
  "(the outputs are bit-identical) and the code is scalar. Even the best method reached only about 3.5-3.9 GFLOP/s, a "
  "small fraction of what SIMD, register blocking and multiple threads could deliver [6]; those belong to Assignment 3.")
P(f"<b>Answer to question B.</b> For uniform random patterns, CSR SpGEMM was faster than the tuned dense method at all "
  f"densities up to {lo_win[2048]:g} % and slower from {hi_lose[2048]:g} % at both tested sizes. Its advantage grew steeply "
  "as density fell because the work falls with d². That advantage is reduced, and at high density reversed, by "
  "(i) a cost per operation between about 3× and 25× that of dense blocked, (ii) the conversion from dense storage, "
  "which scans all n² entries and dominated below about 1 %, and (iii) fill-in, which makes C nearly dense at moderate "
  "input densities and removes the memory advantage. Structure mattered as much as density: at the same 5 % density the "
  "banded inputs ran about 3.5× faster and produced a ten times sparser output than the uniform ones. "
  "Optimisation effects (Part A) and storage effects (Part B) are separate here: the dense comparison used the best tested "
  "dense method, and the sparse kernel received no tuning.")
P("<b>Bottlenecks observed.</b> Baseline: column-stride memory access, worst at power-of-two n. ikj: streaming the whole of B "
  "for every row once B no longer fits in cache. ijk_bt: probably the serial dependence of the dot product. SpGEMM: indirect "
  "access to the accumulator and per-row overheads at low density, conversion cost when inputs start dense, and growing "
  "cache misses as n grows at fixed nonzeros per row.")
P("<b>Limitations.</b> (1) One 2-vCPU cloud VM, whose cache hierarchy is only partly visible; the run-to-run spread of the "
  "baseline was larger than the within-run IQR, and the results do not transfer directly to other hardware such as the "
  "author's laptop. (2) No hardware performance counters were available (perf is not installed), so the cache explanations "
  "are inferences from timing and data sizes, labelled as such. (3) The sparse inputs are synthetic. A real matrix from the "
  "SuiteSparse collection [15] would add realism, but the collection could not be downloaded from the benchmark machine; "
  "the CSR format used here would accept it after conversion. (4) Only one sparse algorithm and one format were studied, "
  "with unsorted output columns, and SciPy was used only for checking, not as a timed competitor. (5) The tile size was "
  "tuned once on two sizes, and differences between tiles were close to the noise. (6) Scalar code only, -O2, no -march.")

# =============================== 6 CONCLUSIONS ===============================
P("6. Conclusions", H1)
P(f"Within the tested scope (C, gcc -O2, float64, one thread, one cloud VM, synthetic inputs), the study establishes four "
  f"points. Making the inner loop read B with unit stride turned a baseline that ran at about 0.2 GFLOP/s for n ≥ 1024 "
  f"into methods running at 2.6-3.6 GFLOP/s, and it moved the largest size within the 180 s limit from 2048 to at least 4096. "
  f"Cache blocking was only needed for the largest sizes tested, where it kept the rate near 3.4-3.8 GFLOP/s while plain "
  f"loop reordering dropped to about 2. CSR SpGEMM beat the best dense method for uniform inputs up to "
  f"{lo_win[2048]:g} % density, but conversion, fill-in and structure changed the size of that advantage by orders of "
  f"magnitude, so the share of zeros alone is not a sufficient predictor. CSR alone made n = 131 072 feasible in about "
  "one second and under 0.5 GiB. Still open: the exact break-even density between 50 % and 70 %; hardware-counter evidence for "
  "the cache explanations; real-world matrices; and how these results change with vectorisation and threads, which "
  "Assignment 3 will address.")

# =============================== 7 AI ===============================
P("7. Use of generative AI", H1)
P("In line with the course policy, the AI assistant Claude (Anthropic, claude.ai) was used to plan the experiments, draft "
  "the C implementations, the harness and analysis scripts, the README and the first draft of this report. All numerical "
  "results come from measurements stored in results/ and produced by the committed code; the AI output is not used as "
  "evidence. The author remains responsible for understanding the code, validating the measurements, verifying the "
  "references and defending the conclusions.")

# =============================== REFERENCES ===============================
P("References", H1)
refs = [
    "M. D. Afonso Suárez, J. J. Hernández Cabrera and J. Évora Gómez, “Big Data,” course materials, Universidad de Las Palmas "
    "de Gran Canaria, 2026/27. [Online]. Available: https://mltechdrawer.github.io/BIGDATA/",
    "“Individual Assignments: Matrix Multiplication,” Big Data, Grado en Ciencia e Ingeniería de Datos, ULPGC, revised proposal 2026-2027.",
    "J. L. Hennessy and D. A. Patterson, <i>Computer Architecture: A Quantitative Approach</i>, 6th ed. Cambridge, MA, USA: Morgan Kaufmann, 2017.",
    "U. Drepper, “What every programmer should know about memory,” Red Hat, Inc., Nov. 2007.",
    "M. S. Lam, E. E. Rothberg and M. E. Wolf, “The cache performance and optimizations of blocked algorithms,” in <i>Proc. ASPLOS IV</i>, 1991, pp. 63-74.",
    "K. Goto and R. A. van de Geijn, “Anatomy of high-performance matrix multiplication,” <i>ACM Trans. Math. Softw.</i>, vol. 34, no. 3, art. 12, 2008.",
    "V. Strassen, “Gaussian elimination is not optimal,” <i>Numerische Mathematik</i>, vol. 13, pp. 354-356, 1969.",
    "Y. Saad, <i>Iterative Methods for Sparse Linear Systems</i>, 2nd ed. Philadelphia, PA, USA: SIAM, 2003.",
    "T. A. Davis, <i>Direct Methods for Sparse Linear Systems</i>. Philadelphia, PA, USA: SIAM, 2006.",
    "F. G. Gustavson, “Two fast algorithms for sparse matrices: Multiplication and permuted transposition,” <i>ACM Trans. Math. Softw.</i>, vol. 4, no. 3, pp. 250-269, 1978.",
    "R. E. Bank and C. C. Douglas, “Sparse matrix multiplication package (SMMP),” <i>Advances in Computational Mathematics</i>, vol. 1, pp. 127-137, 1993.",
    "P. Virtanen <i>et al.</i>, “SciPy 1.0: Fundamental algorithms for scientific computing in Python,” <i>Nature Methods</i>, vol. 17, pp. 261-272, 2020.",
    "“GCC 12 Release Series: Changes, New Features, and Fixes” (vectorisation enabled at -O2), Free Software Foundation. [Online]. Available: https://gcc.gnu.org/gcc-12/changes.html",
    "N. J. Higham, <i>Accuracy and Stability of Numerical Algorithms</i>, 2nd ed. Philadelphia, PA, USA: SIAM, 2002.",
    "T. A. Davis and Y. Hu, “The University of Florida sparse matrix collection,” <i>ACM Trans. Math. Softw.</i>, vol. 38, no. 1, art. 1, 2011.",
    "H. Amador Hidalgo, “Basic matrix multiplication in Python, Java and C,” Big Data, Individual Assignment 1, ULPGC, 2026.",
]
for i, r in enumerate(refs, 1):
    P(f"[{i}] {r}", REF)
P("Reproducibility materials", H1)
P(f"Repository: {args.repo_url}, folder Assignment02_AmadorHidalgo. It contains src/ (dense.c, sparse.c, harness.h), build.sh, "
  "a README with exact commands, requirements.txt, scripts/ (input generator, tests, benchmark harness, analysis, this report), "
  "data/manifest.csv (seeds, nonzero counts and SHA-256 of every input), and results/ (raw samples, one row per process, "
  "summaries, key numbers, environment, correctness log and figures). The inputs (about 3 GB) are regenerated with "
  "scripts/gen_inputs.py. The baseline kernel is reused from Assignment01_AmadorHidalgo/src/c/matmul.c in the same repository [16].")


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(colors.HexColor("#52514e"))
    canvas.drawString(2 * cm, 1.1 * cm, "Hugo Amador Hidalgo - Big Data - Assignment 2 - Optimised and sparse matrix multiplication")
    canvas.drawRightString(A4[0] - 2 * cm, 1.1 * cm, f"{doc.page}")
    canvas.restoreState()


doc = SimpleDocTemplate(str(OUT), pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm, topMargin=1.8 * cm, bottomMargin=1.8 * cm,
                        title="Assignment 2 - Optimised dense and sparse matrix multiplication",
                        author="Hugo Amador Hidalgo", subject="Big Data 2026-2027, ULPGC")
doc.build(story, onFirstPage=footer, onLaterPages=footer)
print("report written:", OUT)
