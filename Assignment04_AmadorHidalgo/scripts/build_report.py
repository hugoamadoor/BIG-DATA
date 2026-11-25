#!/usr/bin/env python3
"""Build report/Assignment04_AmadorHidalgo_Report.pdf from results/ (run analyze.py first).

Numbers in tables and sentences that quote measurements are read from the CSV/JSON files. The
interpretive sentences were written for the published data set.

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
OUT = ROOT / "report" / "Assignment04_AmadorHidalgo_Report.pdf"
OUT.parent.mkdir(exist_ok=True)

S = pd.read_csv(RES / "summary_spark.csv")
REF = pd.read_csv(RES / "summary_references.csv")
ST = pd.read_csv(RES / "summary_startup.csv").set_index("config")
RUNS = pd.read_csv(RES / "spark_runs.csv")
TASKS = pd.read_csv(RES / "spark_tasks.csv")
NOTRUN = pd.read_csv(RES / "not_run.csv")
ENV = json.loads((RES / "environment.json").read_text())
CORR = pd.read_csv(RES / "correctness.csv")
CONF = json.loads((RES / "spark_config_2x1.json").read_text())

D = S[S.experiment == "D"]
RB = D[D.strategy == "rowblock"]
B2 = D[D.strategy == "block2d"]
G = S[(S.config == "2x1") & (S.strategy == "rowblock") & (S.n == 2048) & S.experiment.isin(["D", "G"])].sort_values("parts")
sizes = sorted(D.n.unique())
nmax = max(sizes)
T = {(r.method, r.n): r.median_s for r in REF.itertuples()}


def tfmt(s):
    if s != s:
        return "-"
    return f"{s * 1e3:.3g} ms" if s < 1 else f"{s:.3g} s"


def rb(c, n, col):
    r = RB[(RB.config == c) & (RB.n == n)]
    return float(r[col].iloc[0]) if len(r) else float("nan")


def b2(c, n, col):
    r = B2[(B2.config == c) & (B2.n == n)]
    return float(r[col].iloc[0]) if len(r) else float("nan")


def mib(b):
    return f"{b / 2 ** 20:.0f} MiB"


# ---------- styles ----------
ss = getSampleStyleSheet()
BODY = ParagraphStyle("body", parent=ss["BodyText"], fontName="Helvetica", fontSize=9.4, leading=12.6,
                      alignment=TA_JUSTIFY, spaceAfter=5)
H1 = ParagraphStyle("h1", parent=ss["Heading1"], fontName="Helvetica-Bold", fontSize=13, leading=16,
                    spaceBefore=10, spaceAfter=5, textColor=colors.HexColor("#123f7a"))
H2 = ParagraphStyle("h2", parent=ss["Heading2"], fontName="Helvetica-Bold", fontSize=10.5, leading=13,
                    spaceBefore=6, spaceAfter=3, textColor=colors.HexColor("#0b0b0b"))
TITLE = ParagraphStyle("title", parent=ss["Title"], fontName="Helvetica-Bold", fontSize=17, leading=21, spaceAfter=4, alignment=0)
SUB = ParagraphStyle("sub", parent=BODY, fontSize=10, textColor=colors.HexColor("#52514e"), alignment=0, spaceAfter=2)
BOX = ParagraphStyle("box", parent=BODY, backColor=colors.HexColor("#fff4e5"), borderColor=colors.HexColor("#eb6834"),
                     borderWidth=0.8, borderPadding=6, spaceBefore=6, spaceAfter=10)
CAP = ParagraphStyle("cap", parent=BODY, fontSize=8.3, leading=10.5, textColor=colors.HexColor("#52514e"), alignment=0, spaceAfter=8)
CELL = ParagraphStyle("cell", parent=BODY, fontSize=8, leading=9.6, alignment=0, spaceAfter=0)
CELLB = ParagraphStyle("cellb", parent=CELL, fontName="Helvetica-Bold")
REFS = ParagraphStyle("ref", parent=BODY, fontSize=8.4, leading=10.6, alignment=0, leftIndent=16, firstLineIndent=-16, spaceAfter=2.5)
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
    story.append(KeepTogether([Image(str(FIG / name), width=width_cm * cm, height=width_cm * cm * h / w),
                               Paragraph(caption, CAP)]))


# ---------- numbers ----------
n_checks, n_ok = len(CORR), int(CORR.passed.sum())
meas = RUNS[RUNS.rep > 0]
n_mult, n_warm = len(meas), int((RUNS.rep == 0).sum())
strong = {c: rb(c, nmax, "strong_speedup_vs_1x1") for c in ["1x2", "2x1"]}
vs_seq = {c: rb(c, nmax, "speedup_vs_seq") for c in ["1x1", "1x2", "2x1"]}
vs_omp = {c: rb(c, nmax, "speedup_vs_omp2") for c in ["1x2", "2x1"]}
ov1 = 100 * (rb("1x1", nmax, "end_to_end_s") / T[("seq", nmax)] - 1)
shuffle_ratio = b2("2x1", nmax, "shuffle_read_bytes") / (8 * nmax ** 2)
spark_ver = ENV.get("spark_version", "4.0.1")
import re as _re
_jv = _re.search(r'version "([0-9.]+)', " ".join(ENV.get("java", [])))
java = f"OpenJDK {_jv.group(1)}" if _jv else "OpenJDK 21"
st_lo = ST.loc["1x1", "cluster_start_s"] + ST.loc["1x1", "app_start_s"]
st_hi = ST.loc["4x1", "cluster_start_s"] + ST.loc["4x1", "app_start_s"]

# =============================== TITLE ===============================
P("Distributed Matrix Multiplication with Apache Spark: Partitioning, Overhead and Strong Scaling "
  "in a Local Multi-Worker Cluster", TITLE)
P("Big Data - Individual Assignment 4 (Distributed matrix multiplication)", SUB)
P("Hugo Amador Hidalgo - Grado en Ciencia e Ingeniería de Datos, ULPGC - Academic year 2026-2027", SUB)
P(f"Repository: {args.repo_url} (folder Assignment04_AmadorHidalgo)", SUB)
P("<b>Environment label.</b> LOCAL multi-worker execution: every Spark process (master, workers, executors, "
  "driver) ran on ONE physical machine, a 2-vCPU cloud VM, and communicated over the loopback interface. "
  "The study measures framework and process overhead and the scaling of a Spark job over cores of one machine. "
  "It does not measure inter-machine network performance or the memory capacity of several machines. The "
  "assignment allows local execution only with the instructor's permission.", BOX)

P("Abstract", H1)
P(f"This study implements C = AB on Apache Spark {spark_ver} (RDD API, PySpark) and measures how it scales with "
  "workers and where its time goes. The main strategy splits A into row blocks, broadcasts B to every executor and "
  "collects C on the driver; a 2-D block strategy with shuffles is the comparison. Every task computes with the same "
  "blocked C kernel as the sequential and OpenMP versions of Assignment 3, and the row-block result is bit-identical "
  "to the sequential one, so the comparison isolates distribution and framework costs. Four cluster configurations "
  "(1, 2 and 4 single-core workers, and one two-core worker) ran n = 1024, 2048 and 3072 in two rounds. "
  f"At n = {nmax} two single-core workers were {strong['2x1']:.2f}× faster than one and "
  f"{vs_seq['2x1']:.2f}× faster than the sequential program, but {1 / vs_omp['2x1']:.2f}× slower than "
  f"OpenMP with two threads; with one worker Spark was {ov1:.0f} % slower than the sequential program. At n = 1024 no "
  "configuration beat the sequential program, because about half a second per multiplication went to scheduling, "
  "moving data and Python worker overhead, and "
  f"{st_lo:.0f}-{st_hi:.0f} s of cluster and application start-up came on top, outside the end-to-end boundary. Four workers on two cores were slower than two. The shuffle strategy moved "
  f"{shuffle_ratio:.1f} times the size of a matrix through the shuffle and was slower than row blocks in every "
  "configuration. Memory, not time, set the size limit: each executor and each Python worker held its own copy of B, "
  "and runs above n = 3072 (and with four executors above 2048) were stopped by the 6 GiB memory limit.")

# =============================== 1 INTRO ===============================
P("1. Introduction and objective", H1)
P("Assignments 1-3 multiplied matrices inside one process. Here the multiplication is split into tasks that a "
  "distributed framework schedules on worker processes. Apache Spark was chosen because it is part of the course "
  "material on distributed processing [2] and because its monitoring interface reports per-stage task metrics "
  "(run time, deserialisation, garbage collection, result and shuffle bytes) [3].")
P("<b>Research question.</b> With the total problem fixed (strong scaling), how do end-to-end time, its components "
  "and memory change as Spark workers and cores are added, and how does the distributed version compare with the "
  "validated basic sequential, blocked sequential and OpenMP versions on the same machine? "
  "<b>Hypotheses.</b> H1: because each task runs the same kernel, Spark with one worker costs the sequential time plus a "
  "roughly constant per-job overhead, so the relative overhead falls as n grows. H2: two workers approach a speed-up of 2 "
  "only for large n; more workers than cores do not help. H3: replicating B (broadcast) costs memory proportional to "
  "the number of executors, and moving blocks through a shuffle costs more than broadcasting B once.")

# =============================== 2 BACKGROUND ===============================
P("2. Background", H1)
P("<b>Spark execution model.</b> A Spark application has one driver and a set of executors started by worker daemons. "
  "An action (here collect) becomes a job, split into stages at shuffle boundaries; each stage runs one task per "
  "partition, and an executor with c cores runs c tasks at a time [4], [5]. In PySpark the task's Python code runs in a "
  "separate Python worker process that the executor starts and reuses between tasks. Data move in four ways: inside "
  "the serialised task (sc.parallelize), through broadcast variables (BitTorrent-like distribution in 4 MiB pieces "
  "from the driver and between executors), through shuffles (written by map tasks, fetched by reduce tasks) and back "
  "to the driver as task results [3]. "
  "<b>Distributed matrix multiplication.</b> 1-D decompositions replicate one operand; 2-D decompositions such as "
  "SUMMA [6] and the block multiplication of Spark's MLlib BlockMatrix [7] replicate blocks along rows and columns of a "
  "grid. For dense n×n matrices the work is 2n³ flops while the data are 3·8n² bytes, so the ratio of "
  "computation to communication grows linearly with n. "
  "<b>Strong and weak scaling.</b> Strong scaling keeps n fixed while resources grow [8]. Weak scaling would have to grow "
  "the work, n³, in proportion to the resources (n<sub>W</sub> = n<sub>1</sub>W<super>1/3</super>); with only two "
  "cores it was not attempted.")

# =============================== 3 METHOD ===============================
P("3. Methodology", H1)
P("3.1 Implementation", H2)
table([["", "rowblock (main)", "block2d (comparison)"],
       ["Partitioning", "A: P = 8 row blocks, boundaries at multiples of the 32-row tile; one block per task. "
                        "B: whole, broadcast.", "A and B: 2×2 grid of square blocks. Block A[I,K] is sent to products (I,K,J) "
                                                "and B[K,J] to (I,K,J) for every I, J; a join pairs them."],
       ["Placement of computation", "task p computes C[rows p, :] = A[rows p, :] B with mm_rows on the executor that runs it",
        "each of the 8 joined pairs is multiplied with mm_acc; reduceByKey((I,J)) adds the 2 partial products of each C block"],
       ["Collection of the result", "collect() returns the 8 row blocks to the driver, which copies them into C",
        "collect() returns the 4 C blocks to the driver"],
       ["Stages / shuffles", "1 stage, no shuffle", "3 stages, 2 shuffles (join, reduceByKey)"],
       ["Result", "bit-identical to the sequential baseline", "within rounding (different summation order)"]],
      [3.0 * cm, 7.0 * cm, 7.0 * cm], caption="Table 1. The two strategies (src/spark/spark_matmul.py).")
P("The kernel inside every task is the blocked i-k-j kernel of Assignments 2-3 (32×32 tiles), compiled from "
  "src/kernel/mmkernel.c with the Assignment 3 flags (gcc -O2 -ffp-contract=off -falign-loops=32, scalar code) and called "
  "through ctypes on NumPy arrays without copying. NumPy only holds and slices the matrices; no BLAS call is made in a "
  "task, and OPENBLAS_NUM_THREADS = 1 everywhere. Tasks record wall-clock timestamps and the time spent inside the kernel. "
  "The references are the unchanged Assignment 2 and 3 programs (src/reference/): the basic triple loop (ijk), the "
  "blocked sequential baseline (seq) and the OpenMP version with two threads (omp).")

P("3.2 Cluster configurations and framework settings", H2)
table([["Config", "Worker daemons (JVM)", "Executors (JVM)", "Task slots", "Python workers", "Physical machines", "Note"],
       ["1x1", "1", "1", "1", "≥ 1", "1", ""],
       ["2x1", "2", "2", "2", "≥ 2", "1", "uses both vCPUs"],
       ["4x1", "4", "4", "4", "≥ 4", "1", "oversubscribed: 4 slots on 2 vCPUs"],
       ["1x2", "1", "1", "2", "≥ 2", "1", "two task slots in one executor"]],
      [1.3 * cm, 2.6 * cm, 2.2 * cm, 1.8 * cm, 2.2 * cm, 2.3 * cm, 4.6 * cm],
      caption="Table 2. Configurations. Every configuration also has one master JVM (control only) and the driver "
              "(Python process + JVM). A Python worker can be restarted by its executor, so more than one may appear per slot.")
P(f"Spark standalone mode, version {spark_ver} as bundled with PySpark; {java}; Python 3.13. Settings in every run: "
  f"spark.executor.cores = cores per worker, spark.executor.memory = {CONF['spark.executor.memory']} per core, "
  f"spark.driver.memory = {CONF['spark.driver.memory']}, spark.python.worker.reuse = true, spark.task.maxFailures = 1, "
  "FIFO scheduling, default broadcast block size (4 MiB) and serialiser; full listings in results/spark_config_*.json. "
  f"Hardware: {ENV['cpu_model']} (KVM guest), 2 vCPUs, 1 thread per core, L2 {ENV['l2_cache']}, "
  f"{int(ENV['ram'].split()[1]) / 2 ** 20:.1f} GiB RAM, of which the benchmark could use 6 GiB (container memory limit); "
  f"{ENV['os']}.")

P("3.3 Protocol and measurement boundary", H2)
P("<b>End-to-end boundary:</b> from A and B as NumPy arrays in the driver to C as a NumPy array in the driver; it is the "
  "sum of three steps that run one after another on the driver: broadcast (B serialised and stored by the driver), "
  "job (the Spark action: scheduling, shipping A blocks inside the tasks, executors fetching B, kernels, returning C "
  "blocks) and assemble (copying C blocks into C). Reading the input files, starting the cluster and starting the "
  "application are outside the boundary and reported separately. Inside the job, tasks overlap, so components are "
  "not added up; instead the <b>kernel critical path</b> is reported: the largest kernel time of one task slot "
  "(sum of the kernel times of the tasks an executor ran, divided by its cores). The rest of the job, job minus kernel "
  "critical path, is time on the critical path not spent computing. Both come from the same measurements and add up to "
  "the job time by construction.")
P(f"Every (configuration, n) pair ran in a fresh cluster and application. After start-up, one warm-up multiplication "
  "(reported separately) was followed by 5 measured multiplications (3 at n = 3072). The whole set ran twice; the "
  "second round used the reverse order of configurations. References ran in the same session on the same inputs: "
  "seq and omp in both rounds, ijk (about 70 s at n = 2048) in the first round only. Statistic: median and IQR of the "
  "pooled measured multiplications; round spread = range of the two round medians. Stage metrics come from Spark's "
  "REST monitoring API [3]; memory is the peak resident set size of each executor process tree (executor JVM and its "
  "Python workers) and of the driver (Python process and its JVM), sampled every 0.1 s with psutil. Between "
  "multiplications, outside the timed region, the harness requests garbage collection in Python and in the driver JVM.")
P(f"<b>Memory budget.</b> The benchmark ran under a 6 GiB memory limit. Preliminary attempts at n = 4096, and with 4x1 "
  "at n = 3072, were stopped by the kernel's OOM killer (results/oom_events.txt). The largest size is therefore 3072, "
  f"and 4x1 ran up to 2048 ({len(NOTRUN)} configurations recorded in not_run.csv). This limit is itself a result "
  "(Section 4.5).")
P("3.4 Correctness", H2)
P(f"scripts/run_tests.py ran {n_checks} checks ({n_ok} passed) before the benchmarks: known answers (hand-computed 3×3, "
  "identity, zero) for the kernel; the kernel on whole matrices bit-identical to the Assignment 3 sequential program; the "
  "reference programs within 2γ<sub>n</sub>|A||B| of NumPy [9]; and on a running 2-worker cluster, rowblock with "
  "P = 1, 3, 8 and 64 (more tasks than row tiles) bit-identical to the sequential kernel and block2d with g = 1, 2, 3 within "
  "the bound, for n = 1 to 256. During the benchmarks every rowblock result was compared by SHA-256 with the C written by "
  "the sequential program in the same round, and every block2d result element by element with the bound; "
  f"all {len(RUNS)} Spark multiplications passed.")

# =============================== 4 RESULTS ===============================
P("4. Results", H1)
P(f"{n_mult} measured multiplications and {n_warm} warm-up multiplications in {len(RUNS.groupby(['round', 'config', 'n']))} "
  "cluster sessions (results/spark_runs.csv, spark_tasks.csv, reference_raw.csv).")
P("4.1 Distributed against single-process versions", H2)
rows = [["n", "basic ijk", "blocked seq", "OpenMP 2 thr.", "Spark 1x1", "Spark 2x1", "Spark 1x2", "Spark 4x1", "block2d 2x1"]]
for n in sizes:
    rows.append([str(n), tfmt(T.get(("ijk", n), np.nan)), tfmt(T[("seq", n)]), tfmt(T[("omp", n)])] +
                [f"{tfmt(rb(c, n, 'end_to_end_s'))} ±{rb(c, n, 'iqr_pct'):.0f}%" if rb(c, n, 'end_to_end_s') == rb(c, n, 'end_to_end_s')
                 else "not run (memory)" for c in ["1x1", "2x1", "1x2", "4x1"]] +
                [f"{tfmt(b2('2x1', n, 'end_to_end_s'))} ±{b2('2x1', n, 'iqr_pct'):.0f}%"])
table(rows, [1.0 * cm, 1.7 * cm, 1.8 * cm, 1.9 * cm, 1.9 * cm, 1.9 * cm, 1.9 * cm, 2.1 * cm, 1.9 * cm],
      caption="Table 3. Median time per multiplication (end-to-end boundary for Spark) ± IQR. ijk was not run at 3072 "
              "(above 180 s in Assignment 2). Spark rows: rowblock with P = 8 unless named block2d.")
figure("fig1_end_to_end.png", 13.0, "Figure 1. Time against n on log axes. Spark curves are end-to-end, without cluster and application start-up.")
P(f"With one worker, Spark took {rb('1x1', 1024, 'end_to_end_s') / T[('seq', 1024)]:.2f}×, "
  f"{rb('1x1', 2048, 'end_to_end_s') / T[('seq', 2048)]:.2f}× and {rb('1x1', 3072, 'end_to_end_s') / T[('seq', 3072)]:.2f}× "
  "the time of the sequential program at n = 1024, 2048 and 3072. The kernel inside the tasks ran at the sequential "
  f"speed (the summed kernel time was {rb('1x1', 1024, 'kernel_inflation'):.2f}×, {rb('1x1', 2048, 'kernel_inflation'):.2f}× "
  f"and {rb('1x1', 3072, 'kernel_inflation'):.2f}× the sequential program's time), so the difference is overhead of "
  f"{rb('1x1', 1024, 'end_to_end_s') - rb('1x1', 1024, 'crit_kernel_s'):.2f}, {rb('1x1', 2048, 'end_to_end_s') - rb('1x1', 2048, 'crit_kernel_s'):.2f} "
  f"and {rb('1x1', 3072, 'end_to_end_s') - rb('1x1', 3072, 'crit_kernel_s'):.2f} s. It grows with n, but more slowly than "
  "the n³ work, so its share falls (H1 supported). With two cores Spark beat the sequential program from n = 2048 on "
  f"({rb('2x1', 2048, 'speedup_vs_seq'):.2f}× and {rb('2x1', 3072, 'speedup_vs_seq'):.2f}× for 2x1, "
  f"{rb('1x2', 2048, 'speedup_vs_seq'):.2f}× and {rb('1x2', 3072, 'speedup_vs_seq'):.2f}× for 1x2) and was "
  f"{rb('2x1', 3072, 'speedup_vs_ijk') if rb('2x1', 3072, 'speedup_vs_ijk') == rb('2x1', 3072, 'speedup_vs_ijk') else rb('2x1', 2048, 'speedup_vs_ijk'):.0f}× "
  "faster than the basic triple loop at n = 2048, but it never reached the OpenMP version on the same two cores: "
  f"OpenMP was {1 / rb('2x1', 1024, 'speedup_vs_omp2'):.1f}×, {1 / rb('2x1', 2048, 'speedup_vs_omp2'):.2f}× and "
  f"{1 / rb('2x1', 3072, 'speedup_vs_omp2'):.2f}× faster than 2x1. That comparison is isolated as far as possible: same "
  "machine, session, inputs, kernel code and compiler flags, and the same row decomposition; the remaining differences are "
  "the framework, process boundaries and data copies.")

P("4.2 Where the time goes", H2)
figure("fig2_decomposition.png", 15.5, "Figure 2. Median end-to-end time split into broadcast, kernel critical path, rest of the "
       "job and assembly (rowblock). The two middle parts are a critical-path split of the job, not independent measurements.")
rows = [["n", "config", "end-to-end", "broadcast", "job", "kernel crit.", "job - kernel", "first task starts", "after last task", "warm-up run"]]
for n in sizes:
    for c in ["1x1", "2x1", "1x2", "4x1"]:
        if rb(c, n, "end_to_end_s") != rb(c, n, "end_to_end_s"):
            continue
        rows.append([str(n), c] + [tfmt(rb(c, n, k)) for k in ["end_to_end_s", "broadcast_s", "job_s", "crit_kernel_s", "noncrit_s",
                                                                 "first_task_start_s", "after_last_task_s", "warmup_e2e_s"]])
table(rows, [0.9 * cm, 1.2 * cm, 1.8 * cm, 1.6 * cm, 1.6 * cm, 1.8 * cm, 1.8 * cm, 2.0 * cm, 1.9 * cm, 1.8 * cm],
      caption="Table 4. Medians (rowblock). 'first task starts' = job submission to the earliest task start; 'after last task' = "
              "end of the last task to the job's end on the driver (result transfer and deserialisation). These two lie inside "
              "'job - kernel' together with gaps between tasks; they are shown for orientation, not to be added.")
P(f"Broadcasting B cost the driver {tfmt(rb('1x1', 1024, 'broadcast_s'))} to {tfmt(rb('1x1', 3072, 'broadcast_s'))} and assembling "
  f"C at most {tfmt(D.assemble_s.max())}; almost all overhead is inside the job. Before the first task started, "
  f"{tfmt(rb('1x1', 1024, 'first_task_start_s'))} to {tfmt(rb('4x1', 2048, 'first_task_start_s'))} passed (more with more "
  "executors): the driver serialises the tasks with their A blocks (8n² bytes in total), the scheduler offers them to "
  "executors and the Python workers deserialise them. Fetching and unpickling B in a task took at most "
  f"{tfmt(RB.b_fetch_max_s.max())}, because executors on one machine fetch it through loopback. Returning the C blocks "
  f"after the last task took {tfmt(RB.after_last_task_s.min())}-{tfmt(RB.after_last_task_s.max())}. The rest of 'job - kernel' "
  "is spread between tasks: Figure 3 shows gaps of roughly 0.1-0.2 s between consecutive tasks of one worker, in which a task "
  "result is serialised and sent and the next task is shipped and started. With 8 tasks this per-task cost repeats 4 times per "
  "worker on two workers. The first multiplication of a session was 1-4 s slower (warm-up column), most likely because Python "
  "workers were started and the kernel library loaded in that run. Cluster start-up, outside the boundary, took "
  f"{ST.loc['1x1', 'cluster_start_s']:.1f}-{ST.loc['4x1', 'cluster_start_s']:.1f} s and application start-up "
  f"{ST.loc['1x1', 'app_start_s']:.1f}-{ST.loc['4x1', 'app_start_s']:.1f} s (median), growing with the number of workers; "
  f"for one multiplication at n = 1024 that is {st_lo / rb('1x1', 1024, 'end_to_end_s'):.0f} (1x1) to "
  f"{st_hi / rb('4x1', 1024, 'end_to_end_s'):.0f} (4x1) times the multiplication itself.")
figure("fig4_timeline.png", 13.5, "Figure 3. Task timeline of one multiplication (round 1, first measured repetition). Each "
       "row is one worker; orange: fetching and unpickling B; blue: the C kernel; gaps: framework work between tasks.")

P("4.3 Strong scaling", H2)
figure("fig3_strong_scaling.png", 12.5, "Figure 4. Speed-up over one worker with one core, end-to-end (solid) and for the "
       "kernel critical path alone (dotted).")
P(f"Doubling the cores (1x1 to 2x1) gave {rb('2x1', 1024, 'strong_speedup_vs_1x1'):.2f}×, {rb('2x1', 2048, 'strong_speedup_vs_1x1'):.2f}× "
  f"and {rb('2x1', 3072, 'strong_speedup_vs_1x1'):.2f}× at n = 1024, 2048 and 3072, and one two-core executor (1x2) "
  f"{rb('1x2', 1024, 'strong_speedup_vs_1x1'):.2f}×, {rb('1x2', 2048, 'strong_speedup_vs_1x1'):.2f}× and "
  f"{rb('1x2', 3072, 'strong_speedup_vs_1x1'):.2f}×. The kernel critical path alone scaled better "
  f"({rb('1x1', 3072, 'crit_kernel_s') / rb('2x1', 3072, 'crit_kernel_s'):.2f}× for 2x1 and "
  f"{rb('1x1', 3072, 'crit_kernel_s') / rb('1x2', 3072, 'crit_kernel_s'):.2f}× for 1x2 at n = 3072), so the shortfall from 2 is "
  "the part of the overhead that does not shrink when cores are added. H2 is supported: the efficiency rises with n "
  "because the overhead is a smaller share of a larger job. 1x2 was slightly faster than 2x1 at every size. A plausible "
  "reason, not isolated here, is that it has one executor JVM instead of two, so the driver sends B to, and schedules "
  "tasks on, one executor only; the memory differences are in Section 4.5. "
  f"Four workers on two cores (4x1) were slower than two at both sizes they ran ({rb('4x1', 1024, 'strong_speedup_vs_1x1'):.2f}× "
  f"and {rb('4x1', 2048, 'strong_speedup_vs_1x1'):.2f}× over 1x1). Their summed kernel time was "
  f"{rb('4x1', 1024, 'kernel_inflation'):.1f}× and {rb('4x1', 2048, 'kernel_inflation'):.1f}× the sequential time, against "
  f"{rb('2x1', 1024, 'kernel_inflation'):.1f}× and {rb('2x1', 2048, 'kernel_inflation'):.1f}× for 2x1: with four Python workers "
  "time-sliced on two vCPUs each kernel ran slower in wall time, while start-up, scheduling and JVM overheads grew with "
  "the number of executors (Figure 3, bottom).")

P("4.4 Partitioning and granularity", H2)
rows = [["", "tasks", "end-to-end", "kernel crit.", "job - kernel", "shuffle read", "result bytes"]]
for c in ["1x1", "2x1"]:
    for n in sizes:
        rows.append([f"block2d {c}, n = {n}", f"{b2(c, n, 'n_tasks'):.0f}", tfmt(b2(c, n, "end_to_end_s")), tfmt(b2(c, n, "crit_kernel_s")),
                     tfmt(b2(c, n, "noncrit_s")), mib(b2(c, n, "shuffle_read_bytes")), mib(b2(c, n, "result_bytes"))])
        rows.append([f"rowblock {c}, n = {n}", f"{rb(c, n, 'n_tasks'):.0f}", tfmt(rb(c, n, "end_to_end_s")), tfmt(rb(c, n, "crit_kernel_s")),
                     tfmt(rb(c, n, "noncrit_s")), mib(rb(c, n, "shuffle_read_bytes")), mib(rb(c, n, "result_bytes"))])
table(rows, [3.6 * cm, 1.2 * cm, 2.0 * cm, 2.0 * cm, 2.0 * cm, 2.2 * cm, 2.2 * cm],
      caption="Table 5. block2d against rowblock (medians). Shuffle and result bytes from the Spark stage metrics.")
P(f"block2d was slower than rowblock in all six comparisons, by {min(b2(c, n, 'end_to_end_s') / rb(c, n, 'end_to_end_s') for c in ['1x1', '2x1'] for n in sizes):.2f}-"
  f"{max(b2(c, n, 'end_to_end_s') / rb(c, n, 'end_to_end_s') for c in ['1x1', '2x1'] for n in sizes):.2f}×. Its shuffles read "
  f"{shuffle_ratio:.1f} times the size of one matrix (8n² bytes) at every n, as the design predicts: the join moves g = 2 copies "
  "of each of A and B (4·8n²) and reduceByKey moves the g partial products of C (2·8n²); each shuffle is written to "
  "local files and read back, and its stages have to finish before the next start. rowblock moves A and C once and B once "
  "per executor, without shuffles (H3 supported for time). On one machine a shuffle is a local file and loopback copy; on a "
  "real network the difference could be larger, which this setup cannot show. "
  f"<b>Granularity.</b> With 2x1 at n = 2048, P = 2 tasks gave {tfmt(G[G.parts == 2].end_to_end_s.iloc[0])}, P = 8 "
  f"{tfmt(G[G.parts == 8].end_to_end_s.iloc[0])} and P = 64 {tfmt(G[G.parts == 64].end_to_end_s.iloc[0])} (Figure 5): each extra "
  "task adds scheduling, serialisation and result-handling cost, and on two identical cores there is no imbalance for smaller "
  f"tasks to correct. P = 2 came within {100 * (G[G.parts == 2].end_to_end_s.iloc[0] / T[('omp', 2048)] - 1):.0f} % of OpenMP. "
  "On a heterogeneous or shared cluster more tasks than cores would be needed to absorb slow nodes, so P = 8 was kept as the "
  "main setting rather than tuned to this machine.")
figure("fig5_granularity.png", 10.5, "Figure 5. Number of row blocks P, 2 workers × 1 core, n = 2048.")

P("4.5 Memory and data transfer", H2)
rows = [["n", "config", "A shipped", "B broadcast (x executors)", "C returned", "largest executor tree", "all executors", "driver", "single process (seq)"]]
for n in sizes:
    for c in ["1x1", "2x1", "1x2", "4x1"]:
        if rb(c, n, "end_to_end_s") != rb(c, n, "end_to_end_s"):
            continue
        rows.append([str(n), c, mib(rb(c, n, "bytes_A")), f"{mib(rb(c, n, 'bytes_B_broadcast'))} × {rb(c, n, 'executors_used'):.0f}",
                     mib(rb(c, n, "result_bytes")), f"{rb(c, n, 'peak_rss_exec_max_mib'):.0f} MiB", f"{rb(c, n, 'peak_rss_exec_sum_mib'):.0f} MiB",
                     f"{rb(c, n, 'peak_rss_driver_mib'):.0f} MiB",
                     f"{REF[(REF.method == 'seq') & (REF.n == n)].peak_rss_mib.iloc[0]:.0f} MiB"])
table(rows, [0.9 * cm, 1.2 * cm, 1.6 * cm, 2.5 * cm, 1.7 * cm, 2.1 * cm, 1.8 * cm, 1.5 * cm, 2.0 * cm],
      caption="Table 6. Data moved per multiplication (A and B: payload size; C: Spark resultSize) and peak RSS (largest over "
              "runs). 'B × executors' is the number of executor JVMs that must hold B; each Python worker unpickles its own copy as well.")
figure("fig6_memory.png", 11.0, f"Figure 6. Peak RSS at n = {nmax}.")
P(f"The data moved per multiplication match the design: A once, B once per executor and C back once; there is no shuffle. "
  f"All of it crossed the loopback interface, so these are byte counts, not network measurements. Memory is where "
  f"distribution cost most. At n = {nmax} the sequential program needed {REF[(REF.method == 'seq') & (REF.n == nmax)].peak_rss_mib.iloc[0]:.0f} MiB; "
  f"Spark 1x1 needed {rb('1x1', nmax, 'peak_rss_exec_sum_mib') + rb('1x1', nmax, 'peak_rss_driver_mib'):.0f} MiB across executor and "
  f"driver, and 2x1 {rb('2x1', nmax, 'peak_rss_exec_sum_mib') + rb('2x1', nmax, 'peak_rss_driver_mib'):.0f} MiB. Every JVM "
  "has a fixed footprint of a few hundred MiB, the driver holds A, B and C both as NumPy arrays and in serialised form, and "
  "every executor JVM and every Python worker holds its own copy of B. The 1x2 executor tree was larger than a 2x1 executor "
  "tree because its two Python workers each held B. This replication is why n = 4096, and four executors at n = 3072, did not "
  "fit in 6 GiB (H3 supported for memory): broadcasting B gives up memory scalability to save communication, which is the "
  "usual trade-off of 1-D algorithms [6].")

# =============================== 5 DISCUSSION ===============================
P("5. Discussion", H1)
P(f"<b>Answer to the research question.</b> On one 2-vCPU machine, Spark scaled the multiplication from one to two cores "
  f"with a speed-up of up to {max(strong.values()):.2f}× at the largest size, and the efficiency grew with n. The "
  "kernel itself scaled almost perfectly; what limited Spark was a framework overhead of roughly 0.5-2 s per multiplication "
  "(task scheduling and serialisation, moving A and C between processes, Python worker overhead) and, outside the "
  f"end-to-end boundary, {st_lo:.0f}-{st_hi:.0f} s of cluster and application start-up. For the sizes that fit, the distributed version "
  "beat the sequential program from n = 2048 with two cores, but never the OpenMP version on the same cores, which shares "
  "memory and pays none of these costs. More executors than cores slowed it down. The row-block strategy with a "
  "broadcast beat the 2-D shuffle strategy in time but replicated B in every executor and Python worker, and memory, "
  "not time, set the largest feasible size.")
P("<b>What would change on a real cluster.</b> Real nodes would add network transfer of A, B and C (8n² bytes each, "
  "B once per node), which loopback hides, and would add memory per node, so larger n would fit. The per-multiplication "
  "overhead measured here is a lower bound for that setting. A distributed version pays off when the problem does not fit "
  "in one machine or when many machines are available; for a problem that fits in one machine, the shared-memory version "
  "was faster here. These are inferences from the measurements, not measured multi-machine results.")
P("<b>Limitations.</b> (1) Local execution only: one physical machine, loopback network, 2 vCPUs; it needs the "
  "instructor's permission. (2) The memory limit forced n ≤ 3072 and removed 4x1 at 3072. (3) PySpark adds a Python "
  "process boundary and pickling that a Scala/Java implementation would avoid; the kernel was kept in C precisely to "
  "keep computation identical. (4) The harness's memory sampler (a Python thread every 0.1 s) and the gc requests between "
  "runs add a small, unmeasured load. (5) The critical-path split assumes the tasks of one executor slot run one after "
  "another, which Figure 3 confirms. (6) Weak scaling and inputs larger than one machine's memory were not studied. "
  "(7) Only two strategies, one tile size and one broadcast setting were tested.")

# =============================== 6 CONCLUSIONS ===============================
P("6. Conclusions", H1)
P(f"Within a local Spark cluster on one 2-vCPU machine, row-block multiplication with a broadcast B, using the same C kernel "
  f"as the single-process versions, scaled from one to two cores by {strong['2x1']:.2f}× (two workers) and "
  f"{strong['1x2']:.2f}× (one two-core worker) for n = {nmax}. With two workers it was {vs_seq['2x1']:.2f}× faster than "
  f"the sequential program there, but {1 / vs_omp['2x1']:.2f}× slower than OpenMP on the same cores. Framework overhead per multiplication, start-up costs and replicated memory explain the gap: the computation "
  "was the same. A 2-D strategy with two shuffles moved six matrix-sizes of data through the shuffle and was always slower. "
  "Open questions are how the overhead and the broadcast compare with network costs on several physical machines, and "
  "whether a 2-D strategy, which replicates less, would allow larger n when memory per node is the limit.")

P("7. Use of generative AI", H1)
P("In line with the course policy, the AI assistant Claude (Anthropic) was used to plan the experiments, draft the Spark "
  "implementation, the harness and analysis scripts, the README and the first draft of this report. All numerical results "
  "come from measurements stored in results/ and produced by the committed code; the AI output is not used as evidence. "
  "The author remains responsible for understanding the code, validating the measurements, verifying the references and "
  "defending the conclusions.")

P("References", H1)
refs = [
    "“Individual Assignments: Matrix Multiplication,” Big Data, Grado en Ciencia e Ingeniería de Datos, ULPGC, revised proposal 2026-2027.",
    "M. D. Afonso Suárez, J. J. Hernández Cabrera and J. Évora Gómez, “Big Data,” course materials (Block 2: Spark), ULPGC, 2026/27. "
    "[Online]. Available: https://mltechdrawer.github.io/BIGDATA/",
    "The Apache Software Foundation, “Monitoring and Instrumentation” and “RDD Programming Guide,” Apache Spark documentation. "
    "[Online]. Available: https://spark.apache.org/docs/latest/",
    "M. Zaharia <i>et al.</i>, “Resilient distributed datasets: A fault-tolerant abstraction for in-memory cluster computing,” in "
    "<i>Proc. 9th USENIX NSDI</i>, 2012, pp. 15-28.",
    "M. Zaharia <i>et al.</i>, “Apache Spark: A unified engine for big data processing,” <i>Commun. ACM</i>, vol. 59, no. 11, pp. 56-65, 2016.",
    "R. A. van de Geijn and J. Watts, “SUMMA: Scalable universal matrix multiplication algorithm,” <i>Concurrency: Practice and "
    "Experience</i>, vol. 9, no. 4, pp. 255-274, 1997.",
    "X. Meng <i>et al.</i>, “MLlib: Machine learning in Apache Spark,” <i>J. Mach. Learn. Res.</i>, vol. 17, no. 34, pp. 1-7, 2016.",
    "J. L. Gustafson, “Reevaluating Amdahl's law,” <i>Commun. ACM</i>, vol. 31, no. 5, pp. 532-533, 1988.",
    "N. J. Higham, <i>Accuracy and Stability of Numerical Algorithms</i>, 2nd ed. Philadelphia, PA, USA: SIAM, 2002.",
    "H. Amador Hidalgo, “Thread scaling of a blocked matrix-multiplication kernel with OpenMP,” Big Data, Individual Assignment 3, ULPGC, 2026.",
]
for i, r in enumerate(refs, 1):
    P(f"[{i}] {r}", REFS)
P("Reproducibility materials", H1)
P(f"Repository: {args.repo_url}, folder Assignment04_AmadorHidalgo: src/spark (strategies, cluster control, session "
  "and framework configuration), src/kernel (C kernel), src/reference (unchanged Assignment 2-3 programs [10]), build.sh, "
  "README with exact commands, requirements.txt (pyspark==4.0.1), scripts/ (inputs, tests, harness, analysis, this report), "
  "data/manifest.csv, and results/ (every multiplication and task, start-up times, stage metrics, memory, configuration "
  "files, OOM events, figures).")


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(colors.HexColor("#52514e"))
    canvas.drawString(2 * cm, 1.1 * cm, "Hugo Amador Hidalgo - Big Data - Assignment 4 - Distributed matrix multiplication (local execution)")
    canvas.drawRightString(A4[0] - 2 * cm, 1.1 * cm, f"{doc.page}")
    canvas.restoreState()


doc = SimpleDocTemplate(str(OUT), pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm, topMargin=1.8 * cm, bottomMargin=1.8 * cm,
                        title="Assignment 4 - Distributed matrix multiplication with Apache Spark",
                        author="Hugo Amador Hidalgo", subject="Big Data 2026-2027, ULPGC")
doc.build(story, onFirstPage=footer, onLaterPages=footer)
print("report written:", OUT)
