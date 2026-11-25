#!/usr/bin/env python3
"""Summaries, figures and key numbers for Assignment 4 from results/*.csv.

Statistic: median of the measured multiplications (warm-up, rep 0, excluded; reported separately) pooled
over the rounds, with IQR; the spread between rounds is the range of the per-round medians.

Time decomposition of one Spark multiplication (all measured on the driver unless stated):
  end_to_end = broadcast + job + assemble                  (sequential on the driver: additive)
  job        = wall time of the Spark action (task scheduling, shipping A blocks, fetching B, kernels,
               returning C blocks). Inside the job, activities overlap across tasks, so only a
               critical-path split is reported: crit_kernel = the largest total kernel time of one
               Python worker (measured inside the tasks), and job - crit_kernel = time on the critical
               path not spent in the kernel (framework, transfer, serialisation, waiting). These two
               add up to job by construction; they are not independent measurements.
Outputs: results/summary_*.csv, results/key_numbers.json, results/figures/*.png
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RES = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "results"
FIG = RES / "figures"
FIG.mkdir(parents=True, exist_ok=True)

INK, INK2, SURFACE, GRID = "#0b0b0b", "#52514e", "#fcfcfb", "#e4e3df"
CFG_COL = {"1x1": "#2a78d6", "2x1": "#eb6834", "4x1": "#1baf7a", "1x2": "#8a5cd6"}
REF_COL = {"seq": INK, "omp": "#d6336c", "ijk": "#9a9893"}
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 9,
    "axes.titlesize": 10, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "legend.frameon": False, "lines.linewidth": 2, "lines.markersize": 6, "lines.markeredgecolor": SURFACE,
})
CFG_LABEL = {"1x1": "1 worker x 1 core", "2x1": "2 workers x 1 core", "4x1": "4 workers x 1 core (oversubscribed)",
             "1x2": "1 worker x 2 cores"}

runs = pd.read_csv(RES / "spark_runs.csv")
tasks = pd.read_csv(RES / "spark_tasks.csv")
rraw = pd.read_csv(RES / "reference_raw.csv")
rruns = pd.read_csv(RES / "reference_runs.csv")
startup = pd.read_csv(RES / "startup.csv")
runs["warmup"] = runs.rep == 0
# Kernel critical path, recomputed from the task records: an executor with c cores runs c tasks at a time,
# so its kernel time per core is (sum of its tasks' kernel time) / c; the critical path is the largest of
# these over executors. (The harness's own crit_kernel_s grouped by Python process; PySpark sometimes starts
# a second Python worker on the same core, which made that per-process value too small. It is kept in the
# raw file as crit_kernel_by_pid_s.)
RK = ["round", "experiment", "config", "strategy", "parts", "n", "rep"]
cpw = tasks.config.str.split("x").str[1].astype(int)
per_w = tasks.assign(k=tasks.kernel_s / cpw).groupby(RK + ["worker"]).k.sum()
crit = per_w.groupby(RK).max().rename("crit_kernel_new").reset_index()
runs = runs.rename(columns={"crit_kernel_s": "crit_kernel_by_pid_s"}).merge(crit, on=RK, how="left")
runs["crit_kernel_s"] = runs.crit_kernel_new
runs["noncrit_s"] = runs.job_s - runs.crit_kernel_s
meas = runs[~runs.warmup]
KEYS = ["experiment", "config", "strategy", "parts", "n"]


def q(x, p):
    return float(np.percentile(x, p))


# ------------------------------------------------------------ references -------------------------
ref = []
for (prog, meth, thr, n), g in rraw.groupby(["program", "method", "threads", "n"]):
    pr = g.groupby("round").seconds.median()
    ref.append(dict(program=prog, method=meth, threads=thr, n=n, samples=len(g), median_s=g.seconds.median(),
                    iqr_pct=100 * (q(g.seconds, 75) - q(g.seconds, 25)) / g.seconds.median(),
                    round_spread_pct=100 * (pr.max() / pr.min() - 1),
                    peak_rss_mib=rruns[(rruns.program == prog) & (rruns.method == meth) & (rruns.n == n)].peak_rss_kb.max() / 1024))
ref = pd.DataFrame(ref)
ref.to_csv(RES / "summary_references.csv", index=False, float_format="%.6g")
T = {(r.method, r.n): r.median_s for r in ref.itertuples()}

# ------------------------------------------------------------ spark summary ----------------------
cols = ["end_to_end_s", "broadcast_s", "job_s", "assemble_s", "crit_kernel_s", "sum_kernel_s", "noncrit_s",
        "first_task_start_s", "b_fetch_max_s", "after_last_task_s"]
rows = []
for k, g in meas.groupby(KEYS):
    d = dict(zip(KEYS, k), samples=len(g))
    for c in cols:
        d[c] = float(g[c].median())
    e = g.end_to_end_s
    d["iqr_pct"] = 100 * (q(e, 75) - q(e, 25)) / e.median()
    pr = g.groupby("round").end_to_end_s.median()
    d["round_spread_pct"] = 100 * (pr.max() / pr.min() - 1)
    for c in ["result_bytes", "shuffle_read_bytes", "shuffle_write_bytes", "bytes_A", "bytes_B_broadcast", "bytes_C",
              "executors_used", "python_workers", "n_tasks", "gc_ms", "deserialize_ms", "run_time_ms", "cpu_time_ms",
              "spilled_bytes"]:
        d[c] = float(g[c].median())
    d["peak_rss_exec_max_mib"] = float(g.peak_rss_exec_max_mib.max())
    d["peak_rss_exec_sum_mib"] = float(g.peak_rss_exec_sum_mib.max())
    d["peak_rss_driver_mib"] = float(g.peak_rss_driver_mib.max())
    d["valid_all"] = bool((g.valid == "yes").all())
    w = runs[(runs.rep == 0) & (runs.experiment == k[0]) & (runs.config == k[1]) & (runs.strategy == k[2]) &
             (runs.parts == k[3]) & (runs.n == k[4])]
    d["warmup_e2e_s"] = float(w.end_to_end_s.median()) if len(w) else np.nan
    rows.append(d)
S = pd.DataFrame(rows)
S["workers"] = S.config.str.split("x").str[0].astype(int)
S["cores"] = S.workers * S.config.str.split("x").str[1].astype(int)
S["t_seq_s"] = [T.get(("seq", n), np.nan) for n in S.n]
S["t_omp2_s"] = [T.get(("omp", n), np.nan) for n in S.n]
S["t_ijk_s"] = [T.get(("ijk", n), np.nan) for n in S.n]
S["speedup_vs_seq"] = S.t_seq_s / S.end_to_end_s
S["speedup_vs_omp2"] = S.t_omp2_s / S.end_to_end_s
S["speedup_vs_ijk"] = S.t_ijk_s / S.end_to_end_s
base = S[(S.config == "1x1")].set_index(["strategy", "n"]).end_to_end_s
S["strong_speedup_vs_1x1"] = [base.get((s, n), np.nan) / t for s, n, t in zip(S.strategy, S.n, S.end_to_end_s)]
S["kernel_share_pct"] = 100 * S.crit_kernel_s / S.end_to_end_s
S["kernel_inflation"] = S.sum_kernel_s / S.t_seq_s           # total kernel time in tasks vs sequential program
S["bytes_B_total"] = S.bytes_B_broadcast * S.executors_used
S = S.sort_values(["experiment", "strategy", "n", "cores", "config"])
S.to_csv(RES / "summary_spark.csv", index=False, float_format="%.6g")
startup.groupby("config")[["master_start_s", "workers_register_s", "cluster_start_s", "driver_start_s",
                           "executors_register_s", "app_start_s"]].median().to_csv(RES / "summary_startup.csv",
                                                                                     float_format="%.4g")

D = S[S.experiment == "D"]
RB = D[D.strategy == "rowblock"]
B2 = D[D.strategy == "block2d"]
sizes = sorted(D.n.unique())
cfgs = ["1x1", "2x1", "4x1", "1x2"]


def save(fig, name, rect=None):
    fig.tight_layout(rect=rect)
    fig.savefig(FIG / name, dpi=200)
    plt.close(fig)


# fig1: end-to-end vs n
fig, ax = plt.subplots(figsize=(6.6, 4.0))
for meth, lab in (("ijk", "basic sequential (ijk, Assignment 1-2)"), ("seq", "blocked sequential (Assignment 3)"),
                  ("omp", "OpenMP, 2 threads (Assignment 3)")):
    g = ref[ref.method == meth].sort_values("n")
    g = g[g.n.isin(sizes)]
    ax.plot(g.n, g.median_s, color=REF_COL[meth], marker="s", ls="--", lw=1.4, label=lab)
for c in cfgs:
    g = RB[RB.config == c].sort_values("n")
    ax.plot(g.n, g.end_to_end_s, color=CFG_COL[c], marker="o", label=f"Spark rowblock, {CFG_LABEL[c]}")
g = B2[B2.config == "2x1"].sort_values("n")
ax.plot(g.n, g.end_to_end_s, color=CFG_COL["2x1"], marker="^", ls=":", label="Spark block2d, 2 workers x 1 core")
ax.set_xscale("log", base=2)
ax.set_yscale("log")
ax.set_xticks(sizes)
ax.set_xticklabels(sizes)
ax.set_xlabel("matrix size n")
ax.set_ylabel("time per multiplication (s)")
ax.set_title("End-to-end time: Spark against the single-process versions")
ax.legend(fontsize=6.8)
save(fig, "fig1_end_to_end.png")

# fig2: decomposition at each n (rowblock)
fig, axes = plt.subplots(1, len(sizes), figsize=(8.2, 3.8), sharey=False)
parts_ = [("broadcast_s", "broadcast B (driver)", "#9db9de"), ("crit_kernel_s", "kernel (critical path)", "#2a78d6"),
          ("noncrit_s", "job, not kernel (critical path)", "#eb6834"), ("assemble_s", "assemble C (driver)", "#1baf7a")]
for ax, n in zip(axes, sizes):
    g = RB[RB.n == n].set_index("config").reindex(cfgs)
    bottom = np.zeros(len(cfgs))
    for col, lab, colr in parts_:
        ax.bar(range(len(cfgs)), g[col].values, bottom=bottom, color=colr, label=lab, width=0.7)
        bottom += g[col].values
    ax.axhline(T[("seq", n)], color=INK, ls="--", lw=1)
    ax.axhline(T[("omp", n)], color=REF_COL["omp"], ls=":", lw=1.2)
    ax.set_xticks(range(len(cfgs)))
    ax.set_xticklabels(cfgs)
    ax.set_title(f"n = {n}")
    for i, cname in enumerate(cfgs):
        if np.isnan(g.loc[cname, "end_to_end_s"]):
            ax.text(i, 0.02 * ax.get_ylim()[1], "not run\n(memory)", ha="center", fontsize=6.5, color=INK2)
axes[0].set_ylabel("seconds (median)")
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=4, fontsize=7)
fig.text(0.01, 0.965, "dashed: blocked sequential; dotted: OpenMP 2 threads (same machine)", fontsize=7, color=INK2)
save(fig, "fig2_decomposition.png", rect=(0, 0.09, 1, 0.95))

# fig3: strong scaling
fig, ax = plt.subplots(figsize=(6.2, 3.5))
order = ["1x1", "1x2", "2x1", "4x1"]
for n, colr in zip(sizes, ["#2a78d6", "#eb6834", "#1baf7a"]):
    g = RB[RB.n == n].set_index("config").reindex(order)
    ax.plot(range(len(order)), g.strong_speedup_vs_1x1, color=colr, marker="o", label=f"rowblock, n = {n}")
    k = g.crit_kernel_s.iloc[0] / g.crit_kernel_s
    ax.plot(range(len(order)), k, color=colr, marker="x", ls=":", lw=1.2, label=f"kernel critical path only, n = {n}")
ax.axhline(2, color=INK2, ls="--", lw=1)
ax.text(0, 2.05, "2 vCPUs", fontsize=7.5, color=INK2)
ax.set_xticks(range(len(order)))
ax.set_xticklabels([CFG_LABEL[c].replace(" (oversubscribed)", "\n(oversubscribed)") for c in order], fontsize=7.5)
ax.set_ylim(0, 2.4)
ax.set_ylabel("speed-up over 1 worker x 1 core")
ax.set_title("Strong scaling (n fixed) of the rowblock strategy")
ax.legend(fontsize=6.8, ncol=2, loc="lower right")
save(fig, "fig3_strong_scaling.png")

# fig4: timeline of one multiplication, rowblock, 2x1 and 4x1, round 1 rep 1, at the largest n both ran
nmax = max(sizes)
ntl = int(RB[RB.config == "4x1"].n.max())
fig, axes = plt.subplots(2, 1, figsize=(7.2, 4.2), sharex=True)
for ax, c in zip(axes, ["2x1", "4x1"]):
    tt = tasks[(tasks.experiment == "D") & (tasks.config == c) & (tasks.strategy == "rowblock") &
               (tasks.n == ntl) & (tasks["round"] == 1) & (tasks.rep == 1)].copy()
    lanes = {p: i for i, p in enumerate(sorted(tt.pid.unique(), key=lambda p: (tt[tt.pid == p].worker.iloc[0], p)))}
    for r in tt.itertuples():
        y = lanes[r.pid]
        ax.barh(y, r.got_b_rel_s - r.start_rel_s, left=r.start_rel_s, color="#eb6834", height=0.6)
        k0 = r.end_rel_s - r.kernel_s
        ax.barh(y, r.kernel_s, left=k0, color="#2a78d6", height=0.6)
        ax.barh(y, k0 - r.got_b_rel_s, left=r.got_b_rel_s, color="#c9c8c3", height=0.6)
    je = tt.job_end_rel_s.iloc[0] if len(tt) else 0
    ax.axvline(0, color=INK2, lw=0.8)
    ax.axvline(je, color=INK2, lw=0.8, ls="--")
    ax.set_yticks(list(lanes.values()))
    ax.set_yticklabels([f"worker {tt[tt.pid == p].worker.iloc[0]}" for p in lanes], fontsize=7)
    ax.set_title(f"{CFG_LABEL[c]}, n = {ntl}: one task per bar segment", fontsize=9)
axes[-1].set_xlabel("seconds since the job was submitted (dashed: job finished on the driver)")
from matplotlib.patches import Patch
axes[0].legend(handles=[Patch(color="#eb6834", label="fetch + unpickle B"), Patch(color="#c9c8c3", label="other task work"),
                        Patch(color="#2a78d6", label="C kernel")], fontsize=7, loc="lower right", ncol=3)
save(fig, "fig4_timeline.png")

# fig5: granularity
G = S[(S.config == "2x1") & (S.strategy == "rowblock") & (S.n == 2048) & S.experiment.isin(["D", "G"])].sort_values("parts")
fig, ax = plt.subplots(figsize=(5.8, 3.2))
ax.plot(G.parts, G.end_to_end_s, color=CFG_COL["2x1"], marker="o", label="end-to-end")
ax.plot(G.parts, G.crit_kernel_s, color="#2a78d6", marker="x", ls=":", label="kernel critical path")
ax.axhline(T[("omp", 2048)], color=REF_COL["omp"], ls=":", lw=1.2, label="OpenMP, 2 threads")
ax.set_xscale("log", base=2)
ax.set_xticks(G.parts)
ax.set_xticklabels(G.parts.astype(int))
ax.set_ylim(bottom=0)
ax.set_xlabel("row blocks (tasks) P")
ax.set_ylabel("seconds (median)")
ax.set_title("Task granularity, 2 workers x 1 core, n = 2048")
ax.legend(fontsize=7.5)
save(fig, "fig5_granularity.png")

# fig6: memory at the largest n
fig, ax = plt.subplots(figsize=(6.2, 3.3))
g = RB[RB.n == nmax].set_index("config").reindex(cfgs)
x = np.arange(len(cfgs))
ax.bar(x - 0.27, g.peak_rss_exec_max_mib, 0.26, color="#2a78d6", label="largest executor tree (JVM + Python workers)")
ax.bar(x, g.peak_rss_exec_sum_mib, 0.26, color="#9db9de", label="all executor trees")
ax.bar(x + 0.27, g.peak_rss_driver_mib, 0.26, color="#eb6834", label="driver (Python + JVM)")
ax.axhline(3 * 8 * nmax ** 2 / 2 ** 20, color=INK, ls="--", lw=1)
ax.text(-0.45, 3 * 8 * nmax ** 2 / 2 ** 20 * 1.03, "A + B + C = 3 x 8n²", fontsize=7, color=INK2)
for i, cname in enumerate(cfgs):
    if np.isnan(g.loc[cname, "peak_rss_driver_mib"]):
        ax.text(i, 150, "not run\n(memory\nbudget)", ha="center", fontsize=7, color=INK2)
ax.set_xticks(x)
ax.set_xticklabels(cfgs)
ax.set_ylabel("peak RSS (MiB)")
ax.set_title(f"Memory during one multiplication, rowblock, n = {nmax}")
ax.legend(fontsize=7)
save(fig, "fig6_memory.png")

# ------------------------------------------------------------ key numbers -----------------------
key = dict(
    references={f"{r.method}@{r.n}": r.median_s for r in ref.itertuples()},
    rowblock={f"{r.config}@{r.n}": dict(e2e=r.end_to_end_s, job=r.job_s, crit=r.crit_kernel_s, vs_seq=r.speedup_vs_seq,
                                         vs_omp2=r.speedup_vs_omp2, strong=r.strong_speedup_vs_1x1,
                                         warm=r.warmup_e2e_s, iqr=r.iqr_pct) for r in RB.itertuples()},
    block2d={f"{r.config}@{r.n}": dict(e2e=r.end_to_end_s, vs_rowblock=RB[(RB.config == r.config) & (RB.n == r.n)].end_to_end_s.iloc[0] / r.end_to_end_s,
                                        shuffle=r.shuffle_read_bytes) for r in B2.itertuples()},
    all_valid=bool((runs.valid == "yes").all()), n_runs=len(runs), n_measured=len(meas),
)
(RES / "key_numbers.json").write_text(json.dumps(key, indent=1, default=float))
print(json.dumps(key, indent=1, default=float)[:5000])
