#!/usr/bin/env python3
"""Turn results/raw_measurements.csv + runs.csv into summary tables, figures and key numbers.

Statistic: the median of all measured samples of a configuration, pooled over the rounds, with the
inter-quartile range (linear-interpolation quartiles). Speed-up S(p) = median Tseq / median Tp and
efficiency E(p) = S(p) / p, where Tseq is the sequential baseline (method seq of build/mm) at the
same n. The spread between rounds is reported as the range of the per-round speed-ups.
Nothing is removed.

Outputs (results/): summary_scaling.csv, summary_vectorisation.csv, summary_controls.csv,
key_numbers.json, figures/*.png
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
SEQ = ["#2a78d6", "#eb6834", "#1baf7a", "#8a5cd6", "#d6336c", "#b08a00"]
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 9,
    "axes.titlesize": 10, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "legend.frameon": False, "lines.linewidth": 2, "lines.markersize": 6, "lines.markeredgecolor": SURFACE,
})

raw = pd.read_csv(RES / "raw_measurements.csv")
runs = pd.read_csv(RES / "runs.csv")
raw["t"] = raw.ns_per_mult * 1e-9
runs["cores_busy"] = (runs.kernel_user_s + runs.kernel_sys_s) / runs.kernel_wall_s
KEY = ["experiment", "binary", "method", "threads", "n"]


def q(x, p):
    return float(np.percentile(x, p))


def summarise(df_raw, df_runs):
    out = []
    for k, g in df_raw.groupby(KEY):
        r = df_runs.set_index(KEY).loc[[k]]
        per_round = g.groupby("round").t.median()
        out.append(dict(zip(KEY, k), samples=len(g), median_s=float(g.t.median()), q1_s=q(g.t, 25), q3_s=q(g.t, 75),
                        iqr_pct=100 * (q(g.t, 75) - q(g.t, 25)) / g.t.median(),
                        round_medians_s=";".join(f"{v:.6g}" for v in per_round.values),
                        round_spread_pct=100 * (per_round.max() / per_round.min() - 1),
                        peak_rss_mib=float(r.peak_rss_kb.max()) / 1024, cores_busy=float(r.cores_busy.median()),
                        statuses=";".join(sorted(set(r.status))), gflops=2 * k[4] ** 3 / g.t.median() / 1e9))
    return pd.DataFrame(out)


S = summarise(raw, runs)
S["array_mib"] = 3 * 8 * S.n.astype(float) ** 2 / 2 ** 20

# --------------------------- thread scaling (experiment P) ----------------------------
P = S[S.experiment == "P"].copy()
seq = P[P.method == "seq"].set_index("n").median_s
P["speedup"] = P.n.map(seq) / P.median_s
P["efficiency"] = P.speedup / P.threads
rseq = raw[(raw.experiment == "P") & (raw.method == "seq")].groupby(["n", "round"]).t.median()
rS = []
for (n, m, p), g in raw[raw.experiment == "P"].groupby(["n", "method", "threads"]):
    s = [rseq[(n, r)] / gg.t.median() for r, gg in g.groupby("round")]
    rS.append(dict(n=n, method=m, threads=p, speedup_round_min=min(s), speedup_round_max=max(s)))
P = P.merge(pd.DataFrame(rS), on=["n", "method", "threads"])
P["oversubscribed"] = P.threads > 2
P = P.sort_values(["n", "method", "threads"])
P.to_csv(RES / "summary_scaling.csv", index=False, float_format="%.6g")

# --------------------------- vectorisation (experiment V) -----------------------------
V = S[S.experiment == "V"].copy()
V["config"] = V.binary.str.replace("mm_", "") + "_" + V.method + "_p" + V.threads.astype(str)
Vt = V.pivot(index="n", columns="config", values="median_s")
# Primary 2 x 2 comparison on the same code path (row_tile): omp p = 1 vs omp p = 2, scalar vs AVX2.
# The seq configurations of the -O3 builds are kept as a secondary check: in mm_scalar the sequential
# function was compiled into slower code than row_tile, so it is not used as the reference there.
vec = pd.DataFrame({
    "n": Vt.index,
    "scalar_p1_s": Vt.scalar_omp_p1, "simd_p1_s": Vt.simd_omp_p1, "scalar_p2_s": Vt.scalar_omp_p2,
    "simd_p2_s": Vt.simd_omp_p2,
    "vec_speedup_p1": Vt.scalar_omp_p1 / Vt.simd_omp_p1, "vec_speedup_p2": Vt.scalar_omp_p2 / Vt.simd_omp_p2,
    "thread_speedup_scalar": Vt.scalar_omp_p1 / Vt.scalar_omp_p2, "thread_speedup_simd": Vt.simd_omp_p1 / Vt.simd_omp_p2,
    "combined_speedup": Vt.scalar_omp_p1 / Vt.simd_omp_p2,
    "scalar_seq_s": Vt.scalar_seq_p1, "simd_seq_s": Vt.simd_seq_p1,
    "scalar_seq_over_omp1": Vt.scalar_seq_p1 / Vt.scalar_omp_p1, "simd_seq_over_omp1": Vt.simd_seq_p1 / Vt.simd_omp_p1,
    "scalar_tiles_over_omp1": Vt.scalar_tiles_p1 / Vt.scalar_omp_p1, "simd_tiles_over_omp1": Vt.simd_tiles_p1 / Vt.simd_omp_p1,
    "o2_seq_s": [seq.get(n, np.nan) for n in Vt.index],
}).reset_index(drop=True)
vec["combined_vs_o2_seq"] = vec.o2_seq_s / vec.simd_p2_s
vec["gflops_simd_p2"] = 2 * vec.n.astype(float) ** 3 / vec.simd_p2_s / 1e9
vec.to_csv(RES / "summary_vectorisation.csv", index=False, float_format="%.6g")
V.to_csv(RES / "summary_vectorisation_configs.csv", index=False, float_format="%.6g")

# --------------------------- controls (experiments N and A) ---------------------------
C = S[S.experiment.isin(["N", "A"])].copy()
C["ratio_to_P_seq"] = C.median_s / C.n.map(seq)
C["ratio_to_same_method_aligned"] = [r.median_s / P[(P.n == r.n) & (P.method == r.method) & (P.threads == r.threads)]
                                     .median_s.iloc[0] for r in C.itertuples()]
C.to_csv(RES / "summary_controls.csv", index=False, float_format="%.6g")


# --------------------------- figures --------------------------------------------------
def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG / name, dpi=200)
    plt.close(fig)


sizes = sorted(P.n.unique())
omp = P[P.method == "omp"]
threads = sorted(omp.threads.unique())

# fig1: speed-up vs threads
fig, ax = plt.subplots(figsize=(6.4, 3.8))
ax.axvspan(2.5, max(threads) + 0.5, color=GRID, alpha=0.45, lw=0)
ax.text(2.6, 2.28, "oversubscribed (more threads than the 2 vCPUs)", fontsize=7.5, color=INK2)
ax.plot(threads, [min(p, 2) for p in threads], color=INK2, ls="--", lw=1, label="ideal with 2 cores: min(p, 2)")
for c, n in zip(SEQ, sizes):
    g = omp[omp.n == n].sort_values("threads")
    ax.errorbar(g.threads, g.speedup, yerr=[(g.speedup - g.speedup_round_min).clip(lower=0),
                                            (g.speedup_round_max - g.speedup).clip(lower=0)],
                color=c, marker="o", capsize=2, label=f"n = {n}")
ax.set_xticks(threads)
ax.set_xlim(0.6, max(threads) + 0.4)
ax.set_ylim(0, 2.4)
ax.set_xlabel("OpenMP threads p")
ax.set_ylabel("speed-up S(p) = Tseq / Tp")
ax.set_title("Speed-up of the OpenMP blocked kernel over the sequential baseline")
ax.legend(fontsize=7.5, ncol=3, loc="lower right")
save(fig, "fig1_speedup.png")

# fig2: efficiency
fig, ax = plt.subplots(figsize=(6.4, 3.4))
ax.axvspan(2.5, max(threads) + 0.5, color=GRID, alpha=0.45, lw=0)
for c, n in zip(SEQ, sizes):
    g = omp[omp.n == n].sort_values("threads")
    ax.plot(g.threads, g.efficiency, color=c, marker="o", label=f"n = {n}")
ax.plot(threads, [min(p, 2) / p for p in threads], color=INK2, ls="--", lw=1, label="best possible: min(p, 2) / p")
ax.set_xticks(threads)
ax.set_ylim(0, 1.15)
ax.set_xlabel("OpenMP threads p")
ax.set_ylabel("efficiency E(p) = S(p) / p")
ax.set_title("Parallel efficiency")
ax.legend(fontsize=7.5, ncol=2)
save(fig, "fig2_efficiency.png")

# fig3: runtime vs n
fig, ax = plt.subplots(figsize=(6.4, 3.8))
lines = [("seq", 1, "sequential baseline", INK)] + [("omp", p, f"OpenMP p = {p}", c) for p, c in zip(threads, SEQ)]
for m, p, lab, c in lines:
    g = P[(P.method == m) & (P.threads == p)].sort_values("n")
    ax.errorbar(g.n, g.median_s, yerr=[g.median_s - g.q1_s, g.q3_s - g.median_s], color=c, marker="o",
                capsize=2, label=lab, ls="-" if m == "seq" else "--" if p > 2 else "-", lw=1.6)
ax.set_xscale("log", base=2)
ax.set_yscale("log")
ax.set_xticks(sizes)
ax.set_xticklabels(sizes)
ax.set_xlabel("matrix size n")
ax.set_ylabel("time per multiplication (s)")
ax.set_title("Runtime against n (median, IQR bars)")
ax.legend(fontsize=7.5, ncol=2)
save(fig, "fig3_runtime.png")

# fig4: CPU busy vs p
fig, ax = plt.subplots(figsize=(6.0, 3.2))
for c, n in zip(SEQ, sizes):
    g = omp[omp.n == n].sort_values("threads")
    ax.plot(g.threads, g.cores_busy, color=c, marker="o", label=f"n = {n}")
ax.axhline(2, color=INK2, ls="--", lw=1)
ax.text(1, 2.05, "2 vCPUs", fontsize=7.5, color=INK2)
ax.set_xticks(threads)
ax.set_ylim(0, 2.4)
ax.set_xlabel("OpenMP threads p")
ax.set_ylabel("CPU time / wall time")
ax.set_title("Average number of busy cores during the kernel phase")
ax.legend(fontsize=7.5, ncol=3)
save(fig, "fig4_cores_busy.png")

# fig5: vectorisation 2x2
fig, ax = plt.subplots(figsize=(6.6, 3.5))
cfgs = [("scalar_omp_p1", "scalar, 1 thread", "#9db9de"), ("simd_omp_p1", "AVX2, 1 thread", "#2a78d6"),
        ("scalar_omp_p2", "scalar, 2 threads", "#f2b49a"), ("simd_omp_p2", "AVX2, 2 threads", "#eb6834")]
x = np.arange(len(Vt.index))
for i, (col, lab, c) in enumerate(cfgs):
    ax.bar(x + (i - 1.5) * 0.2, 2 * Vt.index.values.astype(float) ** 3 / Vt[col].values / 1e9, 0.19, color=c, label=lab)
ax.set_xticks(x)
ax.set_xticklabels([f"n = {n}" for n in Vt.index])
ax.set_ylabel("GFLOP/s (2n³ / median time)")
ax.set_title("Vectorisation and threads, measured separately (OpenMP code path, -O3 -mavx2)")
ax.legend(fontsize=7.5, ncol=2)
save(fig, "fig5_vectorisation.png")

# fig6: code placement
A = C[C.experiment == "A"]
fig, ax = plt.subplots(figsize=(6.0, 3.1))
labels, al, na = [], [], []
for n in sorted(A.n.unique()):
    for m, lab in (("seq", "seq"), ("tiles", "tiles"), ("omp", "omp p=1")):
        labels.append(f"{lab}\nn={n}")
        al.append(P[(P.n == n) & (P.method == m) & (P.threads == 1)].median_s.iloc[0] / seq[n])
        na.append(A[(A.n == n) & (A.method == m)].median_s.iloc[0] / seq[n])
x = np.arange(len(labels))
ax.bar(x - 0.2, na, 0.38, color="#c9c8c3", label="default code placement (build/mm_noalign)")
ax.bar(x + 0.2, al, 0.38, color="#2a78d6", label="loops aligned to 32 bytes (build/mm)")
ax.axhline(1, color=INK2, lw=0.8, ls="--")
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=7.5)
ax.set_ylabel("time / aligned sequential baseline")
ax.set_title("Same inner loop, different code placement (1 thread)")
ax.set_ylim(0, 2.05)
ax.legend(fontsize=7.5, loc="upper left")
save(fig, "fig6_code_placement.png")

# --------------------------- key numbers ----------------------------------------------
key = {}
for n in sizes:
    g = omp[omp.n == n].set_index("threads")
    key[f"n{n}"] = dict(seq_s=float(seq[n]), omp_p1_overhead_pct=float(100 * (g.median_s[1] / seq[n] - 1)),
                        tiles_overhead_pct=float(100 * (P[(P.n == n) & (P.method == "tiles")].median_s.iloc[0] / seq[n] - 1)),
                        speedup={int(p): float(g.speedup[p]) for p in g.index},
                        efficiency={int(p): float(g.efficiency[p]) for p in g.index},
                        best_p=int(g.speedup.idxmax()))
key["max_speedup"] = float(omp.speedup.max())
key["superlinear"] = omp[omp.speedup > omp.threads.clip(upper=2) * 1.0].loc[:, ["n", "threads", "speedup"]].to_dict("records")
key["non_ok_runs"] = runs[runs.status != "ok"][["experiment", "binary", "method", "threads", "n", "status"]].to_dict("records")
key["vectorisation"] = vec.to_dict("records")
(RES / "key_numbers.json").write_text(json.dumps(key, indent=2, default=float))
print(json.dumps({k: v for k, v in key.items() if k != "vectorisation"}, indent=1, default=float)[:4000])
