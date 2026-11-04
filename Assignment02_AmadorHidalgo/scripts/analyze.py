#!/usr/bin/env python3
"""Turn results/raw_measurements.csv + runs.csv into summary tables, figures and key numbers.

Statistic: median and inter-quartile range (linear-interpolation quartiles) of the
per-multiplication time of all measured samples of a configuration. Nothing is removed.

Outputs (results/):
  summary_tuning.csv, summary_dense.csv, summary_sparse.csv, key_numbers.json, figures/*.png
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gen_inputs import read_csr, tag  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RES = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "results"
DATA = ROOT / "data"
FIG = RES / "figures"
FIG.mkdir(exist_ok=True, parents=True)

# ---- style: validated categorical palette (blue, orange, aqua, violet), recessive grid ----
INK, INK2, SURFACE, GRID = "#0b0b0b", "#52514e", "#fcfcfb", "#e4e3df"
COL = {"ijk": "#2a78d6", "ijk_bt": "#eb6834", "ikj": "#1baf7a", "blocked": "#8a5cd6",
       "csr_spgemm": "#d6336c", "convert": "#d6336c"}
MARK = {"ijk": "o", "ijk_bt": "s", "ikj": "^", "blocked": "D", "csr_spgemm": "v"}
LABEL = {"ijk": "ijk (baseline, Assignment 1)", "ijk_bt": "ijk + transposed B", "ikj": "ikj (loop reordering)",
         "blocked": "blocked ikj (tiling)", "csr_spgemm": "CSR SpGEMM (Gustavson)"}
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 9,
    "axes.titlesize": 10, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "legend.frameon": False, "lines.linewidth": 2, "lines.markersize": 6,
    "lines.markeredgecolor": SURFACE,
})

raw = pd.read_csv(RES / "raw_measurements.csv")
runs = pd.read_csv(RES / "runs.csv")
base_rss = json.loads((RES / "baseline_rss.json").read_text())
KEYS = ["experiment", "method", "input", "structure", "n", "density_pct"]


def stats(t):
    t = np.asarray(t, dtype=float)
    q1, med, q3 = np.percentile(t, [25, 50, 75])
    return dict(samples=len(t), median_s=med, q1_s=q1, q3_s=q3, iqr_s=q3 - q1,
                iqr_pct=100 * (q3 - q1) / med if med else np.nan, min_s=t.min(), max_s=t.max())


def summarise(exp_filter):
    rows = []
    for _, r in runs[runs.experiment.isin(exp_filter)].iterrows():
        g = raw[(raw.experiment == r.experiment) & (raw.method == r.method) & (raw.input == r.input) &
                (raw.structure == r.structure) & (raw.n == r.n) & (np.isclose(raw.density_pct, r.density_pct))]
        if r.method == "blocked" or r.experiment == "T":
            g = g[g.block.astype(str) == str(r.block)]
        d = {k: r[k] for k in KEYS}
        d.update(block=r.block, status=r.status, detail=r.detail if isinstance(r.detail, str) else "",
                 peak_rss_mib=r.peak_rss_kb / 1024, planned_mib=r.planned_bytes / 2 ** 20,
                 nnz_A=r.nnz_A, nnz_B=r.nnz_B, nnz_C=r.nnz_C, warmup_runs=r.warmup_runs)
        m = g[g.phase == "multiply"]["ns_per_mult"] * 1e-9
        c = g[g.phase == "convert"]["ns_per_mult"] * 1e-9
        if len(m):
            d.update(stats(m))
            d["inner"] = int(g[g.phase == "multiply"]["inner"].iloc[0])
        if len(c):
            s = stats(c)
            d.update(convert_median_s=s["median_s"], convert_iqr_s=s["iqr_s"])
        rows.append(d)
    return pd.DataFrame(rows)


# ------------------------------- T: block tuning ---------------------------------------
tun = summarise(["T"])
tun["block"] = tun.block.astype(int)
tun.sort_values(["n", "block"]).to_csv(RES / "summary_tuning.csv", index=False, float_format="%.6g")
choice = json.loads((RES / "block_choice.json").read_text())
B = choice["chosen_block"]

# ------------------------------- D: dense ----------------------------------------------
den = summarise(["D"])
den["gflops"] = 2 * den.n.astype(float) ** 3 / den.median_s / 1e9
den["ns_per_iter"] = den.median_s / den.n.astype(float) ** 3 * 1e9
bl = den[den.method == "ijk"].set_index("n")["median_s"]
den["speedup_vs_ijk"] = [bl.get(n, np.nan) / t if t == t else np.nan for n, t in zip(den.n, den.median_s)]
den["rss_minus_baseline_mib"] = den.peak_rss_mib - base_rss["dense"] / 1024
den["array_storage_mib"] = [(4 if m == "ijk_bt" else 3) * 8 * n * n / 2 ** 20 for m, n in zip(den.method, den.n)]
order = {m: i for i, m in enumerate(["ijk", "ijk_bt", "ikj", "blocked"])}
den = den.sort_values(["n", "method"], key=lambda s: s.map(order) if s.name == "method" else s)
den.to_csv(RES / "summary_dense.csv", index=False, float_format="%.6g")

# ------------------------------- sparse experiments ------------------------------------
spa = summarise(["M", "S", "L"])


def mult_adds(structure, n, d):
    """Scalar multiply-adds of Gustavson: sum over nonzeros A[i,k] of nnz(B[k,:])."""
    t = tag(structure, n, d)
    _, _, ca, _ = read_csr(DATA / "sparse" / f"{t}_A.csr")
    _, rpb, _, _ = read_csr(DATA / "sparse" / f"{t}_B.csr")
    return int(np.diff(rpb)[ca].sum())


ma = {}
for (s_, n_, d_) in spa[["structure", "n", "density_pct"]].drop_duplicates().itertuples(index=False):
    ma[(s_, n_, d_)] = mult_adds(s_, int(n_), d_)
spa["mult_adds"] = [ma[(s_, n_, d_)] if m == "csr_spgemm" else float(n_) ** 3
                    for m, s_, n_, d_ in zip(spa.method, spa.structure, spa.n, spa.density_pct)]
nn = spa.n.astype(float)
spa["density_C_pct"] = 100 * spa.nnz_C / nn ** 2
# expected output density for independent Bernoulli(d) patterns (uniform structure only)
spa["pred_density_C_pct"] = np.where(spa.structure == "uniform",
                                     100 * (1 - (1 - (spa.density_pct / 100) ** 2) ** nn), np.nan)
spa["ns_per_mult_add"] = spa.median_s / spa.mult_adds * 1e9   # dense rows: per n^3 multiply-add
spa["csr_storage_mib"] = (3 * 8 * (nn + 1) + 12 * (spa.nnz_A + spa.nnz_B + spa.nnz_C)) / 2 ** 20
spa["dense_storage_mib"] = 3 * 8 * nn ** 2 / 2 ** 20
spa["rss_minus_baseline_mib"] = spa.peak_rss_mib - np.where(spa.method == "csr_spgemm", base_rss["sparse"],
                                                            base_rss["dense"]) / 1024
# speed-up of the sparse multiply (csr-input run) over dense blocked on the same matrices
dref = spa[(spa.method == "blocked") & (spa.status == "ok")].set_index(["experiment", "structure", "n", "density_pct"])["median_s"]
conv = spa[(spa.method == "csr_spgemm") & (spa.input == "dense")].set_index(["experiment", "structure", "n", "density_pct"])
spa["dense_blocked_s"] = [dref.get((e, s_, n_, d_), np.nan) for e, s_, n_, d_ in
                          zip(spa.experiment, spa.structure, spa.n, spa.density_pct)]
spa["convert_s_from_dense_run"] = [conv["convert_median_s"].get((e, s_, n_, d_), np.nan) if len(conv) else np.nan
                                   for e, s_, n_, d_ in zip(spa.experiment, spa.structure, spa.n, spa.density_pct)]
is_sp = (spa.method == "csr_spgemm") & (spa.input == "csr")
spa["speedup_vs_dense_blocked"] = np.where(is_sp, spa.dense_blocked_s / spa.median_s, np.nan)
spa["speedup_incl_convert"] = np.where(is_sp, spa.dense_blocked_s / (spa.median_s + spa.convert_s_from_dense_run),
                                       np.nan)
spa = spa.sort_values(["experiment", "structure", "n", "density_pct", "method", "input"],
                      ascending=[True, True, True, False, True, True])
spa.to_csv(RES / "summary_sparse.csv", index=False, float_format="%.6g")


# ------------------------------- figures -----------------------------------------------
def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG / name, dpi=200)
    plt.close(fig)


# fig1: dense runtime vs n
fig, ax = plt.subplots(figsize=(6.4, 3.9))
for m in ["ijk", "ijk_bt", "ikj", "blocked"]:
    g = den[(den.method == m) & (den.status == "ok")]
    ax.errorbar(g.n, g.median_s, yerr=[g.median_s - g.q1_s, g.q3_s - g.median_s], color=COL[m], marker=MARK[m],
                label=LABEL[m] + (f", block {B}" if m == "blocked" else ""), capsize=2)
    lim = den[(den.method == m) & (den.status.isin(["time_limit", "wall_limit"]))]
    for n in lim.n:
        ax.scatter([n], [180], marker="x", color=COL[m], s=50, zorder=5)
ax.axhline(180, color=INK2, lw=0.8, ls="--")
ax.text(den.n.min(), 200, "180 s limit per multiplication (x = stopped at the limit)", color=INK2, fontsize=8)
ax.set_xscale("log", base=2)
ax.set_yscale("log")
ax.set_xticks(sorted(den.n.unique()))
ax.set_xticklabels(sorted(den.n.unique()), rotation=0)
ax.set_xlabel("matrix size n")
ax.set_ylabel("time per multiplication (s), median, IQR bars")
ax.set_title("Dense methods: runtime against n (C, gcc -O2, float64, one thread)")
ax.legend(fontsize=8)
save(fig, "fig1_dense_runtime.png")

# fig2: GFLOP/s
fig, ax = plt.subplots(figsize=(6.4, 3.6))
for m in ["ijk", "ijk_bt", "ikj", "blocked"]:
    g = den[(den.method == m) & (den.status == "ok")]
    ax.plot(g.n, g.gflops, color=COL[m], marker=MARK[m], label=LABEL[m])
ax.set_xscale("log", base=2)
ax.set_xticks(sorted(den.n.unique()))
ax.set_xticklabels(sorted(den.n.unique()))
ax.set_ylim(bottom=0)
ax.set_xlabel("matrix size n")
ax.set_ylabel("GFLOP/s  (2n³ / median time)")
ax.set_title("Dense methods: achieved floating-point rate")
ax.legend(fontsize=8)
save(fig, "fig2_dense_gflops.png")

# fig3: block-size tuning
fig, ax = plt.subplots(figsize=(6.0, 3.2))
for n, c in zip(sorted(tun.n.unique()), ["#2a78d6", "#eb6834"]):
    g = tun[tun.n == n].sort_values("block")
    ax.errorbar(g.block, g.median_s, yerr=[g.median_s - g.q1_s, g.q3_s - g.median_s], color=c, marker="o",
                label=f"n = {n}", capsize=2)
ax.set_xscale("log", base=2)
ax.set_xticks(sorted(tun.block.unique()))
ax.set_xticklabels(sorted(tun.block.unique()))
ax.set_yscale("log")
ax.set_xlabel("tile side (doubles)")
ax.set_ylabel("time per multiplication (s)")
ax.set_title(f"Block-size tuning for blocked ikj (chosen: {B})")
ax.legend(fontsize=8)
save(fig, "fig3_block_tuning.png")

# fig4: sparse vs dense against density (M)
M = spa[spa.experiment == "M"]
fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.6), sharey=True)
for ax, n in zip(axes, [1024, 2048]):
    g = M[M.n == n]
    s = g[(g.method == "csr_spgemm") & (g.input == "csr")].sort_values("density_pct")
    ax.plot(s.density_pct, s.median_s, color=COL["csr_spgemm"], marker="v", label="CSR SpGEMM (multiply only)")
    ax.plot(s.density_pct, s.median_s + s.convert_s_from_dense_run, color=COL["csr_spgemm"], ls="--", marker="v",
            markerfacecolor=SURFACE, label="CSR SpGEMM + dense→CSR conversion")
    for m in ["blocked", "ijk"]:
        d = g[(g.method == m) & (g.status == "ok")].sort_values("density_pct")
        if len(d):
            ax.plot(d.density_pct, d.median_s, color=COL[m], marker=MARK[m], label=LABEL[m])
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("density of A and B (% nonzero)")
    ax.set_title(f"n = {n}, uniform random sparsity")
axes[0].set_ylabel("time per multiplication (s)")
axes[0].legend(fontsize=7, loc="lower right")
save(fig, "fig4_sparse_vs_dense.png")

# fig5: output density (fill-in)
fig, ax = plt.subplots(figsize=(6.0, 3.4))
for n, c in zip([1024, 2048], ["#2a78d6", "#eb6834"]):
    s = M[(M.n == n) & (M.method == "csr_spgemm") & (M.input == "csr")].sort_values("density_pct")
    ax.plot(s.density_pct, s.density_C_pct, color=c, marker="o", label=f"observed, n = {n}")
    ax.plot(s.density_pct, s.pred_density_C_pct, color=c, ls=":", label=f"1-(1-d²)ⁿ, n = {n}")
ax.plot([0.1, 50], [0.1, 50], color=INK2, lw=0.8, ls="--", label="output density = input density")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("density of A and B (%)")
ax.set_ylabel("density of C = AB (%)")
ax.set_title("Fill-in: the product is much denser than its factors")
ax.legend(fontsize=7)
save(fig, "fig5_fill_in.png")

# fig6: memory vs density, n = 2048
fig, ax = plt.subplots(figsize=(6.0, 3.4))
g = M[M.n == 2048]
s = g[(g.method == "csr_spgemm") & (g.input == "csr")].sort_values("density_pct")
d = g[g.method == "blocked"].sort_values("density_pct")
ax.plot(s.density_pct, s.peak_rss_mib, color=COL["csr_spgemm"], marker="v", label="CSR SpGEMM: peak RSS")
ax.plot(s.density_pct, s.csr_storage_mib, color=COL["csr_spgemm"], ls=":", label="CSR SpGEMM: A, B, C storage")
ax.plot(d.density_pct, d.peak_rss_mib, color=COL["blocked"], marker="D", label="dense blocked: peak RSS")
ax.axhline(3 * 8 * 2048 ** 2 / 2 ** 20, color=COL["blocked"], ls=":", label="dense: 3 × 8n² bytes")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("density of A and B (%)")
ax.set_ylabel("MiB")
ax.set_title("Memory at n = 2048")
ax.legend(fontsize=7)
save(fig, "fig6_memory.png")

# fig7: structure
S = spa[(spa.experiment == "S") & (spa.method == "csr_spgemm")]
fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.2))
structs = ["uniform", "banded", "skewed"]
x = np.arange(len(structs))
for i, (d_, c) in enumerate(zip([5, 1], ["#2a78d6", "#eb6834"])):
    g = S[np.isclose(S.density_pct, d_)].set_index("structure").reindex(structs)
    axes[0].bar(x + (i - 0.5) * 0.38, g.median_s * 1e3, width=0.36, color=c, label=f"density {d_} %")
    axes[1].bar(x + (i - 0.5) * 0.38, g.density_C_pct, width=0.36, color=c, label=f"density {d_} %")
for ax, yl, tt in ((axes[0], "SpGEMM time (ms)", "Same density, different structure"),
                   (axes[1], "density of C (%)", "Output density")):
    ax.set_xticks(x)
    ax.set_xticklabels(structs)
    ax.set_ylabel(yl)
    ax.set_title(tt)
    ax.set_yscale("log")
axes[0].legend(fontsize=7)
save(fig, "fig7_structure.png")

# fig8: large sparse
L = spa[spa.experiment == "L"]
fig, ax = plt.subplots(figsize=(6.4, 3.6))
s = L[(L.method == "csr_spgemm") & (L.status == "ok")].sort_values("n")
ax.plot(s.n, s.median_s, color=COL["csr_spgemm"], marker="v", label="CSR SpGEMM (16 nonzeros/row)")
d = L[(L.method == "blocked") & (L.status == "ok")]
ax.plot(d.n, d.median_s, color=COL["blocked"], marker="D", label="dense blocked, same matrices")
for _, r in L[(L.method == "blocked") & (L.status.isin(["time_limit", "wall_limit"]))].iterrows():
    ax.scatter([r.n], [180], marker="x", color=COL["blocked"], s=50, zorder=5, label="dense blocked stopped at the limit")
na = sorted(L[(L.method == "blocked") & (L.status == "not_attempted")].n)
if na:
    ax.text(na[0], 25, f"dense not attempted for n ≥ {na[0]}\n(n = {int(na[0]) // 2} already hit the time limit;\n"
            f"n = {na[0]} would also need {3 * 8 * na[0] ** 2 / 2 ** 30:.0f} GiB > 4 GiB budget)",
            fontsize=7, color=INK2, va="center")
ax.axhline(180, color=INK2, lw=0.8, ls="--")
ax.set_xscale("log", base=2)
ax.set_yscale("log")
ax.set_xlabel("matrix size n")
ax.set_ylabel("time per multiplication (s)")
ax.set_title("Large sparse matrices: sizes where dense storage is no longer feasible")
ax.set_xticks(sorted(L.n.unique()))
ax.set_xticklabels([str(int(v)) for v in sorted(L.n.unique())])
ax.legend(fontsize=8, loc="lower right")
save(fig, "fig8_large_sparse.png")

# ------------------------------- key numbers -------------------------------------------
key = {"block": B}
ok = den[den.status == "ok"]
fit = {}
for m in ["ijk", "ijk_bt", "ikj", "blocked"]:
    g = ok[(ok.method == m) & (ok.n >= 512)]
    if len(g) >= 3:
        fit[m] = float(np.polyfit(np.log(g.n), np.log(g.median_s), 1)[0])
key["dense_fitted_exponent_n_ge_512"] = fit
key["largest_ok_dense"] = {m: int(ok[ok.method == m].n.max()) for m in order if (ok.method == m).any()}
key["dense_limits"] = den[den.status != "ok"][["method", "n", "status"]].to_dict("records")
sl = spa[(spa.method == "csr_spgemm") & (spa.input == "csr")]
key["largest_ok_sparse_n"] = int(sl[sl.status == "ok"].n.max())
# crossover: lowest density at which dense blocked beats SpGEMM (multiply only, and incl. conversion)
cross = {}
for n in [1024, 2048]:
    g = sl[(sl.experiment == "M") & (sl.n == n)].sort_values("density_pct")
    cross[n] = dict(sparse_faster_at=[float(x) for x in g[g.speedup_vs_dense_blocked > 1].density_pct],
                    dense_faster_at=[float(x) for x in g[g.speedup_vs_dense_blocked <= 1].density_pct],
                    sparse_faster_incl_convert_at=[float(x) for x in g[g.speedup_incl_convert > 1].density_pct])
key["crossover"] = cross
X = summarise(["X"])
if len(X):
    X[["method", "n", "status", "samples", "median_s", "iqr_pct"]].to_csv(RES / "summary_crosscheck.csv", index=False,
                                                                     float_format="%.6g")
    key["crosscheck_a1_binary"] = {f"{r.method}@{int(r.n)}": float(r.median_s) for r in X.itertuples()}
key["dense_ns_per_mult_add_blocked"] = {int(r.n): float(r.median_s / r.n ** 3 * 1e9) for r in
                                        den[(den.method == "blocked") & (den.status == "ok")].itertuples()}
(RES / "key_numbers.json").write_text(json.dumps(key, indent=2, default=float))
print(json.dumps(key, indent=2, default=float))
