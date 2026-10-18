#!/usr/bin/env python3
"""Turn results/raw_measurements.csv + runs.csv into summary tables and figures.

Outputs (results/): summary.csv (main study), summary_s1.csv (supplementary C flags),
scaling.json (fitted exponents, ratios), figures/*.png
Statistic: median and inter-quartile range (IQR, linear-interpolation quartiles) of the
per-multiplication time of all measured samples; nothing is removed.
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
FIG.mkdir(exist_ok=True, parents=True)

# ---- style: validated categorical slots 1-3 (blue, orange, aqua), recessive grid, text in ink ----
INK, INK2, SURFACE, GRID = "#0b0b0b", "#52514e", "#fcfcfb", "#e4e3df"
COL = {"C-O2": "#2a78d6", "Java": "#eb6834", "Python": "#1baf7a"}
MARK = {"C-O2": "o", "Java": "s", "Python": "^"}
LABEL = {"C-O2": "C (gcc -O2)", "Java": "Java (OpenJDK, JIT)", "Python": "Python (CPython)"}
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
runs = runs[runs["study"].isin(["main", "S1", "S2"])].copy()
base = json.loads((RES / "baseline_rss.json").read_text())      # kB, keyed by variant


def q(x, p):
    return float(np.percentile(x, p))


def summarise(study):
    rows = []
    for (variant, n), g in raw[raw.study == study].groupby(["variant", "n"]):
        t = g["time_s_per_multiplication"].to_numpy()
        r = runs[(runs.study == study) & (runs.variant == variant) & (runs.n == n)].iloc[0]
        med = float(np.median(t))
        rss_mb = float(r["peak_rss_kb"]) / 1024          # MiB
        b = base.get("C-O2" if variant.startswith("C-") else variant, 0) / 1024
        rows.append(dict(
            study=study, variant=variant, language=r["language"], n=int(n), status=r["status"],
            samples=len(t), inner=int(g["inner_multiplications"].iloc[0]),
            # The programs print WARMUP = explicit warm-up runs only. When none ran (CPython, n >= 512)
            # they perform ONE untimed calibration multiplication before sampling (batch sizing);
            # it is effectively a warm-up run, so it is counted here.
            warmup_recorded=int(r["warmup_multiplications"]),
            warmup_mults=max(int(r["warmup_multiplications"]), 1),
            median_s=med, q1_s=q(t, 25), q3_s=q(t, 75), iqr_s=q(t, 75) - q(t, 25),
            iqr_pct=100 * (q(t, 75) - q(t, 25)) / med, min_s=float(t.min()), max_s=float(t.max()),
            ns_per_iter=med / n ** 3 * 1e9,
            peak_rss_mb=rss_mb, runtime_baseline_mb=b, rss_minus_baseline_mb=rss_mb - b,
            array_storage_mb=24 * n * n / 2 ** 20,   # MiB: 3 matrices x 8 B
            validated=int(r["validated"]) if r["validated"] == r["validated"] else -1,
            note=r["note"] if isinstance(r["note"], str) else ""))
    return pd.DataFrame(rows).sort_values(["variant", "n"])


main = summarise("main")
main.to_csv(RES / "summary.csv", index=False, float_format="%.6g")
s1 = summarise("S1") if (raw.study == "S1").any() else pd.DataFrame()
if len(s1):
    s1.to_csv(RES / "summary_s1.csv", index=False, float_format="%.6g")
s2 = summarise("S2") if (raw.study == "S2").any() else pd.DataFrame()
if len(s2):
    s2.to_csv(RES / "summary_s2.csv", index=False, float_format="%.6g")
assert (main["validated"] == 1).all(), "some runs failed validation"

# ---- scaling analysis ---------------------------------------------------------------------
scaling = {"fit": {}, "ratios": {}}
okm = main[main.status == "ok"]
for v, g in okm.groupby("variant"):
    gg = g[g.n >= 128]
    if len(gg) >= 3:
        slope, intercept = np.polyfit(np.log(gg.n), np.log(gg.median_s), 1)
        scaling["fit"][v] = dict(exponent=float(slope), n_range=[int(gg.n.min()), int(gg.n.max())], points=len(gg))
piv = okm.pivot(index="n", columns="variant", values="median_s")
for a, b in [("Java", "C-O2"), ("Python", "C-O2"), ("Python", "Java")]:
    if a in piv and b in piv:
        r = (piv[a] / piv[b]).dropna()
        scaling["ratios"][f"{a}/{b}"] = {int(n): float(x) for n, x in r.items()}
json.dump(scaling, open(RES / "scaling.json", "w"), indent=2)


# ---- figures --------------------------------------------------------------------------------
def end_label(ax, x, y, text, color_key, dy=0):
    ax.annotate(text, (x, y), xytext=(6, dy), textcoords="offset points", va="center", fontsize=8,
                color=INK, annotation_clip=False)


def fig_runtime():
    fig, ax = plt.subplots(figsize=(6.4, 3.9))
    for v in ["C-O2", "Java", "Python"]:
        g = main[(main.variant == v) & (main.status == "ok")]
        if g.empty:
            continue
        ax.errorbar(g.n, g.median_s, yerr=[g.median_s - g.q1_s, g.q3_s - g.median_s], color=COL[v], marker=MARK[v],
                    capsize=2, label=LABEL[v], elinewidth=1)
        o = main[(main.variant == v) & (main.status == "over_limit")]
        if len(o):
            ax.scatter(o.n, o.median_s, marker="x", s=55, color=COL[v], zorder=5, linewidths=2)
            ax.annotate("exceeded 180 s limit\n(single run)", (o.n.iloc[0], o.median_s.iloc[0]), xytext=(-8, 8),
                        textcoords="offset points", ha="right", fontsize=7.5, color=INK2)
    # cubic guide anchored at the C median for n = 256
    c = main[(main.variant == "C-O2") & (main.n == 256) & (main.status == "ok")]
    if len(c):
        xs = np.array([16, 2048])
        ax.plot(xs, c.median_s.iloc[0] * (xs / 256.0) ** 3, ls="--", lw=1.2, color=INK2, label=r"$n^3$ guide (anchored at C, $n$=256)")
    for _, r_ in runs[(runs.study == "main") & runs.status.isin(["timeout", "over_limit"]) & (runs.variant == "Python")].iterrows():
        ax.text(r_.n, 2.2e2, f"Python n={int(r_.n)}: no result\n({r_.status.replace('_', ' ')}, budget exceeded)", ha="center", va="bottom",
                fontsize=7.5, color=INK2)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(sorted(main.n.unique()))
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.tick_params(axis="x", labelsize=7.5, rotation=0)
    ax.set_xlabel("matrix size n (n × n, float64)")
    ax.set_ylabel("time per multiplication (s)")
    ax.set_title("Runtime of the basic i-j-k multiplication (median, bars = IQR)")
    ax.set_ylim(top=1.2e3)
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "fig1_runtime.png", dpi=200)
    plt.close(fig)


def fig_cost():
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    for v in ["C-O2", "Java", "Python"]:
        g = main[(main.variant == v)]
        if g.empty:
            continue
        ax.plot(g.n, g.ns_per_iter, color=COL[v], marker=MARK[v], label=LABEL[v])
        last = g.iloc[-1]
        end_label(ax, last.n, last.ns_per_iter, v.replace("-O2", ""), v)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(sorted(main.n.unique()))
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.tick_params(axis="x", labelsize=7.5)
    ax.set_xlabel("matrix size n")
    ax.set_ylabel("median time / n³  (ns per inner-loop iteration)")
    ax.set_title("Cost per multiply-add: flat = cubic model holds")
    ax.legend(loc="center left", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "fig2_cost_per_iteration.png", dpi=200)
    plt.close(fig)


def fig_memory():
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    for v in ["C-O2", "Java", "Python"]:
        g = main[(main.variant == v) & main.peak_rss_mb.notna()]
        ax.plot(g.n, g.peak_rss_mb, color=COL[v], marker=MARK[v], label=LABEL[v])
    xs = np.array([16, 2048])
    ax.plot(xs, 24 * xs ** 2 / 2 ** 20, ls="--", lw=1.2, color=INK2, label="array storage: 3 × 8·n² B (ideal)")
    ax.plot(xs, 96 * xs ** 2 / 2 ** 20, ls=":", lw=1.4, color=INK2, label="3 × n² boxed floats (32 B each)")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(sorted(main.n.unique()))
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.tick_params(axis="x", labelsize=7.5)
    ax.set_xlabel("matrix size n")
    ax.set_ylabel("peak resident set size (MiB)")
    ax.set_title("Peak process memory (runtime baseline included)")
    ax.legend(loc="upper left", fontsize=7.5)
    fig.tight_layout()
    fig.savefig(FIG / "fig3_memory.png", dpi=200)
    plt.close(fig)


def fig_s1():
    if s1.empty:
        return
    shade = {"C-O0": "#9ec5f0", "C-O2": "#2a78d6", "C-O3": "#123f7a"}
    mk = {"C-O0": "o", "C-O2": "s", "C-O3": "^"}
    fig, ax = plt.subplots(figsize=(6.0, 3.4))
    for v in ["C-O0", "C-O2", "C-O3"]:
        g = s1[(s1.variant == v) & (s1.status == "ok")]
        if g.empty:
            continue
        ax.errorbar(g.n, g.median_s, yerr=[g.median_s - g.q1_s, g.q3_s - g.median_s], color=shade[v], marker=mk[v],
                    capsize=2, elinewidth=1, label="gcc " + v.replace("C", ""), markeredgecolor=INK2)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(sorted(s1.n.unique()))
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel("matrix size n")
    ax.set_ylabel("time per multiplication (s)")
    ax.set_title("Supplementary S1: same C source, different gcc optimisation level")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "fig4_s1_c_flags.png", dpi=200)
    plt.close(fig)


def fig_s2():
    if s2.empty:
        return
    sizes = [1000, 1024, 1025]
    fig, ax = plt.subplots(figsize=(6.0, 3.3))
    w = 0.36
    for off, v in zip([-w / 2, w / 2], ["C-O2", "Java"]):
        g = s2[s2.variant == v].set_index("n").reindex(sizes)
        bars = ax.bar(np.arange(3) + off, g.ns_per_iter, w, color=COL[v], label=LABEL[v], edgecolor=SURFACE, linewidth=2)
        for b_, val in zip(bars, g.ns_per_iter):
            ax.text(b_.get_x() + b_.get_width() / 2, val + 0.1, f"{val:.1f}", ha="center", va="bottom", fontsize=8, color=INK)
    ax.set_xticks(range(3))
    ax.set_xticklabels([f"n = {n}" for n in sizes])
    ax.set_ylabel("ns per inner-loop iteration")
    ax.set_title("Supplementary S2: neighbouring sizes, very different cost per iteration")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "fig5_s2_neighbouring_sizes.png", dpi=200)
    plt.close(fig)


fig_runtime()
fig_cost()
fig_memory()
fig_s1()
fig_s2()
print("summary.csv:", len(main), "rows; figures written to", FIG)
print(json.dumps(scaling["fit"], indent=1))
