#!/usr/bin/env python3
"""Recompute statistics and export publication figures as vector PDF and 400 dpi PNG."""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np

from analyze_results import PAPER, LABELS, METHODS, run_analysis

COLORS = dict(zip(METHODS, ["#777777", "#009E73", "#CC79A7", "#D55E00",
                           "#0072B2", "#E69F00", "#111111"]))
MARKERS = dict(zip(METHODS, ["o", "s", "^", "v", "D", "P", "*"]))
OUT = PAPER / "figures"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8, "axes.labelsize": 8,
    "axes.titlesize": 9, "legend.fontsize": 7, "xtick.labelsize": 7,
    "ytick.labelsize": 7, "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": .6, "lines.linewidth": 1.25, "lines.markersize": 4,
    "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 400,
})


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight", pad_inches=.04,
                metadata={"Creator": "Cosmic-Net paper/scripts/generate_figures.py",
                          "CreationDate": None, "ModDate": None})
    fig.savefig(OUT / f"{name}.png", bbox_inches="tight", pad_inches=.04,
                metadata={"Software": "Cosmic-Net paper/scripts/generate_figures.py"})
    plt.close(fig)


def style(ax):
    ax.grid(axis="y", color=".9", linewidth=.5)
    ax.set_axisbelow(True)


def metric(data, method, q):
    return next(x for x in data["metrics"] if x["method"] == method and x["q"] == q)


def curves(data, degradation=False):
    fig, ax = plt.subplots(figsize=(3.5, 2.65))
    for method in METHODS:
        rows = [metric(data, method, q) for q in (.4, .6, .8, 1.)]
        y = [r["rmse"] - (metric(data, "full", r["q"])["rmse"] if degradation else 0) for r in rows]
        ax.plot([100*r["physical_retention"] for r in rows], y,
                color=COLORS[method], marker=MARKERS[method],
                markersize=5 if method == "provided_rl" else 3.8,
                linewidth=1.7 if method == "provided_rl" else 1., label=LABELS[method])
    ax.axhline(0 if degradation else metric(data, "full", .4)["rmse"],
               color=".35", linestyle="--", linewidth=1., label="Full graph (100%)")
    ax.set(xlabel="Mean realized physical-pair retention (%)",
           ylabel="RMSE increase from full graph (dex)" if degradation else "Test RMSE (dex)",
           xlim=(38, 102), xticks=[40, 60, 80, 100])
    style(ax)
    ax.legend(loc="upper center", bbox_to_anchor=(.5, -.29), ncol=2,
              frameon=False, columnspacing=.9, handlelength=1.8)
    save(fig, "figure_3_degradation" if degradation else "figure_1_retention_performance")


def matched(data):
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.65), sharey=True)
    order = ["random", "distance", "degree", "stellar_binding_proxy",
             "edge_feature_grad_x_input", "supervised_raw", "provided_rl"]
    for ax, q in zip(axes, [.4, .6, .8]):
        rl = metric(data, "provided_rl", q)
        ax.axvline(rl["rmse"], color=".5", linestyle=":", linewidth=1)
        for i, method in enumerate(order):
            ax.scatter(metric(data, method, q)["rmse"], i, color=COLORS[method],
                       marker=MARKERS[method], s=40 if method == "provided_rl" else 22,
                       zorder=3)
        ax.set_title(f"Nominal {q:.0%}\nRealized {100*rl['physical_retention']:.2f}%", fontsize=8)
        ax.set_xlabel("Test RMSE (dex)")
        ax.grid(axis="x", color=".9", linewidth=.5)
        ax.xaxis.set_major_locator(MaxNLocator(4))
        ax.set_axisbelow(True)
    axes[0].set_yticks(range(len(order)), [LABELS[m] for m in order])
    axes[0].invert_yaxis()
    fig.subplots_adjust(wspace=.18)
    save(fig, "figure_2_matched_budgets")


def latency(data):
    fig, (ax, tta) = plt.subplots(1, 2, figsize=(7.1, 2.65), gridspec_kw={"width_ratios": [1.15, 1]})
    for method in ["full"] + METHODS:
        m = metric(data, method, .4)
        ax.scatter(m["latency_s"], m["rmse"], c=COLORS.get(method, "#56B4E9"),
                   marker=MARKERS.get(method, "X"), s=36 if method in ("full", "provided_rl") else 24,
                   label=LABELS[method], zorder=3)
    ax.set(xscale="log", xlabel="Mean inference time (s/halo; log scale)", ylabel="Test RMSE (dex)")
    ax.set_title("(a) C: nominal 40% + full reference", loc="left", fontsize=8)
    ax.set_ylim(.105, .22)
    ax.legend(loc="upper center", bbox_to_anchor=(1.0, -.30), ncol=4,
              frameon=False, columnspacing=1., handletextpad=.4)
    ds = data["adaptation"]
    tta.plot([x["latency_s"] for x in ds], [x["rmse"] for x in ds],
             color="#0072B2", marker="o", markersize=4)
    for r in ds:
        if r["K"] in (0, 5):
            tta.annotate(f"K={r['K']}", (r["latency_s"], r["rmse"]),
                         xytext=(5, 6), textcoords="offset points", fontsize=7)
    selected = ds[-1]
    tta.scatter(selected["latency_s"], selected["rmse"], color="black", marker="*", s=50, zorder=4)
    tta.annotate("K=10, 20\n20 selected on validation", (selected["latency_s"], selected["rmse"]),
                 xytext=(-4, 17), textcoords="offset points", ha="right", fontsize=6.8)
    tta.set(xlabel="Mean inference time (s/halo)", ylabel="Test RMSE (dex)",
            xlim=(.39, 1.29), ylim=(.13025, .13135))
    tta.ticklabel_format(axis="y", useOffset=False)
    tta.set_title("(b) D: nominal 50% adaptation", loc="left", fontsize=8)
    for a in (ax, tta):
        style(a)
    fig.subplots_adjust(wspace=.4)
    save(fig, "figure_4_accuracy_latency")


def halo_differences(data):
    with (PAPER / "evidence/per-halo-error-differences.csv").open() as f:
        rows = list(csv.DictReader(f))
    controls = ["edge_feature_grad_x_input", "random", "supervised_raw"]
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.6), sharex=True, sharey=True)
    for ax, method in zip(axes, controls):
        chosen = [r for r in rows if float(r["q"]) == .4 and r["control"] == method]
        values = np.array([float(r["delta_absolute_error_dex"]) for r in chosen])
        xs = np.arange(1, len(values)+1)
        ax.axhline(0, color=".4", linewidth=.8)
        ax.vlines(xs, 0, values, colors=np.where(values < 0, "#0072B2", "#D55E00"), alpha=.65, linewidth=.6)
        ax.scatter(xs, values, c=np.where(values < 0, "#0072B2", "#D55E00"), s=7, zorder=3)
        ax.set_title(f"RL − {LABELS[method]}\nBetter on {(values<0).sum()}/82 halos", fontsize=8)
        ax.set_xlabel("Halo rank (sorted graph ID)")
        ax.set_xticks([1, 40, 82])
        style(ax)
    axes[0].set_ylabel("Paired absolute-error difference (dex)")
    fig.subplots_adjust(wspace=.12)
    save(fig, "figure_5_per_halo_differences")


def confidence_intervals(data):
    order = ["edge_feature_grad_x_input", "random", "supervised_raw", "degree",
             "distance", "stellar_binding_proxy", "full"]
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.8), sharex=True, sharey=True)
    for ax, q in zip(axes, [.4, .6, .8]):
        ax.axvline(0, color=".5", linestyle="--", linewidth=.8)
        for i, method in enumerate(order):
            x = next(r for r in data["contrasts"] if r["q"] == q and r["control"] == method)
            color = COLORS.get(method, "#56B4E9")
            ax.errorbar(x["delta_rmse"], i, xerr=[[x["delta_rmse"]-x["ci_low"]],
                                                       [x["ci_high"]-x["delta_rmse"]]],
                        fmt="o", color=color, capsize=2, markersize=3.5, linewidth=1.)
        ax.set_title(f"Nominal {q:.0%}", fontsize=8)
        ax.set_xlabel("RL − control RMSE (dex)")
        ax.set_xlim(-.105, .045)
        ax.set_xticks([-.10, -.05, 0])
        ax.grid(axis="x", color=".92", linewidth=.5)
    axes[0].set_yticks(range(len(order)), [LABELS[m] + ("†" if m == "full" else "") for m in order])
    axes[0].invert_yaxis()
    fig.subplots_adjust(wspace=.15)
    save(fig, "figure_6_paired_confidence_intervals")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    data = run_analysis()
    curves(data)
    matched(data)
    curves(data, degradation=True)
    latency(data)
    halo_differences(data)
    confidence_intervals(data)
    print("Exported six figures as PDF + 400 dpi PNG:", OUT)


if __name__ == "__main__":
    main()
