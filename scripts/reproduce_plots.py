"""Regenerate every figure in the README and paper from saved results.

This script does not train models and does not touch the N-body data. It reads
the JSON evidence in experiments/results and the certificates in models, then
writes publication figures to assets/figures.

Usage:
    python scripts/reproduce_plots.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "experiments" / "results"
FIGS = ROOT / "assets" / "figures"

COLORS = {
    "termgraph": "#1f5f8b",
    "ebm": "#c44e52",
    "nam": "#8c8c8c",
    "gbm": "#2e8b57",
    "edit": "#6a51a3",
    "grid": "#d9d9d9",
}


def load(name: str) -> dict:
    return json.loads((RESULTS / name).read_text())


def style():
    plt.rcParams.update({
        "font.size": 8.5,
        "axes.titlesize": 9.5,
        "axes.labelsize": 8.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": COLORS["grid"],
        "grid.linewidth": 0.6,
        "figure.dpi": 120,
        "savefig.dpi": 220,
        "savefig.bbox": "tight",
    })


def save(fig, name: str):
    path = FIGS / name
    fig.savefig(path)
    plt.close(fig)
    print("wrote", path)


def architecture():
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5.2)
    ax.axis("off")

    def box(x, y, w, h, text, color, fontsize=8.4):
        p = FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.06,rounding_size=0.12",
            linewidth=1.0, edgecolor="#333333", facecolor=color,
        )
        ax.add_patch(p)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=fontsize, color="#111111", linespacing=1.25)

    def arrow(x1, y1, x2, y2):
        ax.add_patch(FancyArrowPatch(
            (x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=11,
            linewidth=1.0, color="#333333",
        ))

    w, h = 3.3, 0.95
    xs = [0.25, 4.25, 8.25]
    y1, y2, y3 = 3.85, 2.35, 0.85

    box(xs[0], y1, w, h, "GBM teacher\nXGBoost / LightGBM", "#dce9f5")
    box(xs[1], y1, w, h, "Compiler\nEBM + order-3 XGBoost", "#dce9f5")
    box(xs[2], y1, w, h, "Term graph + certificate\nlow-order tables", "#dce9f5")
    arrow(xs[0] + w, y1 + h / 2, xs[1], y1 + h / 2)
    arrow(xs[1] + w, y1 + h / 2, xs[2], y1 + h / 2)

    box(xs[0], y2, w, h, "Exact term ledger\n$\\phi_j = g_j$", "#e8f0e8")
    box(xs[1], y2, w, h, "Feature sharing W\n$\\psi = W\\phi$", "#e8f0e8")
    box(xs[2], y2, w, h, "SHAP-compatible\nschema", "#e8f0e8")
    arrow(xs[0] + w, y2 + h / 2, xs[1], y2 + h / 2)
    arrow(xs[1] + w, y2 + h / 2, xs[2], y2 + h / 2)
    arrow(xs[2] + w / 2, y1, xs[2] + w / 2, y2 + h)

    box(xs[0], y3, w, h, "Glassbox Contract\naudit, budgets, residual", "#fff3d6")
    box(xs[1], y3, w, h, "Budget controller\ncompact vs max accuracy", "#fff3d6")
    box(xs[2], y3, w, h, "Runtime\nfused LUTs, threads, NPZ", "#fff3d6")
    arrow(xs[0] + w / 2, y2, xs[0] + w / 2, y3 + h)
    arrow(xs[0] + w, y3 + h / 2, xs[1], y3 + h / 2)
    arrow(xs[1] + w, y3 + h / 2, xs[2], y3 + h / 2)

    ax.text(6.0, 0.35, "Every deployed computation is a low-order, editable term.",
            ha="center", fontsize=9, style="italic", color="#444444")
    ax.set_title("TermGraph: compiler, exact ledger, contract, and deployment", pad=6)
    save(fig, "fig_architecture.png")


def budget():
    comp = load("kirk_compactness.json")["summary"]
    report = json.loads((ROOT / "models" / "termgraph_report.json").read_text())
    tg = [report["performance"]["compact_5term_auc"],
          report["performance"]["five_seed_compiled_auc"]]
    ebm = [comp["ebm_5feat_gam"]["auc_mean"],
           report["performance"]["ebm_ga2m_auc"]]
    labels = ["5-term budget", "22-term budget"]
    x = np.arange(2)
    w = 0.34
    fig, ax = plt.subplots(figsize=(3.35, 2.25))
    b1 = ax.bar(x - w / 2, tg, w, color=COLORS["termgraph"], label="TermGraph")
    b2 = ax.bar(x + w / 2, ebm, w, color=COLORS["ebm"], label="EBM")
    for bars in (b1, b2):
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.0008,
                    f"{b.get_height():.4f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x, labels)
    ax.set_ylabel("AUC vs 10^9 truth")
    ax.set_ylim(0.915, 0.956)
    ax.set_title("KIRK: TermGraph at the same term budget as EBM")
    ax.legend(frameon=False, ncol=2, loc="upper left", fontsize=7)
    save(fig, "fig_budget.png")


def five_seed():
    cv = load("kirk_compile_sweep.json")["rows"]
    ebm = load("kirk_ebm.json")["summary"]["ebm_ga2m"]["rows"]
    seeds = [r["seed"] for r in cv]
    tg = [r["auc"] for r in cv]
    eb = [r["auc"] for r in ebm]
    fig, ax = plt.subplots(figsize=(3.35, 2.25))
    ax.plot(seeds, eb, "o-", color=COLORS["ebm"], label="EBM GA2M")
    ax.plot(seeds, tg, "o-", color=COLORS["termgraph"], label="TermGraph")
    ax.set_xticks(seeds)
    ax.set_xlabel("seed")
    ax.set_ylabel("AUC vs truth")
    ax.set_title("Paired five-seed comparison on KIRK")
    ax.legend(frameon=False)
    ax.text(0.02, 0.04,
            f"TermGraph {np.mean(tg):.4f} ± {np.std(tg):.4f}\n"
            f"EBM       {np.mean(eb):.4f} ± {np.std(eb):.4f}",
            transform=ax.transAxes, fontsize=8, va="bottom",
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": "#cccccc"})
    save(fig, "fig_five_seed.png")


def synthetic():
    s = load("synthetic_threeway_ebm.json")["summary"]
    truths = ["additive", "pairwise", "threeway"]
    models = [
        ("flat", "flat", COLORS["nam"]),
        ("pairs", "pairs", COLORS["gbm"]),
        ("triple", "triple", COLORS["termgraph"]),
        ("ebm_ga2m", "EBM GA2M", COLORS["ebm"]),
        ("unconstrained", "GBM", "#444444"),
    ]
    x = np.arange(len(truths))
    w = 0.15
    fig, ax = plt.subplots(figsize=(5.8, 3.2))
    for i, (key, label, color) in enumerate(models):
        vals = [s[t][key]["auc_mean"] for t in truths]
        ax.bar(x + (i - 2) * w, vals, w, color=color, label=label)
    ax.set_xticks(x, ["additive", "pairwise", "3-way"])
    ax.set_ylabel("AUC")
    ax.set_ylim(0.65, 0.87)
    ax.set_title("Matched synthetic tests: order is the capability", pad=6)
    ax.legend(
        frameon=False, ncol=5, fontsize=7,
        loc="upper center", bbox_to_anchor=(0.5, -0.18),
    )
    fig.subplots_adjust(bottom=0.30)
    save(fig, "fig_synthetic.png")


def parallel():
    fast = load("kirk_fast_runtime.json")["rows"]
    def rows(prefix):
        out = []
        for r in fast:
            if r["name"].startswith(prefix):
                nj = r.get("n_jobs", 1)
                out.append((nj, r["speedup"], r["rows_per_s"]))
        return sorted(out)
    base = rows("baseline")
    fst = rows("fast")
    fig, ax = plt.subplots(figsize=(3.35, 2.25))
    ax.plot([r[0] for r in base], [r[1] for r in base], "o-",
            color=COLORS["nam"], label="baseline TermGraph")
    ax.plot([r[0] for r in fst], [r[1] for r in fst], "o-",
            color=COLORS["termgraph"], label="fused TermGraph")
    ax.set_xlabel("threads")
    ax.set_ylabel("speedup vs serial baseline")
    ax.set_xticks([1, 2, 4, 8])
    ax.set_title("Deployment scaling, 200k rows")
    ax.legend(frameon=False, loc="upper left")
    ax.annotate(f"{fst[-1][1]:.2f}x", xy=(fst[-1][0], fst[-1][1]),
                xytext=(6.6, fst[-1][1] - 0.5), fontsize=9,
                color=COLORS["termgraph"])
    save(fig, "fig_parallel.png")


def edits():
    e = load("kirk_edit_demo.json")
    names = ["base", "zero", "clamp", "monotone"]
    labels = ["base", "zero weakest", "clamp strongest", "monotone Tsec"]
    auc = [e[n]["metrics"]["auc"] if n != "base" else e["base"]["auc"] for n in names]
    fig, ax = plt.subplots(figsize=(3.35, 2.25))
    y = np.arange(len(names))
    bars = ax.barh(y, auc, color=[COLORS["nam"], COLORS["edit"],
                                  COLORS["edit"], COLORS["termgraph"]])
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlim(0.9335, 0.9355)
    ax.set_xlabel("AUC vs truth")
    ax.set_title("AUC after a one-term edit")
    ax.grid(axis="x")
    for b, val in zip(bars, auc):
        ax.text(val + 0.00006, b.get_y() + b.get_height() / 2,
                f"{val:.4f}", va="center", fontsize=7)
    ax.set_xlim(0.9335, 0.9359)
    save(fig, "fig_edit.png")


def scaling():
    s = load("kirk_scaling.json")["models"]
    fig, ax = plt.subplots(figsize=(4.7, 2.8))
    styles = {
        "triple_truth": ("TermGraph truth", COLORS["termgraph"], "o-"),
        "triple_distill": ("TermGraph distill", COLORS["nam"], "s--"),
        "unconstrained_truth": ("GBM truth", COLORS["gbm"], "^-"),
    }
    for key, (label, color, fmt) in styles.items():
        rows = s[key]
        ax.plot([r["n"] for r in rows], [r["auc_mean"] for r in rows],
                fmt, color=color, label=label)
    ax.set_xlabel("training systems")
    ax.set_ylabel("AUC vs truth")
    ax.set_title("Data scaling: truth labels scale, distillation plateaus")
    ax.legend(frameon=False)
    save(fig, "fig_scaling.png")


def copy_term_plots():
    """Copy the pre-rendered term plots into the figure directory."""
    for src, dst in (
        ("kirk_glassbox_main.png", "fig_terms_main.png"),
        ("kirk_glassbox_threeway.png", "fig_terms_threeway.png"),
    ):
        data = (ROOT / "assets" / src).read_bytes()
        (FIGS / dst).write_bytes(data)
        print("wrote", FIGS / dst)


def main():
    FIGS.mkdir(parents=True, exist_ok=True)
    style()
    architecture()
    budget()
    five_seed()
    synthetic()
    parallel()
    edits()
    scaling()
    copy_term_plots()
    print(f"\nAll figures in {FIGS}")


if __name__ == "__main__":
    main()
