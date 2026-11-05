"""Render the compiled KIRK artifact as a human-readable term inventory."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT / "experiments" / "legacy"))

from termgraph.ir import TermGraph  # noqa: E402


def centers(edges):
    return 0.5 * (edges[:-1] + edges[1:])


def main():
    (ROOT / "assets").mkdir(exist_ok=True)
    g = TermGraph.from_json(ROOT / "models" / "kirk_glassbox.json")
    text = g.describe()
    (ROOT / "experiments" / "results" / "kirk_describe.txt").write_text(text)
    print(text)

    mains = [t for t in g.terms if t.order == 1]
    mains.sort(key=lambda t: np.ptp(t.values - t.center), reverse=True)
    fig, axes = plt.subplots(2, 2, figsize=(11, 6.5))
    for ax, t in zip(axes.ravel(), mains[:4]):
        x = centers(t.edges[0])
        y = t.values - t.center
        ax.plot(x, y, lw=2)
        ax.axhline(0, color="0.7", lw=0.8)
        ax.set_title(t.id, fontsize=10)
        ax.set_xlabel(t.inputs[0])
        ax.set_ylabel("logit contribution")
        ax.grid(alpha=0.25)
    fig.suptitle("KIRK TermGraph: main-effect terms (top 4 by range)")
    fig.tight_layout()
    fig.savefig(ROOT / "assets" / "kirk_glassbox_main.png", dpi=160)
    plt.close(fig)

    # 3-way term slices: fix the third input at low/mid/high bins.
    tid = "EMcrossnear*MMRstrengthnear*EMfracstdnear"
    t = next(t for t in g.terms if t.id == tid)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), sharey=True)
    x = centers(t.edges[0])
    y = centers(t.edges[1])
    z = centers(t.edges[2])
    for ax, zi, label in zip(axes, [0, len(z) // 2, len(z) - 1],
                             ["low", "mid", "high"]):
        v = t.values[:, :, zi] - t.center
        im = ax.pcolormesh(x, y, v.T, shading="auto", cmap="RdBu_r")
        ax.set_title(f"EMfracstdnear = {label} ({z[zi]:.3f})", fontsize=10)
        ax.set_xlabel("EMcrossnear")
        if ax is axes[0]:
            ax.set_ylabel("MMRstrengthnear")
        fig.colorbar(im, ax=ax, shrink=0.85)
    fig.suptitle("KIRK TermGraph: order-3 term slices")
    fig.tight_layout()
    fig.savefig(ROOT / "assets" / "kirk_glassbox_threeway.png", dpi=160)
    plt.close(fig)

    out = {
        "n_terms": len(g.terms),
        "max_order": g.max_order,
        "main_terms": len(mains),
        "pair_terms": sum(t.order == 2 for t in g.terms),
        "threeway_terms": sum(t.order == 3 for t in g.terms),
        "describe": text,
        "figures": ["assets/kirk_glassbox_main.png",
                    "assets/kirk_glassbox_threeway.png"],
    }
    path = ROOT / "experiments" / "results" / "kirk_readability.json"
    path.write_text(json.dumps(out, indent=1))
    print("saved", path)


if __name__ == "__main__":
    main()
