"""Term surgery on the compiled KIRK artifact.

Shows three post-hoc operations that EBM/NAM/SHAP do not provide as a supported
workflow:
  - zero the weakest term,
  - clamp a high-impact term,
  - enforce monotonicity in Tsec on one 2D term.

Every operation leaves all other term contributions exactly unchanged.
"""
from __future__ import annotations
import os

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score, log_loss

ROOT = Path(__file__).resolve().parents[2]
KIRK = Path(os.environ.get("KIRK_ROOT", Path(__file__).resolve().parents[1].parent / "TESTAstronomy"))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT / "experiments" / "legacy"))

from kirk_termgraph import load_data  # noqa: E402
from termgraph.ir import (  # noqa: E402
    TermGraph, zero_term, clamp_term, monotonize_term, diff_graphs,
)


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def metrics(score, y):
    p = sigmoid(score)
    return {
        "auc": float(roc_auc_score(y.astype(int), p)),
        "logloss": float(log_loss(y.astype(int), p)),
        "acc": float(((p > 0.5) == (y > 0.5)).mean()),
    }


def main():
    g = TermGraph.from_json(ROOT / "models" / "kirk_glassbox.json")
    X, Y, labels, feats, tr, ho = load_data()
    Xh, yh = X[ho], labels[ho]
    base_score = g.predict(Xh)
    base = metrics(base_score, yh)
    terms = g.predict_terms(Xh)
    imp = terms.std(axis=0)
    order = np.argsort(imp)
    weakest = g.terms[int(order[0])]
    print("weakest term:", weakest.id, "std", float(imp[int(order[0])]), flush=True)

    zeroed = zero_term(g, weakest.id)
    d_zero = diff_graphs(g, zeroed, Xh)
    m_zero = metrics(zeroed.predict(Xh), yh)

    # Clamp the highest-impact term to a bounded logit range.
    strongest = g.terms[int(order[-1])]
    lo, hi = -1.0, 1.0
    clamped = clamp_term(g, strongest.id, lo, hi)
    d_clamp = diff_graphs(g, clamped, Xh)
    m_clamp = metrics(clamped.predict(Xh), yh)

    # Enforce monotonicity in Tsec on the EMcrossfar*Tsec term.
    tsec_id = "EMcrossfar*Tsec"
    mono = monotonize_term(g, tsec_id, axis=1, direction=1)
    d_mono = diff_graphs(g, mono, Xh)
    m_mono = metrics(mono.predict(Xh), yh)

    def violations(graph, tid):
        t = next(t for t in graph.terms if t.id == tid)
        v = t.values - t.center
        return int(np.sum(np.diff(v, axis=1) < -1e-12))

    out = {
        "base": base,
        "weakest_term": weakest.id,
        "strongest_term": strongest.id,
        "zero": {
            "term": weakest.id,
            "metrics": m_zero,
            "prediction_max_abs": d_zero["prediction_max_abs"],
            "remaining_term_max_abs": max(
                (v["max_abs"] for k, v in d_zero["term_delta"].items()
                 if k != weakest.id), default=0.0
            ),
            "term_delta": d_zero["term_delta"].get(weakest.id),
        },
        "clamp": {
            "term": strongest.id,
            "range": [lo, hi],
            "metrics": m_clamp,
            "prediction_max_abs": d_clamp["prediction_max_abs"],
            "remaining_term_max_abs": max(
                (v["max_abs"] for k, v in d_clamp["term_delta"].items()
                 if k != strongest.id), default=0.0
            ),
        },
        "monotone": {
            "term": tsec_id,
            "axis": 1,
            "direction": 1,
            "violations_before": violations(g, tsec_id),
            "violations_after": violations(mono, tsec_id),
            "metrics": m_mono,
            "prediction_max_abs": d_mono["prediction_max_abs"],
            "remaining_term_max_abs": max(
                (v["max_abs"] for k, v in d_mono["term_delta"].items()
                 if k != tsec_id), default=0.0
            ),
        },
    }
    path = ROOT / "experiments" / "results" / "kirk_edit_demo.json"
    path.write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    print("saved", path)


if __name__ == "__main__":
    main()
