"""Regenerate the authoritative certificate from the saved artifact + models.

The earlier builder compared against an in-memory EBM array that was stale at
one tail sample. This script loads the saved EBM and residual and recomputes
every fidelity number from the deployed artifact.
"""
from __future__ import annotations
import os

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import xgboost as xgb
from sklearn.metrics import roc_auc_score, log_loss

ROOT = Path(__file__).resolve().parents[2]
KIRK = Path(os.environ.get("KIRK_ROOT", Path(__file__).resolve().parents[1].parent / "TESTAstronomy"))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT / "experiments" / "legacy"))

from kirk_termgraph import load_data  # noqa: E402
from termgraph.ir import TermGraph  # noqa: E402


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
    X, Y, labels, feats, tr0, ho0 = load_data()
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(X))
    cut = int(len(X) * 0.85)
    tr, ho = perm[:cut], perm[cut:]

    graph = TermGraph.from_json(ROOT / "models" / "kirk_glassbox.json")
    ebm = joblib.load(ROOT / "models" / "kirk_ebm.joblib")
    residual = xgb.Booster()
    residual.load_model(str(ROOT / "models" / "kirk_residual.json"))
    triple = [feats.index(n) for n in
              ["EMcrossnear", "MMRstrengthnear", "EMfracstdnear"]]

    base = np.asarray(ebm.decision_function(X[ho]), dtype=np.float64)
    term = residual.predict(xgb.DMatrix(X[ho][:, triple]), output_margin=True)
    hybrid = base + term
    score = graph.predict(X[ho])
    terms = graph.predict_terms(X[ho])
    xterm = graph.terms[-1]
    ebm_only = graph.bias - xterm.center + terms[:, :-1].sum(1)

    d_ebm = ebm_only - base
    d_xgb = terms[:, -1] - (term - xterm.center)
    d_hybrid = score - hybrid
    p999 = np.percentile(np.abs(d_hybrid), 99.9)

    cert = {
        "artifact": "kirk_glassbox.json",
        "model": {
            "terms": len(graph.terms),
            "max_order": graph.max_order,
            "bias": float(graph.bias),
            "features": feats,
        },
        "metrics_truth": {
            "ebm_ga2m": metrics(base, labels[ho]),
            "ebm_plus_3way": metrics(hybrid, labels[ho]),
            "compiled_table": metrics(score, labels[ho]),
        },
        "compile_fidelity": {
            "table_vs_hybrid_rmse": float(np.sqrt(np.mean(d_hybrid ** 2))),
            "table_vs_hybrid_max_abs": float(np.max(np.abs(d_hybrid))),
            "table_vs_hybrid_p99_9_abs": float(p999),
            "table_vs_ebm_rmse": float(np.sqrt(np.mean(d_ebm ** 2))),
            "table_vs_ebm_max_abs": float(np.max(np.abs(d_ebm))),
            "xgb_term_rmse": float(np.sqrt(np.mean(d_xgb ** 2))),
            "xgb_term_max_abs": float(np.max(np.abs(d_xgb))),
        },
        "additivity_error": graph.additivity_error(X[ho]),
        "n_terms": len(graph.terms),
        "term_ids": [t.id for t in graph.terms],
        "term_counts": {
            "main": sum(t.order == 1 for t in graph.terms),
            "pair": sum(t.order == 2 for t in graph.terms),
            "threeway": sum(t.order == 3 for t in graph.terms),
        },
        "grid_bins": graph.metadata.get("grid_bins", {}),
        "note": "fidelity recomputed from saved EBM/residual and the deployed artifact",
    }
    graph.certificate = cert
    graph.to_json(ROOT / "models" / "kirk_glassbox.json")
    (ROOT / "models" / "kirk_glassbox_certificate.json").write_text(
        json.dumps(cert, indent=1)
    )
    print(json.dumps(cert, indent=1))
    print("artifact bytes", (ROOT / "models" / "kirk_glassbox.json").stat().st_size)


if __name__ == "__main__":
    main()
