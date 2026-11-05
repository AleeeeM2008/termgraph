"""Build the 5-term compact TermGraph artifact.

Blocks:
  order-3: EMcrossnear, MMRstrengthnear, EMfracstdnear
  pair:    EMcrossfar, Tsec
  pair:    MEGNO, MEGNOstd
  pair:    EPstdnear, EMfracstdfar
  pair:    EPstdfar, MMRstrengthfar

Trained on truth labels with an internal 85/15 fit/validation split; evaluated
on the seed-0 holdout. This is the compact mode the controller can select at a
5-term budget.
"""
from __future__ import annotations
import os

import json
import sys
import time
from pathlib import Path

import numpy as np
import xgboost as xgb
from sklearn.metrics import roc_auc_score, log_loss

ROOT = Path(__file__).resolve().parents[2]
KIRK = Path(os.environ.get("KIRK_ROOT", Path(__file__).resolve().parents[1].parent / "TESTAstronomy"))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT / "experiments" / "legacy"))

from kirk_termgraph import load_data  # noqa: E402
from termgraph.contract import audit  # noqa: E402
from termgraph.export_xgb import export_block_terms  # noqa: E402
from termgraph.gbm import train_constrained  # noqa: E402
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
    perm2 = np.random.default_rng(100).permutation(len(tr))
    cut2 = int(len(tr) * 0.85)
    fit, val = tr[perm2[:cut2]], tr[perm2[cut2:]]

    idx = {f: i for i, f in enumerate(feats)}
    blocks = [
        [idx["EMcrossnear"], idx["MMRstrengthnear"], idx["EMfracstdnear"]],
        [idx["EMcrossfar"], idx["Tsec"]],
        [idx["MEGNO"], idx["MEGNOstd"]],
        [idx["EPstdnear"], idx["EMfracstdfar"]],
        [idx["EPstdfar"], idx["MMRstrengthfar"]],
    ]

    t0 = time.time()
    booster = train_constrained(
        X[fit], labels[fit].astype(np.float32), blocks, feats,
        Xval=X[val], yval=labels[val].astype(np.float32),
        max_depth=3, eta=0.05, num_round=180, early_stopping=50,
        seed=0, objective="binary:logistic", eval_metric="logloss",
    )
    print(f"trained {booster.num_boosted_rounds()} trees in {time.time()-t0:.1f}s",
          flush=True)

    terms, bias, diag = export_block_terms(
        booster, feats, blocks, X[tr], max_bins=128, seed=0, link="logit",
    )
    graph = TermGraph(
        features=feats, terms=terms, bias=bias,
        metadata={
            "compiler": "xgboost-constrained",
            "mode": "compact",
            "blocks": [[feats[i] for i in b] for b in blocks],
            "max_bins": 128,
        },
    )
    score = graph.predict(X[ho])
    exact = booster.predict(
        xgb.DMatrix(X[ho], feature_names=feats), output_margin=True
    )
    d = score - exact
    m_compiled = metrics(score, labels[ho])
    cert = {
        "artifact": "kirk_glassbox_compact.json",
        "mode": "compact",
        "model": {
            "terms": len(graph.terms),
            "max_order": graph.max_order,
            "bias": float(graph.bias),
            "features": feats,
            "blocks": [[feats[i] for i in b] for b in blocks],
        },
        "metrics_truth": {
            "compiled_table": m_compiled,
            "booster": metrics(exact, labels[ho]),
        },
        "compile_fidelity": {
            "table_vs_booster_rmse": float(np.sqrt(np.mean(d ** 2))),
            "table_vs_booster_max_abs": float(np.max(np.abs(d))),
            "term_diagnostics": diag,
        },
        "additivity_error": graph.additivity_error(X[ho]),
        "n_terms": len(graph.terms),
        "term_counts": {
            "main": sum(t.order == 1 for t in graph.terms),
            "pair": sum(t.order == 2 for t in graph.terms),
            "threeway": sum(t.order == 3 for t in graph.terms),
        },
        "term_ids": [t.id for t in graph.terms],
        "expected_auc": float(m_compiled["auc"]),
    }
    graph.certificate = cert
    rep = audit(graph, max_order=3, max_terms=5, X=X[ho][:200])
    cert["glassbox_contract"] = rep.to_dict()
    out = ROOT / "models" / "kirk_glassbox_compact.json"
    graph.to_json(out)
    cert_path = ROOT / "models" / "kirk_glassbox_compact_certificate.json"
    cert_path.write_text(json.dumps(cert, indent=1))
    print(json.dumps(cert, indent=1))
    print("artifact bytes", out.stat().st_size)


if __name__ == "__main__":
    main()
