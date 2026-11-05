"""Full-data deployment artifact: all 9,767 stable systems.

The five-seed sweep in kirk_compile_sweep.json is the honest performance
estimate (0.9427 AUC). This artifact is trained on all data for deployment.
"""
from __future__ import annotations
import os

import json
import sys
import time
from pathlib import Path

import numpy as np
import xgboost as xgb
from interpret.glassbox import ExplainableBoostingClassifier
from sklearn.metrics import roc_auc_score, log_loss

ROOT = Path(__file__).resolve().parents[2]
KIRK = Path(os.environ.get("KIRK_ROOT", Path(__file__).resolve().parents[1].parent / "TESTAstronomy"))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT / "experiments" / "legacy"))

from kirk_termgraph import load_data  # noqa: E402
from termgraph.compile import compile_ebm_plus_residual  # noqa: E402


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def main():
    X, Y, labels, feats, tr0, ho0 = load_data()
    stable = np.ones(len(X), dtype=bool)  # load_data already returned stable rows
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(X))
    cut = int(len(X) * 0.85)
    fit, val = perm[:cut], perm[cut:]

    print(f"full data n={len(X)}", flush=True)
    t0 = time.time()
    ebm = ExplainableBoostingClassifier(
        feature_names=feats, interactions=10,
        max_rounds=5000, early_stopping_rounds=50,
        outer_bags=4, n_jobs=-1, random_state=0,
    )
    ebm.fit(X, labels.astype(int))
    print(f"EBM {time.time()-t0:.1f}s", flush=True)

    base_fit = logit(ebm.predict_proba(X[fit])[:, 1])
    base_val = logit(ebm.predict_proba(X[val])[:, 1])
    triple = [feats.index(n) for n in
              ["EMcrossnear", "MMRstrengthnear", "EMfracstdnear"]]
    params = {
        "objective": "binary:logistic", "max_depth": 3, "eta": 0.05,
        "subsample": 0.8, "min_child_weight": 20, "reg_lambda": 1.0,
        "tree_method": "hist", "seed": 0, "eval_metric": "logloss",
        "verbosity": 0,
    }
    dfit = xgb.DMatrix(X[fit][:, triple], label=labels[fit].astype(np.float32))
    dfit.set_base_margin(base_fit)
    dval = xgb.DMatrix(X[val][:, triple], label=labels[val].astype(np.float32))
    dval.set_base_margin(base_val)
    residual = xgb.train(
        params, dfit, num_boost_round=1000,
        evals=[(dval, "val")], early_stopping_rounds=60, verbose_eval=False,
    )
    print(f"residual trees {residual.num_boosted_rounds()}", flush=True)

    graph, diag = compile_ebm_plus_residual(
        ebm, residual, X, feats, triple, seed=0,
    )
    score = graph.predict(X)
    p = sigmoid(score)
    in_sample = {
        "auc": float(roc_auc_score(labels.astype(int), p)),
        "logloss": float(log_loss(labels.astype(int), p)),
        "acc": float(((p > 0.5) == (labels > 0.5)).mean()),
    }
    base = np.asarray(ebm.decision_function(X), dtype=np.float64)
    term = residual.predict(xgb.DMatrix(X[:, triple]), output_margin=True)
    hybrid = base + term
    d = score - hybrid
    cv = json.loads(
        (ROOT / "experiments" / "results" / "kirk_compile_sweep.json").read_text()
    )["summary"]
    cert = {
        "artifact": "kirk_glassbox_full.json",
        "training": "all 9,767 short-stable systems",
        "model": {
            "terms": len(graph.terms),
            "max_order": graph.max_order,
            "bias": float(graph.bias),
            "features": feats,
        },
        "expected_cv": {
            "auc_mean": cv["auc_mean"],
            "auc_std": cv["auc_std"],
            "logloss_mean": cv["logloss_mean"],
            "note": "five-seed sweep, kirk_compile_sweep.json",
        },
        "in_sample": in_sample,
        "compile_fidelity": {
            "table_vs_hybrid_rmse": float(np.sqrt(np.mean(d ** 2))),
            "table_vs_hybrid_max_abs": float(np.max(np.abs(d))),
        },
        "additivity_error": graph.additivity_error(X[:1000]),
        "n_terms": len(graph.terms),
        "term_counts": {
            "main": sum(t.order == 1 for t in graph.terms),
            "pair": sum(t.order == 2 for t in graph.terms),
            "threeway": sum(t.order == 3 for t in graph.terms),
        },
        "term_ids": [t.id for t in graph.terms],
    }
    graph.certificate = cert
    out = ROOT / "models" / "kirk_glassbox_full.json"
    graph.to_json(out)
    (ROOT / "models" / "kirk_glassbox_full_certificate.json").write_text(
        json.dumps(cert, indent=1)
    )
    print(json.dumps(cert, indent=1))
    print("artifact bytes", out.stat().st_size)


if __name__ == "__main__":
    main()
