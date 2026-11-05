"""Is the 5-term glassbox statistically indistinguishable from the unconstrained GBM?

Clean protocol: truth labels, internal 85/15 fit/val, fixed holdout.
Paired bootstrap over the holdout for Delta AUC (unconstrained - glassbox).
"""
from __future__ import annotations
import os

import json
import sys
from pathlib import Path

import numpy as np
import xgboost as xgb
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
KIRK = Path(os.environ.get("KIRK_ROOT", Path(__file__).resolve().parents[1].parent / "TESTAstronomy"))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT / "experiments" / "legacy"))

from termgraph.gbm import train_constrained  # noqa: E402
from kirk_termgraph import load_data  # noqa: E402
from kirk_fourway import blocks_for  # noqa: E402


def fit(name, blocks, X, labels, feats, tr, seed=0):
    rng = np.random.default_rng(1000 + seed)
    perm = rng.permutation(len(tr))
    cut = int(len(tr) * 0.85)
    fit_idx, val_idx = tr[perm[:cut]], tr[perm[cut:]]
    booster = train_constrained(
        X[fit_idx], labels[fit_idx], blocks, feats,
        Xval=X[val_idx], yval=labels[val_idx],
        max_depth=3 if name == "glass" else 4,
        eta=0.05, num_round=1500, early_stopping=80,
        seed=seed, objective="binary:logistic", eval_metric="logloss",
    )
    return booster


def main():
    X, Y, labels, feats, tr, ho = load_data()
    glass_blocks = blocks_for("triple", feats)
    unc_blocks = blocks_for("unconstrained", feats)
    g = fit("glass", glass_blocks, X, labels, feats, tr)
    u = fit("unc", unc_blocks, X, labels, feats, tr)

    sg = g.predict(xgb.DMatrix(X[ho], feature_names=feats), output_margin=True)
    su = u.predict(xgb.DMatrix(X[ho], feature_names=feats), output_margin=True)
    y = labels[ho].astype(int)

    auc_g = roc_auc_score(y, sg)
    auc_u = roc_auc_score(y, su)
    delta = auc_u - auc_g

    rng = np.random.default_rng(0)
    boots = []
    for _ in range(2000):
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) < 2:
            continue
        boots.append(
            roc_auc_score(y[idx], su[idx]) - roc_auc_score(y[idx], sg[idx])
        )
    boots = np.asarray(boots)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    # Non-inferiority: one-sided 95% upper bound.
    upper = np.percentile(boots, 95)

    out = {
        "auc_glass": float(auc_g),
        "auc_unconstrained": float(auc_u),
        "delta_unconstrained_minus_glass": float(delta),
        "bootstrap_ci95": [float(lo), float(hi)],
        "bootstrap_upper95": float(upper),
        "n_holdout": int(len(y)),
        "n_boot": int(len(boots)),
        "interpretation": {
            "statistically_tied": bool(lo <= 0.0 <= hi),
            "non_inferior_margin_0.005": bool(upper < 0.005),
            "non_inferior_margin_0.002": bool(upper < 0.002),
        },
    }
    path = ROOT / "experiments" / "results" / "kirk_parity.json"
    path.write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    print(f"saved {path}")


if __name__ == "__main__":
    main()
