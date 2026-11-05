"""Hybrid test: 5-term glassbox core + small GBM residual.

The glassbox is trained on truth labels. A second XGBoost is trained on the
same labels with the glassbox margin as base_margin. The certificate reports
how much of the final margin is carried by the glassbox.

This is Mode B of the architecture: accuracy of the hybrid, auditability of the
core, and a flagged residual.
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

from termgraph.gbm import train_constrained, fidelity_metrics  # noqa: E402
from kirk_termgraph import load_data  # noqa: E402
from kirk_fourway import blocks_for  # noqa: E402


def main():
    X, Y, labels, feats, tr, ho = load_data()
    rng = np.random.default_rng(1000)
    perm = rng.permutation(len(tr))
    cut = int(len(tr) * 0.85)
    fit_idx, val_idx = tr[perm[:cut]], tr[perm[cut:]]

    glass_blocks = blocks_for("triple", feats)
    glass = train_constrained(
        X[fit_idx], labels[fit_idx], glass_blocks, feats,
        Xval=X[val_idx], yval=labels[val_idx],
        max_depth=3, eta=0.05, num_round=1500, early_stopping=80,
        seed=0, objective="binary:logistic", eval_metric="logloss",
    )

    def margin(booster, Xpart, base=None):
        dm = xgb.DMatrix(Xpart, feature_names=feats)
        if base is not None:
            dm.set_base_margin(base)
        return booster.predict(dm, output_margin=True)

    g_fit = margin(glass, X[fit_idx])
    g_val = margin(glass, X[val_idx])
    g_ho = margin(glass, X[ho])

    # Residual GBM on the same labels, with the glassbox as base margin.
    params = {
        "objective": "binary:logistic",
        "max_depth": 3,
        "eta": 0.05,
        "subsample": 0.8,
        "min_child_weight": 20,
        "reg_lambda": 1.0,
        "tree_method": "hist",
        "seed": 0,
        "eval_metric": "logloss",
        "verbosity": 0,
    }
    dfit = xgb.DMatrix(X[fit_idx], label=labels[fit_idx], feature_names=feats)
    dfit.set_base_margin(g_fit)
    dval = xgb.DMatrix(X[val_idx], label=labels[val_idx], feature_names=feats)
    dval.set_base_margin(g_val)
    residual = xgb.train(
        params, dfit, num_boost_round=1500,
        evals=[(dval, "val")], early_stopping_rounds=80,
        verbose_eval=False,
    )

    h_ho = margin(residual, X[ho], base=g_ho)
    # If predict already includes base_margin, h_ho is the total; check against
    # the glassbox-only margin.
    total = h_ho
    if np.allclose(total, g_ho):
        # Some versions return only the residual; add it.
        total = g_ho + h_ho

    auc_g = roc_auc_score(labels[ho].astype(int), g_ho)
    auc_h = roc_auc_score(labels[ho].astype(int), total)
    m_g = fidelity_metrics(g_ho, Y[ho], labels[ho])
    m_h = fidelity_metrics(total, Y[ho], labels[ho])

    # Coverage: share of the hybrid margin variance carried by the glassbox.
    var_g = float(np.var(g_ho))
    var_h = float(np.var(total))
    var_r = float(np.var(total - g_ho))
    coverage_var = var_g / var_h if var_h > 0 else float("nan")
    coverage_mse = 1.0 - var_r / var_h if var_h > 0 else float("nan")

    out = {
        "glass_auc": float(auc_g),
        "hybrid_auc": float(auc_h),
        "delta_hybrid_minus_glass": float(auc_h - auc_g),
        "glass_metrics": m_g,
        "hybrid_metrics": m_h,
        "residual_trees": int(residual.num_boosted_rounds()),
        "coverage_var": coverage_var,
        "coverage_mse": coverage_mse,
        "var_glass": var_g,
        "var_hybrid": var_h,
        "var_residual": var_r,
        "base_margin_included_in_predict": bool(np.allclose(total, h_ho)),
    }
    path = ROOT / "experiments" / "results" / "kirk_hybrid.json"
    path.write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps(out, indent=1, default=float))
    print(f"saved {path}")


if __name__ == "__main__":
    main()
