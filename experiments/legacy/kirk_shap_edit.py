"""Validate exact-SHAP equivalence and term surgery on the KIRK PBAM.

1. Train the 6-term PBAM (k=5 disjoint pairs).
2. Compare TreeSHAP block sums with the term ledger. For a block-additive
   model, interventional TreeSHAP should assign exactly the centered term to
   each block: the explanation is the model.
3. Zero each term in turn and report the exact prediction/metric delta. This is
   the audit/edit contract.
"""
from __future__ import annotations
import os

import json
import sys
from pathlib import Path

import numpy as np
import shap
import xgboost as xgb

ROOT = Path(__file__).resolve().parents[2]
KIRK = Path(os.environ.get("KIRK_ROOT", Path(__file__).resolve().parents[1].parent / "TESTAstronomy"))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT / "experiments" / "legacy"))

from termgraph.gbm import (  # noqa: E402
    train_constrained,
    term_decompose,
    feature_shap,
    fidelity_metrics,
)
from kirk_termgraph import (  # noqa: E402
    load_data,
    interaction_ranking,
    greedy_disjoint_pairs,
    blocks_from_pairs,
)


def main():
    X, Y, labels, feats, tr, ho = load_data()
    teacher = xgb.Booster()
    teacher.load_model(str(KIRK / "models" / "featureclassifier.json"))
    ranked = interaction_ranking(teacher, X[tr], feats, seed=0, n=2000)
    pairs = greedy_disjoint_pairs(ranked, 5)
    blocks = blocks_from_pairs(pairs, len(feats))

    booster = train_constrained(
        X[tr], Y[tr], blocks, feats,
        Xval=X[ho], yval=Y[ho],
        max_depth=3, eta=0.05, num_round=1500,
        early_stopping=80, seed=0,
    )
    score = booster.predict(
        xgb.DMatrix(X[ho], feature_names=feats), output_margin=True
    )
    terms, bias, _ = term_decompose(booster, X[ho], blocks)
    # Use the first 100 training rows as the SHAP background and for centering.
    # TreeExplainer subsamples to 100 by default; matching that avoids a
    # baseline mismatch between the ledger and SHAP.
    bg = X[tr][:100]
    terms_bg, bias_bg, _ = term_decompose(booster, bg, blocks)
    centers = terms_bg.mean(0)
    phi, expected, phi_block = feature_shap(terms, centers, bias, blocks)

    # Interventional TreeSHAP with the same background.
    explainer = shap.TreeExplainer(
        booster, data=bg, feature_perturbation="interventional"
    )
    sv = explainer.shap_values(X[ho])
    ev = float(np.asarray(explainer.expected_value).ravel()[0])

    shap_add_err = float(np.max(np.abs(sv.sum(1) + ev - score)))
    term_add_err = float(np.max(np.abs(phi.sum(1) + expected - score)))

    block_rows = []
    for k, block in enumerate(blocks):
        bsv = sv[:, block].sum(1)
        expected_term = terms[:, k] - terms_bg[:, k].mean()
        diff = bsv - expected_term
        block_rows.append({
            "term": [feats[i] for i in block],
            "corr": float(np.corrcoef(bsv, expected_term)[0, 1]),
            "rmse": float(np.sqrt(np.mean(diff ** 2))),
            "max_abs_diff": float(np.max(np.abs(diff))),
            "importance_std": float(np.std(expected_term)),
            "importance_range": float(np.ptp(expected_term)),
        })

    # Term surgery: zero one term at a time.
    base = fidelity_metrics(score, Y[ho], labels[ho])
    ablations = []
    for k, block in enumerate(blocks):
        score_k = score - (terms[:, k] - centers[k])
        m = fidelity_metrics(score_k, Y[ho], labels[ho])
        ablations.append({
            "term": [feats[i] for i in block],
            "delta_auc_truth": m["auc_truth"] - base["auc_truth"],
            "delta_agree": m["agree_teacher"] - base["agree_teacher"],
            "delta_rmse": m["rmse_teacher"] - base["rmse_teacher"],
            "importance_std": block_rows[k]["importance_std"],
        })
    ablations.sort(key=lambda r: r["delta_auc_truth"])

    out = {
        "pairs": [
            {"strength": s, "a": feats[i], "b": feats[j]}
            for s, i, j in pairs
        ],
        "terms": [[feats[i] for i in b] for b in blocks],
        "base_metrics": base,
        "shap": {
            "expected_value_shap": ev,
            "expected_value_ledger": expected,
            "additivity_error_shap": shap_add_err,
            "additivity_error_ledger": term_add_err,
            "blocks": block_rows,
        },
        "term_ablation": ablations,
    }
    path = ROOT / "experiments" / "results" / "kirk_shap_edit.json"
    path.write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps(out, indent=1, default=float))
    print(f"saved {path}")


if __name__ == "__main__":
    main()
