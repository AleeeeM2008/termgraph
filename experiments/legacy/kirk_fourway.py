"""Does one 4-way block close the gap to the unconstrained truth student?

Clean protocol: fixed holdout, internal 85/15 fit/val split, truth labels.
Three repetitions per model. Compare with the triple truth result from the
scaling run (0.9349) and the unconstrained truth result (0.9384) at n=8301.
"""
from __future__ import annotations
import os

import json
import sys
from pathlib import Path

import numpy as np
import xgboost as xgb

ROOT = Path(__file__).resolve().parents[2]
KIRK = Path(os.environ.get("KIRK_ROOT", Path(__file__).resolve().parents[1].parent / "TESTAstronomy"))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT / "experiments" / "legacy"))

from termgraph.gbm import train_constrained, fidelity_metrics  # noqa: E402
from kirk_termgraph import load_data  # noqa: E402


def blocks_for(kind, feats):
    i = {f: k for k, f in enumerate(feats)}
    if kind == "triple":
        return [
            [i["EMcrossnear"], i["MMRstrengthnear"], i["EMfracstdnear"]],
            [i["EMcrossfar"], i["Tsec"]],
            [i["MEGNO"], i["MEGNOstd"]],
            [i["EPstdnear"], i["EMfracstdfar"]],
            [i["EPstdfar"], i["MMRstrengthfar"]],
        ]
    if kind == "four_megno":
        return [
            [i["EMcrossnear"], i["MMRstrengthnear"], i["EMfracstdnear"],
             i["MEGNO"]],
            [i["EMcrossfar"], i["Tsec"]],
            [i["EPstdnear"], i["EMfracstdfar"]],
            [i["EPstdfar"], i["MMRstrengthfar"]],
            [i["MEGNOstd"]],
        ]
    if kind == "four_tsec":
        return [
            [i["EMcrossnear"], i["MMRstrengthnear"], i["EMfracstdnear"],
             i["Tsec"]],
            [i["EMcrossfar"]],
            [i["MEGNO"], i["MEGNOstd"]],
            [i["EPstdnear"], i["EMfracstdfar"]],
            [i["EPstdfar"], i["MMRstrengthfar"]],
        ]
    if kind == "unconstrained":
        return [list(range(len(feats)))]
    raise ValueError(kind)


def main():
    X, Y, labels, feats, tr, ho = load_data()
    kinds = ["triple", "four_megno", "four_tsec", "unconstrained"]
    out = {"kinds": {}, "note": "truth labels; internal 85/15 fit/val; fixed holdout"}
    for kind in kinds:
        blocks = blocks_for(kind, feats)
        vals = []
        for rep in range(3):
            rng = np.random.default_rng(1000 + rep)
            perm = rng.permutation(len(tr))
            cut = int(len(tr) * 0.85)
            fit_idx, val_idx = tr[perm[:cut]], tr[perm[cut:]]
            booster = train_constrained(
                X[fit_idx], labels[fit_idx], blocks, feats,
                Xval=X[val_idx], yval=labels[val_idx],
                max_depth=3 if kind != "unconstrained" else 4,
                eta=0.05, num_round=1500, early_stopping=80,
                seed=rep, objective="binary:logistic", eval_metric="logloss",
            )
            score = booster.predict(
                xgb.DMatrix(X[ho], feature_names=feats), output_margin=True
            )
            m = fidelity_metrics(score, Y[ho], labels[ho])
            vals.append(m["auc_truth"])
        out["kinds"][kind] = {
            "auc_mean": float(np.mean(vals)),
            "auc_std": float(np.std(vals)),
            "auc_per_rep": [float(v) for v in vals],
            "n_terms": len(blocks),
            "max_order": max(len(b) for b in blocks),
        }
        print(
            f"{kind:14s} terms={len(blocks)} order={max(len(b) for b in blocks)} "
            f"aucY {np.mean(vals):.4f}±{np.std(vals):.4f}",
            flush=True,
        )
    path = ROOT / "experiments" / "results" / "kirk_fourway.json"
    path.write_text(json.dumps(out, indent=1, default=float))
    print(f"saved {path}")


if __name__ == "__main__":
    main()
