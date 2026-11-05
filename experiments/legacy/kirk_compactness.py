"""Compactness win: 5 terms, all 11 features, versus EBM at the same budget.

TermGraph's 5-term model is one 3-way block + four pairs and covers all 11
features. A standard EBM needs one term per feature, so a 5-term EBM can only
see 5 features. We select the best 5 features by EBM's own full-GAM importance
and train EBM GAM/GA2M under that budget.
"""
from __future__ import annotations
import os

import json
import sys
import time
from pathlib import Path

import numpy as np
from interpret.glassbox import ExplainableBoostingClassifier
from sklearn.metrics import roc_auc_score, log_loss

ROOT = Path(__file__).resolve().parents[2]
KIRK = Path(os.environ.get("KIRK_ROOT", Path(__file__).resolve().parents[1].parent / "TESTAstronomy"))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT / "experiments" / "legacy"))

from kirk_termgraph import load_data  # noqa: E402


def main():
    X, Y, labels, feats, tr0, ho0 = load_data()

    # Full GAM once, to rank features by EBM's own importance.
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(X))
    cut = int(len(X) * 0.85)
    tr, ho = perm[:cut], perm[cut:]
    full = ExplainableBoostingClassifier(
        feature_names=feats, interactions=0, max_rounds=5000,
        early_stopping_rounds=50, outer_bags=2, n_jobs=-1, random_state=0,
    )
    full.fit(X[tr], labels[tr].astype(int))
    imp = np.asarray(full.term_importances()[:len(feats)], dtype=float)
    order = np.argsort(imp)[::-1]
    top5 = list(order[:5])
    print("top5 by EBM importance:", [feats[i] for i in top5], flush=True)

    out = {"top5": [feats[i] for i in top5], "models": {}}
    for seed in range(5):
        rng = np.random.default_rng(seed)
        perm = rng.permutation(len(X))
        cut = int(len(X) * 0.85)
        tr, ho = perm[:cut], perm[cut:]
        for name, idx, inter in (
            ("ebm_5feat_gam", top5, 0),
            ("ebm_5feat_ga2m", top5, 3),
        ):
            t0 = time.time()
            ebm = ExplainableBoostingClassifier(
                feature_names=[feats[i] for i in idx],
                interactions=inter, max_rounds=5000,
                early_stopping_rounds=50, outer_bags=2,
                n_jobs=-1, random_state=seed,
            )
            ebm.fit(X[tr][:, idx], labels[tr].astype(int))
            p = ebm.predict_proba(X[ho][:, idx])[:, 1]
            m = {
                "auc": float(roc_auc_score(labels[ho].astype(int), p)),
                "logloss": float(log_loss(labels[ho].astype(int), p)),
                "acc": float(((p > 0.5) == (labels[ho] > 0.5)).mean()),
                "seconds": round(time.time() - t0, 1),
                "seed": seed,
            }
            out["models"].setdefault(name, []).append(m)
            print(
                f"seed {seed} {name:16s} auc {m['auc']:.4f} "
                f"ll {m['logloss']:.4f} acc {m['acc']:.4f}",
                flush=True,
            )

    summary = {}
    for name, rows in out["models"].items():
        summary[name] = {}
        for key in ("auc", "logloss", "acc"):
            vals = np.array([r[key] for r in rows])
            summary[name][key + "_mean"] = float(vals.mean())
            summary[name][key + "_std"] = float(vals.std())
        summary[name]["n_terms"] = 5 if "gam" in name else 8
        summary[name]["rows"] = rows
    out["summary"] = summary
    out["termgraph_5term"] = {"auc_mean": 0.9364, "auc_std": 0.0058,
                              "n_terms": 5}
    path = ROOT / "experiments" / "results" / "kirk_compactness.json"
    path.write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps({k: {kk: round(vv, 4) for kk, vv in v.items()
                          if kk.endswith("_mean") or kk == "n_terms"}
                      for k, v in summary.items()}, indent=1))
    print(f"saved {path}")


if __name__ == "__main__":
    main()
