"""EBM baseline on KIRK: GAM and GA2M, truth labels.

The comparison that matters for the paper:
  EBM(GAM) vs EBM(GA2M) vs TermGraph flat/pairs/triple vs unconstrained GBM.

EBM is limited to pairwise interactions; TermGraph can add the 3-way block.
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

ROOT = Path(__file__).resolve().parents[2]
KIRK = Path(os.environ.get("KIRK_ROOT", Path(__file__).resolve().parents[1].parent / "TESTAstronomy"))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT / "experiments" / "legacy"))

from kirk_termgraph import load_data  # noqa: E402


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def metrics(y, p):
    from sklearn.metrics import roc_auc_score, log_loss
    return {
        "auc": float(roc_auc_score(y, p)),
        "logloss": float(log_loss(y, p)),
        "acc": float(((p > 0.5) == (y > 0.5)).mean()),
    }


def main():
    X, Y, labels, feats, tr0, ho0 = load_data()
    out = {"models": {}, "note": "truth labels; same five-seed splits"}
    for seed in range(5):
        rng = np.random.default_rng(seed)
        perm = rng.permutation(len(X))
        cut = int(len(X) * 0.85)
        tr, ho = perm[:cut], perm[cut:]
        for name, inter in (("ebm_gam", 0), ("ebm_ga2m", 10)):
            t0 = time.time()
            ebm = ExplainableBoostingClassifier(
                feature_names=feats,
                interactions=inter,
                max_rounds=5000,
                early_stopping_rounds=50,
                outer_bags=4,
                n_jobs=-1,
                random_state=seed,
            )
            ebm.fit(X[tr], labels[tr].astype(int))
            p = ebm.predict_proba(X[ho])[:, 1]
            m = metrics(labels[ho].astype(int), p)
            m["seed"] = seed
            m["seconds"] = round(time.time() - t0, 1)
            out["models"].setdefault(name, []).append(m)
            print(
                f"seed {seed} {name:10s} auc {m['auc']:.4f} "
                f"logloss {m['logloss']:.4f} acc {m['acc']:.4f} "
                f"{m['seconds']}s",
                flush=True,
            )

    summary = {}
    for name, rows in out["models"].items():
        summary[name] = {}
        for key in ("auc", "logloss", "acc"):
            vals = np.array([r[key] for r in rows])
            summary[name][key + "_mean"] = float(vals.mean())
            summary[name][key + "_std"] = float(vals.std())
        summary[name]["rows"] = rows
    out["summary"] = summary
    path = ROOT / "experiments" / "results" / "kirk_ebm.json"
    path.write_text(json.dumps(out, indent=1, default=float))
    for name, s in summary.items():
        print(
            f"{name:10s} auc {s['auc_mean']:.4f}±{s['auc_std']:.4f} "
            f"logloss {s['logloss_mean']:.4f}"
        )
    print(f"saved {path}")


if __name__ == "__main__":
    main()
