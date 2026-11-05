"""Exchange fidelity: can EBM reproduce the deployed GBM as well as TermGraph?

Train EBM regressors on the SPOCK teacher logits (not truth labels), exactly
like the TermGraph compiler. Compare agreement, RMSE, R2, and AUC-vs-teacher.
This is the "replace a deployed GBM without changing its decisions" use case.
"""
from __future__ import annotations
import os

import json
import sys
import time
from pathlib import Path

import numpy as np
from interpret.glassbox import ExplainableBoostingRegressor
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
KIRK = Path(os.environ.get("KIRK_ROOT", Path(__file__).resolve().parents[1].parent / "TESTAstronomy"))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT / "experiments" / "legacy"))

from kirk_termgraph import load_data  # noqa: E402


def metrics(score, teacher, truth):
    p = 1 / (1 + np.exp(-score))
    pt = 1 / (1 + np.exp(-teacher))
    sse = float(np.sum((score - teacher) ** 2))
    sst = float(np.sum((teacher - teacher.mean()) ** 2))
    return {
        "agree_teacher": float(((p > 0.5) == (pt > 0.5)).mean()),
        "rmse_teacher": float(np.sqrt(np.mean((score - teacher) ** 2))),
        "r2_teacher": float(1 - sse / sst),
        "auc_teacher": float(roc_auc_score((pt > 0.5).astype(int), score)),
        "auc_truth": float(roc_auc_score(truth.astype(int), p)),
    }


def main():
    X, Y, labels, feats, tr0, ho0 = load_data()
    out = {"models": {}}
    for seed in range(3):
        rng = np.random.default_rng(seed)
        perm = rng.permutation(len(X))
        cut = int(len(X) * 0.85)
        tr, ho = perm[:cut], perm[cut:]
        for name, inter in (("ebm_reg_gam", 0), ("ebm_reg_ga2m", 10)):
            t0 = time.time()
            ebm = ExplainableBoostingRegressor(
                feature_names=feats,
                interactions=inter,
                max_rounds=5000,
                early_stopping_rounds=50,
                outer_bags=4,
                n_jobs=-1,
                random_state=seed,
            )
            ebm.fit(X[tr], Y[tr])
            score = ebm.predict(X[ho])
            m = metrics(score, Y[ho], labels[ho])
            m["seed"] = seed
            m["seconds"] = round(time.time() - t0, 1)
            out["models"].setdefault(name, []).append(m)
            print(
                f"seed {seed} {name:16s} agree {m['agree_teacher']:.4f} "
                f"rmse {m['rmse_teacher']:.4f} r2 {m['r2_teacher']:.4f} "
                f"aucT {m['auc_teacher']:.4f} aucY {m['auc_truth']:.4f}",
                flush=True,
            )

    summary = {}
    for name, rows in out["models"].items():
        summary[name] = {}
        for key in ("agree_teacher", "rmse_teacher", "r2_teacher",
                    "auc_teacher", "auc_truth"):
            vals = np.array([r[key] for r in rows])
            summary[name][key + "_mean"] = float(vals.mean())
            summary[name][key + "_std"] = float(vals.std())
        summary[name]["rows"] = rows
    out["summary"] = summary
    path = ROOT / "experiments" / "results" / "kirk_ebm_exchange.json"
    path.write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps({k: {kk: round(vv, 4) for kk, vv in v.items()
                          if kk.endswith("_mean")}
                      for k, v in summary.items()}, indent=1))
    print(f"saved {path}")


if __name__ == "__main__":
    main()
