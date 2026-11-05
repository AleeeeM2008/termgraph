"""Five-seed certificate for the compiled EBM + order-3 TermGraph."""
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


def metrics(score, y):
    p = sigmoid(score)
    return {
        "auc": float(roc_auc_score(y.astype(int), p)),
        "logloss": float(log_loss(y.astype(int), p)),
        "acc": float(((p > 0.5) == (y > 0.5)).mean()),
    }


def main():
    X, Y, labels, feats, tr0, ho0 = load_data()
    triple = [feats.index(n) for n in
              ["EMcrossnear", "MMRstrengthnear", "EMfracstdnear"]]
    rows = []
    for seed in range(5):
        rng = np.random.default_rng(seed)
        perm = rng.permutation(len(X))
        cut = int(len(X) * 0.85)
        tr, ho = perm[:cut], perm[cut:]
        perm2 = np.random.default_rng(seed + 100).permutation(len(tr))
        cut2 = int(len(tr) * 0.85)
        fit, val = tr[perm2[:cut2]], tr[perm2[cut2:]]

        t0 = time.time()
        ebm = ExplainableBoostingClassifier(
            feature_names=feats, interactions=10,
            max_rounds=5000, early_stopping_rounds=50,
            outer_bags=4, n_jobs=-1, random_state=seed,
        )
        ebm.fit(X[tr], labels[tr].astype(int))
        base_fit = logit(ebm.predict_proba(X[fit])[:, 1])
        base_val = logit(ebm.predict_proba(X[val])[:, 1])
        base_ho = logit(ebm.predict_proba(X[ho])[:, 1])

        params = {
            "objective": "binary:logistic", "max_depth": 3, "eta": 0.05,
            "subsample": 0.8, "min_child_weight": 20, "reg_lambda": 1.0,
            "tree_method": "hist", "seed": seed, "eval_metric": "logloss",
            "verbosity": 0,
        }
        dfit = xgb.DMatrix(X[fit][:, triple], label=labels[fit].astype(np.float32))
        dfit.set_base_margin(base_fit)
        dval = xgb.DMatrix(X[val][:, triple], label=labels[val].astype(np.float32))
        dval.set_base_margin(base_val)
        residual = xgb.train(
            params, dfit, num_boost_round=1000,
            evals=[(dval, "val")], early_stopping_rounds=60,
            verbose_eval=False,
        )
        term_ho = residual.predict(xgb.DMatrix(X[ho][:, triple]), output_margin=True)
        hybrid = base_ho + term_ho

        graph, diag = compile_ebm_plus_residual(
            ebm, residual, X[tr], feats, triple, seed=seed,
        )
        score = graph.predict(X[ho])
        m = metrics(score, labels[ho])
        m.update({
            "seed": seed,
            "n_terms": len(graph.terms),
            "max_order": graph.max_order,
            "table_vs_hybrid_rmse": float(np.sqrt(np.mean((score - hybrid) ** 2))),
            "table_vs_hybrid_max_abs": float(np.max(np.abs(score - hybrid))),
            "ebm_auc": metrics(base_ho, labels[ho])["auc"],
            "hybrid_auc": metrics(hybrid, labels[ho])["auc"],
            "xgb_term_rmse": diag["xgb"]["rmse"],
            "seconds": round(time.time() - t0, 1),
        })
        rows.append(m)
        print(
            f"seed {seed} compiled auc {m['auc']:.4f} ll {m['logloss']:.4f} "
            f"ebm {m['ebm_auc']:.4f} hybrid {m['hybrid_auc']:.4f} "
            f"tbl_rmse {m['table_vs_hybrid_rmse']:.2e} {m['seconds']}s",
            flush=True,
        )

    summary = {}
    for key in ("auc", "logloss", "acc", "table_vs_hybrid_rmse",
                "table_vs_hybrid_max_abs"):
        vals = np.array([r[key] for r in rows])
        summary[key + "_mean"] = float(vals.mean())
        summary[key + "_std"] = float(vals.std())
    out = {"rows": rows, "summary": summary}
    path = ROOT / "experiments" / "results" / "kirk_compile_sweep.json"
    path.write_text(json.dumps(out, indent=1))
    print(json.dumps({k: round(v, 6) for k, v in summary.items()}, indent=1))
    print("saved", path)


if __name__ == "__main__":
    main()
