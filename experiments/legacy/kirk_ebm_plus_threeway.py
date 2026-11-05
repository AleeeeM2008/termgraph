"""Absorb EBM's strength: EBM GA2M + one order-3 residual term.

EBM wins on KIRK because its term-wise boosting is a better optimizer than our
constrained XGBoost. So use EBM for the low-order terms, then add the order-3
block that EBM cannot represent. The result is a heterogeneous term graph:
EBM terms + one XGBoost term, one ledger.

Compare:
  EBM GA2M alone
  EBM GA2M + 3-way residual (EMcrossnear, MMRstrengthnear, EMfracstdnear)
  EBM GA2M + phase 3-way (EMfracstdnear, MEGNO, Tsec)
  TermGraph triple (current)
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


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def metrics(y, score):
    p = 1 / (1 + np.exp(-score))
    return {
        "auc": float(roc_auc_score(y, p)),
        "logloss": float(log_loss(y, p)),
        "acc": float(((p > 0.5) == (y > 0.5)).mean()),
    }


def add_term(X, y, base_tr, Xval, yval, base_val, Xho, base_ho, block, seed):
    params = {
        "objective": "binary:logistic",
        "max_depth": 3,
        "eta": 0.05,
        "subsample": 0.8,
        "min_child_weight": 20,
        "reg_lambda": 1.0,
        "tree_method": "hist",
        "seed": seed,
        "eval_metric": "logloss",
        "verbosity": 0,
    }
    dtr = xgb.DMatrix(X[:, block], label=y)
    dtr.set_base_margin(base_tr)
    dval = xgb.DMatrix(Xval[:, block], label=yval)
    dval.set_base_margin(base_val)
    b = xgb.train(
        params, dtr, num_boost_round=1000,
        evals=[(dval, "val")], early_stopping_rounds=60,
        verbose_eval=False,
    )
    dho = xgb.DMatrix(Xho[:, block])
    dho.set_base_margin(base_ho)
    return b.predict(dho, output_margin=True), int(b.num_boosted_rounds())


def main():
    X, Y, labels, feats, tr0, ho0 = load_data()
    tri_res = [feats.index(n) for n in
               ["EMcrossnear", "MMRstrengthnear", "EMfracstdnear"]]
    tri_phase = [feats.index(n) for n in
                 ["EMfracstdnear", "MEGNO", "Tsec"]]
    out = {"models": {}}

    for seed in range(5):
        rng = np.random.default_rng(seed)
        perm = rng.permutation(len(X))
        cut = int(len(X) * 0.85)
        tr, ho = perm[:cut], perm[cut:]
        # Internal validation for the residual term.
        perm2 = np.random.default_rng(seed + 100).permutation(len(tr))
        cut2 = int(len(tr) * 0.85)
        fit, val = tr[perm2[:cut2]], tr[perm2[cut2:]]

        ebm = ExplainableBoostingClassifier(
            feature_names=feats, interactions=10,
            max_rounds=5000, early_stopping_rounds=50,
            outer_bags=4, n_jobs=-1, random_state=seed,
        )
        ebm.fit(X[tr], labels[tr].astype(int))
        base_fit = logit(ebm.predict_proba(X[fit])[:, 1])
        base_val = logit(ebm.predict_proba(X[val])[:, 1])
        base_ho = logit(ebm.predict_proba(X[ho])[:, 1])
        base_metrics = metrics(labels[ho].astype(int), base_ho)
        out["models"].setdefault("ebm_ga2m", []).append(
            {"seed": seed, **base_metrics}
        )
        print(f"seed {seed} ebm_ga2m auc {base_metrics['auc']:.4f}", flush=True)

        for name, block in (("ebm_plus_res3", tri_res),
                            ("ebm_plus_phase3", tri_phase)):
            t0 = time.time()
            score, trees = add_term(
                X[fit], labels[fit].astype(np.float32),
                base_fit, X[val], labels[val].astype(np.float32), base_val,
                X[ho], base_ho, block, seed,
            )
            m = metrics(labels[ho].astype(int), score)
            m.update({"seed": seed, "trees": trees,
                      "seconds": round(time.time() - t0, 1)})
            out["models"].setdefault(name, []).append(m)
            print(
                f"seed {seed} {name:16s} auc {m['auc']:.4f} "
                f"ll {m['logloss']:.4f} trees {trees}",
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

    # Paired deltas vs EBM alone.
    deltas = {}
    for name in ("ebm_plus_res3", "ebm_plus_phase3"):
        d = np.array([r["auc"] for r in out["models"][name]]) - np.array(
            [r["auc"] for r in out["models"]["ebm_ga2m"]]
        )
        deltas[name] = {
            "auc_mean": float(d.mean()),
            "auc_std": float(d.std()),
            "per_seed": [float(v) for v in d],
        }
    out["paired_delta_vs_ebm"] = deltas

    path = ROOT / "experiments" / "results" / "kirk_ebm_plus_threeway.json"
    path.write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps({k: {kk: round(vv, 4) for kk, vv in v.items()
                          if kk.endswith("_mean")}
                      for k, v in summary.items()}, indent=1))
    print(json.dumps(deltas, indent=1))
    print(f"saved {path}")


if __name__ == "__main__":
    main()
