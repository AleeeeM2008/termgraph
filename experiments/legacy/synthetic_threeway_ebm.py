"""Matched-order test with EBM: where does TermGraph's 3-way capability pay?

Three truths:
  additive   main effects only
  pairwise   main + pairs (0,1),(2,3),(4,5)
  threeway   main + 3-way terms (0,1,2),(3,4,5), no cross-pairs

Models:
  TermGraph flat / pairs / triple
  EBM GAM (no interactions)
  EBM GA2M (10 pairwise interactions)
  unconstrained XGBoost

The 3-way truth is the case EBM cannot represent: its interactions are
pairwise only. If TermGraph's triple block tracks the unconstrained GBM while
EBM lags, that is the clear architectural benefit.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import xgboost as xgb
from interpret.glassbox import ExplainableBoostingClassifier
from sklearn.metrics import roc_auc_score, log_loss

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT / "experiments" / "legacy"))

from termgraph.gbm import train_constrained  # noqa: E402

P = 10
FEATS = [f"x{i}" for i in range(P)]


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def truth(X, kind):
    x = X
    base = (
        0.8 * x[:, 0]
        + 0.6 * np.sin(2.0 * x[:, 1])
        + 0.5 * x[:, 2] ** 2
        + 0.4 * x[:, 3]
        - 0.4 * x[:, 4]
        + 0.3 * x[:, 5]
    )
    if kind == "additive":
        return base
    pair = 1.2 * x[:, 0] * x[:, 1] + 0.9 * x[:, 2] * x[:, 3] + 0.6 * np.tanh(2 * x[:, 4] * x[:, 5])
    if kind == "pairwise":
        return base + pair
    if kind == "threeway":
        # Strong within-block 3-way terms only; no cross-block pairs.
        return base + 2.0 * x[:, 0] * x[:, 1] * x[:, 2] + 1.5 * x[:, 3] * x[:, 4] * x[:, 5]
    raise ValueError(kind)


def blocks(kind):
    if kind == "flat":
        return [[i] for i in range(P)]
    if kind == "pairs":
        return [[0, 1], [2, 3], [4, 5]] + [[i] for i in range(6, P)]
    if kind == "triple":
        return [[0, 1, 2], [3, 4, 5]] + [[i] for i in range(6, P)]
    if kind == "unconstrained":
        return [list(range(P))]
    raise ValueError(kind)


def auc_ll(y, score):
    p = sigmoid(score)
    return float(roc_auc_score(y, p)), float(log_loss(y, p))


def main():
    n_train, n_test = 12000, 20000
    seeds = [0, 1, 2]
    truths = ["additive", "pairwise", "threeway"]
    out = {"n_train": n_train, "n_test": n_test, "truths": {}}

    for kind in truths:
        out["truths"][kind] = {}
        for seed in seeds:
            rng = np.random.default_rng(seed)
            Xtr = rng.normal(size=(n_train, P)).astype(np.float32)
            ytr = rng.binomial(1, sigmoid(truth(Xtr, kind))).astype(np.int32)
            Xte = rng.normal(size=(n_test, P)).astype(np.float32)
            yte = rng.binomial(1, sigmoid(truth(Xte, kind))).astype(np.int32)
            perm = rng.permutation(n_train)
            cut = int(n_train * 0.8)
            fit, val = perm[:cut], perm[cut:]

            for model in ("flat", "pairs", "triple", "unconstrained"):
                blk = blocks(model)
                depth = 5 if model == "unconstrained" else 3
                booster = train_constrained(
                    Xtr[fit], ytr[fit].astype(np.float32), blk, FEATS,
                    Xval=Xtr[val], yval=ytr[val].astype(np.float32),
                    max_depth=depth, eta=0.05, num_round=1500,
                    early_stopping=80, seed=seed,
                    objective="binary:logistic", eval_metric="logloss",
                )
                score = booster.predict(
                    xgb.DMatrix(Xte, feature_names=FEATS), output_margin=True
                )
                auc, ll = auc_ll(yte, score)
                out["truths"][kind].setdefault(model, []).append(
                    {"auc": auc, "logloss": ll, "seed": seed}
                )
                print(f"{kind:10s} seed {seed} {model:14s} auc {auc:.4f} ll {ll:.4f}", flush=True)

            for ename, inter in (("ebm_gam", 0), ("ebm_ga2m", 10)):
                t0 = time.time()
                ebm = ExplainableBoostingClassifier(
                    feature_names=FEATS, interactions=inter,
                    max_rounds=5000, early_stopping_rounds=50,
                    outer_bags=2, n_jobs=-1, random_state=seed,
                )
                ebm.fit(Xtr, ytr)
                p = ebm.predict_proba(Xte)[:, 1]
                auc = float(roc_auc_score(yte, p))
                ll = float(log_loss(yte, p))
                out["truths"][kind].setdefault(ename, []).append(
                    {"auc": auc, "logloss": ll, "seed": seed,
                     "seconds": round(time.time() - t0, 1)}
                )
                print(
                    f"{kind:10s} seed {seed} {ename:14s} auc {auc:.4f} ll {ll:.4f} "
                    f"{time.time()-t0:.1f}s", flush=True
                )

    summary = {}
    for kind, models in out["truths"].items():
        summary[kind] = {}
        for model, rows in models.items():
            summary[kind][model] = {
                "auc_mean": float(np.mean([r["auc"] for r in rows])),
                "auc_std": float(np.std([r["auc"] for r in rows])),
                "logloss_mean": float(np.mean([r["logloss"] for r in rows])),
            }
    out["summary"] = summary
    path = ROOT / "experiments" / "results" / "synthetic_threeway_ebm.json"
    path.write_text(json.dumps(out, indent=1))
    print("\nsummary")
    for kind in truths:
        print(kind)
        for model, s in summary[kind].items():
            print(f"  {model:14s} auc {s['auc_mean']:.4f}±{s['auc_std']:.4f} ll {s['logloss_mean']:.4f}")
    print(f"saved {path}")


if __name__ == "__main__":
    main()
