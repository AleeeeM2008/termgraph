"""Matched synthetic tests: does a glassbox match a GBM when its order matches?

Four data-generating processes:
  additive  f = smooth main effects only
  pairwise  additive + three pairwise interactions
  threeway  pairwise + two 3-way interactions
  fiveway   additive + one 5-way parity interaction

Five model budgets:
  flat       10 singletons
  pairs      (0,1),(2,3),(4,5) + singletons
  triple     (0,1,2),(3,4,5) + singletons
  five       (0,1,2,3,4) + singletons
  unconstrained

If the matched glassbox tracks the GBM, the architecture is conditionally
sufficient. If even the matched model lags, the gap is the term learner, not
the order.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import xgboost as xgb
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
        1.5 * x[:, 0]
        + 1.2 * np.sin(2.0 * x[:, 1])
        + 0.8 * x[:, 2] ** 2
        + 0.5 * x[:, 3]
        - 0.7 * x[:, 4]
        + 0.3 * x[:, 5]
    )
    if kind == "additive":
        return base
    pair = 1.2 * x[:, 0] * x[:, 1] + 0.9 * x[:, 2] * x[:, 3] + 0.6 * np.tanh(2 * x[:, 4] * x[:, 5])
    if kind == "pairwise":
        return base + pair
    triple = 1.5 * x[:, 0] * x[:, 1] * x[:, 2] + 0.8 * np.sin(2 * x[:, 3] * x[:, 4] * x[:, 5])
    if kind == "threeway":
        return base + pair + triple
    if kind == "fiveway":
        parity = 2.0 * np.prod(np.sign(x[:, :5]), axis=1)
        return base + parity
    raise ValueError(kind)


def blocks(kind):
    if kind == "flat":
        return [[i] for i in range(P)]
    if kind == "pairs":
        return [[0, 1], [2, 3], [4, 5]] + [[i] for i in range(6, P)]
    if kind == "triple":
        return [[0, 1, 2], [3, 4, 5]] + [[i] for i in range(6, P)]
    if kind == "five":
        return [[0, 1, 2, 3, 4]] + [[i] for i in range(5, P)]
    if kind == "unconstrained":
        return [list(range(P))]
    raise ValueError(kind)


def main():
    n_train = 12000
    n_test = 20000
    seeds = [0, 1, 2]
    truths = ["additive", "pairwise", "threeway", "fiveway"]
    models = ["flat", "pairs", "triple", "five", "unconstrained"]
    out = {"n_train": n_train, "n_test": n_test, "truths": {}}

    for kind in truths:
        out["truths"][kind] = {}
        for seed in seeds:
            rng = np.random.default_rng(seed)
            Xtr = rng.normal(size=(n_train, P)).astype(np.float32)
            ftr = truth(Xtr, kind)
            ytr = rng.binomial(1, sigmoid(ftr)).astype(np.float32)
            Xte = rng.normal(size=(n_test, P)).astype(np.float32)
            fte = truth(Xte, kind)
            yte = rng.binomial(1, sigmoid(fte)).astype(np.float32)

            # Internal validation split for early stopping.
            perm = rng.permutation(n_train)
            cut = int(n_train * 0.8)
            fit, val = perm[:cut], perm[cut:]

            for model in models:
                blk = blocks(model)
                depth = 5 if model in ("five", "unconstrained") else 3
                booster = train_constrained(
                    Xtr[fit], ytr[fit], blk, FEATS,
                    Xval=Xtr[val], yval=ytr[val],
                    max_depth=depth, eta=0.05, num_round=1200,
                    early_stopping=60, seed=seed,
                    objective="binary:logistic", eval_metric="logloss",
                )
                score = booster.predict(
                    xgb.DMatrix(Xte, feature_names=FEATS), output_margin=True
                )
                m = {
                    "auc": float(roc_auc_score(yte, score)),
                    "logloss": float(log_loss(yte, sigmoid(score))),
                    "acc": float(((sigmoid(score) > 0.5) == (yte > 0.5)).mean()),
                }
                out["truths"][kind].setdefault(model, []).append(m)
                print(
                    f"{kind:10s} seed {seed} {model:14s} "
                    f"auc {m['auc']:.4f} logloss {m['logloss']:.4f}",
                    flush=True,
                )

    # Aggregate
    summary = {}
    for kind, models_d in out["truths"].items():
        summary[kind] = {}
        for model, rows in models_d.items():
            summary[kind][model] = {
                "auc_mean": float(np.mean([r["auc"] for r in rows])),
                "auc_std": float(np.std([r["auc"] for r in rows])),
                "logloss_mean": float(np.mean([r["logloss"] for r in rows])),
                "acc_mean": float(np.mean([r["acc"] for r in rows])),
            }
    out["summary"] = summary

    # Matched model per truth
    matched = {"additive": "flat", "pairwise": "pairs",
               "threeway": "triple", "fiveway": "five"}
    out["matched_gap"] = {}
    for kind, model in matched.items():
        s = summary[kind]
        out["matched_gap"][kind] = {
            "model": model,
            "gap_vs_unconstrained_auc": float(
                s["unconstrained"]["auc_mean"] - s[model]["auc_mean"]
            ),
            "gap_vs_unconstrained_logloss": float(
                s[model]["logloss_mean"] - s["unconstrained"]["logloss_mean"]
            ),
        }

    path = ROOT / "experiments" / "results" / "synthetic_order.json"
    path.write_text(json.dumps(out, indent=1))
    print("\nsummary")
    for kind in truths:
        print(kind)
        for model in models:
            s = summary[kind][model]
            print(
                f"  {model:14s} auc {s['auc_mean']:.4f}±{s['auc_std']:.4f} "
                f"logloss {s['logloss_mean']:.4f}"
            )
        print("  matched gap", out["matched_gap"][kind])
    print(f"saved {path}")


if __name__ == "__main__":
    main()
