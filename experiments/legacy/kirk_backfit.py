"""Overlapping-block backfitting test on KIRK.

Disjoint PBAM uses one pair per feature. The top interaction ranking has
overlapping pairs around MEGNO/EMfracstdnear, so this test allows a feature to
appear in several terms and fits them by cyclic backfitting:

    f = bias + sum_k T_k(x[block_k])

Two variants:
  pairs_only : top-m pairs + singletons only for features not covered by a pair
  full       : top-m pairs + all 11 singletons

Run:
  python experiments/legacy/kirk_backfit.py
"""
from __future__ import annotations
import os

import json
import sys
import time
from pathlib import Path

import numpy as np
import xgboost as xgb

ROOT = Path(__file__).resolve().parents[2]
KIRK = Path(os.environ.get("KIRK_ROOT", ROOT.parent / "TESTAstronomy"))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT / "experiments" / "legacy"))

from termgraph.gbm import fidelity_metrics  # noqa: E402
from kirk_termgraph import load_data, interaction_ranking  # noqa: E402


def train_term(X, y, rounds=50, eta=0.1, max_depth=2, seed=0):
    params = {
        "objective": "reg:squarederror",
        "max_depth": max_depth,
        "eta": eta,
        "subsample": 0.8,
        "min_child_weight": 20,
        "reg_lambda": 1.0,
        "tree_method": "hist",
        "seed": seed,
        "verbosity": 0,
    }
    d = xgb.DMatrix(X, label=y)
    return xgb.train(params, d, num_boost_round=rounds)


def predict_term(model, X):
    return model.predict(xgb.DMatrix(X), output_margin=True)


def fit_backfit(Xtr, Ytr, Xval, Yval, blocks, cycles=6, rounds=50,
                eta=0.1, max_depth=2, seed=0, verbose=False):
    terms_tr = [np.zeros(len(Ytr)) for _ in blocks]
    terms_val = [np.zeros(len(Yval)) for _ in blocks]
    bias = float(Ytr.mean())
    f_tr = np.full(len(Ytr), bias)
    f_val = np.full(len(Yval), bias)
    best = None
    best_mse = float("inf")

    for cycle in range(cycles):
        for k, block in enumerate(blocks):
            r = Ytr - (f_tr - terms_tr[k])
            model = train_term(
                Xtr[:, block], r, rounds=rounds, eta=eta,
                max_depth=max_depth, seed=seed + k,
            )
            new_tr = predict_term(model, Xtr[:, block])
            new_val = predict_term(model, Xval[:, block])
            f_tr += new_tr - terms_tr[k]
            f_val += new_val - terms_val[k]
            terms_tr[k] = new_tr
            terms_val[k] = new_val

        mus = np.array([t.mean() for t in terms_tr])
        for k in range(len(blocks)):
            terms_tr[k] = terms_tr[k] - mus[k]
            terms_val[k] = terms_val[k] - mus[k]
        bias += float(mus.sum())
        # f is unchanged: bias went up by sum(mus), terms went down by the same.

        mse = float(np.mean((f_val - Yval) ** 2))
        if verbose:
            print(f"  cycle {cycle} val_mse {mse:.5f}", flush=True)
        if mse < best_mse:
            best_mse = mse
            best = {
                "bias": bias,
                "terms_tr": [t.copy() for t in terms_tr],
                "terms_val": [t.copy() for t in terms_val],
                "cycle": cycle,
                "val_mse": mse,
                "f_val": f_val.copy(),
            }
    return best


def blocks_for(pairs, n_features, include_all_singletons):
    used = set()
    blocks = []
    for _, i, j in pairs:
        blocks.append([i, j])
        used.update((i, j))
    if include_all_singletons:
        blocks.extend([[i] for i in range(n_features)])
    else:
        blocks.extend([[i] for i in range(n_features) if i not in used])
    return blocks


def main():
    X, Y, labels, feats, tr, ho = load_data()
    teacher = xgb.Booster()
    teacher.load_model(str(KIRK / "models" / "featureclassifier.json"))
    ranked = interaction_ranking(teacher, X[tr], feats, seed=0, n=2000)

    results = {}
    for m in (3, 5, 8):
        for mode in ("pairs_only", "full"):
            pairs = ranked[:m]
            blocks = blocks_for(pairs, len(feats), mode == "full")
            t0 = time.time()
            best = fit_backfit(
                X[tr], Y[tr], X[ho], Y[ho], blocks,
                cycles=6, rounds=50, eta=0.1, max_depth=2, seed=0,
            )
            score = best["f_val"]
            m_metrics = fidelity_metrics(score, Y[ho], labels[ho])
            m_metrics.update({
                "n_terms": len(blocks),
                "n_pairs": m,
                "mode": mode,
                "best_cycle": best["cycle"],
                "val_mse": best["val_mse"],
                "seconds": round(time.time() - t0, 1),
                "terms": [[feats[i] for i in b] for b in blocks],
            })
            name = f"backfit_m{m}_{mode}"
            results[name] = m_metrics
            print(
                f"{name:22s} terms={len(blocks):2d} aucY {m_metrics['auc_truth']:.4f} "
                f"aucT {m_metrics['auc_teacher']:.4f} agree "
                f"{m_metrics['agree_teacher']:.4f} rmse "
                f"{m_metrics['rmse_teacher']:.4f} cycle {best['cycle']} "
                f"{m_metrics['seconds']}s",
                flush=True,
            )

    out = {
        "top_pairs": [
            {"strength": s, "a": feats[i], "b": feats[j]}
            for s, i, j in ranked[:10]
        ],
        "models": results,
    }
    path = ROOT / "experiments" / "results" / "kirk_backfit.json"
    path.write_text(json.dumps(out, indent=1, default=float))
    print(f"saved {path}")


if __name__ == "__main__":
    main()
