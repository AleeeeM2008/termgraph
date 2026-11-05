"""Data-scaling curve: does the 5-term glassbox keep improving?

Fixed holdout (seed-0 split). For each training size, sample from the training
pool, use an internal 85/15 fit/val split for early stopping, and evaluate on
the fixed holdout. Three sampling seeds per size.

Models:
  triple_truth       5-term, binary truth labels
  triple_distill     5-term, teacher logits
  unconstrained_truth  one block, binary truth labels
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
KIRK = Path(os.environ.get("KIRK_ROOT", Path(__file__).resolve().parents[1].parent / "TESTAstronomy"))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT / "experiments" / "legacy"))

from termgraph.gbm import train_constrained, fidelity_metrics  # noqa: E402
from kirk_termgraph import load_data, interaction_ranking  # noqa: E402
from kirk_threeway import blocks_with_triple  # noqa: E402


def fit_predict(name, Xfit, yfit, Xval, yval, Xho, feats, blocks, seed):
    if name.endswith("truth"):
        objective, eval_metric = "binary:logistic", "logloss"
        yfit = yfit.astype(np.float32)
        yval = yval.astype(np.float32)
    else:
        objective, eval_metric = "reg:squarederror", None
        yfit = yfit.astype(np.float32)
        yval = yval.astype(np.float32)
    booster = train_constrained(
        Xfit, yfit, blocks, feats,
        Xval=Xval, yval=yval,
        max_depth=3 if "triple" in name else 4,
        eta=0.05, num_round=1500, early_stopping=80,
        seed=seed, objective=objective, eval_metric=eval_metric,
    )
    return booster.predict(
        xgb.DMatrix(Xho, feature_names=feats), output_margin=True
    )


def main():
    X, Y, labels, feats, tr, ho = load_data()
    teacher = xgb.Booster()
    teacher.load_model(str(KIRK / "models" / "featureclassifier.json"))
    ranked = interaction_ranking(teacher, X[tr], feats, seed=0, n=2000)
    triple = [feats.index(n) for n in
              ["EMcrossnear", "MMRstrengthnear", "EMfracstdnear"]]
    triple_blocks = blocks_with_triple(triple, ranked, len(feats))
    all_block = [list(range(len(feats)))]

    sizes = [1000, 2200, 4000, 6000, 8301]
    models = ["triple_truth", "triple_distill", "unconstrained_truth"]
    out = {"sizes": sizes, "models": {}}

    for name in models:
        rows = []
        blocks = triple_blocks if "triple" in name else all_block
        for n in sizes:
            vals = []
            for rep in range(3):
                rng = np.random.default_rng(1000 * rep + n)
                idx = rng.choice(len(tr), n, replace=False)
                sub = tr[idx]
                cut = int(len(sub) * 0.85)
                fit_idx, val_idx = sub[:cut], sub[cut:]
                t0 = time.time()
                if "truth" in name:
                    yfit, yval = labels[fit_idx], labels[val_idx]
                else:
                    yfit, yval = Y[fit_idx], Y[val_idx]
                score = fit_predict(
                    name, X[fit_idx], yfit, X[val_idx], yval,
                    X[ho], feats, blocks, seed=rep,
                )
                m = fidelity_metrics(score, Y[ho], labels[ho])
                vals.append(m["auc_truth"])
            rows.append({
                "n": n,
                "auc_mean": float(np.mean(vals)),
                "auc_std": float(np.std(vals)),
                "auc_per_rep": [float(v) for v in vals],
                "seconds": round(time.time() - t0, 1),
            })
            print(
                f"{name:22s} n={n:4d} aucY {np.mean(vals):.4f} "
                f"±{np.std(vals):.4f}",
                flush=True,
            )
        out["models"][name] = rows

    path = ROOT / "experiments" / "results" / "kirk_scaling.json"
    path.write_text(json.dumps(out, indent=1, default=float))
    print(f"saved {path}")


if __name__ == "__main__":
    main()
