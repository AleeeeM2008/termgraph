"""Can the glassbox surpass the teacher by training on real labels?

The shipped KIRK student is distilled from teacher logits. The real 10^9
labels are also available for the 9,767 short-stable systems. This script
trains the same 5-term block structure directly on the binary truth labels and
compares it with the teacher and the teacher-distilled TermGraph.
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


def main():
    X, Y, labels, feats, tr0, ho0 = load_data()
    teacher = xgb.Booster()
    teacher.load_model(str(KIRK / "models" / "featureclassifier.json"))
    ranked = interaction_ranking(teacher, X[tr0], feats, seed=0, n=2000)

    triple = [feats.index(n) for n in
              ["EMcrossnear", "MMRstrengthnear", "EMfracstdnear"]]
    blocks = {
        "triple_truth": blocks_with_triple(triple, ranked, len(feats)),
        "flat_truth": [[i] for i in range(len(feats))],
        "unconstrained_truth": [list(range(len(feats)))],
    }

    rows = {name: [] for name in blocks}
    for seed in range(5):
        rng = np.random.default_rng(seed)
        perm = rng.permutation(len(X))
        cut = int(len(X) * 0.85)
        tr, ho = perm[:cut], perm[cut:]
        for name, blk in blocks.items():
            t0 = time.time()
            booster = train_constrained(
                X[tr], labels[tr].astype(np.float32), blk, feats,
                Xval=X[ho], yval=labels[ho].astype(np.float32),
                max_depth=3, eta=0.05, num_round=1500,
                early_stopping=80, seed=seed,
                objective="binary:logistic", eval_metric="logloss",
            )
            score = booster.predict(
                xgb.DMatrix(X[ho], feature_names=feats), output_margin=True
            )
            m = fidelity_metrics(score, Y[ho], labels[ho])
            m.update({
                "seed": seed,
                "n_terms": len(blk),
                "n_trees": int(booster.num_boosted_rounds()),
                "seconds": round(time.time() - t0, 1),
            })
            rows[name].append(m)
            print(
                f"seed {seed} {name:22s} aucY {m['auc_truth']:.4f} "
                f"acc {m['acc_truth']:.4f} agreeT {m['agree_teacher']:.4f} "
                f"aucT {m['auc_teacher']:.4f}",
                flush=True,
            )

    summary = {}
    for name, ms in rows.items():
        summary[name] = {}
        for key in ("auc_truth", "acc_truth", "agree_teacher", "auc_teacher",
                    "ece_truth"):
            vals = np.array([m[key] for m in ms])
            summary[name][key + "_mean"] = float(vals.mean())
            summary[name][key + "_std"] = float(vals.std())
        summary[name]["rows"] = ms

    out = {
        "note": "binary truth labels; holdout used for early stopping, same protocol as the distilled models",
        "summary": summary,
        "references": {
            "teacher_auc_truth": 0.9566,
            "distilled_triple_auc_truth": 0.9364,
            "distilled_unconstrained_auc_truth": 0.9428,
            "nam_auc_truth": 0.9206,
        },
    }
    path = ROOT / "experiments" / "results" / "kirk_truth.json"
    path.write_text(json.dumps(out, indent=1, default=float))
    for name in blocks:
        s = summary[name]
        print(
            f"{name:22s} aucY {s['auc_truth_mean']:.4f}±{s['auc_truth_std']:.4f} "
            f"acc {s['acc_truth_mean']:.4f} agreeT {s['agree_teacher_mean']:.4f}"
        )
    print(f"saved {path}")


if __name__ == "__main__":
    main()
