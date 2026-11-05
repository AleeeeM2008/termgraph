"""Seed sweep: flat TermGraph vs pair blocks vs the shipped NAM sweep.

The 12k NAM reports a five-seed mean of 0.9206 AUC-vs-truth. Its seed-0
artifact is 0.9083. A single-seed comparison is therefore not enough. This
script trains the term models on the same five splits and reports mean/std and
paired deltas.
"""
from __future__ import annotations
import os

import json
import sys
import time
from pathlib import Path

import numpy as np
import shap
import xgboost as xgb

ROOT = Path(__file__).resolve().parents[2]
KIRK = Path(os.environ.get("KIRK_ROOT", Path(__file__).resolve().parents[1].parent / "TESTAstronomy"))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT / "experiments" / "legacy"))
sys.path.insert(0, str(KIRK / "src"))

from termgraph.gbm import (  # noqa: E402
    train_constrained,
    term_decompose,
    fidelity_metrics,
)

from kirk_termgraph import (  # noqa: E402
    load_data,
    interaction_ranking,
    greedy_disjoint_pairs,
    blocks_from_pairs,
)


def evaluate(booster, X, feats, Xho, Yho, yho, blocks=None):
    score = booster.predict(
        xgb.DMatrix(Xho, feature_names=feats), output_margin=True
    )
    m = fidelity_metrics(score, Yho, yho)
    if blocks is not None:
        terms, bias, _ = term_decompose(booster, Xho, blocks)
        err = float(np.max(np.abs(terms.sum(1) + bias - score)))
        m["reconstruction_error"] = err
    return m


def main():
    X, Y, labels, feats, tr0, ho0 = load_data()
    # Fixed pair structure chosen once on the seed-0 training split.
    teacher = xgb.Booster()
    teacher.load_model(str(KIRK / "models" / "featureclassifier.json"))
    ranked = interaction_ranking(teacher, X[tr0], feats, seed=0, n=2000)
    pair_sets = {
        "flat": [],
        "pbam_k1": greedy_disjoint_pairs(ranked, 1),
        "pbam_k3": greedy_disjoint_pairs(ranked, 3),
        "pbam_k5": greedy_disjoint_pairs(ranked, 5),
    }
    for name, pairs in pair_sets.items():
        print(name, [(feats[i], feats[j]) for _, i, j in pairs], flush=True)

    rows = {name: [] for name in pair_sets}
    for seed in range(5):
        rng = np.random.default_rng(seed)
        perm = rng.permutation(len(X))
        cut = int(len(X) * 0.85)
        tr, ho = perm[:cut], perm[cut:]
        for name, pairs in pair_sets.items():
            blocks = blocks_from_pairs(pairs, len(feats))
            t0 = time.time()
            booster = train_constrained(
                X[tr], Y[tr], blocks, feats,
                Xval=X[ho], yval=Y[ho],
                max_depth=3, eta=0.05, num_round=1500,
                early_stopping=80, seed=seed,
            )
            m = evaluate(booster, X, feats, X[ho], Y[ho], labels[ho], blocks)
            m["seed"] = seed
            m["n_terms"] = len(blocks)
            m["n_pairs"] = len(pairs)
            m["seconds"] = round(time.time() - t0, 1)
            rows[name].append(m)
            print(
                f"seed {seed} {name:8s} aucY {m['auc_truth']:.4f} "
                f"aucT {m['auc_teacher']:.4f} agree {m['agree_teacher']:.4f} "
                f"rmse {m['rmse_teacher']:.4f} acc {m['acc_truth']:.4f}",
                flush=True,
            )

    summary = {}
    for name, ms in rows.items():
        summary[name] = {}
        for key in ("auc_truth", "auc_teacher", "agree_teacher",
                    "rmse_teacher", "acc_truth", "ece_truth"):
            vals = np.array([m[key] for m in ms], dtype=float)
            summary[name][key + "_mean"] = float(vals.mean())
            summary[name][key + "_std"] = float(vals.std())
        summary[name]["rows"] = ms

    # Paired deltas vs flat, across the same five seeds.
    deltas = {}
    for name in ("pbam_k1", "pbam_k3", "pbam_k5"):
        d = {}
        for key in ("auc_truth", "agree_teacher", "rmse_teacher", "acc_truth"):
            a = np.array([m[key] for m in rows[name]])
            b = np.array([m[key] for m in rows["flat"]])
            diff = a - b
            d[key + "_mean"] = float(diff.mean())
            d[key + "_std"] = float(diff.std())
            d[key + "_per_seed"] = [float(v) for v in diff]
        deltas[name] = d

    out = {
        "pair_sets": {
            name: [{"strength": s, "a": feats[i], "b": feats[j]}
                   for s, i, j in pairs]
            for name, pairs in pair_sets.items()
        },
        "summary": summary,
        "paired_delta_vs_flat": deltas,
        "nam_12k_reference": {
            "auc_truth_mean": 0.9205843779263286,
            "auc_truth_std": 0.0068331514796418965,
            "agree_mean": 0.9009549795361528,
            "agree_std": 0.0079712748474715,
            "auc_teacher_mean": 0.9696179503961766,
            "auc_teacher_std": 0.004514893097663005,
            "acc_mean": 0.8409276944065484,
            "acc_std": 0.006826737245946434,
        },
    }
    path = ROOT / "experiments" / "results" / "kirk_seed_sweep.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1, default=float))
    print(f"\nsaved {path}", flush=True)

    print("\nmodel        aucY (mean±std)   agree      rmse      acc")
    for name in ("flat", "pbam_k1", "pbam_k3", "pbam_k5"):
        s = summary[name]
        print(
            f"{name:8s}  {s['auc_truth_mean']:.4f}±{s['auc_truth_std']:.4f}  "
            f"{s['agree_teacher_mean']:.4f}  {s['rmse_teacher_mean']:.4f}  "
            f"{s['acc_truth_mean']:.4f}"
        )
    print("\npaired delta vs flat (mean±std over 5 seeds)")
    for name, d in deltas.items():
        print(
            f"{name}: dAUC {d['auc_truth_mean']:+.4f}±{d['auc_truth_std']:.4f}  "
            f"dAgree {d['agree_teacher_mean']:+.4f}±{d['agree_teacher_std']:.4f}  "
            f"dRMSE {d['rmse_teacher_mean']:+.4f}±{d['rmse_teacher_std']:.4f}"
        )


if __name__ == "__main__":
    main()
