"""Five-seed test: pair-only PBAM vs one explicit 3-way block."""
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
from kirk_threeway import blocks_with_triple  # noqa: E402


def run(blocks, X, Y, labels, feats, tr, ho, seed):
    t0 = time.time()
    booster = train_constrained(
        X[tr], Y[tr], blocks, feats,
        Xval=X[ho], yval=Y[ho],
        max_depth=3, eta=0.05, num_round=1500,
        early_stopping=80, seed=seed,
    )
    score = booster.predict(
        xgb.DMatrix(X[ho], feature_names=feats), output_margin=True
    )
    terms, bias, _ = term_decompose(booster, X[ho], blocks)
    err = float(np.max(np.abs(terms.sum(1) + bias - score)))
    m = fidelity_metrics(score, Y[ho], labels[ho])
    m.update({
        "seed": seed,
        "n_terms": len(blocks),
        "reconstruction_error": err,
        "seconds": round(time.time() - t0, 1),
    })
    return m


def main():
    X, Y, labels, feats, tr0, ho0 = load_data()
    teacher = xgb.Booster()
    teacher.load_model(str(KIRK / "models" / "featureclassifier.json"))
    ranked = interaction_ranking(teacher, X[tr0], feats, seed=0, n=2000)

    baseline_blocks = blocks_from_pairs(
        greedy_disjoint_pairs(ranked, 5), len(feats)
    )
    triple_sets = {
        "triple_resonance": ["EMcrossnear", "MMRstrengthnear", "EMfracstdnear"],
        "triple_phase": ["EMfracstdnear", "MEGNO", "Tsec"],
    }
    variants = {"pbam_k5": baseline_blocks}
    for name, names in triple_sets.items():
        triple = [feats.index(n) for n in names]
        variants[name] = blocks_with_triple(triple, ranked, len(feats))

    rows = {name: [] for name in variants}
    for seed in range(5):
        rng = np.random.default_rng(seed)
        perm = rng.permutation(len(X))
        cut = int(len(X) * 0.85)
        tr, ho = perm[:cut], perm[cut:]
        for name, blocks in variants.items():
            m = run(blocks, X, Y, labels, feats, tr, ho, seed)
            rows[name].append(m)
            print(
                f"seed {seed} {name:16s} aucY {m['auc_truth']:.4f} "
                f"rmse {m['rmse_teacher']:.4f} agree {m['agree_teacher']:.4f} "
                f"terms {m['n_terms']}",
                flush=True,
            )

    summary = {}
    for name, ms in rows.items():
        summary[name] = {}
        for key in ("auc_truth", "auc_teacher", "agree_teacher",
                    "rmse_teacher", "acc_truth"):
            vals = np.array([m[key] for m in ms])
            summary[name][key + "_mean"] = float(vals.mean())
            summary[name][key + "_std"] = float(vals.std())
        summary[name]["n_terms"] = ms[0]["n_terms"]
        summary[name]["rows"] = ms

    deltas = {}
    for name in triple_sets:
        d = {}
        for key in ("auc_truth", "agree_teacher", "rmse_teacher"):
            a = np.array([m[key] for m in rows[name]])
            b = np.array([m[key] for m in rows["pbam_k5"]])
            diff = a - b
            d[key + "_mean"] = float(diff.mean())
            d[key + "_std"] = float(diff.std())
            d[key + "_per_seed"] = [float(v) for v in diff]
        deltas[name] = d

    out = {
        "blocks": {
            name: [[feats[i] for i in b] for b in blocks]
            for name, blocks in variants.items()
        },
        "summary": summary,
        "paired_delta_vs_pbam_k5": deltas,
    }
    path = ROOT / "experiments" / "results" / "kirk_threeway_sweep.json"
    path.write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if not kk.endswith("_per_seed")}
                      for k, v in deltas.items()}, indent=1))
    print(f"saved {path}")


if __name__ == "__main__":
    main()
