"""Does one explicit 3-way block beat the disjoint pair model?

Fix a triple, greedily add disjoint pairs from the interaction ranking, and
compare with the 6-term PBAM k=5 on the same split. Seed 0 first; if a triple
wins clearly, the script can be extended to a seed sweep.
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


def blocks_with_triple(triple, ranked, n_features):
    used = set(triple)
    blocks = [list(triple)]
    for _, i, j in ranked:
        if i in used or j in used:
            continue
        blocks.append([i, j])
        used.update((i, j))
    for i in range(n_features):
        if i not in used:
            blocks.append([i])
    return sorted(blocks, key=lambda b: (-len(b), min(b)))


def fit_eval(X, Y, labels, feats, tr, ho, blocks, seed=0):
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
        "n_terms": len(blocks),
        "blocks": [[feats[i] for i in b] for b in blocks],
        "reconstruction_error": err,
        "n_trees": int(booster.num_boosted_rounds()),
        "seconds": round(time.time() - t0, 1),
    })
    return m


def main():
    X, Y, labels, feats, tr, ho = load_data()
    teacher = xgb.Booster()
    teacher.load_model(str(KIRK / "models" / "featureclassifier.json"))
    ranked = interaction_ranking(teacher, X[tr], feats, seed=0, n=2000)

    # Baseline: disjoint pairs, no triple.
    baseline_blocks = blocks_from_pairs(greedy_disjoint_pairs(ranked, 5), len(feats))
    baseline = fit_eval(X, Y, labels, feats, tr, ho, baseline_blocks)
    print(
        "baseline pbam_k5  terms=%d aucY %.4f rmse %.4f agree %.4f"
        % (baseline["n_terms"], baseline["auc_truth"],
           baseline["rmse_teacher"], baseline["agree_teacher"]),
        flush=True,
    )

    triples = [
        ("EMfracstdnear", "MEGNO", "MEGNOstd"),
        ("EMfracstdnear", "MEGNO", "Tsec"),
        ("EMcrossnear", "MMRstrengthnear", "MEGNO"),
        ("EMcrossnear", "MMRstrengthnear", "EMfracstdnear"),
        ("EMcrossnear", "MMRstrengthnear", "Tsec"),
    ]
    results = {"baseline": baseline, "triples": {}}
    best = None
    for names in triples:
        triple = [feats.index(n) for n in names]
        blocks = blocks_with_triple(triple, ranked, len(feats))
        m = fit_eval(X, Y, labels, feats, tr, ho, blocks)
        key = " + ".join(names)
        results["triples"][key] = m
        print(
            "%-46s terms=%d aucY %.4f rmse %.4f agree %.4f"
            % (key, m["n_terms"], m["auc_truth"],
               m["rmse_teacher"], m["agree_teacher"]),
            flush=True,
        )
        if best is None or m["auc_truth"] > best[1]["auc_truth"]:
            best = (key, m)

    results["best_triple"] = best[0]
    results["delta_auc_vs_baseline"] = (
        best[1]["auc_truth"] - baseline["auc_truth"]
    )
    path = ROOT / "experiments" / "results" / "kirk_threeway.json"
    path.write_text(json.dumps(results, indent=1, default=float))
    print(
        "best triple: %s  dAUC %+0.4f"
        % (best[0], results["delta_auc_vs_baseline"]),
        flush=True,
    )
    print(f"saved {path}")


if __name__ == "__main__":
    main()
