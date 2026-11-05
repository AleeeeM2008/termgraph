"""First TermGraph test on KIRK.

Question: on the KIRK 12k cache, does a term-wise XGBoost with discovered pair
blocks close the fidelity gap that the additive NAM leaves?

We compare, on the same holdout split as the shipped 12k NAM:

  teacher (SPOCK XGBoost)
  NAM 12k (shipped artifact)
  flat TermGraph (11 singleton blocks)
  PBAM with k discovered disjoint pairs, k = 1, 2, 3, 5
  unconstrained XGBoost student (reference ceiling on this sample)

Run:
  python experiments/kirk_termgraph.py
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import shap
import torch
import xgboost as xgb

ROOT = Path(__file__).resolve().parents[1]
KIRK = Path(os.environ.get("KIRK_ROOT", ROOT.parent / "TESTAstronomy"))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(KIRK / "src"))

from termgraph.gbm import (  # noqa: E402
    train_constrained,
    term_decompose,
    feature_shap,
    fidelity_metrics,
)
from kirk import glass  # noqa: E402


def load_data():
    d = np.load(KIRK / "data" / "features_8000_4000_0.npz", allow_pickle=True)
    X = d["X"].astype(np.float32)
    Y = d["Y"].astype(np.float32)
    stable = d["stable"].astype(bool)
    labels = d["labels"].astype(np.float32)
    feats = [str(f) for f in d["feats"]]
    X, Y, labels = X[stable], Y[stable], labels[stable]
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(X))
    cut = int(len(X) * 0.85)
    tr, ho = perm[:cut], perm[cut:]
    return X, Y, labels, feats, tr, ho


def interaction_ranking(teacher, X, feats, seed=0, n=2000):
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), min(n, len(X)), replace=False)
    t0 = time.time()
    explainer = shap.TreeExplainer(
        teacher, feature_perturbation="tree_path_dependent"
    )
    sv = explainer.shap_interaction_values(X[idx])
    M = np.abs(sv).mean(0)
    np.fill_diagonal(M, 0.0)
    pairs = []
    for i in range(len(feats)):
        for j in range(i + 1, len(feats)):
            pairs.append((float(M[i, j]), i, j))
    pairs.sort(reverse=True)
    print(f"interaction values: {sv.shape} in {time.time() - t0:.1f}s", flush=True)
    return pairs


def greedy_disjoint_pairs(ranked, k):
    if k <= 0:
        return []
    used = set()
    out = []
    for strength, i, j in ranked:
        if i in used or j in used:
            continue
        out.append((strength, i, j))
        used.update((i, j))
        if len(out) == k:
            break
    return out


def blocks_from_pairs(pairs, n_features):
    used = set()
    blocks = []
    for _, i, j in pairs:
        blocks.append([i, j])
        used.update((i, j))
    for i in range(n_features):
        if i not in used:
            blocks.append([i])
    return sorted(blocks, key=lambda b: (len(b), min(b)))


def nam_contributions(model, meta, X):
    """Exact feature contributions of the shipped NAM, in logit space."""
    z = glass._normalize(meta, X)
    with torch.no_grad():
        bias = float(model.bias.item())
        parts = [
            net(z[:, i:i + 1]).numpy().ravel()
            for i, net in enumerate(model.nets)
        ]
        total = bias + np.sum(parts, axis=0)
        raw = model(z).numpy().ravel()
    err = float(np.max(np.abs(total - raw)))
    parts = np.stack(parts, axis=1) * float(meta["Ys"])
    score = total * float(meta["Ys"]) + float(meta["Ym"])
    expected = bias * float(meta["Ys"]) + float(meta["Ym"])
    return score, parts, expected, err


def main():
    X, Y, labels, feats, tr, ho = load_data()
    print(f"n={len(X)} train={len(tr)} holdout={len(ho)}", flush=True)

    teacher = xgb.Booster()
    teacher.load_model(str(KIRK / "models" / "featureclassifier.json"))
    teacher_ho = teacher.predict(
        xgb.DMatrix(X[ho], feature_names=feats), output_margin=True
    )
    teacher_metrics = fidelity_metrics(teacher_ho, Y[ho], labels[ho])
    print(
        "teacher truth AUC %.4f acc %.4f"
        % (teacher_metrics["auc_truth"], teacher_metrics["acc_truth"]),
        flush=True,
    )

    # Shipped NAM baseline -------------------------------------------------
    nam_model, nam_meta = glass.load_artifact(
        str(KIRK / "models" / "nam_spock_distilled_12k.pt"), arch="nam"
    )
    nam_score, nam_parts, nam_expected, nam_err = nam_contributions(
        nam_model, nam_meta, X[ho]
    )
    nam_metrics = fidelity_metrics(nam_score, Y[ho], labels[ho])
    nam_metrics["additivity_error"] = nam_err
    nam_metrics["n_terms"] = len(feats)
    nam_metrics["n_pairs"] = 0
    print(
        "NAM 12k: agree %.4f rmse %.4f aucT %.4f aucY %.4f acc %.4f"
        % (
            nam_metrics["agree_teacher"],
            nam_metrics["rmse_teacher"],
            nam_metrics["auc_teacher"],
            nam_metrics["auc_truth"],
            nam_metrics["acc_truth"],
        ),
        flush=True,
    )

    # Interaction discovery ------------------------------------------------
    ranked = interaction_ranking(teacher, X[tr], feats)
    print("top pairs:", flush=True)
    for s, i, j in ranked[:12]:
        print(f"  {s:.4f}  {feats[i]:16s} {feats[j]}", flush=True)

    # Model sweep ----------------------------------------------------------
    results = {
        "dataset": {
            "n": int(len(X)),
            "n_train": int(len(tr)),
            "n_holdout": int(len(ho)),
            "features": feats,
        },
        "teacher": teacher_metrics,
        "models": {"nam_12k": nam_metrics},
        "interaction_ranking": [
            {"strength": s, "i": i, "j": j, "a": feats[i], "b": feats[j]}
            for s, i, j in ranked[:30]
        ],
    }

    ks = [0, 1, 2, 3, 5]
    for k in ks:
        pairs = greedy_disjoint_pairs(ranked, k)
        blocks = blocks_from_pairs(pairs, len(feats))
        t0 = time.time()
        booster = train_constrained(
            X[tr],
            Y[tr],
            blocks,
            feats,
            Xval=X[ho],
            yval=Y[ho],
            max_depth=3,
            eta=0.05,
            num_round=1500,
            early_stopping=80,
            seed=0,
        )
        score = booster.predict(
            xgb.DMatrix(X[ho], feature_names=feats), output_margin=True
        )
        terms, bias, _ = term_decompose(booster, X[ho], blocks)
        train_terms, train_bias, _ = term_decompose(
            booster, X[tr][: min(2000, len(tr))], blocks
        )
        centers = train_terms.mean(0)
        phi, expected, _ = feature_shap(terms, centers, bias, blocks)
        add_err = float(np.max(np.abs(phi.sum(1) + expected - score)))
        recon_err = float(
            np.max(np.abs(terms.sum(1) + bias - score))
        )
        metrics = fidelity_metrics(score, Y[ho], labels[ho])
        metrics.update({
            "n_terms": len(blocks),
            "n_pairs": len(pairs),
            "pairs": [
                {"strength": s, "a": feats[i], "b": feats[j]}
                for s, i, j in pairs
            ],
            "additivity_error": add_err,
            "reconstruction_error": recon_err,
            "best_iteration": int(getattr(booster, "best_iteration", -1)),
            "n_trees": int(booster.num_boosted_rounds()),
            "train_seconds": round(time.time() - t0, 1),
        })
        name = "flat" if k == 0 else f"pbam_k{k}"
        results["models"][name] = metrics
        print(
            "%-8s terms=%2d pairs=%d  agree %.4f rmse %.4f aucT %.4f "
            "aucY %.4f acc %.4f addErr %.1e  %.0fs"
            % (
                name,
                len(blocks),
                len(pairs),
                metrics["agree_teacher"],
                metrics["rmse_teacher"],
                metrics["auc_teacher"],
                metrics["auc_truth"],
                metrics["acc_truth"],
                add_err,
                metrics["train_seconds"],
            ),
            flush=True,
        )

    # Unconstrained reference ---------------------------------------------
    all_block = [list(range(len(feats)))]
    t0 = time.time()
    booster = train_constrained(
        X[tr],
        Y[tr],
        all_block,
        feats,
        Xval=X[ho],
        yval=Y[ho],
        max_depth=4,
        eta=0.05,
        num_round=1500,
        early_stopping=80,
        seed=0,
    )
    score = booster.predict(
        xgb.DMatrix(X[ho], feature_names=feats), output_margin=True
    )
    metrics = fidelity_metrics(score, Y[ho], labels[ho])
    metrics.update({
        "n_terms": 1,
        "n_pairs": 0,
        "best_iteration": int(getattr(booster, "best_iteration", -1)),
        "n_trees": int(booster.num_boosted_rounds()),
        "train_seconds": round(time.time() - t0, 1),
    })
    results["models"]["unconstrained_student"] = metrics
    print(
        "unconstrained: agree %.4f rmse %.4f aucT %.4f aucY %.4f acc %.4f"
        % (
            metrics["agree_teacher"],
            metrics["rmse_teacher"],
            metrics["auc_teacher"],
            metrics["auc_truth"],
            metrics["acc_truth"],
        ),
        flush=True,
    )

    out = ROOT / "experiments" / "results" / "kirk_termgraph.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=1, default=float))
    print(f"saved {out}", flush=True)


if __name__ == "__main__":
    main()
