"""Constrained-GBM term decomposition.

The model is a normal XGBoost ensemble, but every tree is restricted to a
disjoint block of features via ``interaction_constraints``. With disjoint
blocks, a tree's split features all lie in exactly one block, so the ensemble
decomposes exactly:

    margin(x) = base + sum_k T_k(x[block_k])

No distillation, no surrogate. This module trains that model and reads the
terms back out of the XGBoost JSON.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence

import numpy as np
import xgboost as xgb
from sklearn.metrics import roc_auc_score


def _as_int_list(blocks: Iterable[Iterable[int]]) -> list[list[int]]:
    return [sorted(int(i) for i in block) for block in blocks]


def _check_blocks(blocks: Sequence[Sequence[int]], n_features: int) -> list[list[int]]:
    blocks = _as_int_list(blocks)
    seen: set[int] = set()
    for block in blocks:
        for i in block:
            if i in seen:
                raise ValueError(f"feature {i} appears in more than one block")
            seen.add(i)
    missing = sorted(set(range(n_features)) - seen)
    if missing:
        blocks.extend([[i] for i in missing])
    return blocks


def train_constrained(
    X: np.ndarray,
    y: np.ndarray,
    blocks: Sequence[Sequence[int]],
    feature_names: Sequence[str],
    *,
    Xval: np.ndarray | None = None,
    yval: np.ndarray | None = None,
    max_depth: int = 3,
    eta: float = 0.05,
    num_round: int = 1500,
    early_stopping: int = 80,
    subsample: float = 0.8,
    min_child_weight: float = 20.0,
    reg_lambda: float = 1.0,
    seed: int = 0,
    objective: str = "reg:squarederror",
    eval_metric: str | None = None,
    verbose: bool = False,
):
    """Train an XGBoost model whose trees are confined to disjoint blocks."""
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    n_features = X.shape[1]
    blocks = _check_blocks(blocks, n_features)
    constraints = [[str(feature_names[i]) for i in block] for block in blocks]

    params = {
        "objective": objective,
        "max_depth": int(max_depth),
        "eta": float(eta),
        "subsample": float(subsample),
        "min_child_weight": float(min_child_weight),
        "reg_lambda": float(reg_lambda),
        "tree_method": "hist",
        "seed": int(seed),
        "interaction_constraints": constraints,
        "verbosity": 0,
    }
    if eval_metric is not None:
        params["eval_metric"] = eval_metric

    dtrain = xgb.DMatrix(X, label=y, feature_names=list(feature_names))
    evals = [(dtrain, "train")]
    if Xval is not None and yval is not None:
        dval = xgb.DMatrix(
            np.asarray(Xval, dtype=np.float32),
            label=np.asarray(yval, dtype=np.float32),
            feature_names=list(feature_names),
        )
        evals.append((dval, "val"))

    booster = xgb.train(
        params,
        dtrain,
        num_boost_round=int(num_round),
        evals=evals,
        early_stopping_rounds=int(early_stopping) if len(evals) > 1 else None,
        verbose_eval=verbose,
    )
    return booster


def parse_model(booster: xgb.Booster):
    """Return (trees, base_margin, raw_json_doc)."""
    doc = json.loads(booster.save_raw(raw_format="json"))
    learner = doc["learner"]
    trees = learner["gradient_booster"]["model"]["trees"]
    base = learner["learner_model_param"]["base_score"]
    if isinstance(base, str):
        base = json.loads(base)
    base = float(np.asarray(base, dtype=np.float64).ravel()[0])
    return trees, base, doc


def tree_features(tree: dict) -> set[int]:
    """Feature indices used by the internal nodes of one tree."""
    left = np.asarray(tree["left_children"], dtype=np.int64)
    split = np.asarray(tree["split_indices"], dtype=np.int64)
    return {int(i) for i in split[left != -1]}


def tree_outputs(tree: dict, X: np.ndarray) -> np.ndarray:
    """Evaluate one tree on every row of X."""
    X = np.asarray(X, dtype=np.float32)
    n = X.shape[0]
    out = np.zeros(n, dtype=np.float64)
    left = np.asarray(tree["left_children"], dtype=np.int64)
    right = np.asarray(tree["right_children"], dtype=np.int64)
    split = np.asarray(tree["split_indices"], dtype=np.int64)
    cond = np.asarray(tree["split_conditions"], dtype=np.float64)
    default_left = np.asarray(tree["default_left"], dtype=np.int64)
    weight = np.asarray(tree["base_weights"], dtype=np.float64)

    stack: list[tuple[int, np.ndarray]] = [(0, np.arange(n, dtype=np.int64))]
    while stack:
        node, idx = stack.pop()
        if idx.size == 0:
            continue
        if left[node] == -1:
            out[idx] = weight[node]
            continue
        f = int(split[node])
        v = X[idx, f]
        thr = cond[node]
        go_left = np.where(np.isnan(v), default_left[node] == 1, v < thr)
        stack.append((int(left[node]), idx[go_left]))
        stack.append((int(right[node]), idx[~go_left]))
    return out


def term_decompose(
    booster: xgb.Booster,
    X: np.ndarray,
    blocks: Sequence[Sequence[int]],
):
    """Decompose a constrained booster into exact block terms.

    Returns
    -------
    terms : (n, K) array
    bias : float
    tree_blocks : list[int]
        Block index assigned to each tree.
    """
    X = np.asarray(X, dtype=np.float32)
    blocks = _as_int_list(blocks)
    feature_to_block = {i: k for k, block in enumerate(blocks) for i in block}

    trees, base, _ = parse_model(booster)
    terms = np.zeros((X.shape[0], len(blocks)), dtype=np.float64)
    constant = 0.0
    tree_blocks: list[int] = []

    for tree in trees:
        fs = tree_features(tree)
        if not fs:
            constant += float(tree_outputs(tree, X[:1])[0])
            tree_blocks.append(-1)
            continue
        candidates = {feature_to_block[i] for i in fs}
        if len(candidates) != 1:
            raise ValueError(
                "tree uses features from multiple blocks; constraints are not disjoint: "
                f"{sorted(fs)}"
            )
        k = candidates.pop()
        terms[:, k] += tree_outputs(tree, X)
        tree_blocks.append(k)

    return terms, base + constant, tree_blocks


def feature_shap(
    terms: np.ndarray,
    centers: np.ndarray,
    bias: float,
    blocks: Sequence[Sequence[int]],
):
    """Feature-level exact contributions under the independence convention.

    For a pair block, the centered term is split equally between its two
    features. That is the Shapley value of a symmetric two-player game and it
    preserves additivity. For correlated features, the baseline convention still
    matters; this is the interventional/marginal version.
    """
    terms = np.asarray(terms, dtype=np.float64)
    centers = np.asarray(centers, dtype=np.float64)
    phi_block = terms - centers
    n_features = max(max(b) for b in blocks) + 1
    phi = np.zeros((terms.shape[0], n_features), dtype=np.float64)
    for k, block in enumerate(blocks):
        share = phi_block[:, k] / len(block)
        for i in block:
            phi[:, i] += share
    expected = float(bias + centers.sum())
    return phi, expected, phi_block


def _safe_auc(y_true, score) -> float:
    y_true = np.asarray(y_true)
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, score))


def fidelity_metrics(
    score: np.ndarray,
    teacher_logit: np.ndarray,
    truth: np.ndarray,
    *,
    bins: int = 5,
) -> dict:
    """Prediction quality against the teacher and against the real labels."""
    score = np.asarray(score, dtype=np.float64)
    teacher_logit = np.asarray(teacher_logit, dtype=np.float64)
    truth = np.asarray(truth).astype(np.float64)
    p = 1.0 / (1.0 + np.exp(-score))
    pt = 1.0 / (1.0 + np.exp(-teacher_logit))

    sse = float(np.sum((score - teacher_logit) ** 2))
    sst = float(np.sum((teacher_logit - teacher_logit.mean()) ** 2))
    r2 = 1.0 - sse / sst if sst > 0 else float("nan")

    edges = np.linspace(0.0, 1.0, bins + 1)
    which = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    cal = []
    ece = 0.0
    for b in range(bins):
        m = which == b
        if not m.any():
            cal.append({"bin": [float(edges[b]), float(edges[b + 1])], "n": 0})
            continue
        pred_mean = float(p[m].mean())
        obs_mean = float(truth[m].mean())
        ece += float(m.mean()) * abs(pred_mean - obs_mean)
        cal.append({
            "bin": [float(edges[b]), float(edges[b + 1])],
            "n": int(m.sum()),
            "pred": pred_mean,
            "obs": obs_mean,
        })

    return {
        "agree_teacher": float(((p > 0.5) == (pt > 0.5)).mean()),
        "rmse_teacher": float(np.sqrt(np.mean((score - teacher_logit) ** 2))),
        "r2_teacher": float(r2),
        "auc_teacher": _safe_auc((pt > 0.5).astype(np.int64), score),
        "auc_truth": _safe_auc((truth > 0.5).astype(np.int64), score),
        "acc_truth": float(((p > 0.5) == (truth > 0.5)).mean()),
        "ece_truth": float(ece),
        "calibration": cal,
    }
