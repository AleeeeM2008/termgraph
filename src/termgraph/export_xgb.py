"""Compile a constrained XGBoost term into a TermGraph table.

The booster must be trained on the block's columns only (no base margin at
prediction time, or the base margin is the EBM offset and is added separately).
The term is evaluated on an explicit raw-value grid; the table error is
certified.
"""

from __future__ import annotations

import json

import numpy as np
import xgboost as xgb

from .export_ebm import _grid, _representatives, quantile_edges
from .ir import Term


def _threshold_edges(booster: xgb.Booster, block: list[int], X_train: np.ndarray):
    """Exact per-feature split thresholds from the tree ensemble."""
    doc = json.loads(booster.save_raw(raw_format="json"))
    trees = doc["learner"]["gradient_booster"]["model"]["trees"]
    cuts = [set() for _ in block]
    for tree in trees:
        left = tree["left_children"]
        for node, lc in enumerate(left):
            if lc != -1:
                f = int(tree["split_indices"][node])
                if 0 <= f < len(block):
                    cuts[f].add(float(tree["split_conditions"][node]))
    edges = []
    for j, f in enumerate(block):
        x = X_train[:, f]
        c = np.asarray(sorted(cuts[j]), dtype=np.float64)
        e = np.concatenate(([-1e300], c, [1e300]))
        e = np.unique(e)
        if len(e) < 2:
            e = quantile_edges(x, 48)
        edges.append(e)
    return edges


def export_block_terms(
    booster: xgb.Booster,
    feature_names: list[str],
    blocks: list[list[int]],
    X_train: np.ndarray,
    *,
    max_bins: int = 24,
    seed: int = 0,
    link: str = "logit",
) -> tuple[list[Term], float, dict]:
    """Export every disjoint block of a constrained booster as a Term.

    The booster must have been trained on all features with the given disjoint
    interaction constraints. Terms are evaluated on a grid of block features;
    if a block has more thresholds than ``max_bins``, quantile bins are used and
    the approximation error is reported.
    """
    from .gbm import parse_model, tree_features, tree_outputs

    X_train = np.asarray(X_train, dtype=np.float64)
    trees, base, _ = parse_model(booster)
    feature_to_block = {f: k for k, block in enumerate(blocks) for f in block}
    thresholds: list[dict[int, set]] = [
        {f: set() for f in block} for block in blocks
    ]
    block_trees: list[list[dict]] = [[] for _ in blocks]
    constant = 0.0
    for tree in trees:
        fs = tree_features(tree)
        if not fs:
            constant += float(tree_outputs(tree, X_train[:1])[0])
            continue
        ks = {feature_to_block[i] for i in fs}
        if len(ks) != 1:
            raise ValueError(f"tree uses multiple blocks: {sorted(fs)}")
        k = ks.pop()
        block_trees[k].append(tree)
        left = tree["left_children"]
        for node, lc in enumerate(left):
            if lc != -1:
                f = int(tree["split_indices"][node])
                thresholds[k][f].add(float(tree["split_conditions"][node]))

    rng = np.random.default_rng(seed)
    sample = X_train[rng.choice(len(X_train), min(2000, len(X_train)), replace=False)]
    centers = np.zeros(len(blocks))
    for k, trees_k in enumerate(block_trees):
        centers[k] = float(np.mean(
            sum(tree_outputs(t, sample) for t in trees_k)
            if trees_k else np.zeros(len(sample))
        ))

    if link == "logit":
        bias = float(np.log(base / (1 - base)) + constant)
    elif link == "identity":
        bias = float(base + constant)
    else:
        raise ValueError(f"unknown link {link!r}")
    total_bias = float(bias + centers.sum())
    bg = np.median(X_train, axis=0)
    terms: list[Term] = []
    diagnostics = {}
    for k, block in enumerate(blocks):
        edges = []
        approx = False
        for f in block:
            cuts = np.asarray(sorted(thresholds[k][f]), dtype=np.float64)
            if len(cuts) + 1 > max_bins:
                e = quantile_edges(X_train[:, f], max_bins)
                approx = True
            else:
                e = np.concatenate(([-1e300], cuts, [1e300]))
                e = np.unique(e)
            edges.append(e)
        reps = [_representatives(e) for e in edges]
        mesh = np.meshgrid(*reps, indexing="ij")
        grid = np.column_stack([m.ravel() for m in mesh])
        Xg = np.tile(bg, (len(grid), 1))
        Xg[:, block] = grid
        vals = np.zeros(len(grid))
        for tree in block_trees[k]:
            vals += tree_outputs(tree, Xg)
        shape = tuple(len(e) - 1 for e in edges)
        tid = "*".join(feature_names[f] for f in block)
        term = Term(
            id=tid,
            inputs=[feature_names[f] for f in block],
            edges=edges,
            values=vals.reshape(shape),
            center=float(centers[k]),
            provenance={
                "backend": "xgboost",
                "block": [int(f) for f in block],
                "grid_bins": [int(s) for s in shape],
                "approximate": approx,
                "trees": len(block_trees[k]),
            },
            backend="xgboost",
            split_side="right",
        )
        terms.append(term)
        got = term.lookup(sample, {f: i for i, f in enumerate(feature_names)})
        want = (
            sum(tree_outputs(t, sample) for t in block_trees[k])
            if block_trees[k] else np.zeros(len(sample))
        ) - centers[k]
        diagnostics[tid] = {
            "rmse": float(np.sqrt(np.mean((got - want) ** 2))),
            "max_abs": float(np.max(np.abs(got - want))),
            "approximate": approx,
            "grid_cells": int(np.prod(shape)),
            "trees": len(block_trees[k]),
        }

    score = total_bias + sum(
        t.lookup(sample, {f: i for i, f in enumerate(feature_names)})
        for t in terms
    )
    exact = booster.predict(
        xgb.DMatrix(sample, feature_names=feature_names), output_margin=True
    )
    diagnostics["total_rmse"] = float(np.sqrt(np.mean((score - exact) ** 2)))
    diagnostics["total_max_abs"] = float(np.max(np.abs(score - exact)))
    return terms, total_bias, diagnostics


def export_xgb_term(
    booster: xgb.Booster,
    feature_names: list[str],
    block: list[int],
    X_train: np.ndarray,
    *,
    n_bins: int = 48,
    seed: int = 0,
    term_id: str | None = None,
    exact: bool = True,
) -> tuple[Term, dict]:
    X_train = np.asarray(X_train, dtype=np.float64)
    if exact:
        edges = _threshold_edges(booster, block, X_train)
    else:
        edges = [quantile_edges(X_train[:, f], n_bins) for f in block]
    grid = _grid(edges)
    vals = booster.predict(xgb.DMatrix(grid), output_margin=True)
    shape = tuple(len(e) - 1 for e in edges)
    vals = np.asarray(vals, dtype=np.float64).reshape(shape)

    rng = np.random.default_rng(seed)
    sample = X_train[rng.choice(len(X_train), min(2000, len(X_train)), replace=False)]
    exact = booster.predict(xgb.DMatrix(sample[:, block]), output_margin=True)
    center = float(np.mean(exact))

    tid = term_id or "*".join(feature_names[f] for f in block)
    term = Term(
        id=tid,
        inputs=[feature_names[f] for f in block],
        edges=edges,
        values=vals,
        center=center,
        provenance={
            "backend": "xgboost",
            "block": [int(b) for b in block],
            "grid_bins": [int(len(e) - 1) for e in edges],
        },
        backend="xgboost",
        split_side="left",
    )
    col = {f: i for i, f in enumerate(feature_names)}
    got = term.lookup(sample, col)
    want = exact - center
    diag = {
        "term": tid,
        "n_terms": 1,
        "rmse": float(np.sqrt(np.mean((got - want) ** 2))),
        "max_abs": float(np.max(np.abs(got - want))),
        "n_bins": [int(len(e) - 1) for e in edges],
    }
    return term, diag
