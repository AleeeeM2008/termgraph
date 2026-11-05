"""Compile an InterpretML EBM into TermGraph tables.

EBM stores 1024-bin term scores with an internal quantile discretization. We
compile each term onto an explicit raw-value grid and certify the table error.
The deployed artifact is pure NumPy and has no InterpretML dependency.
"""

from __future__ import annotations

import numpy as np

from .ir import Term, TermGraph


def quantile_edges(x: np.ndarray, n_bins: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.array([0.0, 1.0])
    qs = np.linspace(0.0, 1.0, int(n_bins) + 1)
    edges = np.quantile(x, qs)
    edges = np.unique(edges)
    if len(edges) < 2:
        v = float(edges[0]) if len(edges) else 0.0
        edges = np.array([v - 0.5, v + 0.5])
    return edges


SENTINEL = 1e300


def _representatives(edges: np.ndarray) -> np.ndarray:
    """One representative point per bin, including the sentinel overflow bins."""
    reps = []
    for i in range(len(edges) - 1):
        lo, hi = float(edges[i]), float(edges[i + 1])
        if lo <= -SENTINEL:
            scale = max(1.0, abs(hi))
            reps.append(hi - 1e3 * scale)
        elif hi >= SENTINEL:
            scale = max(1.0, abs(lo))
            reps.append(lo + 1e3 * scale)
        else:
            reps.append(0.5 * (lo + hi))
    return np.asarray(reps, dtype=np.float64)


def _grid(edges: list[np.ndarray]) -> np.ndarray:
    """Cartesian product of bin representatives, shape (prod, k)."""
    cs = [_representatives(e) for e in edges]
    mesh = np.meshgrid(*cs, indexing="ij")
    return np.column_stack([m.ravel() for m in mesh])


def _term_edges(ebm, feature: int, k: int, X_train: np.ndarray) -> np.ndarray:
    """Use EBM's own cut points for this term's bin level when available."""
    levels = getattr(ebm, "bins_", None)
    if levels is not None and feature < len(levels) and len(levels[feature]) > 0:
        level = min(k, len(levels[feature])) - 1
        cuts = np.asarray(levels[feature][level], dtype=np.float64)
        cuts = cuts[np.isfinite(cuts)]
        edges = np.concatenate(([-SENTINEL], cuts, [SENTINEL]))
        edges = np.unique(edges)
        if len(edges) >= 2:
            return edges
    return quantile_edges(X_train[:, feature], 48 if k == 1 else 24)


def export_ebm_terms(
    ebm,
    X_train: np.ndarray,
    feature_names: list[str],
    *,
    n_bins: int = 48,
    pair_bins: int = 24,
    seed: int = 0,
) -> tuple[list[Term], float, dict]:
    """Return (terms, bias, diagnostics) for a fitted EBM classifier/regressor.

    By default the table edges are EBM's own cut points, which makes the
    compiled term almost exactly the EBM term. ``n_bins``/``pair_bins`` are the
    fallback when a feature has no cut points.
    """
    X_train = np.asarray(X_train, dtype=np.float64)
    bg = np.median(X_train, axis=0)
    rng = np.random.default_rng(seed)
    sample = X_train[rng.choice(len(X_train), min(2000, len(X_train)), replace=False)]
    ref = np.asarray(ebm.eval_terms(sample), dtype=np.float64)
    terms: list[Term] = []
    centers = []
    diagnostics = {"per_term_rmse": {}, "per_term_max_abs": {}}

    for ti, feats in enumerate(ebm.term_features_):
        feats = [int(f) for f in feats]
        k = len(feats)
        edges = [_term_edges(ebm, f, k, X_train) for f in feats]
        grid = _grid(edges)
        Xg = np.tile(bg, (len(grid), 1))
        for j, f in enumerate(feats):
            Xg[:, f] = grid[:, j]
        vals = np.asarray(ebm.eval_terms(Xg)[:, ti], dtype=np.float64)
        shape = tuple(len(e) - 1 for e in edges)
        vals = vals.reshape(shape)
        center = float(ref[:, ti].mean())
        term_id = "*".join(feature_names[f] for f in feats)
        term = Term(
            id=term_id,
            inputs=[feature_names[f] for f in feats],
            edges=edges,
            values=vals,
            center=center,
            constraints={"monotone": None},
            provenance={
                "backend": "ebm",
                "term_index": int(ti),
                "grid_bins": [int(len(e) - 1) for e in edges],
            },
            backend="ebm",
        )
        # table error on the reference sample
        got = term.lookup(sample, {f: i for i, f in enumerate(feature_names)})
        want = ref[:, ti] - center
        diagnostics["per_term_rmse"][term_id] = float(
            np.sqrt(np.mean((got - want) ** 2))
        )
        diagnostics["per_term_max_abs"][term_id] = float(np.max(np.abs(got - want)))
        terms.append(term)
        centers.append(center)

    bias = float(np.asarray(ebm.intercept_).ravel()[0]) + float(sum(centers))
    # Total table reconstruction on the reference sample.
    col = {f: i for i, f in enumerate(feature_names)}
    total = np.full(len(sample), bias)
    for t in terms:
        total += t.lookup(sample, col)
    want = np.asarray(ebm.decision_function(sample), dtype=np.float64)
    diagnostics["total_rmse"] = float(np.sqrt(np.mean((total - want) ** 2)))
    diagnostics["total_max_abs"] = float(np.max(np.abs(total - want)))
    diagnostics["n_terms"] = len(terms)
    return terms, bias, diagnostics


def ebm_to_graph(
    ebm,
    X_train: np.ndarray,
    feature_names: list[str],
    *,
    n_bins: int = 48,
    pair_bins: int = 24,
    seed: int = 0,
) -> tuple[TermGraph, dict]:
    terms, bias, diag = export_ebm_terms(
        ebm, X_train, feature_names, n_bins=n_bins, pair_bins=pair_bins, seed=seed
    )
    g = TermGraph(
        features=list(feature_names),
        terms=terms,
        bias=bias,
        metadata={"compiler": "ebm", "grid_bins": [n_bins, pair_bins]},
    )
    g.certificate["compile"] = diag
    return g, diag
