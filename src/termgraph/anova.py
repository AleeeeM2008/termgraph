"""Functional-ANOVA energy of a black-box model under a marginal baseline.

For a function f and independent features, the ANOVA component for a subset S
is the inclusion-exclusion of conditional expectations:

    f_S(x_S) = sum_{T subset S} (-1)^(|S|-|T|) E[f | X_T = x_T]

This module estimates those components with a Monte Carlo background drawn
from the product of the empirical marginals (the interventional/marginal
convention). The result is a diagnostic for the compiler:

    energy_by_order[d] = share of Var(f) carried by terms of order <= d

If a compiled order-d glassbox captures R^2 close to that share, the residual is
a representation limit. If it captures much less, the residual is an
optimization limit.
"""

from __future__ import annotations

from collections.abc import Callable
from itertools import combinations

import numpy as np


def _subsets(p: int, max_order: int) -> list[tuple[int, ...]]:
    out: list[tuple[int, ...]] = [()]
    for k in range(1, max_order + 1):
        out.extend(combinations(range(p), k))
    return out


def _subsets_of(s: tuple[int, ...]) -> list[tuple[int, ...]]:
    out: list[tuple[int, ...]] = [()]
    for k in range(1, len(s) + 1):
        out.extend(combinations(s, k))
    return out


def marginal_background(X: np.ndarray, n: int, seed: int = 0) -> np.ndarray:
    """Draw rows with independent columns from the empirical marginals."""
    rng = np.random.default_rng(seed)
    X = np.asarray(X)
    idx = rng.integers(0, len(X), size=(n, X.shape[1]))
    return X[idx, np.arange(X.shape[1])[None, :]]


def anova_energy(
    predict: Callable[[np.ndarray], np.ndarray],
    X: np.ndarray,
    *,
    max_order: int = 3,
    n_eval: int = 400,
    n_background: int = 32,
    seed: int = 0,
    batch: int = 8192,
) -> dict:
    """Estimate ANOVA energy by order under a marginal baseline.

    Parameters
    ----------
    predict : callable
        Maps an (n, p) float32 matrix to a length-n margin/logit vector.
    X : array
        Evaluation sample.
    max_order : int
        Highest interaction order to estimate.
    n_eval, n_background : int
        Monte Carlo sizes. Increase for tighter estimates.
    """
    X = np.asarray(X, dtype=np.float32)
    rng = np.random.default_rng(seed)
    if len(X) > n_eval:
        X = X[rng.choice(len(X), n_eval, replace=False)]
    n, p = X.shape
    bg = marginal_background(X, n_background, seed=seed + 1)

    fX = np.asarray(predict(X), dtype=np.float64).ravel()
    total = float(np.mean((fX - fX.mean()) ** 2))
    subsets = _subsets(p, max_order)

    # Conditional expectations m[T][i] = E[f | X_T = X[i, T]].
    m: dict[tuple[int, ...], np.ndarray] = {}
    for T in subsets:
        if len(T) == 0:
            m[T] = np.full(n, fX.mean(), dtype=np.float64)
            continue
        drop = [j for j in range(p) if j not in T]
        # Shape (B, n, p), then flatten for one batched predict.
        Xrep = np.repeat(X[None, :, :], n_background, axis=0)
        if drop:
            Xrep[:, :, drop] = bg[:, None, drop]
        flat = Xrep.reshape(-1, p)
        preds = []
        for start in range(0, len(flat), batch):
            preds.append(np.asarray(predict(flat[start:start + batch])).ravel())
        pred = np.concatenate(preds).reshape(n_background, n)
        m[T] = pred.mean(axis=0)

    # ANOVA components by inclusion-exclusion.
    energy: dict[tuple[int, ...], float] = {}
    for S in subsets:
        g = np.zeros(n, dtype=np.float64)
        for T in _subsets_of(S):
            sign = -1.0 if (len(S) - len(T)) % 2 else 1.0
            g += sign * m[T]
        # Center numerically; the component should have mean zero.
        g -= g.mean()
        energy[S] = float(np.mean(g ** 2))

    by_order = {}
    for k in range(1, max_order + 1):
        by_order[k] = float(sum(v for S, v in energy.items() if len(S) == k))
    cumulative = {}
    running = 0.0
    for k in range(1, max_order + 1):
        running += by_order[k]
        cumulative[k] = running / total if total > 0 else float("nan")

    return {
        "n_eval": int(n),
        "n_background": int(n_background),
        "p": int(p),
        "max_order": int(max_order),
        "total_variance": total,
        "energy_by_order": by_order,
        "share_by_order": {
            k: (v / total if total > 0 else float("nan"))
            for k, v in by_order.items()
        },
        "cumulative_share": cumulative,
        "residual_share": 1.0 - cumulative[max_order],
        "top_components": [
            {"subset": list(S), "share": v / total if total > 0 else float("nan")}
            for S, v in sorted(energy.items(), key=lambda kv: kv[1], reverse=True)[:15]
            if len(S) > 0
        ],
    }
