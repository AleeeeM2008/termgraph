"""Fused TermGraph runtime.

Optimizations over ``TermGraph.predict``:

1. Bin each unique (feature, edges, split-side) once per batch, not once per
   term. Features that recur across terms share the bin indices.
2. Flatten each term table and use linear indexing, avoiding repeated
   multi-dimensional advanced indexing.
3. For prediction, accumulate terms directly; never materialize the (n, K)
   term matrix.
4. Precompute ``bias - sum(centers)`` and subtract no scalar per term.

All three preserve the exact prediction (up to floating-point order).
"""

from __future__ import annotations

import math
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from .ir import TermGraph


class FastTermGraph:
    def __init__(self, graph: TermGraph):
        self.graph = graph
        self.features = list(graph.features)
        self.col = dict(graph.col)
        self.bias_adj = float(graph.bias) - sum(float(t.center) for t in graph.terms)
        self._binners: list[tuple[np.ndarray, int, str]] = []
        self._binner_ids: dict[tuple, int] = {}
        self._terms: list[tuple[list[int], tuple[int, ...], np.ndarray, float]] = []
        for t in graph.terms:
            ids = []
            for f, e in zip(t.inputs, t.edges):
                key = (self.col[f], t.split_side, e.tobytes())
                bid = self._binner_ids.get(key)
                if bid is None:
                    bid = len(self._binners)
                    self._binner_ids[key] = bid
                    self._binners.append((e, self.col[f], t.split_side))
                ids.append(bid)
            self._terms.append(
                (ids, tuple(int(s) for s in t.n_bins), t.values.ravel(), float(t.center))
            )

    def _bins(self, X: np.ndarray) -> list[np.ndarray]:
        out = []
        for edges, f, side in self._binners:
            i = np.searchsorted(edges, X[:, f], side=side) - 1
            out.append(np.clip(i, 0, len(edges) - 2).astype(np.int64, copy=False))
        return out

    @staticmethod
    def _linear(ids: list[int], shape: tuple[int, ...], B: list[np.ndarray]):
        if len(ids) == 1:
            return B[ids[0]]
        if len(ids) == 2:
            return B[ids[0]] * shape[1] + B[ids[1]]
        return (B[ids[0]] * shape[1] + B[ids[1]]) * shape[2] + B[ids[2]]

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        B = self._bins(X)
        total = np.zeros(len(X), dtype=np.float64)
        for ids, shape, values, _center in self._terms:
            total += values[self._linear(ids, shape, B)]
        return self.bias_adj + total

    def predict_terms(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        B = self._bins(X)
        out = np.empty((len(X), len(self._terms)), dtype=np.float64)
        for k, (ids, shape, values, center) in enumerate(self._terms):
            out[:, k] = values[self._linear(ids, shape, B)] - center
        return out

    def predict_parallel(
        self,
        X: np.ndarray,
        *,
        n_jobs: int = 1,
        chunk_size: int | None = None,
    ) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        if n_jobs <= 1 or len(X) < 2000:
            return self.predict(X)
        if chunk_size is None:
            chunk_size = max(1, math.ceil(len(X) / n_jobs))
        parts = [(i, min(len(X), i + chunk_size))
                 for i in range(0, len(X), chunk_size)]
        with ThreadPoolExecutor(max_workers=int(n_jobs)) as ex:
            out = list(ex.map(lambda s: self.predict(X[s[0]:s[1]]), parts))
        return np.concatenate(out)


def compare_serial(graph: TermGraph, X: np.ndarray) -> dict:
    """Convenience check that the fast runtime is numerically identical."""
    fast = FastTermGraph(graph)
    a = graph.predict(X)
    b = fast.predict(X)
    return {
        "max_abs_diff": float(np.max(np.abs(a - b))),
        "rmse": float(np.sqrt(np.mean((a - b) ** 2))),
    }
