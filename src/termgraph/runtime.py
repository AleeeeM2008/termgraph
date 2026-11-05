"""Deployment runtime: vectorized and chunk-parallel TermGraph inference.

The model is additive, so rows are independent and terms are independent.
Two parallel strategies are provided:

- ``thread``: split rows into chunks, evaluate with NumPy. Cheap to start;
  effective when the underlying ufuncs release the GIL.
- ``process``: fork workers with a shared read-only graph. Effective for
  CPU-bound batch inference on Linux.

Both produce bit-identical results to serial prediction.
"""

from __future__ import annotations

import math
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor
from typing import Literal

import numpy as np

from .ir import TermGraph

_GRAPH: TermGraph | None = None


def _init_worker(graph: TermGraph):
    global _GRAPH
    _GRAPH = graph


def _predict_chunk(X: np.ndarray) -> np.ndarray:
    assert _GRAPH is not None
    return _GRAPH.predict(X, n_jobs=1)


def _slices(n: int, n_jobs: int):
    chunk = max(1, math.ceil(n / max(1, n_jobs)))
    return [slice(i, min(n, i + chunk)) for i in range(0, n, chunk)]


def predict(
    graph: TermGraph,
    X: np.ndarray,
    *,
    n_jobs: int = 1,
    backend: Literal["thread", "process"] = "thread",
) -> np.ndarray:
    """Predict a batch, optionally in parallel. Always equals serial output."""
    X = np.asarray(X, dtype=np.float64)
    if n_jobs <= 1 or len(X) == 0:
        return graph.predict(X, n_jobs=1)
    parts = _slices(len(X), n_jobs)
    if backend == "thread":
        with ThreadPoolExecutor(max_workers=int(n_jobs)) as ex:
            out = list(ex.map(lambda s: graph.predict(X[s], n_jobs=1), parts))
    elif backend == "process":
        ctx = mp.get_context("fork")
        with ctx.Pool(
            processes=int(n_jobs),
            initializer=_init_worker,
            initargs=(graph,),
        ) as pool:
            out = pool.map(_predict_chunk, [X[s] for s in parts])
    else:
        raise ValueError(f"unknown backend {backend!r}")
    return np.concatenate(out)


def predict_terms(
    graph: TermGraph,
    X: np.ndarray,
    *,
    n_jobs: int = 1,
    backend: Literal["thread", "process"] = "thread",
) -> np.ndarray:
    """Parallel (n, K) term matrix; same contract as TermGraph.predict_terms."""
    X = np.asarray(X, dtype=np.float64)
    if n_jobs <= 1 or len(X) == 0:
        return graph.predict_terms(X, n_jobs=1)
    parts = _slices(len(X), n_jobs)
    if backend == "thread":
        with ThreadPoolExecutor(max_workers=int(n_jobs)) as ex:
            out = list(ex.map(lambda s: graph.predict_terms(X[s], n_jobs=1), parts))
    else:
        ctx = mp.get_context("fork")
        with ctx.Pool(
            processes=int(n_jobs),
            initializer=_init_worker,
            initargs=(graph,),
        ) as pool:
            out = pool.map(_predict_chunk_terms, [X[s] for s in parts])
    return np.vstack(out)


def _predict_chunk_terms(X: np.ndarray) -> np.ndarray:
    assert _GRAPH is not None
    return _GRAPH.predict_terms(X, n_jobs=1)
