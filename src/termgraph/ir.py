"""TermGraph IR: a portable, editable, exact-where-it-can glassbox.

A model is ``bias + sum(term(x[inputs]) - center)``. Each term is a
piecewise-constant table over explicit raw-value edges. The artifact is JSON;
inference is pure NumPy; terms are independent and can be evaluated in
parallel.
"""

from __future__ import annotations

import copy
import json
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


def _edges_from_values(values: Sequence[float]) -> np.ndarray:
    edges = np.asarray(values, dtype=np.float64)
    if edges.ndim != 1 or len(edges) < 2:
        raise ValueError("edges must be a 1D array with at least 2 entries")
    if np.any(np.diff(edges) <= 0):
        raise ValueError("edges must be strictly increasing")
    return edges


@dataclass
class Term:
    """One piecewise-constant term over one, two, or three features."""

    id: str
    inputs: list[str]
    edges: list[np.ndarray]
    values: np.ndarray
    center: float = 0.0
    constraints: dict = field(default_factory=dict)
    provenance: dict = field(default_factory=dict)
    backend: str = "table"
    split_side: str = "right"

    def __post_init__(self):
        if self.split_side not in ("left", "right"):
            raise ValueError("split_side must be 'left' or 'right'")
        self.edges = [_edges_from_values(e) for e in self.edges]
        self.values = np.asarray(self.values, dtype=np.float64)
        expected = tuple(len(e) - 1 for e in self.edges)
        if self.values.shape != expected:
            raise ValueError(
                f"term {self.id}: values shape {self.values.shape} != {expected}"
            )
        if len(self.inputs) != len(self.edges):
            raise ValueError(f"term {self.id}: inputs/edges length mismatch")

    @property
    def order(self) -> int:
        return len(self.inputs)

    @property
    def n_bins(self) -> tuple[int, ...]:
        return tuple(len(e) - 1 for e in self.edges)

    def _indices(self, X: np.ndarray, col: dict[str, int]) -> list[np.ndarray]:
        idx = []
        for name, edges in zip(self.inputs, self.edges):
            x = X[:, col[name]]
            i = np.searchsorted(edges, x, side=self.split_side) - 1
            idx.append(np.clip(i, 0, len(edges) - 2))
        return idx

    def lookup(self, X: np.ndarray, col: dict[str, int]) -> np.ndarray:
        """Centered contribution of this term for every row of X."""
        idx = self._indices(X, col)
        if len(idx) == 1:
            out = self.values[idx[0]]
        elif len(idx) == 2:
            out = self.values[idx[0], idx[1]]
        elif len(idx) == 3:
            out = self.values[idx[0], idx[1], idx[2]]
        else:
            raise ValueError(f"term {self.id}: unsupported order {len(idx)}")
        return out - self.center

    def range(self) -> tuple[float, float]:
        return (float(self.values.min() - self.center),
                float(self.values.max() - self.center))

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "inputs": list(self.inputs),
            "edges": [e.tolist() for e in self.edges],
            "values": self.values.tolist(),
            "center": float(self.center),
            "constraints": dict(self.constraints),
            "provenance": dict(self.provenance),
            "backend": self.backend,
            "split_side": self.split_side,
        }

    @classmethod
    def from_dict(cls, d: dict) -> Term:
        return cls(
            id=str(d["id"]),
            inputs=[str(x) for x in d["inputs"]],
            edges=[np.asarray(e, dtype=np.float64) for e in d["edges"]],
            values=np.asarray(d["values"], dtype=np.float64),
            center=float(d.get("center", 0.0)),
            constraints=dict(d.get("constraints", {})),
            provenance=dict(d.get("provenance", {})),
            backend=str(d.get("backend", "table")),
            split_side=str(d.get("split_side", "right")),
        )


@dataclass
class TermGraph:
    """A sum of terms plus a bias, with exact attribution and editing."""

    features: list[str]
    terms: list[Term]
    bias: float
    certificate: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)
    format: str = "termgraph/1.0"

    def __post_init__(self):
        self.features = [str(f) for f in self.features]
        self.col = {f: i for i, f in enumerate(self.features)}
        for t in self.terms:
            for f in t.inputs:
                if f not in self.col:
                    raise ValueError(f"term {t.id}: unknown feature {f!r}")
        self._validate()

    def _validate(self):
        if not self.terms:
            raise ValueError("a TermGraph needs at least one term")
        for t in self.terms:
            if t.order < 1 or t.order > 3:
                raise ValueError(f"term {t.id}: order must be 1..3")

    @property
    def max_order(self) -> int:
        return max(t.order for t in self.terms)

    def predict_terms(self, X: np.ndarray, n_jobs: int = 1) -> np.ndarray:
        """(n, K) centered term contributions."""
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2 or X.shape[1] != len(self.features):
            raise ValueError("X must be (n, n_features)")
        if n_jobs <= 1 or len(self.terms) <= 1:
            return np.column_stack([t.lookup(X, self.col) for t in self.terms])
        with ThreadPoolExecutor(max_workers=int(n_jobs)) as ex:
            vals = list(ex.map(lambda t: t.lookup(X, self.col), self.terms))
        return np.column_stack(vals)

    def predict(self, X: np.ndarray, n_jobs: int = 1) -> np.ndarray:
        """Logit/margin prediction."""
        return self.bias + self.predict_terms(X, n_jobs=n_jobs).sum(axis=1)

    @property
    def expected_value(self) -> float:
        """The declared baseline bias."""
        return float(self.bias)

    def term_values(self, X: np.ndarray, n_jobs: int = 1) -> np.ndarray:
        """Exact term ledger, shape (n, K)."""
        return self.predict_terms(X, n_jobs=n_jobs)

    def shap_values(self, X: np.ndarray, n_jobs: int = 1):
        """Feature-level contributions under the equal-split convention.

        Returns (phi, expected). ``phi.sum(1) + expected == predict(X)``.
        For a multi-input term, the centered term is split equally among its
        inputs. That is the Shapley value of the within-term symmetric game,
        not the Shapley value of the full model when terms overlap. The
        term-level ledger (``term_values``) is the canonical exact object.
        """
        terms = self.predict_terms(X, n_jobs=n_jobs)
        phi = np.zeros((terms.shape[0], len(self.features)), dtype=np.float64)
        for k, t in enumerate(self.terms):
            share = terms[:, k] / t.order
            for f in t.inputs:
                phi[:, self.col[f]] += share
        return phi, float(self.bias)

    def feature_values(self, X: np.ndarray, n_jobs: int = 1):
        """Alias for ``shap_values`` with the sharing convention documented."""
        return self.shap_values(X, n_jobs=n_jobs)

    def additivity_error(self, X: np.ndarray) -> float:
        terms = self.predict_terms(X).sum(axis=1)
        return float(np.max(np.abs(terms + self.bias - self.predict(X))))

    def to_dict(self) -> dict:
        return {
            "format": self.format,
            "features": list(self.features),
            "bias": float(self.bias),
            "terms": [t.to_dict() for t in self.terms],
            "certificate": dict(self.certificate),
            "metadata": dict(self.metadata),
        }

    def to_json(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=1))
        return path

    @classmethod
    def from_dict(cls, d: dict) -> TermGraph:
        if d.get("format") != "termgraph/1.0":
            raise ValueError(f"unsupported format {d.get('format')!r}")
        return cls(
            features=[str(x) for x in d["features"]],
            terms=[Term.from_dict(t) for t in d["terms"]],
            bias=float(d["bias"]),
            certificate=dict(d.get("certificate", {})),
            metadata=dict(d.get("metadata", {})),
        )

    @classmethod
    def from_json(cls, path: str | Path) -> TermGraph:
        return cls.from_dict(json.loads(Path(path).read_text()))

    def to_npz(self, path: str | Path, *, compress: bool = True) -> Path:
        """Binary artifact for deployment: same model, no JSON parsing.

        The JSON form remains the editable source; this is the serving form.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        meta = {
            "format": self.format,
            "features": list(self.features),
            "bias": float(self.bias),
            "terms": [
                {
                    "id": t.id,
                    "inputs": list(t.inputs),
                    "n_inputs": len(t.inputs),
                    "center": float(t.center),
                    "constraints": dict(t.constraints),
                    "provenance": dict(t.provenance),
                    "backend": t.backend,
                    "split_side": t.split_side,
                }
                for t in self.terms
            ],
            "certificate": dict(self.certificate),
            "metadata": dict(self.metadata),
        }
        arrays = {"meta": np.asarray(json.dumps(meta)), "bias": np.asarray(self.bias)}
        for i, t in enumerate(self.terms):
            for j, e in enumerate(t.edges):
                arrays[f"t{i}_e{j}"] = e
            arrays[f"t{i}_v"] = t.values
        if compress:
            np.savez_compressed(path, **arrays)
        else:
            np.savez(path, **arrays)
        return path

    @classmethod
    def from_npz(cls, path: str | Path) -> TermGraph:
        d = np.load(Path(path), allow_pickle=False)
        meta = json.loads(str(d["meta"]))
        terms = []
        for i, m in enumerate(meta["terms"]):
            edges = [np.asarray(d[f"t{i}_e{j}"], dtype=np.float64)
                     for j in range(m["n_inputs"])]
            values = np.asarray(d[f"t{i}_v"], dtype=np.float64)
            terms.append(Term(
                id=str(m["id"]),
                inputs=[str(x) for x in m["inputs"]],
                edges=edges,
                values=values,
                center=float(m["center"]),
                constraints=dict(m.get("constraints", {})),
                provenance=dict(m.get("provenance", {})),
                backend=str(m.get("backend", "table")),
                split_side=str(m.get("split_side", "right")),
            ))
        return cls(
            features=[str(x) for x in meta["features"]],
            terms=terms,
            bias=float(d["bias"]),
            certificate=dict(meta.get("certificate", {})),
            metadata=dict(meta.get("metadata", {})),
        )

    def describe(self) -> str:
        lines = [
            (
                f"TermGraph: {len(self.terms)} terms, max_order={self.max_order}, "
                f"bias={self.bias:+.4f}"
            ),
            f"{'id':38s} {'inputs':46s} {'bins':12s} {'range':22s} backend",
        ]
        for t in self.terms:
            lo, hi = t.range()
            lines.append(
                f"{t.id:38s} {','.join(t.inputs):46s} "
                f"{t.n_bins!s:12s} [{lo:+.3f},{hi:+.3f}]  {t.backend}"
            )
        return "\n".join(lines)

    def copy(self) -> TermGraph:
        return copy.deepcopy(self)


def compress_term(term: Term) -> Term:
    """Losslessly merge adjacent bins whose entire slices are identical."""
    edges = [e.copy() for e in term.edges]
    values = term.values.copy()
    changed = True
    while changed:
        changed = False
        for axis in range(values.ndim):
            n = values.shape[axis]
            for i in range(1, n):
                a = np.take(values, i - 1, axis=axis)
                b = np.take(values, i, axis=axis)
                if np.array_equal(a, b):
                    edges[axis] = np.delete(edges[axis], i)
                    values = np.delete(values, i, axis=axis)
                    changed = True
                    break
            if changed:
                break
    return Term(
        id=term.id, inputs=list(term.inputs), edges=edges, values=values,
        center=term.center, constraints=dict(term.constraints),
        provenance=dict(term.provenance, compressed=True), backend=term.backend,
        split_side=term.split_side,
    )


def compress_graph(graph: TermGraph) -> TermGraph:
    g = graph.copy()
    g.terms = [compress_term(t) for t in g.terms]
    return g


def zero_term(graph: TermGraph, term_id: str) -> TermGraph:
    """Remove a term's contribution without touching any other term."""
    g = graph.copy()
    for t in g.terms:
        if t.id == term_id:
            t.values = np.full_like(t.values, t.center)
            t.constraints = dict(t.constraints, edited="zero")
            return g
    raise KeyError(term_id)


def clamp_term(graph: TermGraph, term_id: str, lo: float, hi: float) -> TermGraph:
    """Clamp one term's centered contribution to [lo, hi]."""
    g = graph.copy()
    for t in g.terms:
        if t.id == term_id:
            v = np.clip(t.values - t.center, lo, hi) + t.center
            t.values = v
            t.constraints = dict(t.constraints, edited=f"clamp[{lo},{hi}]")
            return g
    raise KeyError(term_id)


def monotonize_term(graph: TermGraph, term_id: str, axis: int = 0,
                    direction: int = 1) -> TermGraph:
    """Make one term monotone along an axis (cumulative max/min)."""
    g = graph.copy()
    for t in g.terms:
        if t.id == term_id:
            v = (t.values - t.center) * direction
            v = np.maximum.accumulate(v, axis=axis)
            t.values = v / direction + t.center
            t.constraints = dict(
                t.constraints, edited=f"monotone(axis={axis},dir={direction})"
            )
            return g
    raise KeyError(term_id)


def replace_term(graph: TermGraph, term_id: str, values: np.ndarray) -> TermGraph:
    """Replace one term's table, keeping its edges and center."""
    g = graph.copy()
    for t in g.terms:
        if t.id == term_id:
            v = np.asarray(values, dtype=np.float64)
            if v.shape != t.values.shape:
                raise ValueError(f"shape {v.shape} != {t.values.shape}")
            t.values = v
            t.constraints = dict(t.constraints, edited="replace")
            return g
    raise KeyError(term_id)


def diff_graphs(old: TermGraph, new: TermGraph, X: np.ndarray) -> dict:
    """Exact prediction and term deltas between two artifacts."""
    if old.features != new.features:
        raise ValueError("feature order differs")
    p0 = old.predict(X)
    p1 = new.predict(X)
    t0 = old.predict_terms(X)
    t1 = new.predict_terms(X)
    ids0 = {t.id: i for i, t in enumerate(old.terms)}
    ids1 = {t.id: i for i, t in enumerate(new.terms)}
    common = [t for t in old.terms if t.id in ids1]
    term_delta = {}
    for t in common:
        d = t1[:, ids1[t.id]] - t0[:, ids0[t.id]]
        term_delta[t.id] = {
            "max_abs": float(np.max(np.abs(d))),
            "mean": float(np.mean(d)),
        }
    return {
        "prediction_max_abs": float(np.max(np.abs(p1 - p0))),
        "prediction_mean": float(np.mean(p1 - p0)),
        "term_delta": term_delta,
        "added": [t.id for t in new.terms if t.id not in ids0],
        "removed": [t.id for t in old.terms if t.id not in ids1],
    }
