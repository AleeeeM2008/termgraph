"""Budget controller: choose the simplest admissible TermGraph that fits.

The controller does not train. It loads candidate artifacts, audits each
against the Glassbox Contract, filters by term/order budget, and returns the
highest expected accuracy candidate that fits. If no candidate fits, it
returns ``None`` rather than silently violating the budget.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .block import ModelSpec
from .contract import ContractReport, audit
from .ir import TermGraph


@dataclass
class Candidate:
    spec: ModelSpec
    graph: TermGraph
    contract: ContractReport

    @property
    def score(self) -> float:
        return float(self.spec.expected_auc)

    def describe(self) -> str:
        return (
            f"{self.spec.name}: terms={len(self.graph.terms)} "
            f"order={self.graph.max_order} expected_auc={self.score:.4f} "
            f"admissible={self.contract.admissible}"
        )


def load_candidate(spec: ModelSpec, *, X: np.ndarray | None = None) -> Candidate:
    graph = TermGraph.from_json(Path(spec.artifact))
    report = audit(graph, max_order=spec.max_order, X=X)
    return Candidate(spec=spec, graph=graph, contract=report)


def select(
    candidates: list[Candidate],
    *,
    term_budget: int,
    max_order: int = 3,
    min_auc: float | None = None,
) -> Candidate | None:
    """Return the highest-expected-AUC candidate within the budget."""
    eligible = [
        c for c in candidates
        if len(c.graph.terms) <= term_budget
        and c.graph.max_order <= max_order
        and c.contract.admissible
        and (min_auc is None or c.score >= min_auc)
    ]
    if not eligible:
        return None
    return max(eligible, key=lambda c: c.score)


def selection_table(candidates: list[Candidate]) -> list[dict]:
    return [
        {
            "name": c.spec.name,
            "terms": len(c.graph.terms),
            "max_order": c.graph.max_order,
            "expected_auc": c.score,
            "admissible": c.contract.admissible,
            "violations": c.contract.violations,
        }
        for c in candidates
    ]
