"""The Glassbox Contract: admissibility checks for a deployed TermGraph.

Every deployed term must be low-order, table-renderable, exactly attributable,
editable, and declared in the certificate. Black-box learners are allowed only
as compilers; what ships is the term graph.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .ir import TermGraph

ALLOWED_BACKENDS = {"ebm", "xgboost", "table", "spline", "rule", "tensor", "nam"}


@dataclass
class ContractReport:
    admissible: bool
    checks: dict = field(default_factory=dict)
    violations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "admissible": bool(self.admissible),
            "checks": dict(self.checks),
            "violations": list(self.violations),
        }


def audit(
    graph: TermGraph,
    *,
    max_order: int = 3,
    max_terms: int = 64,
    max_bins: int = 250_000,
    require_certificate: bool = True,
    X: np.ndarray | None = None,
) -> ContractReport:
    """Check a TermGraph against the deployed-glassbox contract."""
    checks: dict = {}
    violations: list[str] = []

    checks["n_terms"] = len(graph.terms)
    if len(graph.terms) > max_terms:
        violations.append(f"too many terms: {len(graph.terms)} > {max_terms}")

    checks["max_order"] = graph.max_order
    if graph.max_order > max_order:
        violations.append(f"order {graph.max_order} > {max_order}")

    checks["all_tables"] = all(t.backend in ALLOWED_BACKENDS for t in graph.terms)
    if not checks["all_tables"]:
        violations.append("non-glassbox backend present")

    checks["unique_ids"] = len({t.id for t in graph.terms}) == len(graph.terms)
    if not checks["unique_ids"]:
        violations.append("duplicate term ids")

    checks["bins_ok"] = all(
        int(np.prod(t.n_bins)) <= max_bins for t in graph.terms
    )
    if not checks["bins_ok"]:
        violations.append(f"a term exceeds {max_bins} table cells")

    checks["has_certificate"] = bool(graph.certificate)
    if require_certificate and not checks["has_certificate"]:
        violations.append("missing certificate")

    checks["finite_tables"] = all(
        np.isfinite(t.values).all() and np.isfinite(t.center) for t in graph.terms
    )
    if not checks["finite_tables"]:
        violations.append("non-finite term values")

    if X is not None:
        err = graph.additivity_error(X)
        checks["additivity_error"] = err
        if err > 1e-9:
            violations.append(f"additivity error {err:.2e} > 1e-9")

    return ContractReport(not violations, checks, violations)
