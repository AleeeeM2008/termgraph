"""Reusable block/operator specification for TermGraph models.

A model is a list of blocks. Each block names its structure, basis, learner, and
composition operator. This is the interface the controller searches over; the
deployed artifact is still the compiled term graph.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class TermBlock:
    name: str
    inputs: tuple[str, ...] | str
    basis: str          # table, spline, mlp
    learner: str        # ebm, xgboost, nam, spline, rule
    composition: str = "sum"
    max_order: int = 1
    notes: str = ""

    def describe(self) -> str:
        ins = self.inputs if isinstance(self.inputs, str) else ",".join(self.inputs)
        return (
            f"{self.name}: inputs=({ins}) basis={self.basis} "
            f"learner={self.learner} composition={self.composition} "
            f"order={self.max_order}"
        )


@dataclass
class ModelSpec:
    name: str
    artifact: str
    blocks: list[TermBlock]
    expected_auc: float
    max_order: int
    notes: str = ""
    metadata: dict = field(default_factory=dict)

    def describe(self) -> str:
        lines = [
            (
                f"{self.name}: artifact={self.artifact} "
                f"expected_auc={self.expected_auc:.4f} max_order={self.max_order}"
            )
        ]
        lines.extend("  " + b.describe() for b in self.blocks)
        return "\n".join(lines)
