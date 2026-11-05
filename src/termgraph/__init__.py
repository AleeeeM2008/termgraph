"""TermGraph: a certified glassbox exchange architecture.

The core is NumPy-only. Training and compiler backends live in the optional
``train`` extra and are imported explicitly by the experiment scripts.
"""

from .block import ModelSpec, TermBlock
from .contract import ContractReport, audit
from .controller import Candidate, load_candidate, select
from .fast_runtime import FastTermGraph
from .ir import (
    Term,
    TermGraph,
    clamp_term,
    compress_graph,
    compress_term,
    diff_graphs,
    monotonize_term,
    replace_term,
    zero_term,
)

__all__ = [
    "Candidate",
    "ContractReport",
    "FastTermGraph",
    "ModelSpec",
    "Term",
    "TermBlock",
    "TermGraph",
    "audit",
    "clamp_term",
    "compress_graph",
    "compress_term",
    "diff_graphs",
    "load_candidate",
    "monotonize_term",
    "replace_term",
    "select",
    "zero_term",
]
