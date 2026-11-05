"""Compiler: EBM low-order terms + constrained XGBoost residual term -> IR."""

from __future__ import annotations

import numpy as np
import xgboost as xgb

from .export_ebm import ebm_to_graph
from .export_xgb import export_xgb_term
from .ir import TermGraph


def compile_ebm_plus_residual(
    ebm,
    residual: xgb.Booster,
    X_train: np.ndarray,
    feature_names: list[str],
    block: list[int],
    *,
    n_bins: int = 48,
    pair_bins: int = 24,
    seed: int = 0,
    term_id: str | None = None,
) -> tuple[TermGraph, dict]:
    """Compile an EBM plus one constrained XGBoost residual term.

    The EBM terms are compiled with EBM's own cut points (exact). The residual
    term is compiled on the union of its tree thresholds (exact for finite
    values inside the training range).
    """
    graph, ebm_diag = ebm_to_graph(
        ebm, X_train, feature_names,
        n_bins=n_bins, pair_bins=pair_bins, seed=seed,
    )
    term, x_diag = export_xgb_term(
        residual, feature_names, block, X_train, n_bins=n_bins,
        seed=seed, term_id=term_id,
    )
    graph.terms.append(term)
    graph.bias += term.center
    graph.metadata = {
        "compiler": "ebm+xgboost",
        "ebm_terms": len(graph.terms) - 1,
        "residual_block": [feature_names[i] for i in block],
        "grid_bins": {"ebm_main": n_bins, "ebm_pair": pair_bins,
                      "xgb_term": n_bins},
    }
    diag = {
        "ebm": ebm_diag,
        "xgb": x_diag,
        "n_terms": len(graph.terms),
        "max_order": graph.max_order,
    }
    return graph, diag
