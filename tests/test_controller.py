import os
from pathlib import Path

import numpy as np
import pytest

from termgraph.block import ModelSpec, TermBlock
from termgraph.contract import audit
from termgraph.controller import load_candidate, select
from termgraph.ir import TermGraph

ROOT = Path(__file__).resolve().parents[1]
KIRK = Path(os.environ.get("KIRK_ROOT", ROOT.parent / "TESTAstronomy"))
DATA = Path(os.environ.get("KIRK_DATA", KIRK / "data" / "features_8000_4000_0.npz"))
COMPACT = ROOT / "models" / "kirk_glassbox_compact.json"
MAX = ROOT / "models" / "kirk_glassbox_full.json"


def sample():
    d = np.load(DATA, allow_pickle=True)
    return d["X"].astype(np.float64)[:100]


@pytest.mark.skipif(not COMPACT.exists() or not DATA.exists(),
                    reason="compact artifact/data missing")
def test_compact_artifact():
    g = TermGraph.from_json(COMPACT)
    assert len(g.terms) == 5
    assert g.max_order == 3
    assert g.additivity_error(sample()) < 1e-9
    r = audit(g, max_order=3, max_terms=5, X=sample())
    assert r.admissible, r.violations
    assert g.certificate["metrics_truth"]["compiled_table"]["auc"] > 0.92


@pytest.mark.skipif(not COMPACT.exists() or not MAX.exists() or not DATA.exists(),
                    reason="artifacts/data missing")
def test_controller_budget_selection():
    compact = ModelSpec(
        "compact_5term", str(COMPACT),
        [TermBlock("five", "auto", "table", "xgboost", max_order=3)],
        expected_auc=0.9296, max_order=3,
    )
    full = ModelSpec(
        "max_accuracy_22term", str(MAX),
        [TermBlock("max", "auto", "table", "ebm", max_order=3)],
        expected_auc=0.9427, max_order=3,
    )
    cands = [load_candidate(compact, X=sample()),
             load_candidate(full, X=sample())]
    assert all(c.contract.admissible for c in cands)
    assert select(cands, term_budget=4) is None
    assert select(cands, term_budget=5).spec.name == "compact_5term"
    assert select(cands, term_budget=22).spec.name == "max_accuracy_22term"
