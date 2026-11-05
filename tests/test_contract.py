import os
from pathlib import Path

import numpy as np
import pytest

from termgraph.contract import audit
from termgraph.ir import TermGraph

ROOT = Path(__file__).resolve().parents[1]
KIRK = Path(os.environ.get("KIRK_ROOT", ROOT.parent / "TESTAstronomy"))
DATA = Path(os.environ.get("KIRK_DATA", KIRK / "data" / "features_8000_4000_0.npz"))
ART = ROOT / "models" / "kirk_glassbox.json"


@pytest.mark.skipif(not ART.exists() or not DATA.exists(),
                    reason="artifact/data missing")
def test_artifact_passes_contract():
    g = TermGraph.from_json(ART)
    d = np.load(DATA, allow_pickle=True)
    X = d["X"].astype(np.float64)[:100]
    r = audit(g, X=X)
    assert r.admissible, r.violations
    assert r.checks["max_order"] == 3
    assert r.checks["additivity_error"] < 1e-9


@pytest.mark.skipif(not ART.exists(), reason="artifact missing")
def test_contract_rejects_tight_budget():
    g = TermGraph.from_json(ART)
    r = audit(g, max_order=2, require_certificate=False)
    assert not r.admissible
    assert any("order" in v for v in r.violations)
    r2 = audit(g, max_terms=5, require_certificate=False)
    assert not r2.admissible
    assert any("too many terms" in v for v in r2.violations)
