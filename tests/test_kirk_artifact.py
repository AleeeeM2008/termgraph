import json
import os
from pathlib import Path

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from termgraph.ir import TermGraph
from termgraph.runtime import predict as parallel_predict

ROOT = Path(__file__).resolve().parents[1]
KIRK = Path(os.environ.get("KIRK_ROOT", ROOT.parent / "TESTAstronomy"))
ART = ROOT / "models" / "kirk_glassbox.json"
DATA = Path(os.environ.get("KIRK_DATA", KIRK / "data" / "features_8000_4000_0.npz"))


@pytest.mark.skipif(not ART.exists() or not DATA.exists(), reason="artifact/data missing")
def test_artifact_metrics_and_additivity():
    g = TermGraph.from_json(ART)
    d = np.load(DATA, allow_pickle=True)
    X = d["X"].astype(np.float64)
    labels = d["labels"].astype(np.float64)
    stable = d["stable"].astype(bool)
    X, labels = X[stable], labels[stable]
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(X))
    cut = int(len(X) * 0.85)
    ho = perm[cut:]
    s = g.predict(X[ho])
    p = 1 / (1 + np.exp(-s))
    auc = roc_auc_score(labels[ho].astype(int), p)
    assert auc > 0.93, auc
    assert g.additivity_error(X[ho]) < 1e-9
    assert len(g.terms) == 22
    assert g.max_order == 3
    phi, expected = g.shap_values(X[ho][:50])
    np.testing.assert_allclose(phi.sum(1) + expected, s[:50], atol=1e-10)
    assert "EMcrossnear*MMRstrengthnear*EMfracstdnear" in g.describe()


@pytest.mark.skipif(not ART.exists() or not DATA.exists(),
                    reason="artifact/data missing")
def test_parallel_artifact_equal():
    g = TermGraph.from_json(ART)
    d = np.load(DATA, allow_pickle=True)
    X = d["X"].astype(np.float64)
    X = X[:500]
    ref = g.predict(X)
    for backend in ("thread", "process"):
        np.testing.assert_array_equal(
            parallel_predict(g, X, n_jobs=3, backend=backend), ref
        )


@pytest.mark.skipif(not (ROOT / "models" / "kirk_glassbox_certificate.json").exists(),
                    reason="certificate missing")
def test_certificate_fields():
    c = json.loads((ROOT / "models" / "kirk_glassbox_certificate.json").read_text())
    assert c["n_terms"] == 22
    assert c["model"]["max_order"] == 3
    assert "compile_fidelity" in c
    assert c["compile_fidelity"]["table_vs_hybrid_rmse"] < 1e-10
    assert c["additivity_error"] < 1e-9


@pytest.mark.skipif(not (ROOT / "models" / "kirk_glassbox_full.json").exists()
                    or not DATA.exists(),
                    reason="full artifact/data missing")
def test_full_artifact():
    g = TermGraph.from_json(ROOT / "models" / "kirk_glassbox_full.json")
    d = np.load(DATA, allow_pickle=True)
    X = d["X"].astype(np.float64)
    assert len(g.terms) == 22
    assert g.max_order == 3
    assert g.additivity_error(X[:1000]) < 1e-9
    c = g.certificate
    assert c["expected_cv"]["auc_mean"] > 0.94
    assert c["in_sample"]["auc"] > 0.95
    assert c["term_counts"] == {"main": 11, "pair": 10, "threeway": 1}
