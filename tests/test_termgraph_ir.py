import numpy as np

from termgraph.fast_runtime import FastTermGraph
from termgraph.ir import (
    Term,
    TermGraph,
    clamp_term,
    diff_graphs,
    monotonize_term,
    zero_term,
)
from termgraph.runtime import predict as parallel_predict


def make_graph():
    t1 = Term(
        id="a", inputs=["a"], edges=[np.array([0.0, 1.0, 2.0, 3.0])],
        values=np.array([0.0, 1.0, 2.0]), center=0.0,
    )
    t2 = Term(
        id="a*b", inputs=["a", "b"],
        edges=[np.array([0.0, 0.5, 1.0]), np.array([0.0, 0.5, 1.0])],
        values=np.array([[0.0, 1.0], [1.0, 2.0]]), center=0.0,
    )
    return TermGraph(features=["a", "b"], terms=[t1, t2], bias=0.5)


def test_additivity_and_shap():
    g = make_graph()
    X = np.array([[0.5, 0.5], [1.5, 0.2], [2.0, 0.9]])
    s = g.predict(X)
    np.testing.assert_allclose(s, g.bias + g.predict_terms(X).sum(1))
    np.testing.assert_allclose(g.term_values(X), g.predict_terms(X))
    phi, expected = g.feature_values(X)
    np.testing.assert_allclose(phi.sum(1) + expected, s, atol=1e-12)
    assert g.expected_value == g.bias
    assert g.additivity_error(X) < 1e-12


def test_json_roundtrip(tmp_path):
    g = make_graph()
    p = tmp_path / "m.json"
    g.to_json(p)
    g2 = TermGraph.from_json(p)
    X = np.array([[0.1, 0.4], [1.2, 0.7]])
    np.testing.assert_allclose(g.predict(X), g2.predict(X))
    assert g2.describe()


def test_zero_isolation():
    g = make_graph()
    X = np.array([[0.5, 0.5], [1.5, 0.2]])
    gz = zero_term(g, "a")
    d = diff_graphs(g, gz, X)
    assert d["term_delta"]["a"]["max_abs"] > 0
    assert d["term_delta"]["a*b"]["max_abs"] == 0.0
    np.testing.assert_allclose(
        gz.predict(X), g.predict(X) - g.predict_terms(X)[:, 0]
    )


def test_clamp_and_monotone():
    g = make_graph()
    gc = clamp_term(g, "a", -0.5, 0.5)
    v = gc.predict_terms(np.array([[1.5, 0.0]]))[0, 0]
    assert -0.5 - 1e-12 <= v <= 0.5 + 1e-12
    gm = monotonize_term(g, "a")
    t = next(t for t in gm.terms if t.id == "a")
    assert np.all(np.diff(t.values - t.center) >= -1e-12)


def test_parallel_equals_serial():
    g = make_graph()
    rng = np.random.default_rng(0)
    X = rng.uniform(0, 2, size=(2000, 2))
    serial = g.predict(X)
    for backend in ("thread", "process"):
        par = parallel_predict(g, X, n_jobs=3, backend=backend)
        np.testing.assert_array_equal(par, serial)


def test_fast_runtime_and_npz(tmp_path):
    g = make_graph()
    rng = np.random.default_rng(1)
    X = rng.uniform(0, 2, size=(3000, 2))
    fast = FastTermGraph(g)
    np.testing.assert_allclose(fast.predict(X), g.predict(X), atol=1e-15)
    np.testing.assert_allclose(
        fast.predict_parallel(X, n_jobs=2), g.predict(X), atol=1e-15
    )
    p = tmp_path / "m.npz"
    g.to_npz(p)
    g2 = TermGraph.from_npz(p)
    np.testing.assert_allclose(g2.predict(X), g.predict(X), atol=1e-15)
    assert p.stat().st_size > 0
