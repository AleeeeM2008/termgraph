"""Build the deployable KIRK TermGraph artifact.

Best model so far: EBM GA2M on truth labels + one order-3 XGBoost residual
term. Compile both into explicit tables, certify the table approximation, and
write a pure-NumPy artifact.
"""
from __future__ import annotations
import os

import json
import sys
import time
from pathlib import Path

import numpy as np
import xgboost as xgb
from interpret.glassbox import ExplainableBoostingClassifier
from sklearn.metrics import roc_auc_score, log_loss

ROOT = Path(__file__).resolve().parents[2]
KIRK = Path(os.environ.get("KIRK_ROOT", Path(__file__).resolve().parents[1].parent / "TESTAstronomy"))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT / "experiments" / "legacy"))

from kirk_termgraph import load_data  # noqa: E402
from termgraph.export_ebm import ebm_to_graph  # noqa: E402
from termgraph.export_xgb import export_xgb_term  # noqa: E402
from termgraph.ir import TermGraph  # noqa: E402


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def metrics(score, truth):
    p = sigmoid(score)
    return {
        "auc": float(roc_auc_score(truth.astype(int), p)),
        "logloss": float(log_loss(truth.astype(int), p)),
        "acc": float(((p > 0.5) == (truth > 0.5)).mean()),
    }


def main():
    X, Y, labels, feats, tr0, ho0 = load_data()
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(X))
    cut = int(len(X) * 0.85)
    tr, ho = perm[:cut], perm[cut:]

    # Internal fit/val for the residual term.
    perm2 = np.random.default_rng(100).permutation(len(tr))
    cut2 = int(len(tr) * 0.85)
    fit, val = tr[perm2[:cut2]], tr[perm2[cut2:]]

    print("training EBM...", flush=True)
    t0 = time.time()
    ebm = ExplainableBoostingClassifier(
        feature_names=feats, interactions=10,
        max_rounds=5000, early_stopping_rounds=50,
        outer_bags=4, n_jobs=-1, random_state=0,
    )
    ebm.fit(X[tr], labels[tr].astype(int))
    import joblib
    joblib.dump(ebm, ROOT / "models" / "kirk_ebm.joblib")
    print(f"EBM trained in {time.time()-t0:.1f}s", flush=True)

    base_fit = logit(ebm.predict_proba(X[fit])[:, 1])
    base_val = logit(ebm.predict_proba(X[val])[:, 1])
    base_ho = logit(ebm.predict_proba(X[ho])[:, 1])

    triple = [feats.index(n) for n in
              ["EMcrossnear", "MMRstrengthnear", "EMfracstdnear"]]
    params = {
        "objective": "binary:logistic",
        "max_depth": 3,
        "eta": 0.05,
        "subsample": 0.8,
        "min_child_weight": 20,
        "reg_lambda": 1.0,
        "tree_method": "hist",
        "seed": 0,
        "eval_metric": "logloss",
        "verbosity": 0,
    }
    dfit = xgb.DMatrix(X[fit][:, triple], label=labels[fit].astype(np.float32))
    dfit.set_base_margin(base_fit)
    dval = xgb.DMatrix(X[val][:, triple], label=labels[val].astype(np.float32))
    dval.set_base_margin(base_val)
    print("training 3-way residual...", flush=True)
    residual = xgb.train(
        params, dfit, num_boost_round=1000,
        evals=[(dval, "val")], early_stopping_rounds=60,
        verbose_eval=False,
    )
    term_ho = residual.predict(xgb.DMatrix(X[ho][:, triple]), output_margin=True)
    hybrid_ho = base_ho + term_ho
    residual.save_model(str(ROOT / "models" / "kirk_residual.json"))
    np.savez(
        ROOT / "models" / "kirk_verify.npz",
        ho=ho, base_ho=base_ho, term_ho=term_ho, hybrid_ho=hybrid_ho,
    )
    print(f"residual trees: {residual.num_boosted_rounds()}", flush=True)

    # Compile EBM terms to tables.
    print("compiling EBM terms...", flush=True)
    t0 = time.time()
    graph, ebm_diag = ebm_to_graph(
        ebm, X[tr], feats, n_bins=48, pair_bins=24, seed=0
    )
    print(f"EBM terms compiled in {time.time()-t0:.1f}s", flush=True)
    print("EBM table diag:", {k: ebm_diag[k] for k in
                              ("total_rmse", "total_max_abs", "n_terms")})

    # Compile the 3-way residual term.
    print("compiling 3-way term...", flush=True)
    xterm, xdiag = export_xgb_term(
        residual, feats, triple, X[tr], n_bins=48,
        term_id="EMcrossnear*MMRstrengthnear*EMfracstdnear",
    )
    print("XGB term diag:", xdiag)

    # Add the XGB term; its center moves into the bias.
    graph.terms.append(xterm)
    graph.bias += xterm.center
    graph.metadata = {
        "compiler": "ebm+xgboost",
        "ebm_interactions": 10,
        "residual_block": [feats[i] for i in triple],
        "grid_bins": {"ebm_main": 48, "ebm_pair": 24, "xgb_term": 24},
        "n_terms": len(graph.terms),
    }

    # Certificates.
    score = graph.predict(X[ho])
    base_metrics = metrics(base_ho, labels[ho])
    hybrid_metrics = metrics(hybrid_ho, labels[ho])
    graph_metrics = metrics(score, labels[ho])
    table_vs_hybrid = score - hybrid_ho
    # EBM alone on the graph's EBM terms (drop the last term)
    ebm_only = graph.copy()
    ebm_only.terms = ebm_only.terms[:-1]
    ebm_only.bias -= xterm.center
    ebm_score = ebm_only.predict(X[ho])

    cert = {
        "artifact": "kirk_glassbox.json",
        "model": {
            "terms": len(graph.terms),
            "max_order": graph.max_order,
            "bias": float(graph.bias),
            "features": feats,
        },
        "metrics_truth": {
            "ebm_ga2m": base_metrics,
            "ebm_plus_3way": hybrid_metrics,
            "compiled_table": graph_metrics,
        },
        "compile_fidelity": {
            "table_vs_hybrid_rmse": float(np.sqrt(np.mean(table_vs_hybrid ** 2))),
            "table_vs_hybrid_max_abs": float(np.max(np.abs(table_vs_hybrid))),
            "table_vs_ebm_rmse": float(np.sqrt(np.mean((ebm_score - base_ho) ** 2))),
            "table_vs_ebm_max_abs": float(np.max(np.abs(ebm_score - base_ho))),
            "ebm_term_diagnostics": ebm_diag,
            "xgb_term_diagnostics": xdiag,
        },
        "additivity_error": graph.additivity_error(X[ho]),
        "n_terms": len(graph.terms),
        "term_ids": [t.id for t in graph.terms],
        "grid_bins": graph.metadata["grid_bins"],
    }
    graph.certificate = cert
    out = ROOT / "models" / "kirk_glassbox.json"
    graph.to_json(out)
    cert_path = ROOT / "models" / "kirk_glassbox_certificate.json"
    cert_path.write_text(json.dumps(cert, indent=1))
    print(json.dumps(cert["metrics_truth"], indent=1))
    print(json.dumps(cert["compile_fidelity"], indent=1)[:1200])
    print("additivity", cert["additivity_error"])
    print("terms", len(graph.terms), "max_order", graph.max_order)
    print("artifact bytes", out.stat().st_size)
    print("saved", out)


if __name__ == "__main__":
    main()
