"""Config-driven KIRK training entry point.

Examples:
    python scripts/train_kirk.py --mode compact
    python scripts/train_kirk.py --mode max --seed 1

Everything domain-specific (data path, blocks, grid sizes, budgets) comes from
configs/kirk.json. The external data root is `KIRK_ROOT`.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from termgraph.compile import compile_ebm_plus_residual
from termgraph.config import data_path, load_config
from termgraph.contract import audit
from termgraph.export_xgb import export_block_terms
from termgraph.gbm import train_constrained
from termgraph.ir import TermGraph


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def metrics(score, y):
    p = sigmoid(score)
    return {
        "auc": float(roc_auc_score(y.astype(int), p)),
        "logloss": float(log_loss(y.astype(int), p)),
        "accuracy": float(accuracy_score(y.astype(int), p > 0.5)),
    }


def load_frame(path: Path):
    d = np.load(path, allow_pickle=True)
    X = d["X"].astype(np.float64)
    labels = d["labels"].astype(np.float64)
    stable = d["stable"].astype(bool)
    feats = [str(f) for f in d["feats"]]
    return X[stable], labels[stable], feats


def split(X, labels, seed: int, holdout: float):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(X))
    cut = int(len(X) * (1.0 - holdout))
    tr, ho = perm[:cut], perm[cut:]
    perm2 = np.random.default_rng(seed + 100).permutation(len(tr))
    cut2 = int(len(tr) * 0.85)
    fit, val = tr[perm2[:cut2]], tr[perm2[cut2:]]
    return tr, ho, fit, val


def train_residual(X, y, fit, val, triple, base_fit, base_val, cfg, seed):
    p = cfg["xgb"]
    params = {
        "objective": "binary:logistic",
        "max_depth": int(p["max_depth"]),
        "eta": float(p["eta"]),
        "subsample": float(p["subsample"]),
        "min_child_weight": float(p["min_child_weight"]),
        "reg_lambda": float(p["reg_lambda"]),
        "tree_method": "hist",
        "seed": int(seed),
        "eval_metric": "logloss",
        "verbosity": 0,
    }
    dfit = xgb.DMatrix(X[fit][:, triple], label=y[fit].astype(np.float32))
    dfit.set_base_margin(base_fit)
    dval = xgb.DMatrix(X[val][:, triple], label=y[val].astype(np.float32))
    dval.set_base_margin(base_val)
    return xgb.train(
        params, dfit, num_boost_round=int(p["num_round"]),
        evals=[(dval, "val")],
        early_stopping_rounds=int(p["early_stopping_rounds"]),
        verbose_eval=False,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--mode", choices=("compact", "max"), required=True)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    seed = cfg["seed"] if args.seed is None else args.seed
    X, labels, feats = load_frame(data_path(cfg))
    tr, ho, fit, val = split(X, labels, seed, cfg["holdout_fraction"])
    index = {f: i for i, f in enumerate(feats)}

    if args.mode == "compact":
        blocks = [[index[n] for n in block] for block in cfg["blocks"]["compact"]]
        booster = train_constrained(
            X[fit], labels[fit].astype(np.float32), blocks, feats,
            Xval=X[val], yval=labels[val].astype(np.float32),
            max_depth=int(cfg["xgb"]["max_depth"]),
            eta=float(cfg["xgb"]["eta"]),
            num_round=int(cfg["xgb"]["compact_num_round"]),
            early_stopping=int(cfg["xgb"]["early_stopping_rounds"]),
            seed=seed,
            objective="binary:logistic", eval_metric="logloss",
        )
        terms, bias, diag = export_block_terms(
            booster, feats, blocks, X[tr],
            max_bins=int(cfg["grid"]["compact_bins"]), seed=seed,
        )
        graph = TermGraph(
            features=feats, terms=terms, bias=bias,
            metadata={"compiler": "xgboost-constrained", "mode": "compact",
                      "seed": seed},
        )
        exact = booster.predict(
            xgb.DMatrix(X[ho], feature_names=feats), output_margin=True
        )
        max_terms = int(cfg["budgets"]["compact_terms"])
    else:
        from interpret.glassbox import ExplainableBoostingClassifier

        e = cfg["ebm"]
        ebm = ExplainableBoostingClassifier(
            feature_names=feats,
            interactions=int(e["interactions"]),
            max_rounds=int(e["max_rounds"]),
            early_stopping_rounds=int(e["early_stopping_rounds"]),
            outer_bags=int(e["outer_bags"]),
            n_jobs=-1,
            random_state=seed,
        )
        ebm.fit(X[tr], labels[tr].astype(int))
        base_fit = np.asarray(ebm.decision_function(X[fit]), dtype=np.float64)
        base_val = np.asarray(ebm.decision_function(X[val]), dtype=np.float64)
        base_ho = np.asarray(ebm.decision_function(X[ho]), dtype=np.float64)
        triple = [index[n] for n in cfg["blocks"]["residual"]]
        residual = train_residual(X, labels, fit, val, triple, base_fit, base_val, cfg, seed)
        graph, diag = compile_ebm_plus_residual(
            ebm, residual, X[tr], feats, triple,
            n_bins=int(cfg["grid"]["main_bins"]),
            pair_bins=int(cfg["grid"]["pair_bins"]),
            seed=seed,
        )
        term = residual.predict(
            xgb.DMatrix(X[ho][:, triple]), output_margin=True
        )
        exact = base_ho + term
        max_terms = int(cfg["budgets"]["max_terms"])

    score = graph.predict(X[ho])
    cert = {
        "mode": args.mode,
        "seed": int(seed),
        "model": {"terms": len(graph.terms), "max_order": graph.max_order,
                  "bias": float(graph.bias), "features": feats},
        "metrics_truth": metrics(score, labels[ho]),
        "compile_fidelity": {
            "rmse": float(np.sqrt(np.mean((score - exact) ** 2))),
            "max_abs": float(np.max(np.abs(score - exact))),
            "detail": diag,
        },
        "additivity_error": graph.additivity_error(X[ho]),
        "n_terms": len(graph.terms),
        "term_counts": {
            "main": sum(t.order == 1 for t in graph.terms),
            "pair": sum(t.order == 2 for t in graph.terms),
            "threeway": sum(t.order == 3 for t in graph.terms),
        },
        "term_ids": [t.id for t in graph.terms],
    }
    graph.certificate = cert
    report = audit(graph, max_order=3, max_terms=max_terms, X=X[ho][:200])
    cert["glassbox_contract"] = report.to_dict()
    graph.certificate = cert

    out = Path(args.out or ROOT / "models" / f"kirk_glassbox_{args.mode}_retrained.json")
    graph.to_json(out)
    graph.to_npz(out.with_suffix(".npz"))
    cert_path = out.with_name(out.stem + "_certificate.json")
    cert_path.write_text(json.dumps(cert, indent=1))
    print(json.dumps({"artifact": str(out), "metrics": cert["metrics_truth"],
                      "fidelity": cert["compile_fidelity"]["rmse"],
                      "contract": cert["glassbox_contract"]["admissible"]},
                     indent=1))


if __name__ == "__main__":
    main()
