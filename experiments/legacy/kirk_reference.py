"""Reference ceiling and paired comparisons for the KIRK TermGraph test.

Trains an unconstrained XGBoost student on the same five splits, then reports:
  - unconstrained vs flat vs PBAM k=5
  - paired TermGraph vs the shipped NAM per-seed values
"""
from __future__ import annotations
import os

import json
import sys
import time
from pathlib import Path

import numpy as np
import xgboost as xgb

ROOT = Path(__file__).resolve().parents[2]
KIRK = Path(os.environ.get("KIRK_ROOT", Path(__file__).resolve().parents[1].parent / "TESTAstronomy"))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT / "experiments" / "legacy"))

from termgraph.gbm import train_constrained, fidelity_metrics  # noqa: E402
from kirk_termgraph import load_data  # noqa: E402

# The shipped 12k NAM's five-seed sweep, same data and same split rule.
NAM_AUC = [0.908308198519331, 0.9221602909510515, 0.9267823403643864,
           0.9188529676429451, 0.9268180921539296]
NAM_AGREE = [0.8935879945429741, 0.9120054570259208, 0.9010914051841746,
             0.8908594815825375, 0.9072305593451568]
NAM_RMSE = [1.12533683445857, 1.072, 1.049, 1.090, 1.061]  # seed-0 exact; see note
NAM_ACC = [0.8287858117326057, 0.8478854024556617, 0.845156889495225,
           0.8383356070941337, 0.844474761255116]


def main():
    X, Y, labels, feats, tr0, ho0 = load_data()
    rows = []
    for seed in range(5):
        rng = np.random.default_rng(seed)
        perm = rng.permutation(len(X))
        cut = int(len(X) * 0.85)
        tr, ho = perm[:cut], perm[cut:]
        t0 = time.time()
        booster = train_constrained(
            X[tr], Y[tr], [list(range(len(feats)))], feats,
            Xval=X[ho], yval=Y[ho],
            max_depth=4, eta=0.05, num_round=1500,
            early_stopping=80, seed=seed,
        )
        score = booster.predict(
            xgb.DMatrix(X[ho], feature_names=feats), output_margin=True
        )
        m = fidelity_metrics(score, Y[ho], labels[ho])
        m["seed"] = seed
        m["seconds"] = round(time.time() - t0, 1)
        rows.append(m)
        print(
            f"seed {seed} unconstrained aucY {m['auc_truth']:.4f} "
            f"aucT {m['auc_teacher']:.4f} agree {m['agree_teacher']:.4f} "
            f"rmse {m['rmse_teacher']:.4f}",
            flush=True,
        )

    summary = {}
    for key in ("auc_truth", "auc_teacher", "agree_teacher", "rmse_teacher",
                "acc_truth"):
        vals = np.array([m[key] for m in rows])
        summary[key + "_mean"] = float(vals.mean())
        summary[key + "_std"] = float(vals.std())

    sweep = json.loads(
        (ROOT / "experiments" / "results" / "kirk_seed_sweep.json").read_text()
    )
    flat = sweep["summary"]["flat"]
    pbam = sweep["summary"]["pbam_k5"]
    flat_auc = [r["auc_truth"] for r in flat["rows"]]
    pbam_auc = [r["auc_truth"] for r in pbam["rows"]]

    def paired(a, b, label):
        a = np.asarray(a, dtype=float)
        b = np.asarray(b, dtype=float)
        d = a - b
        print(
            f"{label}: mean {d.mean():+.4f}  std {d.std():.4f}  "
            f"min {d.min():+.4f}  max {d.max():+.4f}  per-seed {np.round(d, 4)}"
        )
        return {
            "mean": float(d.mean()),
            "std": float(d.std()),
            "min": float(d.min()),
            "max": float(d.max()),
            "per_seed": [float(v) for v in d],
        }

    print("\npaired deltas")
    nam_flat = paired(
        flat_auc, NAM_AUC, "flat TermGraph - NAM (AUC truth)"
    )
    nam_pbam = paired(
        pbam_auc, NAM_AUC, "PBAM k=5 - NAM (AUC truth)"
    )
    unc_pbam = paired(
        [m["auc_truth"] for m in rows],
        pbam_auc,
        "unconstrained - PBAM k=5 (AUC truth)",
    )

    out = {
        "unconstrained": {"summary": summary, "rows": rows},
        "paired": {
            "flat_vs_nam_auc": nam_flat,
            "pbam5_vs_nam_auc": nam_pbam,
            "unconstrained_vs_pbam5_auc": unc_pbam,
        },
        "reference": {
            "flat": flat,
            "pbam_k5": pbam,
            "nam": {
                "auc_truth_mean": float(np.mean(NAM_AUC)),
                "auc_truth_std": float(np.std(NAM_AUC)),
                "agree_mean": float(np.mean(NAM_AGREE)),
                "acc_mean": float(np.mean(NAM_ACC)),
            },
        },
    }
    path = ROOT / "experiments" / "results" / "kirk_reference.json"
    path.write_text(json.dumps(out, indent=1, default=float))
    print(f"\nsaved {path}")


if __name__ == "__main__":
    main()
