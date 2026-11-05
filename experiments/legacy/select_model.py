"""Controller demo: choose compact vs max-accuracy by term budget."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
KIRK = Path(os.environ.get("KIRK_ROOT", ROOT.parent / "TESTAstronomy"))
DATA = Path(os.environ.get("KIRK_DATA", KIRK / "data" / "features_8000_4000_0.npz"))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT / "experiments" / "legacy"))

from termgraph.block import ModelSpec, TermBlock  # noqa: E402
from termgraph.controller import (  # noqa: E402
    load_candidate, select, selection_table,
)

compact_blocks = [
    TermBlock("order3_resonance",
              ("EMcrossnear", "MMRstrengthnear", "EMfracstdnear"),
              "table", "xgboost", max_order=3),
    TermBlock("pair_emcrossfar_tsec", ("EMcrossfar", "Tsec"),
              "table", "xgboost", max_order=2),
    TermBlock("pair_megno", ("MEGNO", "MEGNOstd"),
              "table", "xgboost", max_order=2),
    TermBlock("pair_epstdnear", ("EPstdnear", "EMfracstdfar"),
              "table", "xgboost", max_order=2),
    TermBlock("pair_epstdfar", ("EPstdfar", "MMRstrengthfar"),
              "table", "xgboost", max_order=2),
]
max_blocks = [
    TermBlock("ebm_low_order", "auto", "table", "ebm", max_order=2,
              notes="EBM main effects + pairwise interactions"),
    TermBlock("order3_resonance",
              ("EMcrossnear", "MMRstrengthnear", "EMfracstdnear"),
              "table", "xgboost", max_order=3),
]


def main():
    d = np.load(DATA, allow_pickle=True)
    X = d["X"].astype(np.float64)[:100]
    specs = [
        ModelSpec(
            "compact_5term",
            str(ROOT / "models" / "kirk_glassbox_compact.json"),
            compact_blocks,
            expected_auc=0.9296,
            max_order=3,
            notes="5 terms, exact ledger, 5.7 MB",
        ),
        ModelSpec(
            "max_accuracy_22term",
            str(ROOT / "models" / "kirk_glassbox_full.json"),
            max_blocks,
            expected_auc=0.9427,
            max_order=3,
            notes="EBM low-order + order-3 residual, 5.8 MB",
        ),
    ]
    candidates = [load_candidate(s, X=X) for s in specs]
    out = {"candidates": selection_table(candidates), "selections": {}}
    print("candidates")
    for c in candidates:
        print(" ", c.describe(), c.contract.violations)
    for budget in (4, 5, 6, 22):
        chosen = select(candidates, term_budget=budget, max_order=3)
        out["selections"][str(budget)] = (
            None if chosen is None else chosen.spec.name
        )
        print(f"budget={budget:2d} -> {out['selections'][str(budget)]}")
    path = ROOT / "experiments" / "results" / "kirk_controller.json"
    path.write_text(json.dumps(out, indent=1))
    print("saved", path)


if __name__ == "__main__":
    main()
