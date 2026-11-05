# Fixed experimental record

These scripts produced the JSON evidence in `experiments/results/`. They are
kept as the exact historical record, with the seeds, feature blocks, grid sizes,
and thresholds that were used for the committed results. Those constants are
intentional here.

**For new runs, use the supported path:**

```bash
export KIRK_ROOT=/path/to/kirk-data
python scripts/train_kirk.py --mode compact
python scripts/train_kirk.py --mode max
```

The canonical entry point reads `configs/kirk.json`; edit that file to change
seeds, blocks, EBM/XGBoost parameters, grid sizes, or budgets.

## What is here

| Group | Scripts |
|---|---|
| Artifact builders | `build_kirk_glassbox*.py` |
| KIRK baselines and ablations | `kirk_*.py` |
| Synthetic order tests | `synthetic_*.py` |
| Controller demo | `select_model.py` |
| Certificate verification | `verify_artifact.py` |

The shared loader `../kirk_termgraph.py` is not frozen; it reads `KIRK_ROOT`
and is used by the historical scripts.

## Running a historical script

```bash
export KIRK_ROOT=/path/to/kirk-data
python experiments/legacy/kirk_ebm.py
```

The scripts insert the repository `src/`, `experiments/`, and
`experiments/legacy/` paths themselves. Some require the optional training
dependencies (`xgboost`, `interpret`, `scikit-learn`, `joblib`).
