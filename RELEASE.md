# TermGraph v0.1.4

**Release date:** 2026-11-05  
**Author:** Patricio Alejandro Mery Caro  
**License:** MIT

This release ships the TermGraph artifacts with a JMLR-format preprint. The
scientific results are recorded in the certificates and the experiment log.

## Highlights

- 22-term max-accuracy artifact: EBM low-order terms + one order-3 XGBoost term.
- 5-term compact artifact for strict readability budgets.
- Exact term ledger and declared feature sharing.
- Glassbox Contract, budget controller, and post-hoc term edits.
- Fused NumPy runtime: 8.72× on eight threads, 5.66M rows/s.
- Binary NPZ artifact: 0.66 MB, 7.4× faster load than JSON.
- Ten-page JMLR preprint paper and a figure-driven README.

## Verified numbers

| Item | Value |
|---|---:|
| KIRK max AUC, 5 seeds | 0.9427 ± 0.0048 |
| EBM GA²M | 0.9417 ± 0.0049 |
| NAM | 0.9206 |
| Compact 5-term AUC | 0.9296 |
| EBM at 5-term budget | 0.9252 |
| 3-way synthetic AUC | 0.8297 vs EBM 0.6989 |
| Certified-fold compile error | 5.3e-15 |
| Fused runtime | 8.72×, 5.66M rows/s |
| Certified-fold NPZ size | 0.66 MB |
| Tests | 14 passed |

## Artifacts

| File | Purpose |
|---|---|
| `models/kirk_glassbox.json` | certified fold artifact |
| `models/kirk_glassbox.npz` | binary serving artifact |
| `models/kirk_glassbox_full.json` | all-data deployment artifact |
| `models/kirk_glassbox_compact.json` | 5-term compact artifact |
| `paper/main.pdf` | compiled manuscript |
| `SHA256SUMS` | checksums for release files |

## Verify the release

```bash
python -m pip install -e ".[train,plots,dev]"
python -m pytest -q
python scripts/reproduce_plots.py
python scripts/check_repo.py
cd paper && latexmk -pdf -interaction=nonstopmode main.tex
```

Expected:

```text
14 passed with KIRK_ROOT set; 8 passed, 6 skipped without the external data
OK: release checks passed
paper/main.pdf: 10 pages, JMLR preprint style
```

## Known limitations

- EBM remains better on low-order KIRK AUC; the compact model is not a universal
  accuracy win.
- The max artifact is exact for a declared EBM term set, not a canonical
  functional-ANOVA decomposition.
- Feature sharing `W` is a convention; full-model Shapley is separate.
- One real task; no human user study, categorical benchmark, multi-class
  objective, or full 100k-data training.
- Order-3 readability is slice-based.

## Citation

```bibtex
@misc{merycaro2026termgraph,
  title  = {TermGraph: A Certified Glassbox Exchange Architecture for GBM-Class Models},
  author = {Mery Caro, Patricio Alejandro},
  year   = {2026},
  note   = {Independent Researcher, La Serena, Chile}
}
```

## Publication status

Published at [github.com/AleeeeM2008/termgraph](https://github.com/AleeeeM2008/termgraph).
