# TermGraph architecture

TermGraph is an exchange layer between a GBM-class model and a deployable
glassbox. The deployed object is a sum of low-order table terms:

```
f(x) = b + sum_j g_j(x[S_j]),    |S_j| <= d
```

The code implements the compiler, the term graph, the contract, the controller,
the editor, and the runtime. The full derivation and evidence are in
[`paper/main.pdf`](paper/main.pdf).

## Term graph

- `S_j` is the support of term `j`; supports may overlap.
- `g_j` is a centered term. The bias `b` absorbs the centering constants.
- Complexity is the vector `(K, d, overlap, cells)`, not just the term count.
- A term is a table over explicit raw-value edges, with sentinel outer bins.
  The table value is `theta_j[kappa_j(x)] - c_j`.

## Attribution

There are two levels, and they are not interchangeable.

**Term ledger.**

```
phi_j(x) = g_j(x[S_j])
f(x) = b + sum_j phi_j(x)
```

This is exact by construction.

**Feature attribution.** A sharing matrix `W` satisfies

```
W[i,j] >= 0,    W[i,j] = 0 if i not in S_j,    sum_i W[i,j] = 1
psi_i(x) = sum_j W[i,j] phi_j(x)
```

The default is equal split, `W[i,j] = 1/|S_j|`. It preserves additivity but is
not full-model Shapley when terms overlap. Full-model Shapley is a separate
computation.

## Order

Two order definitions are reported separately.

- Functional-ANOVA order under a product baseline `P_perp`. The tail energy
  `E_{>d}` bounds approximation under `P_perp`, not under the data distribution.
- Effective order under the data distribution `P`: the smallest `d` whose
  compiled model is within `delta` of the best available model.

On KIRK, the two disagree: marginal ANOVA gives 43.3% of the energy to order at
most three, while an order-3 glassbox captures 93.5% in distribution.

## Glassbox Contract

A deployed graph is admissible only when:

```
K <= K_max,    d <= d_max,    cells <= C_max,
additivity <= epsilon,    certificate present
```

The contract also requires a rendering protocol: a 1D curve, a 2D heatmap, or
3D slices. Cell count alone is not readability.

## Compiler and controller

A compiler maps a source model to a term graph with a measured fidelity
`delta`. The KIRK compiler uses EBM cut points for the low-order terms and the
union of tree thresholds for the order-3 term.

The controller selects a candidate by expected performance subject to the
contract and a `(K, d, cells)` budget. If no candidate fits, it returns no model
rather than violating the budget.

## What is exact

| Claim | Exact? | Caveat |
|---|---|---|
| `f = b + sum_j phi_j` | yes | by construction |
| table compiler | yes iff constant on cells | otherwise `delta` |
| term ledger | yes | baseline declared |
| feature values | convention | sharing matrix `W` |
| full-model Shapley | separate computation | not the ledger when overlapping |
| marginal ANOVA tail | yes under `P_perp` | not the data-distribution limit |

## Implemented artifacts

- `models/kirk_glassbox.json` and `.npz`: 22-term max artifact.
- `models/kirk_glassbox_compact.json`: 5-term compact artifact.
- `models/kirk_glassbox_full.json`: all-data deployment artifact.
- `models/termgraph_report.json`: consolidated evidence.
- `scripts/train_kirk.py` and `configs/kirk.json`: supported training path.
