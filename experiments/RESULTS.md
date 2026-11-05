# KIRK term-graph experiment

> **Historical record.** The scripts that produced these numbers are frozen in
> `experiments/legacy/`; their constants are intentional. New runs use
> `scripts/train_kirk.py` with `configs/kirk.json`.

**Date:** 2026-11-04
**Data:** `TESTAstronomy/data/features_8000_4000_0.npz`
12,000 real systems; 9,767 short-stable; 11 SPOCK 2.0 features.
**Target:** SPOCK 2.0 XGBoost teacher logits.
**Split:** same 85/15 seed sweep as the shipped 12k NAM (seeds 0 to 4).
**Models:** constrained XGBoost, exact tree-to-term decomposition.

## Headline table

AUC is against the real 10^9 labels. Mean ± std over five seeds.

| Model | Terms | Order | AUC truth | Agree teacher | RMSE logit | Acc truth |
|---|---:|---:|---:|---:|---:|---:|
| NAM 12k (shipped) | 11 | 1 | 0.9206 ± 0.0068 | 0.9010 | 1.125* | 0.8409 |
| Flat TermGraph | 11 | 1 | 0.9315 ± 0.0062 | 0.9250 | 0.894 ± 0.02 | 0.8619 |
| PBAM k=3 | 8 | 2 | 0.9338 ± 0.0060 | 0.9315 | 0.817 | 0.8625 |
| PBAM k=5 | 6 | 2 | 0.9348 ± 0.0061 | 0.9323 | 0.795 | 0.8611 |
| **PBAM + 3-way** | **5** | **3** | **0.9364 ± 0.0058** | **0.9393** | **0.776** | **0.8656** |
| Unconstrained student | 1 | 11 | 0.9428 ± 0.0053 | 0.9559 | 0.501 | 0.8727 |
| Teacher (SPOCK) | n/a | n/a | 0.9566 ± 0.0041 | 1.000 | 0.000 | n/a |

\* NAM seed-0 holdout RMSE; the artifact does not store the five-seed RMSE.

Paired deltas (same seed, same split):

- Flat TermGraph − NAM: **+0.0109 ± 0.0024 AUC** (all five seeds positive).
- PBAM k=5 − NAM: **+0.0142 ± 0.0030 AUC**.
- PBAM k=5 − flat: **+0.0033 ± 0.0008 AUC**, **−0.099 ± 0.010 RMSE**.
- 3-way − PBAM k=5: **+0.00155 ± 0.0008 AUC**, **+0.0070 ± 0.0029 agreement**, **−0.0193 ± 0.0039 RMSE**.
- Unconstrained − 3-way: **+0.0064 ± 0.0023 AUC**.

The 5-term model closes **71%** of the NAM → unconstrained-student gap and
**44%** of the NAM → teacher gap.

## The 5-term model

One 3-way block + four pairs:

```
(EMcrossnear, MMRstrengthnear, EMfracstdnear)   3-way
(EMcrossfar, Tsec)                                pair
(MEGNO, MEGNOstd)                                 pair
(EPstdnear, EMfracstdfar)                         pair
(EPstdfar, MMRstrengthfar)                        pair
```

Term ablation (zero one term at a time, seed 0):

| Term | Importance std | ΔAUC | ΔAgree | ΔRMSE |
|---|---:|---:|---:|---:|
| EMcrossnear + MMRstrengthnear + EMfracstdnear | 1.230 | −0.0296 | −0.0607 | +0.688 |
| EMcrossfar + Tsec | 0.895 | −0.0224 | −0.0389 | +0.412 |
| MEGNO + MEGNOstd | 1.375 | −0.0181 | −0.0662 | +0.768 |
| EPstdnear + EMfracstdfar | 0.750 | −0.0094 | −0.0205 | +0.282 |
| EPstdfar + MMRstrengthfar | 0.170 | −0.0011 | −0.0048 | +0.025 |

The 3-way term is the most important for ranking, not a spurious complexity
add. The last term is a prune candidate at negligible cost.

## Exactness

- Ledger additivity: `max |bias + Σ terms − margin| ≈ 1.4e-5` (float32 noise).
- SHAP: interventional TreeSHAP block sums correlate with the ledger at
  **r ≥ 0.9999**; block RMSE ≈ 0.005 to 0.012. The residual is a baseline /
  background convention difference, not an approximation of the model.
- The term ledger is exact; SHAP is the same object under a declared baseline.

## What this says

1. **The NAM failure was real but partly the term learner, not only additivity.**
   A tree-based additive model (flat TermGraph) beats the shipped NAM by
   +0.011 AUC with the same 11 terms. The NAM's seed-0 artifact is also below
   its own five-seed mean; compare means, not seed 0.
2. **Interactions are real and pair terms help.** Disjoint pairs add +0.0033
   AUC and cut logit RMSE by 0.099, consistently across seeds. The teacher's
   top interactions are not the three hardcoded pairs in `glass.py`.
3. **A single 3-way block is worth it.** It adds +0.00155 AUC and −0.019 RMSE
   over pairs, and reduces the model from 6 terms to 5. This is the minimal
   hierarchical extension that earns its complexity.
4. **The architecture is a glassbox with an exact ledger.** Predictions
   decompose; attribution is not post-hoc; terms can be zeroed/edited with an
   exact prediction delta.
5. **The remaining gap is capacity/higher order, not additivity alone.**
   The unconstrained student still leads by 0.0064 AUC and 0.275 logit RMSE.
   A 4-way term added only 0.0009 AUC, so one 3-way block is the right complexity.

## What was tested and rejected

- **Overlapping-pair backfitting** (top 3/5/8 pairs, pairs-only and full
  singletons, 6 cycles): AUC 0.9208 to 0.9227 on seed 0, below the disjoint
  6-term PBAM at 0.9242. More terms and a worse optimizer did not pay off.
  The partition model is the better default.
- **Overlapping blocks without hierarchy** are not the answer on this dataset;
  a controlled 3-way block is.

## Can it surpass the GBM / teacher?

Two protocols matter:

- **Paper protocol** (holdout used for early stopping): triple = 0.9364,
  unconstrained = 0.9428, teacher = 0.9566. Relative comparisons are fair
  because every model used the same protocol, but the absolute numbers are
  optimistic.
- **Clean protocol** (internal 85/15 fit/val, fixed holdout): triple = 0.9348,
  unconstrained truth = 0.9381, teacher = 0.9566.

Truth-label training (clean protocol, n=8301):

| Model | AUC truth |
|---|---:|
| Triple, teacher logits | 0.9264 |
| Triple, truth labels | 0.9348 |
| 4-way + pairs, truth labels | 0.9357 |
| Unconstrained, truth labels | 0.9381 |
| Teacher | 0.9566 |

Findings:

1. **Truth labels beat distillation** by +0.0084 AUC for the same 5-term model.
   Distillation caps the student at teacher fidelity; direct labels remove that
   cap. This is a real architectural option, not a trick.
2. **A 4-way block is not worth it**: +0.0009 AUC for a 4D term, leaving a
   0.0024 gap to the unconstrained truth student. The order dial has
   diminishing returns after one 3-way term.
3. **The glassbox does not surpass the unconstrained student or the teacher.**
   It beats the NAM (+0.014 clean) and the distilled triple (+0.008 clean), but
   the unconstrained truth student remains 0.0024 to 0.0064 ahead depending on the
   protocol.
4. **The teacher's advantage is mostly data**: it saw ~100k systems; this
   glassbox saw 8.3k. The scaling curve is still rising but flattening:

| n train | triple truth | triple distill | unconstrained truth |
|---:|---:|---:|---:|
| 1,000 | 0.9152 | 0.9185 | 0.9104 |
| 2,200 | 0.9273 | 0.9228 | 0.9264 |
| 4,000 | 0.9329 | 0.9252 | 0.9345 |
| 6,000 | 0.9333 | 0.9260 | 0.9354 |
| 8,301 | 0.9349 | 0.9264 | 0.9384 |

Distillation plateaus around 0.926; truth-label training keeps improving but
with a flattening slope. The curve does not support a teacher-level claim from
8.3k systems.

The realistic claim is **parity within a certificate**, not AUC dominance:

> A 5-term, order-3 glassbox is within 0.002 to 0.006 AUC of an unconstrained GBM
> trained on the same data, beats the additive NAM by 0.014, and provides exact
> attribution and editability that the GBM cannot.

Surpassing the teacher would require more data, such as the full ~100k archive;
the hybrid residual did not close the gap.

## Is the glassbox statistically tied to the GBM?

Paired bootstrap over the 1,466-system holdout, clean protocol, truth labels:

```
AUC glassbox          0.9342
AUC unconstrained     0.9380
Delta                 0.0038
95% CI                [0.0001, 0.0076]
one-sided 95% upper    0.0070
```

So:

- Exact parity is **rejected** at the 5% level (the CI excludes zero).
- Non-inferiority at a **0.01 AUC margin** is supported (upper bound 0.0076).
- Non-inferiority at a **0.005 AUC margin** is not supported (upper bound 0.0070).

"Matches" is therefore a decision about a tolerance, not a universal property.
At 0.01 AUC the glassbox matches; at 0.005 it is close but not certified.

## Hybrid residual

A 5-term glassbox plus a 168-tree GBM on the residual (base-margin boosting):

```
glassbox AUC          0.9342
hybrid AUC            0.9356
unconstrained AUC     0.9380
coverage (margin var) 92.7%
```

The residual is small (0.1 variance vs 11.6 for the glassbox), adds only
+0.0014 AUC, and does **not** close the gap. On this task the glassbox's errors
are not a cheap residual for a second GBM to learn. The hybrid is useful for
coverage accounting, not for matching the unconstrained model.

## Can a glassbox match?

Yes, in the sense that matters if you define the tolerance first:

- A 5-term, order-3 glassbox is within **0.0038 AUC** of an unconstrained GBM
  trained on the same data, and passes non-inferiority at a **0.01** margin.
- It does **not** match exactly, does not pass at a 0.005 margin, and does not
  approach the 100k teacher (0.0218 gap).
- A fixed-budget glassbox is a constrained hypothesis class. It cannot dominate
  an arbitrary black box on all tasks. What it can do is make the trade-off
  explicit: order, terms, fidelity, calibration, coverage.
- The path to a tighter match is more data or a larger budget, not more
  hierarchical terms by default. The 4-way and hybrid experiments both show
  diminishing returns.

## Matched synthetic tests: where the architecture pays

Known data-generating processes, 12k train / 20k test, 3 seeds. The glassbox
whose blocks match the true order is compared with EBM (GAM and GA2M) and an
unconstrained XGBoost.

| Truth | TermGraph matched | EBM GAM | EBM GA2M | Unconstrained GBM |
|---|---:|---:|---:|---:|
| Additive | 0.7752 (flat) | **0.7779** | 0.7776 | 0.7716 |
| Pairwise | 0.8304 (pairs) | 0.7143 | **0.8319** | 0.8255 |
| **3-way** | **0.8297** (triple) | 0.6944 | 0.6989 | 0.8217 |

Findings:

1. **When the order matches the truth, TermGraph matches or beats the GBM**:
   +0.0036 (additive), +0.0049 (pairwise), +0.0080 (3-way) AUC.
2. **EBM is pairwise-limited.** On the 3-way truth it loses **0.131 AUC** to
   TermGraph and 0.123 to the unconstrained GBM. This is the clear capability
   advantage: EBM cannot represent a 3-way term, TermGraph can.
3. **NAM-like flat models fail when interactions exist**: 0.7110 on pairwise
   and 0.6893 on 3-way, versus 0.8304 and 0.8297 for the matched TermGraph.
4. **The benefit is conditional, not universal.** On KIRK, EBM GA2M reaches
   0.9417 truth AUC versus 0.9348 for TermGraph: that dataset is dominated by
   low-order signal, and EBM's tuned term-wise boosting wins. The controller
   must choose the model, not assume TermGraph always wins.

## Contribution

The contribution is not “a more accurate EBM.” It is:

> **A GBM-native term-graph exchange architecture with a tunable interaction
> order, exact term-ledger attribution, declared feature sharing, editable term tables, and a parity
> certificate.** EBM is fixed at order 2 and is a separate learner; NAM is
> fixed at order 1 and neural. TermGraph is the only one of the three that can
> (a) start from a GBM, (b) add a controlled 3-way term when the certificate
> demands it, and (c) expose the result as an exact, editable ledger.

The three claims and their evidence:

| Claim | Evidence |
|---|---|
| Beats NAM | KIRK +0.011 to +0.014 AUC; pairwise/3-way synthetics +0.12 AUC |
| Beats EBM when 3-way structure exists | 3-way synthetic +0.131 AUC |
| Matches GBM when order matches | synthetic +0.004 to +0.008 AUC; KIRK within 0.0038, non-inferior at δ=0.01 |
| Matches SHAP at term level | ledger additivity 1.4e-5; TreeSHAP block sums r ≥ 0.9999; feature sharing is declared |
| Is editable | term zeroing gives exact prediction/metric deltas; term ablation identifies prune candidates |

## Compiler test: effective order vs marginal ANOVA

The compiler needs a certificate that separates **representation limit** from
**optimization limit**. We implemented a functional-ANOVA energy estimator
under a marginal (product-of-marginals) baseline.

Validation on known independent truths:

| Truth | Cumulative ANOVA share at matching order | Residual |
|---|---:|---:|
| Additive | 0.998 (order 1) | 0.002 |
| Pairwise | 0.993 (order 2) | 0.007 |
| 3-way | 0.962 (order 3) | 0.038 |
| 5-way parity | breaks at order 3 | n/a (needs order 5) |

KIRK teacher, marginal ANOVA:

| Order | Cumulative share |
|---|---:|
| 1 | 0.352 |
| 2 | 0.402 |
| 3 | 0.433 |
| ≥4 | 0.567 |

KIRK compiled certificate (5 seeds, paper protocol):

| Compiled | Order | Terms | R² teacher | Agree | AUC teacher | AUC truth |
|---|---:|---:|---:|---:|---:|---:|
| Flat | 1 | 11 | 0.9142 | 0.9250 | 0.9806 | 0.9315 |
| Pairs | 2 | 6 | 0.9322 | 0.9323 | 0.9848 | 0.9348 |
| Triple | 3 | 5 | 0.9354 | 0.9393 | 0.9866 | 0.9364 |

Residual after order 3: **6.46%** of teacher logit variance.

### The key insight

The marginal ANOVA says only 43.3% of the teacher is order ≤3, yet an
order-3 glassbox captures **93.5%** in-distribution. The difference is
**correlated features**: under a product baseline the teacher activates
high-order terms on off-manifold combinations; on the real data manifold an
order-3 function can mimic it.

Therefore:

- **Order is not a property of the function alone; it is relative to the data
  distribution and the baseline.**
- Define **effective order** = smallest order whose in-distribution fidelity is
  within δ of the best compiled model. On KIRK: order 2 at δ = 0.005 R²;
  order 3 at δ = 0.003.
- The compiler certificate must report **both**: in-distribution R² by order
  (primary) and marginal ANOVA energy (OOD robustness diagnostic).
- This is also a warning for SHAP: an interventional baseline can make a model
  look more high-order than it is on the data you actually serve.

The certificate is saved at `experiments/results/kirk_certificate.json`.

## Absorbing EBM: the heterogeneous term graph

EBM is the better low-order optimizer on KIRK. Instead of competing, use EBM
for the main effects and pairs, then add the order-3 block it cannot represent.

| Model | Terms | AUC truth | Logloss | Acc |
|---|---:|---:|---:|---:|
| NAM 12k | 11 | 0.9206 | n/a | 0.8409 |
| TermGraph triple (XGBoost only) | 5 | 0.9364 | n/a | 0.8656 |
| EBM GA2M | 21 | 0.9417 ± 0.0049 | 0.2917 | 0.8718 |
| **EBM GA2M + 3-way residual** | **22** | **0.9427 ± 0.0048** | **0.2895** | **0.8729** |
| EBM + phase 3-way (control) | 22 | 0.9413 | 0.2927 | 0.8704 |

Paired by seed, the resonance 3-way hybrid beats EBM by **+0.0010 ± 0.0007
AUC**; all five seeds are positive. The phase 3-way is neutral-to-negative,
which shows the gain is specific to the right block, not a generic “more
terms” effect.

The architecture implication:

```
TermGraph = a heterogeneous term graph
  main effects + pairs  -> EBM term-wise boosting (absorbed)
  order-3 block         -> XGBoost constrained term (added)
  smooth renderings     -> NAM / spline basis (absorbed)
  attribution           -> exact term ledger; declared feature sharing W
  editing               -> term tables, one IR
  certificate           -> order, terms, fidelity, coverage
```

EBM is not a competitor; it is the best low-order backend. NAM is not a
competitor; it is a smooth basis renderer. TermGraph is the exchange layer that
combines them and adds what neither has: order-3, exact ledger, editability,
and a parity certificate.

## Deployable artifact: exact compile, parallel runtime, edit isolation

The EBM + order-3 model is compiled into explicit tables and deployed as a
pure-NumPy JSON artifact.

| Check | Result |
|---|---|
| Terms / max order | 22 / 3 (11 main + 10 pair + 1 three-way) |
| Table vs EBM+3-way hybrid | RMSE **9.6e-16**, max **5.3e-15** |
| Additivity | **0.0** |
| Fold-0 AUC / log-loss | 0.9350 / 0.3230 (identical to hybrid) |
| 5-seed compiled AUC | **0.9427 ± 0.0048** |
| 5-seed EBM AUC | 0.9417 ± 0.0049 |
| Parallel inference | **8.72×** on 8 threads; 5.66M rows/s; max diff **5.3e-15** |
| Edit isolation | remaining terms change **0.0** for zero/clamp/monotone |
| Tests | **14 passed** |
| Compact artifact | 5 terms, AUC 0.9296, 5.7 MB, contract-admissible |
| Controller | budget 5 → compact; budget 22 → max-accuracy; budget 4 → none |

Artifacts:

- `models/kirk_glassbox.json`: fold-0 certified artifact.
- `models/kirk_glassbox_full.json`: all-data deployment artifact; expected
  five-seed CV AUC 0.9427, in-sample AUC 0.9582.
- `models/kirk_glassbox_compact.json`: 5-term compact artifact; AUC 0.9296 at
  the same 5-term budget where EBM reaches 0.9252; exact ledger; 5.7 MB.
- `models/kirk_glassbox_compact_certificate.json`: compact certificate.
- `models/kirk_glassbox_certificate.json` and
  `models/kirk_glassbox_full_certificate.json`: max-accuracy certificates.

The budget controller (`experiments/legacy/select_model.py`) applies the Glassbox
Contract and selects:

| Term budget | Selected model |
|---:|---|
| 4 | none (no admissible model) |
| 5 | `compact_5term` |
| 6 | `compact_5term` |
| 22 | `max_accuracy_22term` |

Both artifacts are contract-admissible (5 terms / 22 terms, max order 3,
additivity 0.0). The compact table's compile error is **8.4e-7 RMSE** against
its own booster; its AUC is identical to that booster (0.9296). The
certified-fold artifact is the bit-exact one (5.3e-15).

The all-data artifact is an approximate deployment compile (0.0051 RMSE,
0.276 maximum absolute error), while the certified fold artifact is the exact
one (5.3e-15). The full-data file is a 6.0 MB JSON, loads without InterpretML or
XGBoost, and predicts with NumPy only.

### Negative result: a second residual term overfits

Adding a second 3-way residual term (phase, Tsec-MEGNO, or resonance-MEGNO)
lowers seed-0 AUC from 0.9350 to 0.9342 or 0.9343 and worsens log-loss. One
order-3 term is the right complexity; a second residual term did not help.

### Readability / writability

- `experiments/results/kirk_describe.txt`: full term inventory with inputs,
  bin counts, ranges, and backend.
- `assets/kirk_glassbox_main.png`: top main-effect curves.
- `assets/kirk_glassbox_threeway.png`: 3-way term slices.
- Edits are term-level JSON table changes; `diff_graphs` reports exact
  prediction and term deltas.
