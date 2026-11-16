# Experiment 1 — Research questions, hypotheses and decision rules

Written before the test-set results were inspected.

## RQ1 — Feature effectiveness

Can ANNiE's input features improve LAET's and DARTH's prediction accuracy
without changing their original prediction objectives?

* **H1a:** L1 (LAET ∪ ANNiE) has lower cost-prediction error than L0.
* **H1b:** L2 (ANNiE only) is better than L0.
* **H1c:** L2 is not materially worse than L1 (original features add little).

## RQ2 — Feature complementarity

Do the two feature sets carry complementary information?

Measured by the L1 vs L2 / D1 vs D2 difference in prediction error and by
the fraction of gain carried by graph features in the transferred model.

## RQ3 — End-to-end effectiveness

Do prediction improvements survive feature-extraction and inference cost?

Primary outcomes: Query-level target-hit rate, NDC, and (measured on a
30-query timing subset) end-to-end latency.  Prediction error alone cannot
support H3.

## RQ4 — Prediction timing

Does the value of the transferred features depend on the preliminary-search
depth (f ∈ {256, 512, 1260, 2048})?

## Pre-registered hypotheses

| ID | Statement | Falsified if |
|---|---|---|
| H1 | At the same target and checkpoint, original ∪ ANNiE features add predictive information | L1/D1 do not beat L0/D0 on paired query-level error with a 95% CI that excludes zero |
| H2 | ANNiE features have independent predictive power but need not replace the originals | L2/D2 are no better than the original-feature model, **or** L2/D2 match L1/D1 exactly (no complementarity to measure) |
| H3 | Prediction gains do not automatically become end-to-end gains | the extra feature cost is not recovered: target-hit rate/NDC do not improve, or measured latency is worse |
| H4 | The effect changes with preliminary-search depth | the f × error curves are flat within paired CIs |

## Statistics

* Paired by Query ID over all 1,000 test queries.
* 2,000-replicate paired bootstrap, 95% percentile CI.
* Cliff's delta as effect size.
* Prediction errors are compared **within** a method (LAET cost error in
  NDC/log space; DARTH recall error in recall units); never on a shared
  error axis across methods.
* No test-set tuning: multipliers and thresholds are calibrated on the 500
  validation queries only.

## Decision rule

A hypothesis is *Supported* only when the paired 95% CI excludes zero in the
predicted direction and the effect is not driven by a handful of queries
(checked by the median and by removing the top-1% absolute errors).  A
non-significant result is reported as *Inconclusive*, not as evidence
against the hypothesis.  A significant result in the opposite direction is
*Contradicted*.
