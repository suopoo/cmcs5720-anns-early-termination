# Week 6 — three experiments: what was actually run and what was found

Dataset: **GIST1M**, one hnswlib index (M=16, efConstruction=200) over
**992,000** vectors (8,000 queries held out); 960-d squared L2.  Query splits
are disjoint: train 7,000 (5,000 used in exp1/3, 2,000 in exp2), validation
500, official test 1,000.  Every number below comes from the CSV/JSON
artefacts in this folder; nothing is simulated or hand-entered.

## Experiment 1 — Feature transfer (k=10, R_t=0.90, f=1,260)

Six groups, identical labels/loss/checkpoint/search; only the feature matrix
changes.

| group | mean recall | hit rate | mean NDC | native MAE (NDC) | MAPE |
|---|---|---|---|---|---|
| L0 (LAET original) | 0.9040 | 0.499 | 4,073 | 895.8 | 0.916 |
| L1 (LAET ∪ ANNiE) | 0.8908 | 0.465 | 3,798 | **870.2** | **0.848** |
| L2 (ANNiE only) | 0.8943 | 0.468 | 3,792 | 894.3 | 0.885 |
| D0 (DARTH original) | 0.9242 | 0.537 | 5,013 | — | — |
| D1 (DARTH ∪ ANNiE) | 0.9214 | 0.515 | 5,085 | — | — |
| D2 (ANNiE only) | 0.9214 | 0.515 | 5,085 | — | — |
| D0-original (adaptive ipi) | 0.9250 | 0.546 | 4,632 | — | — |

Key findings:

1. **H1 (LAET) supported on prediction accuracy.**  L1 vs L0 absolute
   log2-cost error: Δ = −0.0715, 95% CI [−0.1123, −0.0313]; L2 vs L0:
   Δ = −0.0467 [−0.0836, −0.0075].  L1 vs L2 is inconclusive
   [−0.0539, +0.0066].
2. **The accuracy gain did not become a quality gain.**  L1 vs L0 search
   recall is −0.0132 [−0.0189, −0.0074] while NDC is −275 [−418, −114].
   The transferred model spends less and reaches the target slightly less
   often; H3 (end-to-end improvement) is not supported.
3. **DARTH is unchanged by the transfer.**  D1 vs D0 query-mean absolute
   recall error Δ = −0.00007 [−0.0016, +0.0016], recall Δ = −0.0028
   [−0.0076, +0.0022], NDC Δ = +72 [−220, +366].  D1 and D2 are identical
   by construction because DARTH's 11 features are a subset of ANNiE's
   basic block (all paired differences exactly 0).
4. **Feature extraction cost is real.**  Measured on 30 test queries:
   DARTH D0 65.7 ms/query, D1/D2 182.7/183.5 ms/query (≈105 ms is graph
   feature extraction alone).  LAET L0/L1/L2: 34.0/29.2/32.6 ms (one
   extraction per query).
5. **Checkpoint sensitivity is flat.**  L0 retrained at f=256/512/1260/2048:
   hit rate 0.500/0.479/0.499/0.496.  D0: 0.530/0.535/0.546/0.563.
   Conclusion is not an artefact of choosing f=1,260.

## Experiment 2 — Per-query prediction errors (k=100, R_t=0.95)

| method | mean recall | target hit rate | mean NDC | rel. extra NDC | oracle attainable |
|---|---|---|---|---|---|
| LAET | 0.9562 | 0.676 | 12,150 | 0.799 | 0.980 |
| DARTH | 0.9608 | 0.644 | 12,691 | 1.063 | 0.926 |
| ANNiE-S | **0.9736** | **0.838** | 15,996 | 0.935 | 0.980 |

1. **Failures are strongly non-random and survive cost stratification.**
   Quality-failure overlap (all p=0.0005, stratified permutation by oracle
   cost): LAET∩DARTH 225 queries (Jaccard 0.495), LAET∩ANNiE 122 (0.335),
   DARTH∩ANNiE 149 (0.404); **118 queries fail in all three**.  Efficiency
   failures also overlap (Jaccard 0.129–0.434, p≤0.0015).
2. **The methods trade quality for cost differently.**  ANNiE vs LAET:
   recall +0.0175 [+0.0152, +0.0198] at +3,847 NDC [+3,414, +4,287]; LAET vs
   DARTH: recall −0.0046 [−0.0070, −0.0024], NDC −541 [−923, −179], relative
   extra NDC −0.334 [−0.441, −0.230].
3. **Training size helps monotonically, most for ANNiE.**  500→2000 queries
   (3-seed mean): ANNiE hit 0.769→0.838, DARTH 0.633→0.653, LAET
   0.651→0.676.
4. **Hard-query-enriched training (1,000 queries) raises hit rates but
   costs NDC.**  D0→D2: LAET 0.596→0.624, DARTH 0.654→0.724, ANNiE
   0.848→0.908; NDC rises from 10,881→11,850, 12,894→14,324,
   15,652→18,119.  No free lunch.
5. **Recall-target sensitivity is method-specific.**  R_t = 0.90/0.95/0.99:
   LAET hit 0.604/0.676/0.687; DARTH 0.703/0.644/0.584 (degrades as the
   target rises); ANNiE 0.854/0.838/0.703.

## Experiment 3 — Asymmetric loss and prediction timing (k=10, R_t=0.90)

**3A-LAET (A2: each model's multiplier recalibrated on validation)**

| loss | tau | hit rate | mean NDC |
|---|---|---|---|
| L0 L2 | — | 0.499 | 4,073 |
| L1 quantile | 0.50 | 0.473 | 3,622 |
| L2 | 0.75 | 0.495 | 4,105 |
| L3 | 0.90 | 0.484 | 3,741 |
| L4 | 0.95 | 0.504 | 3,985 |

After recalibration there is **no meaningful gain** (L4 vs L0 recall
+0.0010 [−0.0046, +0.0063]; NDC −88 [−226, +58]).  With the L0 multiplier
fixed (A1), L4 reaches hit 0.818 vs 0.499 at 12,530 vs 4,073 NDC — i.e. the
apparent “benefit” is a larger effective budget, not a better cost estimate.
This is the central LAET finding.

**3A-DARTH**

| loss | tau | hit rate | mean NDC | calls |
|---|---|---|---|---|
| D0 L2 | — | 0.546 | 4,632 | 8.5 |
| D1 | 0.50 | 0.261 | 1,543 | 1.0 |
| D2 | 0.25 | 0.681 | 6,958 | 14.0 |
| D3 | 0.10 | 0.809 | 11,516 | 24.2 |
| D4 | 0.05 | **0.876** | 18,094 | 41.0 |

Lower tau monotonically reduces over-estimation and raises the hit rate, at
a steep NDC/call cost.  tau=0.50 is pathological (the predictor never crosses
the target, so the adaptive interval grows and only one call is made).

**3B timing** — LAET: no consistent quantile gain at any f
(630/945/1260/1575); DARTH: the quantile model (tau=0.25) adds ≈+0.13 hit
over L2 in every interval configuration, and P2 (1.5×ipi/mpi) is the best
cell for both losses (hit 0.692 quantile vs 0.566 L2).  The loss benefit is
therefore not explained away by interval tuning, but it is paid for in NDC.

## Honest limitations

* GIST1M at 992k, not the full 1M; one graph seed (42).
* DARTH's bounded search uses ef=2048, the highest tested value, reaching
  mean recall@100 0.9855 — not the paper's ≈0.99.  Its oracle is attainable
  for 92.6% of test queries; the rest are marked accordingly.
* Exp2 trains on 2,000 queries (not 5,000) to keep the trace pass tractable.
* Python search: NDC is comparable; wall-clock is not comparable to the
  papers' C++.  Unlimited-beam traces are stopped at full recall (k=10) or
  recall@100=0.99 with expansion caps; a budget beyond the trace end is
  flagged as truncated.
* DARTH D1≡D2 is a structural feature-set fact, not an empirical finding.
* The transfer/quantile conclusions are on one dataset and one GIST graph.

## Where everything is

```
experiments/
  SUMMARY.md                      this file
  README.md                       layout / environment / commands
  common/                         all code (one source of truth)
  shared/cache/                   GIST base .npy, hnswlib index, parsed CSR
  shared/traces/                  all 11 cached trajectories
  shared/ground_truth/, splits/   exact GT and query split
  exp01_feature_transfer/
    data/ analysis/ models/ figures/ reports/ logs/ configs/
  exp02_error_analysis/
    data/ analysis/ figures/ reports/ audit/ reproducibility/ logs/ configs/
  exp03_asymmetric_loss/
    data/ analysis/ models/ figures/ reports/ logs/ configs/
```
