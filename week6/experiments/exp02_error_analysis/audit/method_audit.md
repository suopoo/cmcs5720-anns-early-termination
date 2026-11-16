# Method audit — what can and cannot be compared across LAET, DARTH, ANNiE

Experiment 2 compares **per-query prediction errors**, not raw error values
across methods.  This audit fixes the meaning of every number before the
main run.

## 1. Prediction targets (not interchangeable)

| Method | Native target | Unit | When predicted | Stop decision |
|---|---|---|---|---|
| LAET | cost of finding the GT top-1 neighbour, `log2(C1)` | log2 NDC | once, at NDC ≥ f=1,260 | `NDC ≥ max(f, min(2^pred·mult, pred_max))` |
| DARTH | current `recall@100` | recall ∈ [0,1] | repeatedly, at adaptive intervals | `predicted recall ≥ R_t` |
| ANNiE-S | cost to reach `recall@100 ≥ R_t` | NDC | once, at the homing→refining transition | `NDC ≥ predicted cost` |

Consequences:

* LAET and ANNiE both predict a **cost**, so a signed cost error is defined.
  Their *native labels* differ (top-1 vs target recall), so their native
  errors are reported separately; the common comparison uses the effective
  budget error against the **target-recall** oracle.
* DARTH predicts a **recall**, whose signed error has different units.  It is
  never placed on the same numeric axis as a cost error.
* A positive LAET/ANNiE error means over-budget (wasted NDC); a positive
  DARTH error means over-estimated recall (risk of a **false stop**).  The
  practical consequences are opposite in sign, which is why the experiment
  analyses them separately.

## 2. Post-processing and calibration

* **LAET** applies `2^pred` and a multiplier calibrated on validation
  queries, clipped by `pred_max = 2^max(training label)`.  The raw model
  output is therefore *not* a budget; the experiment records both
  `predicted_cost_raw` and `effective_budget`.
* **DARTH** clips the model output to `[0,1]` and compares it with `R_t`.
  Its prediction interval is `pi = mpi + (ipi-mpi)(R_t - pred)`, so the model
  changes not only *whether* but *when* it is called.  Checkpoint counts are
  part of the method's behaviour, not an implementation detail.
* **ANNiE-S** predicts a quantile (tau=p=0.95) of the cost.  The guarantee is
  probabilistic (`Pr[Chat ≥ C] ≥ p`), not a per-query hard bound; the
  empirical coverage is reported but is not expected to be exactly 0.95.
  If `Chat` is below the homing checkpoint, the method stops immediately at
  that checkpoint (lower bound = homing NDC).

## 3. Search trajectories are method-specific

* LAET / ANNiE use an **unlimited-beam** HNSW search: every fresh neighbour
  enters the candidate queue, and only the learned budget ends the search.
* DARTH uses a **bounded efSearch** search tuned on validation to the
  highest tested value, ef=2048, reaching mean recall@100 = 0.9855 (the
  target 0.99 is not attainable at the tested cap on this 992k GIST index;
  recorded as a limitation, not hidden).  Its oracle cost is therefore computed on *its own* trajectory,
  not by extending the unlimited-beam trajectory.
* The oracle is the first NDC at which recall@100 reaches `R_t` on the
  method's own complete trajectory, with the learned stopping disabled.
  `oracle_attainable = false` means the trajectory ended naturally before
  `R_t`; `oracle_status = unknown` means the NDC cap was hit.

## 4. NDC accounting

NDC counts every squared-L2 evaluation in the base layer (upper-layer
greedy entry search included, as in the week-5 validation).  Feature
extraction costs no distance computations; its wall-clock cost is measured
separately and never folded into NDC.  All three methods use the same search
loop and the same counter.

## 5. What is directly comparable

| Quantity | Comparable? | Caveat |
|---|---|---|
| Query-level target hit (`recall@100 ≥ 0.95`) | yes | same 1000 query IDs, same graph |
| Stop NDC, extra NDC vs method-specific oracle | yes | different oracle trajectories |
| Quality-failure / efficiency-failure sets | yes | unified result-level definitions |
| Native prediction error | **no** | different targets/units |
| Number of predictor calls | only within a method | DARTH is repeated by design |
| Wall-clock latency | yes, measured identically | Python implementation; NDC is the portable cost |

## 6. Implementation deviations (documented, not hidden)

1. DARTH training observations are sampled every 25 NDC (week-5 used 5).
   The adaptive interval is `d_bar/10`, typically hundreds of NDC, so 25 is
   below the decision granularity.
2. ANNiE's local subgraph uses popped nodes first and scans the result heap
   only when fewer than s=200 nodes are available.  This is the paper's
   Def. 7 approximation with a cost-motivated implementation choice.
3. ANNiE's feature vector includes the raw 960-d query vector (§4.2), which
   the week-5 prototype omitted.
4. Homing `(w,k')` is calibrated once at R_t=0.95 on 60 training queries to
   the paper's target homing fraction 0.2, then reused for the R_t
   sensitivity runs; the transition position itself does not depend on R_t.
5. The GIST graph is built with hnswlib at M=16, efc=200 and parsed into the
   instrumented search; a spot check against hnswlib's own search verifies
   the adjacency.
