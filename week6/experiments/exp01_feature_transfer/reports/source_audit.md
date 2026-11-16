# Source & paper audit — Experiment 1 (Feature Transfer)

Scope: the three implementations reused from the week-5 project, checked
against the papers before any result was produced.  All reported numbers in
this experiment come from these code paths, not from the papers.

## 1. What is shared, what is not

| Item | Shared across all groups | Notes |
|---|---|---|
| Dataset | GIST1M, `gist.hdf5`, 1,000,000 × 960 | L2 (squared) metric |
| Index base | 992,000 vectors (`train[0:992000]`) | training/validation queries are held out, so they are not graph nodes |
| HNSW graph | one hnswlib index, M=16, efConstruction=200, seed 42 | parsed into the instrumented CSR search |
| Test queries | official `test[0:1000]` | same IDs for every group |
| Validation | `train[999000:999500]` (500 queries) | tuning only |
| Training | `train[992000:997000]` (first 5,000 of 7,000) | never in the graph |
| k / target | k=10 / R_t=0.90 | first-round configuration required by the task |
| Checkpoint | f = 1,260 NDC | LAET's published GIST1M preliminary-search length |

The six groups differ **only** in the input feature matrix.  Model type,
label, loss, training rows, checkpoint, search algorithm and stopping rule
are held fixed within each method.

## 2. LAET (SIGMOD 2020, HNSW case)

| Question | Answer in the code |
|---|---|
| Prediction target | `log2(C1)`, `C1` = NDC at which the ground-truth 1-NN is first *touched* by the search (`core.search.run_query(..., watch={gt[0]})`) |
| Log transform | yes — `log2`, inverted with `2**pred` when forming a budget (paper §4.3.3) |
| Features | query vector (960) + `d_start`, `d_1st`, `d_10th`, `d_1st/d_start`, `d_10th/d_start` (Table 5) |
| Model | LightGBM, L2 loss, 100 rounds, lr=0.2, 31 leaves (§4.1.3) |
| Observation point | one prediction at the first step with NDC ≥ f=1,260 |
| Budget rule | `B = max(f, min(2**max(pred,0) * multiplier, pred_max))`, where `pred_max = 2**max(training label)`; copied from the official `HNSW.cpp::search_from_candidate_unbounded_pred` |
| Multiplier | calibrated per model on the 500 validation queries: smallest value whose mean recall@10 reaches 0.90 |
| Search | unlimited-beam HNSW (LAET's mode): every fresh neighbour is admitted to the candidate queue |

**Known approximations.** (i) `C1` is recorded when the GT 1-NN is first
*distance-evaluated* (the paper's definition); the week-5 code approximated
this by top-k entry — this experiment uses the stricter watch-based counter.
(ii) The original C++ uses a percentile grid for `f`; here `f` is fixed at
the paper's GIST1M value 1,260, which is also the unified checkpoint required
by the experiment design.

## 3. DARTH (SIGMOD 2025)

| Question | Answer in the code |
|---|---|
| Prediction target | current `recall@k` at the checkpoint |
| Features | the 11 of Table 1: `step, ndc, inserts, d_first, d_closest, d_furthest, avg, var, median, p25, p75` (over the current BSF k-set) |
| Label generation | `hit/k` at each checkpoint, computed online against GT |
| Model | LightGBM regression. **The official code uses the default L2 objective**, so D0 is L2 — not a quantile loss |
| Initial / minimum interval | `ipi = d_bar(R_t)/2`, `mpi = d_bar(R_t)/10`, the paper's untuned heuristic (§3.2.2); `d_bar` = mean NDC at which training queries first reach `R_t` |
| Adaptive rule | `pi = mpi + (ipi - mpi) * (R_t - pred)`; next call when `NDC - last_ndc >= pi` |
| Stop rule | stop when `pred >= R_t` |
| Truncation | prediction clipped to `[0,1]` before the decision (range check required by the task) |

**Mode A (main transfer experiment).** First call forced at NDC ≥ 1,260,
then a common fixed grid of every 200 NDC.  This removes the feedback loop
between the model and the interval so D0/D1/D2 see **identical states**.

**Mode B / Original reference.** The original adaptive interval starting at
NDC 0, trained on observations from 0 (sampled every 25 NDC), is run as the
"Original DARTH" reference. It is labelled separately and never mixed with
the controlled variant.

**Training-observation schedule.** DARTH's paper does not prescribe a
sampling stride.  The week-5 reproduction used every 5 NDC; here the training
cache uses every 25 NDC (the adaptive interval is `d_bar/10`, typically
hundreds of NDC, so 25 is well below the decision granularity).  This is a
deliberate, documented deviation and does not change the online rule.

## 4. ANNiE (VLDB 2026)

| Question | Answer in the code |
|---|---|
| Feature groups | §4.2 basic: query vector (960), NDC, step, inserts, 6 BSF distance stats (first/closest/furthest/avg/var/median/p25/p75), recall target; §4.5 graph: average clustering coefficient, per-BSF-node clustering coefficient (k values), h-hop reachability (h=4) |
| Local subgraph | the s=200 closest nodes accessed (popped nodes first; result heap only if fewer than s are popped), plus out-neighbours if still short — the Def. 7 approximation of the paper |
| Graph-feature cost | O(s·d) adjacency construction per extraction; measured and reported as feature-extraction time |
| Homing | sliding window w=calibrated, BSF heap k'=calibrated; transition when `min(window) > dists[k'-1]` |
| Cost label | NDC at first `recall@k >= R_t` on the unlimited-beam trajectory |
| Model | LightGBM quantile, tau=p=0.95 |

**Differences from the week-5 implementation.** (i) The raw query vector is
included, as required by §4.2; week-5 omitted it. (ii) The local-subgraph
builder stops scanning the (possibly 2,000-entry) result heap once s popped
nodes are available; week-5 always scanned the whole result heap. This is a
speed optimisation of the same Def. 7 approximation. (iii) For the transfer
experiment, ANNiE appears only as a **feature source**; its own stopping rule
is not used.

## 5. Feature leakage checks

* No feature uses ground-truth labels, future nodes, the final stopping cost,
  or any post-hoc statistic.  `run_query` only exposes the state at the
  current NDC: result heap, BSF k-set, counters, popped list so far, and the
  query vector.
* Labels (`C1`, current recall, `C*(R_t)`) are computed from the same
  trajectory but are only used as supervision, never as inputs.
* Train / validation / test queries are disjoint at the Query-ID level.  The
  DARTH checkpoint samples are grouped by query before splitting; no query
  appears on both sides.
* `D1 == D2` for DARTH after de-duplication: DARTH's 11 features are a strict
  subset of ANNiE's basic block.  This is reported, not hidden.

## 6. Reproduction status of the baselines

The instrumented search loop was validated in week 5 against faiss at every
`efSearch` (identical recall).  The GIST graph used here is built by hnswlib
with the same M / efConstruction as the week-5 faiss graph; a spot check
compares its retrieved IDs with hnswlib's own search (see
`reproducibility/validation_results.md`).
