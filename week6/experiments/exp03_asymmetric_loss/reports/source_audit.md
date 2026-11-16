# Source audit — Experiment 3 (asymmetric loss and prediction timing)

Checked against the papers and the audited week-5 implementations before the
first run.

## LAET

| Item | Finding | Where |
|---|---|---|
| Loss | original uses **L2** on the log-cost label (§4.1.3) | `exp03_run.fit_laet`, `tau=None` |
| Label | `log2(C1)`, C1 = NDC when the GT 1-NN is first distance-evaluated | `core/search.run_query(watch=...)` |
| Features | query vector + 5 distance statistics (Table 5) | `features.laet_tail` |
| Preliminary search | f=1,260 for GIST1M (paper §4.3.2) | `F0` |
| Budget conversion | `2^pred · multiplier`, clipped by `max(f, pred_max)` | `simulate.laet_with_budget` |
| Multiplier | per-model calibration on 500 validation queries; A1 also uses the L0 multiplier for every model | `simulate.calibrate_multiplier` |
| Model | LightGBM, 100 rounds, lr=0.2, 31 leaves | `gbdt.train_lgb` |

Quantile variants use LightGBM `objective="quantile"` with
`alpha=tau`, label unchanged.  In this label space `pred < true` means the
**cost is under-estimated**, so larger tau penalises under-estimation more.

## DARTH

| Item | Finding | Where |
|---|---|---|
| Original loss | LightGBM regression (squared error) — the official code does not override the objective | `fit_darth(tau=None)` |
| Target | current recall@10 at the checkpoint | `trace.collect_darth` |
| Features | the 11 of Table 1 | `features.darth_vec` |
| Label | `hit/k`, online against GT | same |
| `ipi` / `mpi` | `d_bar(R_t)/2` and `d_bar(R_t)/10`, fixed for every loss variant | `exp03_run.main_3a` |
| Adaptive interval | `pi = mpi + (ipi-mpi)(R_t - pred)` | `simulate.darth_adaptive_from_steps` |
| Stop | `pred ≥ R_t`; predictions clipped to [0,1] | same |
| Training observations | every 25 NDC from the start of the bounded ef99 search | `trace.collect_darth` |

Here the direction is reversed: DARTH predicts recall, so **over-estimation**
is dangerous.  A lower tau penalises the upper quantile and therefore
reduces `pred > actual`.

## Loss definitions

Quantile loss:
`L_tau(y, yhat) = tau (y - yhat)` if `yhat < y`, else
`(1 - tau)(yhat - y)`.

* LAET: `y = log2(C1)`, `yhat < y` = cost under-estimation, so
  tau ∈ {0.50, 0.75, 0.90, 0.95} progressively penalise under-budget.
* DARTH: `y = recall`, `yhat > y` = recall over-estimation, so
  tau ∈ {0.50, 0.25, 0.10, 0.05} progressively penalise over-estimation.

The two tau grids are deliberately opposite in direction; they are never
compared to each other.

## Controlled variables

Within each method: same index, same train/val/test query IDs, same features,
same label, same LightGBM structure and non-loss hyper-parameters, same
f/checkpoint (3A), same ipi/mpi (3A), same seed.  The only changed variable
in 3A is the training loss.

In 3B, the changed variable is f (LAET) or (ipi, mpi) (DARTH), with the same
two losses (original and the validation-selected quantile) in every cell.
The quantile model for 3B is selected on **validation** native MAE, never on
test.

## Reproduction status

The same shared traces as Experiments 1–2 are used; no search is rerun for
model variants.  Trace replay was validated by comparing the simulated stop
NDC with an online run on a 30-query timing subset (see
`reproducibility/validation_results.md`).
