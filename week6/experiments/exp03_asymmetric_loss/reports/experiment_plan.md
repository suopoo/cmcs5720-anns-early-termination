# Experiment 3 — executed plan

1. **Audit** the official LAET / DARTH code and the ANNiE quantile idea
   (`source_audit.md`).  Confirmed: LAET uses L2 on log2(C1); DARTH uses
   LightGBM's default L2 on current recall; ANNiE uses quantile loss.
2. **3A-LAET**: train L0 (L2) + L1/L2/L3/L4 (quantile tau=0.50/0.75/0.90/0.95)
   with identical features (LAET original), label log2(C1), f=1,260,
   5,000/500/1,000 split.  Evaluate with
   (A1) the L0 multiplier fixed for all models;
   (A2) each model's multiplier recalibrated on validation to mean recall 0.90.
3. **3A-DARTH**: train D0 (L2) + D1-D4 (tau=0.50/0.25/0.10/0.05) with the
   11 original features, current-recall label, identical ipi=1533 / mpi=307.
   Run the original adaptive interval; do not force equal call counts.
   Complement with an offline comparison at identical checkpoints.
4. **Select 3B quantile on validation native MAE** (never on test):
   LAET L1 (tau=0.50), DARTH D2 (tau=0.25).
5. **3B-LAET**: f in {630, 945, 1260, 1575} × {L2, selected quantile}, each
   with its own features and validation multiplier.
6. **3B-DARTH**: P0 (ipi,mpi), P1 (0.5x,0.5x), P2 (1.5x,1.5x),
   P3 (ipi,0.5mpi) × {L2, selected quantile}; 1<=mpi<=ipi enforced.
7. Query-level paired bootstrap for every comparison; report CIs and
   Cliff's delta.  A recall gain paid for by a much larger NDC is reported
   as a trade-off, not as an overall improvement.
