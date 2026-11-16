# Experiment 2 — findings (generated from the result tables)

- **LAET**: mean recall 0.9562, target hit rate 0.6760, mean NDC 12150, P95 NDC 28754; oracle-attainable 0.9800.
- **DARTH**: mean recall 0.9608, target hit rate 0.6440, mean NDC 12691, P95 NDC 27549; oracle-attainable 0.9260.
- **ANNiE-S**: mean recall 0.9736, target hit rate 0.8380, mean NDC 15996, P95 NDC 34560; oracle-attainable 0.9800.
- Quality-failure overlap LAET∩DARTH: 225 queries, Jaccard 0.495, P(DARTH|LAET)=0.694, stratified permutation p=0.0005.
- Quality-failure overlap LAET∩ANNiE-S: 122 queries, Jaccard 0.335, P(ANNiE-S|LAET)=0.377, stratified permutation p=0.0005.
- Quality-failure overlap DARTH∩ANNiE-S: 149 queries, Jaccard 0.404, P(ANNiE-S|DARTH)=0.419, stratified permutation p=0.0005.
- ANNiE-S target hit rate from n=500 to n=2000: 0.7690 → 0.8380 (3-seed mean).
- DARTH target hit rate from n=500 to n=2000: 0.6327 → 0.6533 (3-seed mean).
- LAET target hit rate from n=500 to n=2000: 0.6510 → 0.6760 (3-seed mean).
- LAET R_t sensitivity (0.90/0.95/0.99): hit rate 0.604 / 0.676 / 0.687.
- DARTH R_t sensitivity (0.90/0.95/0.99): hit rate 0.703 / 0.644 / 0.584.
- ANNiE-S R_t sensitivity (0.90/0.95/0.99): hit rate 0.854 / 0.838 / 0.703.

These are observations, not causal claims; see `experiment_report.md` for the limitations.

## Interpretation

1. **Different failure profiles, not one shared failure.** ANNiE-S reaches the target for 83.8% of queries at 16.0k mean NDC; DARTH reaches 64.4% at 12.7k; LAET 67.6% at 12.1k. The cheapest method is not the most reliable one.
2. **Quality failures overlap far above independence, and the overlap survives oracle-cost stratification.** All three pairwise Jaccards are 0.34-0.49 with stratified permutation p=0.0005; 118 queries fail in all three methods. This is consistent with a query-intrinsic difficulty component, but not proof that the models share a defect: the conditional probabilities differ (P(DARTH|LAET)=0.69 versus P(ANNiE|LAET)=0.38).
3. **Asymmetric consequences are visible in the stopping decisions.** Among failed queries, 77% of LAET failures have a budget under-estimate (mean -5,610 NDC), 85% of DARTH failures have a recall over-estimate at the stop (mean +0.026), while ANNiE failures are split (49% cost under-estimate). DARTH's oracle-attainable rate on its own ef=2048 trajectory is 0.926. A positive signed error therefore means the opposite thing for the two predictor families; ANNiE's conservative quantile over-spends on average but still misses the target on a hard subset.
4. **More training helps ANNiE, barely moves DARTH, and is non-monotone for LAET.** ANNiE hit rate rises 0.769 -> 0.838 from n=500 to n=2,000; DARTH is flat at 0.63-0.66; LAET is noisy (0.65, 0.57, 0.68).
5. **Higher targets hurt DARTH.** From R_t=0.90 to 0.99, ANNiE's hit rate falls 0.854 -> 0.703 and LAET rises 0.604 -> 0.687, while DARTH falls 0.703 -> 0.584; DARTH's adaptive interval becomes the limiting factor at the strict target.
6. **Hard-query-enriched training helps all three methods but costs NDC.** At n=1,000, D2 raises hit rate by 2.8 pp (LAET), 7.0 pp (DARTH) and 6.0 pp (ANNiE), with 9-16% more mean NDC. The gain is not free.
