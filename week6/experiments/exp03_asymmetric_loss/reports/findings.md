# Experiment 3 — findings

- 3A-LAET L0: hit rate 0.4990, mean NDC 4073.
- 3A-LAET L1: hit rate 0.4730, mean NDC 3622.
- 3A-LAET L2: hit rate 0.4950, mean NDC 4105.
- 3A-LAET L3: hit rate 0.4840, mean NDC 3741.
- 3A-LAET L4: hit rate 0.5040, mean NDC 3985.
- 3A-LAET L0: hit rate 0.4990, mean NDC 4073.
- 3A-LAET L1: hit rate 0.4730, mean NDC 3622.
- 3A-LAET L2: hit rate 0.6070, mean NDC 5609.
- 3A-LAET L3: hit rate 0.7430, mean NDC 8625.
- 3A-LAET L4: hit rate 0.8180, mean NDC 12530.
- 3A-DARTH D0: hit rate 0.5460, mean NDC 4632.
- 3A-DARTH D1: hit rate 0.2610, mean NDC 1543.
- 3A-DARTH D2: hit rate 0.6810, mean NDC 6958.
- 3A-DARTH D3: hit rate 0.8090, mean NDC 11516.
- 3A-DARTH D4: hit rate 0.8760, mean NDC 18094.
- 3A-LAET L1 vs L0 (recall): Δ=-0.0091, 95% CI [-0.0135, -0.0042].
- 3A-LAET L1 vs L0 (stop_ndc): Δ=-450.6812, 95% CI [-568.5748, -335.9085].
- 3A-LAET L2 vs L0 (recall): Δ=-0.0003, 95% CI [-0.0054, +0.0050].
- 3A-LAET L2 vs L0 (stop_ndc): Δ=+31.7914, 95% CI [-94.2523, +161.2479].
- 3A-LAET L3 vs L0 (recall): Δ=-0.0073, 95% CI [-0.0123, -0.0021].
- 3A-LAET L3 vs L0 (stop_ndc): Δ=-332.0566, 95% CI [-468.2963, -198.7504].
- 3A-LAET L4 vs L0 (recall): Δ=+0.0010, 95% CI [-0.0046, +0.0063].
- 3A-LAET L4 vs L0 (stop_ndc): Δ=-88.2644, 95% CI [-225.8840, +57.9445].
- 3A-DARTH D1 vs D0 (recall): Δ=-0.1780, 95% CI [-0.1904, -0.1668].
- 3A-DARTH D1 vs D0 (stop_ndc): Δ=-3089.2890, 95% CI [-3296.7841, -2887.1523].
- 3A-DARTH D2 vs D0 (recall): Δ=+0.0306, 95% CI [+0.0269, +0.0344].
- 3A-DARTH D2 vs D0 (stop_ndc): Δ=+2325.4550, 95% CI [+2185.2350, +2475.0244].
- 3A-DARTH D3 vs D0 (recall): Δ=+0.0512, 95% CI [+0.0462, +0.0563].
- 3A-DARTH D3 vs D0 (stop_ndc): Δ=+6883.3810, 95% CI [+6581.0141, +7204.4883].
- 3A-DARTH D4 vs D0 (recall): Δ=+0.0597, 95% CI [+0.0541, +0.0653].
- 3A-DARTH D4 vs D0 (stop_ndc): Δ=+13461.8450, 95% CI [+12977.9473, +13932.4988].

Verdicts follow the CI signs; a benefit that exists only with a larger effective budget is labelled as such.

## Interpretation

1. **LAET: asymmetric loss changes the estimate but not the recalibrated frontier.** With the L0 multiplier fixed (A1), tau=0.95 raises the hit rate from 0.499 to 0.818 but mean NDC from 4,073 to 12,530. With per-model validation recalibration (A2), the same model is indistinguishable from L0 on recall (Δ=+0.001, CI [-0.0046, +0.0063]) and NDC (Δ=-88, CI [-226, +58]). The A1 gain is a larger effective budget, not a better cost estimate.
2. **DARTH: lower quantiles reduce false stops at a steep cost.** tau=0.25 raises the hit rate 0.546 -> 0.681 (+3.1 pp paired), tau=0.10 to 0.809, tau=0.05 to 0.876, but mean NDC rises 4.6k -> 7.0k / 11.5k / 18.1k and calls 8.5 -> 14 / 24 / 41. tau=0.50 is pathological: a false stop at the first call (hit 0.261, 1.0 call). A median predictor is not a safe stopping rule.
3. **Prediction timing changes the price, not the direction.** LAET's four f values give the same ordering (quantile never beats L2 after recalibration); DARTH P2 (1.5x intervals) gives the best hit rate for both losses, while P1 (0.5x) buys more calls without a quality gain.
4. **Answer to the central question.** On GIST1M, asymmetric loss improves the recall-efficiency trade-off only when it is allowed to move the operating point (fixed multiplier or more frequent calls). After conventional validation calibration it does not beat the original L2 model: the benefit is budget-mediated, not a better predictive mechanism.
