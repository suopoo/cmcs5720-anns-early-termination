# Week 6 — three experiments on learned early termination

All code, data, traces, models, figures and reports for:

1. Experiment 1 — Feature Transfer Ablation (LAET / DARTH / ANNiE features)
2. Experiment 2 — Per-query Error Analysis of LAET / DARTH / ANNiE
3. Experiment 3 — Asymmetric Loss and Prediction Timing Ablation

Start with [`experiments/FINAL_SUMMARY.md`](experiments/FINAL_SUMMARY.md)
and [`experiments/README.md`](experiments/README.md).

## Large artefacts

`experiments/shared/cache/` is intentionally **not** committed:

* `gist_base992000.npy` — 3.6 GB memory-mapped GIST base (992,000 × 960)
* `gist_992000_M16_efc200.hnsw` — 3.9 GB raw hnswlib index
* `gist_idx992000_n992000_M16_efc200_graph.npz` — 44 MB parsed CSR graph

GitHub rejects files over 100 MB and the free LFS quota is far below 7.5 GB.
The 44 MB parsed CSR graph and the exact ground truth are small enough to be
included if desired; the rebuild commands are in
`experiments/common/prepare_gist.py`, `build_hnswlib.py` and
`parse_hnswlib.py`.  The dataset itself is GIST1M from ANN Benchmarks.
