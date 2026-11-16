# Graph / search validation

- Index: hnswlib M=16 efc=200 on GIST1M `train[:992000]`
- Parsed CSR: degree0=32, max_level=4, entry=18691
- Queries checked: 50 official test queries, efSearch=100, k=10
- Mean recall@10 vs exact GT: **0.8400**
- Mean ID overlap with hnswlib's own search: **0.9880**
- Mean NDC per query (our loop): 1966.5
- Wall time: 0.7s

The overlap is computed as |our top-10 ∩ hnswlib top-10| / 10.  A correctly parsed adjacency gives ~1.0; tie-breaking can account for tiny differences.
