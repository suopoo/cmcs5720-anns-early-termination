# Week 6 data — GIST1M raw + derived caches

GitHub Release: <https://github.com/suopoo/cmcs5720-anns-early-termination/releases/tag/week6-data-v1>

All assets are split into parts smaller than GitHub's 2 GiB per-asset limit.
Reassemble the parts, then verify the SHA256 of the reconstructed file.

## Files and parts

| reconstructed file | size (bytes) | parts |
|---|---:|---|
| `gist.hdf5` | 3,844,648,288 | `gist.hdf5.part00`, `gist.hdf5.part01` |
| `gist_base992000.npy` | 3,809,280,128 | `gist_base992000.npy.part00`, `gist_base992000.npy.part01` |
| `gist_992000_M16_efc200.hnsw` | 3,956,635,136 | `gist_992000_M16_efc200.hnsw.part00`, `gist_992000_M16_efc200.hnsw.part01` |

## Reassemble

Linux / macOS / Git Bash:

```bash
cat gist.hdf5.part00 gist.hdf5.part01 > gist.hdf5
cat gist_base992000.npy.part00 gist_base992000.npy.part01 > gist_base992000.npy
cat gist_992000_M16_efc200.hnsw.part00 gist_992000_M16_efc200.hnsw.part01 > gist_992000_M16_efc200.hnsw
sha256sum -c SHA256SUMS
```

Windows PowerShell:

```powershell
Get-Content gist.hdf5.part00,gist.hdf5.part01 -Raw | Set-Content gist.hdf5 -Encoding Byte
Get-FileHash gist.hdf5 -Algorithm SHA256
```

## Placement

| reconstructed file | target path |
|---|---|
| `gist.hdf5` | `D:\HNSW_Project\datasets\gist\gist.hdf5` (or set `ANNS_DATA`) |
| `gist_base992000.npy` | `week6/experiments/shared/cache/gist_base992000.npy` |
| `gist_992000_M16_efc200.hnsw` | `week6/experiments/shared/cache/gist_992000_M16_efc200.hnsw` |

The parsed graph `shared/cache/gist_idx992000_n992000_M16_efc200_graph.npz` (45 MB),
the search traces (`shared/traces/`, 148 MB), and the exact ground truth
(`shared/ground_truth/`) are already committed in the repository and are not
duplicated in this release.

## SHA256

```
8e95831936bfdbfa0a56086942e2cf98cd703517c67f985914183eb4cdbf026a  gist.hdf5
3b845fed8340ee266a98a3db31dce6286144dfb1282a323f7bbcd432537e1bd4  gist_base992000.npy
9ac9948e83a746fd56ee476c630fa6e9d7c88a567905e4c156c2c6952749d5ec  gist_992000_M16_efc200.hnsw
```

## Rebuild instead of downloading

`gist.hdf5` is the public GIST1M dataset from ANN Benchmarks. The two derived
caches can be rebuilt from it:

```bash
python -I prepare_gist.py --stage gt
python -I build_hnswlib.py
python -I parse_hnswlib.py
```

See `week6/experiments/README.md` for the full reproduction commands.
