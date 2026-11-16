"""Parse an hnswlib index file into the CSR cache used by our search loop.

hnswlib save format (hnswalg.h saveIndex/loadIndex):
  13 header fields: 12 x size_t + 1 x double
  cur_element_count * size_data_per_element_ bytes of level-0 blocks
  for each element: uint32 linkListSize, then that many bytes of upper layers
Level 0 block: uint32 count + maxM0 uint32 ids, then the vector at
offsetData_, then the label at label_offset_.  Upper layers are
maxM*4+4-byte link lists, one per layer 1..level.
"""
from __future__ import annotations

import argparse
import json
import os
import struct
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from anns_exp.core.paths import CACHE_DIR  # noqa


def parse(path, out_path):
    t0 = time.time()
    with open(path, "rb") as f:
        # 6 x size_t, int maxlevel, int enterpoint, 3 x size_t, double, size_t
        hdr = struct.unpack("<6Qii3QdQ", f.read(96))
        (offset_level0, max_elements, n, size_elem, label_off, offset_data,
         maxlevel, entry, maxM, maxM0, M, mult, efc) = hdr
        data_start = 96
        print(f"header n={n} size_elem={size_elem} offset_data={offset_data} "
              f"label_off={label_off} maxM={maxM} maxM0={maxM0} "
              f"maxlevel={maxlevel} entry={entry}", flush=True)
        size_links0 = maxM0 * 4 + 4
        size_links_up = maxM * 4 + 4
        assert size_elem >= size_links0 + 960 * 4 + 8, size_elem

        # --- memory-map the file: level-0 blocks are ~4 GB ---
        mm = np.memmap(path, dtype=np.uint8, mode="r")
        level0 = mm[data_start:data_start + n * size_elem]
        assert level0.size == n * size_elem
        counts0 = int(np.frombuffer(level0[:4], dtype="<u4", count=1)[0])
        print(f"first level0 count={counts0}", flush=True)

        # --- upper layers: derive levels while scanning the tail ---
        levels = np.zeros(n, dtype=np.int32)
        upper = []
        pos = data_start + n * size_elem
        for i in range(n):
            L = int(np.frombuffer(mm[pos:pos + 4], dtype="<u4", count=1)[0])
            pos += 4
            if L:
                buf = bytes(mm[pos:pos + L])
                pos += L
                lev = L // size_links_up
                levels[i] = lev
                upper.append((i, buf, lev))
            else:
                upper.append((i, b"", 0))
        print(f"read file in {time.time()-t0:.1f}s; max level={levels.max()} "
              f"levels>0={(levels>0).sum():,}", flush=True)

    stride = maxM0 + int(levels.max()) * maxM
    neighbors = np.full((n, stride), -1, dtype=np.int32)
    lb = memoryview(level0)
    step = size_elem
    for i in range(n):
        base = i * step
        c = int(np.frombuffer(lb[base:base + 4], dtype="<u4", count=1)[0])
        c = min(c, maxM0)
        if c:
            ids = np.frombuffer(lb[base + 4:base + 4 + 4 * c], dtype="<u4")
            neighbors[i, :c] = ids.astype(np.int32)
        if (i + 1) % 200000 == 0:
            print(f"  level0 {i+1}/{n} {time.time()-t0:.0f}s", flush=True)
    for i, buf, lev in upper:
        if not buf:
            continue
        for l in range(1, lev + 1):
            off = (l - 1) * size_links_up
            c = int(struct.unpack("<I", buf[off:off + 4])[0])
            c = min(c, maxM)
            if c:
                ids = np.frombuffer(buf[off + 4:off + 4 + 4 * c], dtype="<u4")
                s = maxM0 + (l - 1) * maxM
                neighbors[i, s:s + c] = ids.astype(np.int32)
    offsets = (np.arange(n + 1, dtype=np.int64) * stride)
    np.savez_compressed(out_path, levels=levels, offsets=offsets,
                        neighbors=neighbors.reshape(-1),
                        entry=np.array([entry], dtype=np.int32),
                        degree0=np.array([maxM0], dtype=np.int32),
                        degree_up=np.array([maxM], dtype=np.int32))
    meta = {"n": n, "dim": 960, "M": M, "maxM0": maxM0, "efConstruction": efc,
            "max_level": int(levels.max()), "stride": stride, "entry": int(entry),
            "levels_hist": {int(k): int(v) for k, v in
                            zip(*np.unique(levels, return_counts=True))},
            "source": os.path.basename(path), "seconds": time.time() - t0}
    with open(out_path.replace("_graph.npz", "_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print("saved", out_path, os.path.getsize(out_path) / 1e6, "MB", flush=True)
    return meta


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--index",
                    default="D:/tmp/anns_wk6/gist_992000_M16_efc200.hnsw")
    ap.add_argument("--out", default=os.path.join(
        CACHE_DIR, "gist_idx992000_n992000_M16_efc200_graph.npz"))
    a = ap.parse_args()
    parse(a.index, a.out)
