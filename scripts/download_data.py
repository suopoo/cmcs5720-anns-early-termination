#!/usr/bin/env python3
"""Download the ann-benchmarks datasets used by E1.

Datasets are NOT stored in this repository (they are 0.5-4 GB each).

Usage:
    python scripts/download_data.py                 # sift + deep (E1 default)
    python scripts/download_data.py --datasets sift deep glove
    python scripts/download_data.py --out D:/data   # custom location
    python scripts/download_data.py --check         # verify without downloading

Supports HTTP range resume, so an interrupted download continues where it
stopped.  Files land in ./data by default; point the code at another folder
with the ANNS_DATA environment variable.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import urllib.request

BASE = "https://ann-benchmarks.com/{}.hdf5"
FILES = {
    "sift": "sift-128-euclidean",
    "glove": "glove-100-angular",
    "gist": "gist-960-euclidean",
    "fashion": "fashion-mnist-784-euclidean",
    "deep": "deep-image-96-angular",
}


def remote_size(url):
    req = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(req, timeout=60) as r:
        return int(r.headers["Content-Length"]), r.headers.get("Accept-Ranges", "")


def download(name, out_dir):
    fname = FILES[name] + ".hdf5"
    url = BASE.format(FILES[name])
    path = os.path.join(out_dir, fname)
    total, ranges = remote_size(url)
    have = os.path.getsize(path) if os.path.exists(path) else 0

    if have == total:
        print(f"[skip] {fname} already complete ({total/1e9:.2f} GB)")
        return True
    if have > total:
        print(f"[warn] {fname} is larger than the remote file; restarting")
        have = 0
        open(path, "wb").close()

    mode = "ab" if (have and "bytes" in ranges.lower()) else "wb"
    if mode == "wb":
        have = 0
    print(f"[get ] {fname}: {have/1e9:.2f}/{total/1e9:.2f} GB"
          f"{' (resuming)' if have else ''}")

    req = urllib.request.Request(url)
    if have:
        req.add_header("Range", f"bytes={have}-")
    t0, last = time.time(), have
    with urllib.request.urlopen(req, timeout=120) as r, open(path, mode) as f:
        got = have
        while True:
            chunk = r.read(8 << 20)
            if not chunk:
                break
            f.write(chunk)
            got += len(chunk)
            if got - last >= 256 << 20:
                rate = (got - have) / max(time.time() - t0, 1e-9) / 1e6
                print(f"       {got/1e9:.2f}/{total/1e9:.2f} GB "
                      f"({rate:.1f} MB/s)", flush=True)
                last = got
    final = os.path.getsize(path)
    if final != total:
        print(f"[FAIL] {fname}: got {final} of {total} bytes")
        return False
    print(f"[ ok ] {fname} ({final/1e9:.2f} GB)")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["sift", "deep"],
                    choices=sorted(FILES))
    ap.add_argument("--out", default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"))
    ap.add_argument("--check", action="store_true",
                    help="only verify sizes, do not download")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    ok = True
    for name in args.datasets:
        if args.check:
            fname = FILES[name] + ".hdf5"
            path = os.path.join(args.out, fname)
            total, _ = remote_size(BASE.format(FILES[name]))
            have = os.path.getsize(path) if os.path.exists(path) else 0
            status = "OK" if have == total else "INCOMPLETE"
            print(f"{status:10} {fname}: {have:,} / {total:,} bytes")
            ok &= have == total
        else:
            ok &= download(name, args.out)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
