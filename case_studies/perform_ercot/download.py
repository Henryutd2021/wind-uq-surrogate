"""Download the PERFORM files used by the ERCOT case study (public S3 bucket, anonymous HTTPS; standard library).

Files (ERCOT, 2018, all 264 wind sites; 793 files, about 2.7 GB):

- ``ERCOT/MetaData/wind_meta.xlsx``
- ``ERCOT/2018/Wind/Actuals/Site_level/*``                              simulated plant power, 5 min
- ``ERCOT/2018/ECMWF/Wind_51MemberForecasts/Day_ahead/Site_level/*``    51 ensemble members, day ahead
- ``ERCOT/2018/ECMWF/Wind_51MemberForecasts/2Day_ahead/Site_level/*``   51 ensemble members, two days ahead

Files are written to ``RAW_DIR/<key>``; existing files with the expected size are skipped. A manifest
(key, bytes, SHA-256, URL, access date) is written to ``RAW_DIR/manifest.csv``; compare it with
``data_manifest.csv`` in this folder to check that the files are the ones used for the published results.

Usage:  python download.py RAW_DIR [--workers 8]
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import html
import re
import sys
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

BUCKET = "https://arpa-e-perform.s3.amazonaws.com"
PREFIXES = (
    "ERCOT/MetaData/wind_meta.xlsx",
    "ERCOT/2018/Wind/Actuals/Site_level/",
    "ERCOT/2018/ECMWF/Wind_51MemberForecasts/Day_ahead/Site_level/",
    "ERCOT/2018/ECMWF/Wind_51MemberForecasts/2Day_ahead/Site_level/",
)


def list_keys(prefix: str) -> list[tuple[str, int]]:
    """All (key, size) pairs under a prefix (S3 ListObjectsV2, paged)."""
    keys, token = [], None
    while True:
        url = f"{BUCKET}/?list-type=2&prefix={urllib.parse.quote(prefix)}"
        if token:
            url += "&continuation-token=" + urllib.parse.quote(token)
        text = urllib.request.urlopen(url, timeout=60).read().decode()
        keys += [(html.unescape(k), int(s))  # keys are XML-escaped (e.g. &apos;)
                 for k, s in re.findall(r"<Key>([^<]*)</Key>.*?<Size>(\d+)</Size>", text, re.S)]
        m = re.search(r"<NextContinuationToken>([^<]*)</NextContinuationToken>", text)
        if not m:
            return keys
        token = m.group(1)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(key: str, size: int, root: Path, tries: int = 4) -> dict:
    dest = root / key
    url = f"{BUCKET}/{urllib.parse.quote(key)}"
    if not (dest.exists() and dest.stat().st_size == size):
        dest.parent.mkdir(parents=True, exist_ok=True)
        for attempt in range(tries):
            try:
                tmp = dest.with_suffix(dest.suffix + ".part")
                with urllib.request.urlopen(url, timeout=120) as r, open(tmp, "wb") as f:
                    while chunk := r.read(1 << 20):
                        f.write(chunk)
                if tmp.stat().st_size != size:
                    raise OSError(f"size {tmp.stat().st_size} != {size}")
                tmp.replace(dest)
                break
            except OSError as exc:
                if attempt == tries - 1:
                    raise RuntimeError(f"download failed: {key}: {exc}") from exc
                time.sleep(2 ** attempt)
    return {"key": key, "bytes": size, "sha256": sha256(dest), "url": url,
            "accessed": dt.date.today().isoformat()}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("raw_dir")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    root = Path(args.raw_dir)
    root.mkdir(parents=True, exist_ok=True)
    keys = []
    for prefix in PREFIXES:
        found = [(k, s) for k, s in list_keys(prefix) if k.endswith((".h5", ".xlsx"))]
        print(f"{prefix}: {len(found)} files, {sum(s for _, s in found) / 1e9:.2f} GB", flush=True)
        keys += found
    rows = []
    with ThreadPoolExecutor(args.workers) as pool:
        for n, row in enumerate(pool.map(lambda ks: fetch(*ks, root), keys), 1):
            rows.append(row)
            if n % 100 == 0 or n == len(keys):
                print(f"  {n}/{len(keys)} files", flush=True)
    rows.sort(key=lambda r: r["key"])
    with open(root / "manifest.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"manifest: {root / 'manifest.csv'} ({len(rows)} files, {sum(r['bytes'] for r in rows) / 1e9:.2f} GB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
