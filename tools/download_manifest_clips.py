"""Download every clip a manifest lists (cf_url -> <dir>/<stem>.mp4). No auth needed.

sporty-clips URLs need no token and do not expire (rolling_pull.py docstring). Resumable:
a clip already on disk is skipped; each file lands as .part then is renamed, so a kill never
leaves a truncated mp4 that later reads as a short clip.

    python tools/download_manifest_clips.py data/clips/pull_821848
"""
import os
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.ingest.video import _cdn_headers  # noqa: E402


def get(url, out):
    if os.path.exists(out):
        return "skip"
    # _cdn_headers is what pull_game_clips.py uses. Probed 2026-09-27: a bare
    # "Mozilla/5.0" UA now gets 403 on sporty-clips while these headers (and no headers
    # at all) get 200 -- the CDN changed since the Sep 18 "any UA works" probe.
    req = urllib.request.Request(url, headers=_cdn_headers())
    with urllib.request.urlopen(req, timeout=60) as r, open(out + ".part", "wb") as f:
        f.write(r.read())
    os.replace(out + ".part", out)
    return "ok"


def main():
    d = sys.argv[1]
    assert os.path.isdir(d), d
    m = pd.read_csv(os.path.join(d, os.path.basename(d.rstrip("/\\")) + "_manifest.csv"))
    jobs = [(u, os.path.join(d, s + ".mp4")) for s, u in zip(m.stem, m.cf_url) if isinstance(u, str)]
    res = {"ok": 0, "skip": 0, "fail": 0}
    with ThreadPoolExecutor(6) as ex:
        for f in [ex.submit(get, u, o) for u, o in jobs]:
            try:
                res[f.result()] += 1
            except Exception as e:  # noqa: BLE001 - count and report, never silently drop
                res["fail"] += 1
                print("  fail:", e)
    print("%s: %d listed -> %s" % (d, len(jobs), res))


if __name__ == "__main__":
    main()
