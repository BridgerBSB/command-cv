"""Gate for detect_setup_catch(setup_only=True): same ok + set-up as a full read, fewer frames.

The reference is a set-up cache the FULL read wrote with the current code (output/setups_<g>.csv,
after the 2026-10-11 straight-through decode, aadd50cc0). Each sampled clip is re-read with the
early stop and compared to its cached row:

  SAME        ok and set-up (x, y, frame) identical to the cache
  NOISE       differs, but a fresh FULL read differs from the cache the same way -> GPU
              run-to-run noise, not the early stop
  EARLY-STOP  differs, and a fresh full read matches the cache -> the stop changed the answer
              (a bug; any count above 0 fails the gate)

Reports the share of each clip's frames actually read. Exit 1 on any EARLY-STOP.

    python scripts/check_glove_setup_only.py --dir data/clips/pull_814874 --game 814874 --n 40
"""
import argparse
import os
import random
import sys
import time

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src.auto_glove import detect_setup_catch  # noqa: E402

MITT = "models/od/YOLO/glove_tracking/model_weights/glove_tracking.pt"   # stabilize_park's default


def same(r, row):
    if row is None:
        return not r.get("ok")
    if not r.get("ok"):
        return False
    x, y, f = r["setup"]
    return np.isclose(x, row.x, atol=1e-6) and np.isclose(y, row.y, atol=1e-6) and int(f) == int(row.frame)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--game", type=int, required=True)
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    from ultralytics import YOLO
    mitt = YOLO(os.path.join(ROOT, MITT))
    d = os.path.join(ROOT, a.dir)
    cache = pd.read_csv(os.path.join(ROOT, "output", "setups_%d.csv" % a.game)).set_index("stem")
    stems = sorted(s[:-4] for s in os.listdir(d) if s.endswith(".mp4"))
    assert stems, "no clips in %s" % d
    random.Random(a.seed).shuffle(stems)
    stems = stems[:a.n]
    print("%d clips from %s; cache %d rows (clips absent from it = full read found no set-up)"
          % (len(stems), a.dir, len(cache)))
    tally, frac, t_early = {"SAME": 0, "NOISE": 0, "EARLY-STOP": 0}, [], 0.0
    for s in stems:
        row = cache.loc[s] if s in cache.index else None
        t0 = time.time()
        r = detect_setup_catch(os.path.join(d, s + ".mp4"), mitt, setup_only=True)
        t_early += time.time() - t0
        if r.get("ok"):
            frac.append(r["frames_read"] / r["n_frames"])
        if same(r, row):
            tally["SAME"] += 1
            continue
        full = detect_setup_catch(os.path.join(d, s + ".mp4"), mitt)
        verdict = "NOISE" if not same(full, row) else "EARLY-STOP"
        tally[verdict] += 1
        print("  %-10s %s  cache %s  early %s  full %s" % (
            verdict, s, None if row is None else (round(row.x, 2), round(row.y, 2), int(row.frame)),
            r.get("setup") if r.get("ok") else r.get("reason"),
            full.get("setup") if full.get("ok") else full.get("reason")))
    print("result: %s" % tally)
    if frac:
        print("frames read with the early stop: median %.0f%%, mean %.0f%% of each clip (n %d); "
              "%.1f s/clip wall clock (shared GPU, not a clean timing)"
              % (100 * np.median(frac), 100 * np.mean(frac), len(frac), t_early / len(stems)))
    sys.exit(1 if tally["EARLY-STOP"] else 0)


if __name__ == "__main__":
    main()
