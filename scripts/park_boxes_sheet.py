"""One sheet per park for marking its broadcast overlay boxes by eye (the scorebug, clocks).

stabilize_park refuses to run without data/landmarks/<g>.json `overlay_boxes`, because a
scorebug is static on screen while the field moves and it wrecked the v1 frame matcher. The
boxes are per BROADCAST, so one marking covers every game at a park (Sugar Land's box held
across both pilot games). This draws what is needed to read them: two clips from one game at
the park, frames 150 and 205 (the set-up and the pitch, sequential decode), in a 2x2 grid with
a FULL-FRAME pixel grid every 100 px. Mark generously; a box only masks features.

Clips go to data/clips/boxes_<venue_id>/ and are deleted after the sheet is written.

    python scripts/park_boxes_sheet.py                    # every park miss_season says needs boxes
    python scripts/park_boxes_sheet.py --venue-ids 2700,3400
"""
import argparse
import csv as _csv
import os
import shutil
import sys

import cv2
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
from grid_crop import read_frame  # noqa: E402
from src.ingest.video import download  # noqa: E402

FRAMES = (150, 205)


TILE_W = 800


def gridded(fr):
    """Scale to TILE_W, then draw the grid: labels stay legible, numbers stay FULL-FRAME px."""
    h, w = fr.shape[:2]
    s = TILE_W / w
    fr = cv2.resize(fr, (TILE_W, int(round(h * s))))
    for x in range(0, w, 100):
        cv2.line(fr, (int(x * s), 0), (int(x * s), fr.shape[0]), (0, 255, 255), 1)
        cv2.putText(fr, str(x), (int(x * s) + 2, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1)
    for y in range(0, h, 100):
        cv2.line(fr, (0, int(y * s)), (fr.shape[1], int(y * s)), (0, 255, 255), 1)
        cv2.putText(fr, str(y), (2, int(y * s) + 13), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1)
    return fr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--venue-ids", default="")
    a = ap.parse_args()
    season = pd.read_csv(os.path.join(ROOT, "output", "season_pull_2026.csv"))
    reg = set(pd.read_csv(os.path.join(ROOT, "docs", "miss_park_registry.csv")).venue_id)
    want = {int(x) for x in a.venue_ids.split(",") if x.strip()}
    out_dir = os.path.join(ROOT, "output", "miss_season", "boxes")
    os.makedirs(out_dir, exist_ok=True)
    for vid, grp in season.groupby("venue_id"):
        if (want and vid not in want) or (not want and vid in reg):
            continue
        g = next((int(x) for x in grp.sort_values("date").game_pk
                  if os.path.exists(os.path.join(ROOT, "output", "cvpull", "cvpull_%d.csv" % x))), None)
        if g is None:
            print("%d %s: no pull CSV yet" % (vid, grp.venue.iloc[0]))
            continue
        rows = [r for r in _csv.DictReader(open(os.path.join(ROOT, "output", "cvpull",
                                                             "cvpull_%d.csv" % g), encoding="utf-8"))
                if r.get("cf_url")]
        picks = [rows[len(rows) // 4], rows[(3 * len(rows)) // 4]]   # two different innings
        d = os.path.join(ROOT, "data", "clips", "boxes_%d" % vid)
        os.makedirs(d, exist_ok=True)
        tiles = []
        for k, r in enumerate(picks):
            p = os.path.join(d, "c%d.mp4" % k)
            download(r["cf_url"], p)
            for f in FRAMES:
                fr = gridded(read_frame(p, f))
                cv2.putText(fr, "game %d clip %d frame %d" % (g, k, f), (60, fr.shape[0] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                tiles.append(fr)
        shutil.rmtree(d)
        sheet = np.vstack([np.hstack(tiles[0:2]), np.hstack(tiles[2:4])])
        out = os.path.join(out_dir, "boxes_%d.png" % vid)
        cv2.imwrite(out, sheet)
        print("%d %-34s game %d -> %s" % (vid, grp.venue.iloc[0][:34], g, out), flush=True)


if __name__ == "__main__":
    main()
