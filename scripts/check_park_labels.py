"""The NUMBERS behind a park's label admission - print them beside the two eye-check images.

The eye check (park_pose_<g>_ref.png + proj_labels_<g>_sheet.png) decides; these say where to
look. Each was a failure found by eye on 2026-10-11 and measured afterwards (PROGRESS-LOG 116):

  confirmed %          share of projected flight frames with a v4 candidate within the match
                       radius (85-94% at the 7 admitted new parks).
  per-clip offset      median label-to-nearest-detection distance per clip, over frames with a
                       detection within 50 px. A whole clip shifted (Equity Bank 816989: 2 of
                       57 clips over 4 px) is a bad homography for that clip, not a bad camera.
  misplacement         UNCONFIRMED labels with a detection 40 px or closer: the ball is visible
                       BESIDE the label (Equity Bank 0009 / 0011, 15-30 px). Hidden balls have no
                       detection nearby; misplaced ones do. 0-4% at admitted parks.
  wide-zoom clips      clips mapped at zoom >= 1.3 (a second, wider camera framing - Wilson
                       826785 had 16); their confirmed labels should sit as close to detections
                       as the rest.

    python scripts/check_park_labels.py --game 816989
"""
import argparse
import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "output")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", type=int, required=True)
    ap.add_argument("--stab-tag", default="c")
    a = ap.parse_args()
    g = a.game
    L = pd.read_csv(os.path.join(OUT, "proj_labels_%d.csv" % g))
    c = pd.read_csv(os.path.join(OUT, "stab2_clips_%d%s.csv" % (g, a.stab_tag)))
    assert len(L) and len(c), "no labels or no stab2 clip table for %d" % g

    print("game %d: %d clips stabilized, %d without a usable H, %d labelled clips, %d labels"
          % (g, len(c), int((c.n_h_good == 0).sum()), L.stem.nunique(), len(L)))
    print("  confirmed: %.1f%%" % (100 * L.confirmed.mean()))
    near = L[L.det_dist <= 50]
    pc = near.groupby("stem").det_dist.median()
    print("  per-clip label-to-detection median: %.2f px (p90 %.2f); clips over 4 px: %d -> %d labels"
          % (pc.median(), pc.quantile(.9), int((pc > 4).sum()), int(L.stem.isin(pc[pc > 4].index).sum())))
    if (pc > 4).any():
        print("    " + ", ".join("%s %.1f" % (s, v) for s, v in pc[pc > 4].sort_values().items()))
    u = L[~L.confirmed]
    mis = u[u.det_dist <= 40]
    print("  unconfirmed: %d; misplacement signature (detection <= 40 px): %d (%.0f%%)"
          % (len(u), len(mis), 100 * len(mis) / max(len(u), 1)))
    wide = set(c[c.zoom_med >= 1.3].stem)
    if wide:
        lw, ln = L[L.stem.isin(wide) & L.confirmed], L[~L.stem.isin(wide) & L.confirmed]
        print("  wide-zoom clips (>= 1.3): %d; confirmed label-to-detection median %.2f px vs %.2f elsewhere"
              % (len(wide), lw.det_dist.median() if len(lw) else np.nan, ln.det_dist.median()))


if __name__ == "__main__":
    main()
