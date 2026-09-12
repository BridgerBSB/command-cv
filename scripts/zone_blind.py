"""The BLIND zone render: draw the zone from a camera that never saw the box.

zone_zoom.py showed our projected corners within 0.1-0.3 px of OpenCommand's
observed box - but it used the pose from parity_opencommand_824821.csv, the
BOX-ASSISTED solve, which was fitted with those four corners at weight 3. Agreeing
with an input you were handed is consistency, not confirmation. That caveat is
written into docs/2026-09-12-zone-graphic-is-in-our-clips.md and this script is the
thing that removes it.

Here the pose comes from ablate_no_box.fit_joint: ONE shared camera position fitted
across every clip at once, from ball pixels ALONE, each clip keeping its own
pan/tilt/roll/f/t0. No box anywhere in the fit. Then we project the zone and compare
it to the box the broadcast drew - a target the solve has never been shown.

Also reports two things zone_zoom could not:

  SKEW. Our zone is a true projected QUADRILATERAL; OpenCommand's stored box is four
  scalars (x_left, y_top, x_right, y_bottom), so it is axis-aligned by construction
  and its format cannot hold perspective. Printing all four corners shows how much
  is actually being discarded.

  INCHES. A 17 in zone spans about 57 px here, so 1 px is about 0.3 in. Pixel error
  is reported in inches at the zone plane as well, because that is the unit every
  other number in this project is quoted in.

    python scripts/zone_blind.py [--game 824821] [--limit 60] [--n 3]
"""
import argparse
import os
import sys

import cv2
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from src import pinhole as ph  # noqa: E402
import parity_opencommand_pose as par  # noqa: E402
import ablate_no_box as ab  # noqa: E402

SEASON = 2026
PAD, ZOOM = 110, 4


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", type=int, default=824821)
    ap.add_argument("--limit", type=int, default=60)
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--out", default=os.path.join(ROOT, "output", "zone_blind.png"))
    a = ap.parse_args()

    clips_dir = os.path.join(ROOT, "data", "clips", f"mlb_{a.game}")
    assert os.path.isdir(clips_dir), clips_dir

    # --- rebuild exactly what ablate_no_box.main does up to the joint fit ---
    clips = par.load_clips(a.game)[: a.limit]
    prep, votes = [], []
    for c in clips:
        runs = par.candidate_runs(c)
        if not runs or len(runs[0]) < ab.MIN_FIT_PTS + ab.HOLDOUT_FR:
            continue
        seg = runs[0]
        fit_seg, held = seg.iloc[:-ab.HOLDOUT_FR], seg.iloc[-ab.HOLDOUT_FR:]
        x_box = ab._fit(c, fit_seg, True)
        prep.append((c, fit_seg, held, x_box[6]))
        votes.append(x_box[:2])
    assert prep, "no clips had a long enough ball run"
    med = np.median(np.array(votes), axis=0)
    print(f"game {a.game}: {len(prep)} clips into the joint solve")
    print(f"joint fit: {2 + 5*len(prep)} params over "
          f"{sum(2*len(s) for _, s, _, _ in prep)} residuals, ball pixels only ...", flush=True)
    xj = ab.fit_joint(prep, med[0], med[1])
    C_joint = np.array([xj[0], ph.CY_FT, xj[1]])
    print(f"BOX-ASSISTED game position : Cx {med[0]:8.3f}  Cz {med[1]:8.3f}")
    print(f"BLIND joint  position      : Cx {C_joint[0]:8.3f}  Cz {C_joint[2]:8.3f}"
          f"   (off {abs(C_joint[0]-med[0]):.2f} ft, {abs(C_joint[2]-med[1]):.2f} ft)\n")

    zy = ph.zone_y(SEASON)
    labels = pd.read_csv(os.path.join(ROOT, "output", f"ball_labels_{a.game}.csv"))

    panels, rows = [], []
    for k, (c, fit_seg, held, t0) in enumerate(prep):
        if len(panels) >= a.n:
            break
        pid = c["play_id"]
        clip = os.path.join(clips_dir, f"{pid}_BROADCAST.mp4")
        lab = labels[labels.play_id == pid]
        if not os.path.exists(clip) or lab.empty:
            continue
        p = c["p"]
        top, bot = float(p.sz_top_ref), float(p.sz_bot_ref)
        corners = np.array([[-17 / 24, zy, top], [17 / 24, zy, top],
                            [17 / 24, zy, bot], [-17 / 24, zy, bot]])
        pan, tilt, roll, f = xj[2 + 5 * k: 6 + 5 * k]
        q = ph.project(C_joint, pan, tilt, roll, f, corners)       # never saw the box

        b = c["box_obs"]                                           # what the broadcast drew
        inx = 17.0 / (b[:, 0].max() - b[:, 0].min())               # in per px, from the 17in zone
        d = np.array([abs(q[:, 0].min() - b[:, 0].min()), abs(q[:, 1].min() - b[:, 1].min()),
                      abs(q[:, 0].max() - b[:, 0].max()), abs(q[:, 1].max() - b[:, 1].max())])
        rows.append(dict(play=pid[:8], L=d[0], T=d[1], R=d[2], B=d[3],
                         in_per_px=inx, max_in=d.max() * inx))
        # how non-rectangular is the projected quad?
        skew_x = abs((q[0, 0] - q[3, 0])) + abs((q[1, 0] - q[2, 0]))
        skew_y = abs((q[0, 1] - q[1, 1])) + abs((q[2, 1] - q[3, 1]))
        print(f"{pid[:8]}  blind-vs-drawn px L{d[0]:5.2f} T{d[1]:5.2f} R{d[2]:5.2f} B{d[3]:5.2f}"
              f"   max {d.max()*inx:5.3f} in   ({inx:.3f} in/px)")
        print(f"          quad corners {np.round(q,1).tolist()}  skew x{skew_x:.2f} y{skew_y:.2f} px")

        fi = int(lab.frame_idx.max())
        cap = cv2.VideoCapture(clip)
        frame, i = None, 0
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            if i == fi:
                frame = fr
                break
            i += 1
        cap.release()
        if frame is None:
            continue

        cx, cy = int(q[:, 0].mean()), int(q[:, 1].mean())
        x0 = max(0, min(cx - PAD, ph.W - 2 * PAD))
        y0 = max(0, min(cy - PAD, ph.H - 2 * PAD))

        def crop():
            cr = frame[y0:y0 + 2 * PAD, x0:x0 + 2 * PAD]
            return cv2.resize(cr, (2 * PAD * ZOOM, 2 * PAD * ZOOM), interpolation=cv2.INTER_NEAREST)

        tiles = []
        for tag, draw in (("1 RAW - the box MLB drew", False), ("2 + BLIND zone (ball-only camera)", True)):
            im = crop()
            if draw:
                cv2.polylines(im, [((q - [x0, y0]) * ZOOM).round().astype(int)], True,
                              (0, 200, 255), 3, cv2.LINE_AA)
            cv2.rectangle(im, (0, 0), (im.shape[1], 30), (0, 0, 0), -1)
            cv2.putText(im, f"{tag}  {pid[:8]}", (8, 21), 0, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
            tiles.append(im)
        panels.append(np.hstack(tiles))

    df = pd.DataFrame(rows)
    print(f"\nBLIND camera vs the box the broadcast drew, {len(df)} clips shown:")
    print(f"  max edge error: median {df.max_in.median():.3f} in, worst {df.max_in.max():.3f} in")
    print("  (the joint solve never saw a box; this is a held-out object)")
    if panels:
        cv2.imwrite(a.out, np.vstack(panels))
        print("wrote", a.out)


if __name__ == "__main__":
    main()
