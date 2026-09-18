"""Turn a park-wide harvest into MEASURED labels by snapping each point onto real ball pixels.

THE TWO HALVES THIS JOINS. Neither is enough alone, and that is the whole reason for this
step:
  aplus_seed_tracks / onboard_park  find ACCURATE pixels, but only on the clips where the
        ball is separable enough to hold a run of top-1 frames -- 41 of 50 at Isotopes,
        13 of 50 at L.P. Frans.
  harvest_park_labels               finds the RIGHT FRAMES on every clip (50 of 50 at
        Isotopes, plate frames clustered at f215-218 exactly where release + flight time
        says they must be) but its POSITIONS come from projecting a pose, and against
        hand-verified tracks those sit ~8.6 px out -- about one ball width, on a ball whose
        training box is 16 px.
So: take the harvest's frames, look for a ball-shaped blob near its projected point, and
keep the BLOB's position. The label becomes a measurement again.

WHY NOT refine_park_pose, WHICH ALREADY DOES SOMETHING LIKE THIS. Its
ball_pixels_near_path carries an older copy of the differencing that compares each frame to
the IMMEDIATELY PREVIOUS one. That is the exact thing cadence-adaptive differencing replaced
on 09-17, because a feed carrying 30 fps content in a 60 fps container has an empty absdiff
on every other frame. The fix landed in onboard_park and its sibling copy never got it, so
this file imports `onboard_park.diff_candidates` rather than making a third copy.

WHAT THE SNAP DISTANCE TELLS YOU, and read it every run. It is the gap between where the
pose said the ball would be and where it actually was. A median near the pose's known error
is healthy. A median near the GATE means the gate is choosing the labels rather than the
image is, and the run is quietly fabricating a path -- exactly the ICP-drift failure
refine_park_pose warns about. That is printed, not hidden.

NO FABRICATION: a harvested frame with no candidate inside the gate is DROPPED, never
back-filled with the projection. Mixing measured and projected points under one column would
make the label set's accuracy unknowable.

    python scripts/snap_harvest_labels.py --game 815419 --dir data/clips/aaa_815419
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from src import pinhole as ph  # noqa: E402
import fit_park_pose as fpp  # noqa: E402
import onboard_park as op  # noqa: E402

FPS = 59.69


def snap_clip(path, lane, frames, path_xy, gate):
    """Nearest strong ball-shaped blob to each projected point, then half-frame corrected.

    `frames` are sequential decode indices, matching every other index in this project.
    Candidates come from the shared cadence-adaptive differencing, so a duplicated-frame feed
    contributes on the frames that actually carry new information instead of going empty.
    """
    want = {int(f): p for f, p in zip(frames, path_xy)}
    lo, hi = min(want), max(want)
    cands, dup = op.diff_candidates(path, lane, lo, hi)
    out = []
    for fr, dgap, cs in cands:
        if fr not in want:
            continue
        px, py = want[fr]
        best = None
        for resp, cx, cy in cs:                     # cs is strongest-first
            dist = float(np.hypot(cx - px, cy - py))
            if dist <= gate and (best is None or resp > best[0]):
                best = (resp, cx, cy, dist)
        if best:
            out.append((fr, best[1], best[2], dgap, best[3]))
    if len(out) < 2:
        return [], dup
    # half_frame_correct wants (f, x, y, dgap); carry the snap distance alongside
    corrected = op.half_frame_correct([(f, x, y, g) for f, x, y, g, _ in out])
    return [(f, x, y, g, out[k][4]) for k, (f, x, y, g) in enumerate(corrected)], dup


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", type=int, required=True)
    ap.add_argument("--dir", required=True)
    ap.add_argument("--pose", default=None)
    ap.add_argument("--harvest", default=None)
    ap.add_argument("--lane", default="400,1000,230,470")
    ap.add_argument("--gate", type=float, default=22.0,
                    help="px from the projected point; generous relative to the pose error "
                         "so the true ball is inside it and the fit is not the thing "
                         "choosing the label")
    ap.add_argument("--min-pts", type=int, default=4, help="drop a clip with fewer snaps")
    a = ap.parse_args()

    d = a.dir if os.path.isabs(a.dir) else os.path.join(ROOT, a.dir)
    lane = tuple(int(v) for v in a.lane.split(","))
    pose = a.pose or os.path.join(ROOT, "output", f"park_pose_{a.game}.npz")
    harv = a.harvest or os.path.join(ROOT, "output", f"harvest_{a.game}.csv")
    assert os.path.exists(pose), f"no pose at {pose}"
    assert os.path.exists(harv), (
        f"no harvest at {harv} -- run harvest_park_labels.py --game {a.game} "
        f"--dir {a.dir} --pose {pose} first")

    z = np.load(pose)
    C, (pan, tilt, roll), f = z["C"], z["ang"], float(z["f"])
    H = pd.read_csv(harv)
    man = pd.read_csv(os.path.join(d, os.path.basename(d) + "_manifest.csv")).set_index("stem")

    rows, snapped_clips, dropped, dists, dups = [], 0, [], [], []
    stems = list(H.stem.unique())
    for j, stem in enumerate(stems, 1):
        g = H[H.stem == stem].sort_values("frame")
        clip = os.path.join(d, stem + ".mp4")
        if not os.path.exists(clip) or stem not in man.index:
            continue
        row = man.loc[stem]
        if any(pd.isna(row[k]) for k in fpp.TRAJ):
            continue
        pts, dup = snap_clip(clip, lane, g.frame.to_numpy(), g[["x", "y"]].to_numpy(), a.gate)
        dups.append(dup)
        if len(pts) < a.min_pts:
            dropped.append((stem, len(pts)))
            continue
        snapped_clips += 1
        # the diameter comes from the pose at the trajectory point for that frame, refitting
        # t0 alone against the SNAPPED pixels so the clock matches the measurement
        traj = fpp.traj_fn(row)
        fr = np.array([p[0] for p in pts], float)
        obs = np.array([[p[1], p[2]] for p in pts], float)
        from scipy.optimize import least_squares
        t0 = float(row["zone_time"]) - fr.max() / FPS

        def res(zz):
            return (ph.project(C, pan, tilt, roll, f, traj(zz[0] + fr / FPS)) - obs).ravel()
        t0 = float(least_squares(res, [t0], bounds=([t0 - 0.30], [t0 + 0.30])).x[0])
        dia = fpp.ball_diameter_px(C, pan, tilt, roll, f, traj(t0 + fr / FPS))
        for k, p in enumerate(pts):
            dists.append(p[4])
            rows.append(dict(stem=stem, game_pk=a.game, frame=int(p[0]),
                             x=float(p[1]), y=float(p[2]), diameter_px=float(dia[k])))
        if j % 10 == 0:
            print(f"  [{j}/{len(stems)}] {snapped_clips} clips snapped, {len(rows)} points",
                  flush=True)

    assert rows, "nothing snapped -- check --lane and --gate against the harvest positions"
    out = pd.DataFrame(rows)
    lp = os.path.join(ROOT, "output", f"snap_labels_{a.game}.csv")
    out.to_csv(lp, index=False)

    dists = np.array(dists)
    print(f"\ngame {a.game}: {snapped_clips}/{len(stems)} clips, {len(out)} MEASURED points "
          f"(harvest offered {len(H)})")
    print(f"  snap distance: median {np.median(dists):5.2f} px, p90 {np.percentile(dists, 90):5.2f}, "
          f"max {dists.max():5.2f}   gate {a.gate:.0f}")
    at_gate = float((dists > 0.9 * a.gate).mean())
    print(f"  {100 * at_gate:.0f}% of snaps sit within 10% of the gate. "
          f"{'HIGH -- the gate is choosing these, not the image. Do not trust them.' if at_gate > 0.15 else 'Low, which is what a healthy snap looks like.'}")
    if dups:
        print(f"  frame cadence: median {100 * float(np.median(dups)):.0f}% duplicate frames")
    if dropped:
        print(f"  {len(dropped)} clips dropped for <{a.min_pts} snapped points")
    print(f"  ball {out.diameter_px.min():.1f}-{out.diameter_px.max():.1f} px")
    print(f"\n{lp}")
    print("These are MEASURED positions. Run verify_ball_dataset.py after building with them.")


if __name__ == "__main__":
    main()
