"""Ball labels from a VALIDATED park camera: project each pitch's trajectory, keep what is seen.

WHY. Isotopes 815443 (a June day game) loses the ball against the dark wall behind the batter:
only 27 of 101 targeted pitches have a video track that reaches the plate (PROGRESS-LOG 101).
The park camera is now accurate to ~0.3 in (row 93), so projecting the known trajectory through
it says where the ball IS on every frame -- the same "labels from camera + trajectory" idea
ball_clock_align.py uses at MLB, now available at a MiLB park.

VISIBLE ONLY. Near the plate the batter often hides the ball; a label on a hidden ball teaches
the detector to hallucinate. So a frame is kept only when the SAVED detector candidates
(dets_<game><tag>.csv, every box >= the run's conf floor, not just the chain it picked) put
something within max(MATCH_PX, 0.8 x ball diameter) of the projected point. The label takes
the PROJECTED coordinate (sub-pixel under this camera), the detection only proves visibility.

FRAMES. Track and projection are in the REFERENCE view (stabilize_park.py); each label is
mapped back to raw clip pixels with the inverse of that clip's homography (nearest of the
per-clip H frames in stab2_H_<game><tag>.csv). Sequential-decode frame numbers throughout.

HOLD-OUT. Only stems NOT in holdout_<game>.csv are labelled, so a detector trained on these can
be graded on the held-out half of the same game.

    python scripts/project_labels.py --game 815443 --dir data/clips/pull_815443 \\
        --pose output/park_pose_815419_s2lm1.npz --stab-tag c
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from src import pinhole as ph  # noqa: E402
import fit_park_pose as fpp  # noqa: E402

FPS = 59.69
MATCH_PX = 6.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", type=int, required=True)
    ap.add_argument("--dir", required=True)
    ap.add_argument("--pose", required=True)
    ap.add_argument("--stab-tag", default="c", help="suffix of the stab2 files to use")
    ap.add_argument("--dets", default=None, help="default output/dets_<game>_v4.csv")
    ap.add_argument("--all-frames", action="store_true",
                    help="label EVERY projected flight frame, not only detector-confirmed ones. "
                         "Justified at 815443 by render (28 Sep): the unconfirmed frames show the "
                         "ball over low-contrast backgrounds (white uniform, red ad board), not "
                         "hidden - which is exactly what the detector needs to learn")
    a = ap.parse_args()
    d = a.dir if os.path.isabs(a.dir) else os.path.join(ROOT, a.dir)
    assert os.path.isdir(d), d
    man = pd.read_csv(os.path.join(d, os.path.basename(d) + "_manifest.csv")).set_index("stem")
    z = np.load(os.path.join(ROOT, a.pose))
    C, ang, f = z["C"], z["ang"], float(z["f"])
    out_dir = os.path.join(ROOT, "output")
    trk = pd.read_csv(os.path.join(out_dir, "detect_tracks_%d_v4stab2%s.csv" % (a.game, a.stab_tag)))
    Hs = pd.read_csv(os.path.join(out_dir, "stab2_H_%d%s.csv" % (a.game, a.stab_tag))).dropna(subset=["h0"])
    dets = pd.read_csv(a.dets or os.path.join(out_dir, "dets_%d_v4.csv" % a.game))
    hold = set(pd.read_csv(os.path.join(out_dir, "holdout_%d.csv" % a.game)).stem)

    rows, n_proj, n_seen, n_clips = [], 0, 0, 0
    for stem, g in trk.groupby("stem"):
        if stem in hold or stem not in man.index or len(g) < 5:
            continue
        r = man.loc[stem]
        if any(pd.isna(r[k]) for k in fpp.TRAJ):
            continue
        traj = fpp.traj_fn(r)
        g = g.sort_values("frame")
        fr, obs = g.frame.to_numpy(float), g[["x", "y"]].to_numpy()

        def res(t0):
            return (ph.project(C, *ang, f, traj(t0[0] + fr / FPS)) - obs).ravel()
        t0 = least_squares(res, [-fr[0] / FPS + 0.05], loss="soft_l1", f_scale=3.0).x[0]
        fit_px = float(np.median(np.hypot(*res([t0]).reshape(-1, 2).T)))
        if fit_px > 5.0:
            continue   # this track is not the pitch (row 101); do not anchor labels on it
        h = Hs[Hs.stem == stem]
        if h.empty:
            continue
        n_clips += 1
        hf = h.frame.to_numpy()
        f_rel = int(np.ceil(-t0 * FPS))
        f_plate = int(np.floor((float(r["zone_time"]) - t0) * FPS))
        dd = dets[dets.stem == stem]
        for fi in range(f_rel + 2, f_plate + 1):
            t = t0 + fi / FPS
            X = traj(t)
            p_ref = ph.project(C, *ang, f, X)
            Hrow = h.iloc[int(np.argmin(np.abs(hf - fi)))]
            Hm = np.array([Hrow["h%d" % k] for k in range(9)]).reshape(3, 3)
            q = np.linalg.inv(Hm) @ np.array([p_ref[0], p_ref[1], 1.0])
            x, y = q[0] / q[2], q[1] / q[2]
            zoom = float(np.sqrt(abs(np.linalg.det(Hm[:2, :2] / Hm[2, 2]))))
            diam = fpp.ball_diameter_px(C, *ang, f, X) / zoom
            n_proj += 1
            cand = dd[dd.frame == fi]
            dist_k, conf_k = np.nan, np.nan
            if not cand.empty:
                dist = np.hypot(cand.x - x, cand.y - y)
                k = int(np.argmin(dist.to_numpy()))
                dist_k, conf_k = float(dist.iloc[k]), float(cand.conf.iloc[k])
            confirmed = bool(dist_k <= max(MATCH_PX, 0.8 * diam)) if dist_k == dist_k else False
            n_seen += confirmed
            if confirmed or a.all_frames:
                rows.append(dict(stem=stem, game_pk=a.game, frame=fi, x=x, y=y, diameter_px=diam,
                                 det_dist=dist_k, det_conf=conf_k, confirmed=confirmed,
                                 in_chain=bool(((g.frame == fi)).any())))
    lab = pd.DataFrame(rows)
    out = os.path.join(out_dir, "proj_labels_%d.csv" % a.game)
    lab.to_csv(out, index=False)
    print("game %d: %d fit-half clips anchored; %d projected flight frames; %d confirmed visible "
          "(%.0f%%)" % (a.game, n_clips, n_proj, n_seen, 100.0 * n_seen / max(n_proj, 1)))
    if len(lab):
        print("  labels written %d (confirmed %d); already in the detector's chain %d"
              % (len(lab), int(lab.confirmed.sum()), int(lab.in_chain.sum())))
    print("->", out)


if __name__ == "__main__":
    main()
