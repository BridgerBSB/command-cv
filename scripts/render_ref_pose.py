"""Eye check for a park camera, drawn on the REFERENCE view the camera was fitted in.

WHY. fit_park_pose --render draws on the FIRST clip's raw frame. With --stab the camera and the
observed points are in the park's reference view (stabilize_park.py), so at a park whose framing
moves between clips (Dell Diamond 815046: zoom 0.95-1.13) that render shows a correct camera as
offset. Found 2026-10-11 on the first labels-only park.

WHAT IT DRAWS on the reference frame, from the camera alone (none of these entered the fit):
  red    home plate          blue  rubber front edge (on the 10 in mound)
  white  9 ft mound circle   grey  foul lines to 120 ft
plus, for --n fit-half clips, the stabilized ball track (green rings) and the camera's projection
of that pitch's trajectory (yellow dots, per-clip release time fitted). Plate under the plate,
rubber under the pivot foot and circle on the dirt edge = the camera is right for reasons the
optimiser could not fake.

REFERENCE = data/landmarks/<g>.json ref_stem/ref_frame if set, else the stab2_H row matched to
itself (identity, most inliers), which is how stabilize_park's median-set-up fallback shows up.

    python scripts/render_ref_pose.py --game 815046 --dir data/clips/pull_815046 \\
        --pose output/park_pose_815046_ballonly.npz
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np
import pandas as pd
from scipy.optimize import least_squares

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from src import pinhole as ph  # noqa: E402
import fit_park_pose as fpp  # noqa: E402

FPS = 59.69
RUBBER_FRONT = np.array([[1.0, 60.5, 10 / 12.0], [-1.0, 60.5, 10 / 12.0]])


def reference(game, tag):
    lm = os.path.join(ROOT, "data", "landmarks", "%d.json" % game)
    j = json.load(open(lm, encoding="utf-8")) if os.path.exists(lm) else {}
    if j.get("ref_stem"):
        return j["ref_stem"], int(j["ref_frame"])
    h = pd.read_csv(os.path.join(ROOT, "output", "stab2_H_%d%s.csv" % (game, tag)))
    M = h[["h%d" % i for i in range(9)]].to_numpy() / h[["h8"]].to_numpy()
    dev = np.abs(M - np.eye(3).ravel()).max(axis=1)
    ident = h[dev < 1e-6]
    assert len(ident), "no identity H in stab2_H_%d%s.csv and no ref_stem in %s" % (game, tag, lm)
    r = ident.sort_values("inliers", ascending=False).iloc[0]
    return r.stem, int(r.frame)


def read_frame(path, idx):
    cap = cv2.VideoCapture(path)
    i, out = 0, None
    while True:   # sequential decode, never seek (index-and-time-origins.md)
        ok, f = cap.read()
        if not ok or i > idx:
            break
        if i == idx:
            out = f
        i += 1
    cap.release()
    assert out is not None, "%s has no frame %d" % (path, idx)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", type=int, required=True)
    ap.add_argument("--dir", required=True)
    ap.add_argument("--pose", required=True)
    ap.add_argument("--stab-tag", default="c")
    ap.add_argument("--n", type=int, default=8, help="fit-half clips whose tracks are drawn")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    d = a.dir if os.path.isabs(a.dir) else os.path.join(ROOT, a.dir)
    z = np.load(a.pose if os.path.isabs(a.pose) else os.path.join(ROOT, a.pose))
    C, ang, f = z["C"], z["ang"], float(z["f"])
    stem, fr = reference(a.game, a.stab_tag)
    img = read_frame(os.path.join(d, stem + ".mp4"), fr)

    def poly(P, col, closed, w=2):
        p = ph.project(C, *ang, f, P).astype(int)
        cv2.polylines(img, [p.reshape(-1, 1, 2)], closed, col, w)
        return p

    t = np.linspace(0, 2 * np.pi, 120)
    poly(np.c_[9 * np.cos(t), 59 + 9 * np.sin(t), np.zeros_like(t)], (255, 255, 255), True, 1)
    for s in (1, -1):
        poly(np.array([[0, 0, 0], [s * 120.0, 120.0, 0]]), (160, 160, 160), False, 1)
    poly(fpp.PLATE, (0, 0, 255), True)
    poly(RUBBER_FRONT, (255, 0, 0), False, 3)

    man = pd.read_csv(os.path.join(d, os.path.basename(d) + "_manifest.csv")).set_index("stem")
    seeds = set(pd.read_csv(os.path.join(ROOT, "output", "seeds_%d_v4fit.csv" % a.game)).stem)
    trk = pd.read_csv(os.path.join(ROOT, "output", "detect_tracks_%d_v4stab2%s.csv" % (a.game, a.stab_tag)))
    trk = trk[trk.stem.isin(seeds)]
    drawn, worst = 0, []
    for s, g in sorted(trk.groupby("stem"), key=lambda kv: -len(kv[1])):
        if drawn >= a.n or s not in man.index or any(pd.isna(man.loc[s][k]) for k in fpp.TRAJ):
            continue
        g = g.sort_values("frame")
        frm, obs = g.frame.to_numpy(float), g[["x", "y"]].to_numpy()
        traj = fpp.traj_fn(man.loc[s])

        def res(t0):
            return (ph.project(C, *ang, f, traj(t0[0] + frm / FPS)) - obs).ravel()
        t0 = least_squares(res, [-frm[0] / FPS + 0.05], loss="soft_l1", f_scale=3.0).x[0]
        worst.append(float(np.median(np.hypot(*res([t0]).reshape(-1, 2).T))))
        for p in ph.project(C, *ang, f, traj(t0 + frm / FPS)).astype(int):
            cv2.circle(img, tuple(p), 2, (0, 255, 255), -1)
        for p in obs.astype(int):
            cv2.circle(img, tuple(p), 5, (0, 255, 0), 1)
        drawn += 1
    cv2.putText(img, "%d ref view %s f%d   %d clips, median track fit %.2f px"
                % (a.game, stem, fr, drawn, float(np.median(worst)) if worst else float("nan")),
                (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    out = a.out or os.path.join(ROOT, "output", "park_pose_%d_ref.png" % a.game)
    cv2.imwrite(out, img)
    print("reference %s f%d, %d clips drawn -> %s" % (stem, fr, drawn, out))


if __name__ == "__main__":
    main()
