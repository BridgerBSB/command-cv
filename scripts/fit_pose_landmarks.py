"""Fit a park's REFERENCE-VIEW camera to GROUND LANDMARKS, optionally plus ball tracks.

WHY. A ball-only fit cannot pin the camera's distance: (Cy, f) trade off almost exactly
(fit_park_pose.fit docstring), and at Durham the refit walked Cy to its 600 ft bound
(PROGRESS-LOG 87). The field itself is a calibration target at exactly known positions:
  foul lines   -- the lines |x| = y on the ground, through the plate tip
  mound edge   -- a 9 ft radius circle centred (0, 59) on the ground
  plate ends   -- the white plate bar, (+-8.5 in, ~12 in, 0)
All on z = 0, so they fix the ground-plane homography, and with square pixels and a centred
principal point that homography determines the whole camera (7 params from 8 dof).

TWO MODES, and the difference is the whole point:
  --ball-weight 0   camera from LANDMARKS ONLY. Every ball track is then HELD OUT: its
                    reprojection error is a genuinely independent check of the camera.
  --ball-weight >0  joint fit, fit-half ball tracks added. Grade on the held-out half with
                    command_miss.py + grade_holdout.py, never on the fit half.

WHICH VIEW. These cameras move between clips (Durham set-up targets shift up to 45 px), so
the pose is for ONE reference view -- data/landmarks/<game>.json names it -- and every clip
is mapped into it first by stabilize_park.py. Use the STABILIZED tracks here.

LANDMARKS are extracted by eye-checked colour segmentation on the reference frame and
stored in data/landmarks/<game>_feats.npz. The json records what was REJECTED and why (the
Durham "rubber" slab was not the rubber; see its note).

    python scripts/fit_pose_landmarks.py --game 816698 --dir data/clips/aaa_816698 \\
        --seeds output/seeds_816698_v4fit.csv --stab output/detect_tracks_816698_v4stab.csv
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
sys.path.insert(0, os.path.join(ROOT, "tools"))
from src import pinhole as ph  # noqa: E402
import fit_park_pose as fpp  # noqa: E402

MOUND_R_FT, MOUND_Y_FT = 9.0, 59.0   # rule size; --mound-r overrides per park (crews vary)
PLATE_BAR = np.array([[8.5 / 12, 1.0, 0.0], [-8.5 / 12, 1.0, 0.0]])


def load(game):
    j = json.load(open(os.path.join(ROOT, "data", "landmarks", "%d.json" % game), encoding="utf-8"))
    F = dict(np.load(os.path.join(ROOT, j["features_npz"])))
    return j, F


def landmark_res(x, F, plate_w=5.0):
    """Pixel residuals of every ground landmark under camera x[:7]."""
    C, (pan, tilt, roll, f) = x[:3], x[3:7]
    out = []
    fl = F.get("foul")
    if fl is not None and len(fl):
        A = ph.project(C, pan, tilt, roll, f, np.array([[20, 20, 0], [120, 120, 0.0]]))
        B = ph.project(C, pan, tilt, roll, f, np.array([[-20, 20, 0], [-120, 120, 0.0]]))
        mid = np.median(fl[:, 0]) if F.get("foul_split") is None else float(F["foul_split"])
        lineL, lineR = (A, B) if A[1, 0] < B[1, 0] else (B, A)
        for P, (a, b) in ((fl[fl[:, 0] < mid], lineL), (fl[fl[:, 0] >= mid], lineR)):
            n = np.array([-(b - a)[1], (b - a)[0]])
            out.append((P - a) @ (n / np.linalg.norm(n)))
    mp = F.get("mound")
    if mp is not None and len(mp):
        # nearest circle point, found on the GROUND (unproject to z=0, go radially to the
        # circle) and then projected: an exact pixel residual at the cost of one projection
        right, up, fwd = ph.camera(C, pan, tilt, roll)
        ray = ((mp[:, :1] - ph.W / 2) / f) * right + ((ph.H / 2 - mp[:, 1:]) / f) * up + fwd
        s = -C[2] / ray[:, 2]
        g = C + s[:, None] * ray
        v = g[:, :2] - np.array([0.0, MOUND_Y_FT])
        v = v / np.linalg.norm(v, axis=1, keepdims=True)
        near = np.c_[MOUND_R_FT * v[:, 0], MOUND_Y_FT + MOUND_R_FT * v[:, 1], np.zeros(len(v))]
        out.append(np.hypot(*(ph.project(C, pan, tilt, roll, f, near) - mp).T) * np.where(s > 0, 1, 50))
    pe = F.get("plate_ends")
    if pe is not None and len(pe):
        pr = ph.project(C, pan, tilt, roll, f, PLATE_BAR)
        pr = pr[np.argsort(pr[:, 0])]
        out.append(plate_w * (pr - pe[np.argsort(pe[:, 0])]).ravel())
    return np.concatenate(out)


def stab_seeds(seeds, stab, game):
    """Fit-half (stem, frame) from --seeds with pixels from the STABILIZED track; count printed."""
    s = pd.read_csv(seeds)
    s = s[s.game_pk == game][["stem", "frame"]]
    t = pd.read_csv(stab, float_precision="round_trip")
    m = s.merge(t[["stem", "frame", "x", "y", "conf"]], on=["stem", "frame"], how="inner")
    print("stabilized seeds: %d of %d fit-half points matched (%d clips)"
          % (len(m), len(s), m.stem.nunique()))
    return m.assign(game_pk=game)


def ball_t0(x, c):
    """Per-clip release time with the camera FIXED -- how a held-out track is scored."""
    def rr(t0):
        return (ph.project(x[:3], *x[3:7], c["traj"](t0[0] + c["fr"] / c["fps"])) - c["obs"]).ravel()
    t = least_squares(rr, [-c["fr"][0] / c["fps"]], loss="soft_l1", f_scale=3.0).x[0]
    return t, np.hypot(*rr([t]).reshape(-1, 2).T)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", type=int, required=True)
    ap.add_argument("--dir", required=True)
    ap.add_argument("--seeds", required=True, help="fit-half points (stem, frame); held-out stems excluded")
    ap.add_argument("--stab", required=True, help="stabilized v4 tracks from stabilize_park.py")
    ap.add_argument("--ball-weight", type=float, default=0.0,
                    help="0 = landmarks only (every ball held out); else scale on ball residuals")
    ap.add_argument("--plate-weight", type=float, default=5.0)
    ap.add_argument("--init", default=None, help="starting pose npz (default: v4fit pose)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--grid", action="store_true", help="add a position lattice of starts")
    ap.add_argument("--mound-r", type=float, default=None,
                    help="visible dirt-circle radius, ft (rule 9.0; Isotopes reads wider)")
    a = ap.parse_args()
    d = a.dir if os.path.isabs(a.dir) else os.path.join(ROOT, a.dir)
    assert os.path.isdir(d), d
    j, F = load(a.game)
    if a.mound_r:
        global MOUND_R_FT
        MOUND_R_FT = a.mound_r
    print("mound radius %.2f ft" % MOUND_R_FT)
    print("landmarks: " + ", ".join("%s %d" % (k, len(v)) for k, v in F.items() if np.ndim(v) == 2))

    tmp = os.path.join(ROOT, "output", "_seeds_%d_stabfit.csv" % a.game)
    stab_seeds(os.path.join(ROOT, a.seeds), os.path.join(ROOT, a.stab), a.game).to_csv(tmp, index=False)
    clips = fpp.build(a.game, tmp, os.path.join(d, os.path.basename(d) + "_manifest.csv"))
    print("ball: %d clips, %d points  (weight %.2f)" % (len(clips), sum(len(c["fr"]) for c in clips),
                                                       a.ball_weight))

    base = np.load(a.init or os.path.join(ROOT, "output", "park_pose_%d_v4fit.npz" % a.game))
    bC, bang, bf = base["C"], base["ang"], float(base["f"])
    lo7, hi7 = [-400, 20, 3, -1.2, -1.2, -0.6, 500], [400, 1200, 300, 1.2, 1.2, 0.6, 90000]
    xs7 = [40, 60, 20, 0.1, 0.1, 0.05, 5000]

    # stage 1: landmarks only, multi-start along the depth valley (f scaled with distance)
    # Starts: the old pose scaled along its depth valley, PLUS (with --grid) a lattice over
    # camera position with f set from the plate bar's measured width, for parks where the
    # old pose is nowhere near the basin (Isotopes: shot from well off the centre line).
    starts = [np.r_[bC[0] * cy / max(bC[1], 1.0), cy, bC[2] * cy / max(bC[1], 1.0), bang,
                    bf * cy / max(bC[1], 1.0)] for cy in (120, 200, 300, 450, 650)]
    if a.grid:
        pe = F["plate_ends"]
        px_per_ft = abs(pe[1, 0] - pe[0, 0]) / (17 / 12.0)
        for cx in (-90, -45, -15, 15, 45, 90):
            for cy in (150, 250, 400, 600):
                for cz in (10, 25, 45):
                    starts.append(np.r_[cx, cy, cz, 0.0, 0.0, 0.0, px_per_ft * np.hypot(cx, cy)])
    best = None
    for x0 in starts:
        cy = x0[1]
        x0 = np.clip(x0, np.array(lo7) + 1e-6, np.array(hi7) - 1e-6)
        r = least_squares(landmark_res, x0, args=(F, a.plate_weight), bounds=(lo7, hi7), x_scale=xs7,
                          loss="soft_l1", f_scale=2.0, max_nfev=3000)
        e = np.abs(landmark_res(r.x, F, 1.0))
        if not a.grid or np.median(e) < 5:
            print("  start Cy %4d -> C (%6.1f,%6.1f,%5.1f) f %7.0f  landmark med %6.2f p90 %6.2f px"
                  % (cy, *r.x[:3], r.x[6], np.median(e), np.percentile(e, 90)), flush=True)
        if best is None or r.cost < best.cost:
            best = r
    x = best.x

    if a.ball_weight > 0:
        t0s = [ball_t0(x, c)[0] for c in clips]
        x0 = np.r_[x, t0s]
        lo = lo7 + [t - 1.0 for t in t0s]
        hi = hi7 + [t + 1.0 for t in t0s]
        x0 = np.clip(x0, np.array(lo) + 1e-9, np.array(hi) - 1e-9)
        w = np.sqrt(a.ball_weight)

        def joint(xx):
            return np.concatenate([landmark_res(xx, F, a.plate_weight), w * fpp.residual(xx, clips)])
        r = least_squares(joint, x0, bounds=(lo, hi), x_scale=xs7 + [0.05] * len(clips),
                          loss="soft_l1", f_scale=2.0, max_nfev=4000)
        x = r.x

    C, (pan, tilt, roll), f = x[:3], x[3:6], x[6]
    e = np.abs(landmark_res(x, F, 1.0))
    print("\ncamera: C=(%.1f, %.1f, %.1f)  pan %.4f tilt %.4f roll %.4f  f %.0f"
          % (*C, pan, tilt, roll, f))
    print("  landmark residual  median %.2f px  p90 %.2f px  (n %d)" % (np.median(e), np.percentile(e, 90), len(e)))
    print("  plate centre -> (%.1f, %.1f)   raised rubber centre -> (%.1f, %.1f)"
          % (*ph.project(C, pan, tilt, roll, f, np.array([0, 1.0, 0])),
             *ph.project(C, pan, tilt, roll, f, np.array([0, 60.5, 10 / 12.0]))))
    tag = "HELD OUT (camera never saw a ball)" if a.ball_weight == 0 else "fit half (NOT a test)"
    meds = []
    print("  ball, %s:" % tag)
    for c in clips:
        _, eb = ball_t0(x, c)
        meds.append(np.median(eb))
        print("    %-18s n %2d  median %5.1f px  max %5.1f" % (c["stem"], len(eb), meds[-1], eb.max()))
    print("  per-clip median of medians %.2f px" % np.median(meds))

    out = a.out or os.path.join(ROOT, "output", "park_pose_%d_lm.npz" % a.game)
    np.savez(out, C=C, ang=np.array([pan, tilt, roll]), f=f)
    print("  pose saved:", out)

    from grid_crop import read_frame
    fr = read_frame(os.path.join(d, j["ref_stem"] + ".mp4"), int(j["ref_frame"]))
    s = np.linspace(0, 160, 80)
    th = np.linspace(0, 2 * np.pi, 200)
    for P, col, closed in ((np.c_[s, s, 0 * s], (0, 255, 255), False),
                           (np.c_[-s, s, 0 * s], (255, 255, 0), False),
                           (np.c_[MOUND_R_FT * np.cos(th), MOUND_Y_FT + MOUND_R_FT * np.sin(th), 0 * th], (255, 0, 255), True),
                           (fpp.PLATE, (0, 0, 255), True),
                           (fpp.RUBBER + [0, 0, 10 / 12.0], (255, 0, 0), True)):
        p = ph.project(C, pan, tilt, roll, f, P).astype(np.int32)
        cv2.polylines(fr, [p.reshape(-1, 1, 2)], closed, col, 1)
    for k in ("foul", "mound", "plate_ends"):
        for q in F.get(k, []):
            cv2.circle(fr, tuple(int(v) for v in q), 1, (0, 255, 0), -1)
    png = out.replace(".npz", ".png")
    cv2.imwrite(png, fr)
    print("  render:", png, "(green = extracted landmarks; lines = projected field)")


if __name__ == "__main__":
    main()
