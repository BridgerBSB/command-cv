"""Fit ONE camera pose for a MiLB park from a handful of seed ball tracks.

THE STRUCTURE THIS EXPLOITS. At MLB every clip carries its own pan/tilt/roll/f because
the camera pans and zooms between pitches, so ablate_no_box.fit_joint shares only (Cx, Cz)
and spends 5 nuisance parameters per clip. aplus_camera_between_pitches.py measured the A+
camera as LOCKED across a whole game (Asheville 822515: 0.13 px median over 59 clips), so
here the ENTIRE pose is shared and the only per-clip unknown is t0, the trajectory time at
sequential frame 0. Five clips is then ~50 residuals against 7 + 5 = 12 parameters.

Cy IS FREE, unlike everywhere else in this codebase. src/pinhole.py pins CY_FT = 400, a
centerfield assumption that is simply false at a park shot from up the line. 55bfba19
established on MLB, where truth exists, that freeing it never rails and does not damage the
held-out metric -- while also establishing that Cy is only WEAKLY determined, because
(Cy, f) trade off almost exactly. Read the caveat below before quoting any number from here.

WHAT THIS POSE IS AND IS NOT GOOD FOR:
  GOOD    generating training labels. A label needs the projection to land on the right
          PIXEL, and the (Cy, f) degeneracy is by definition invisible to that.
  NOT     measuring miss distance in inches at this park. That needs a metrically true
          camera, which needs a known world object in frame to break the degeneracy.
          Separate job. Do not quote Cx/Cy/Cz from here as a camera survey.

THE VALIDATION IS HELD OUT AND CANNOT BE GAMED. The fit sees ball pixels and nothing else.
It is then asked to project HOME PLATE and the PITCHING RUBBER -- objects at exactly known
world coordinates that never entered the fit. If the drawn plate lands on the real plate,
the pose is right for reasons the optimizer had no way to fake. That is the same logic as
ablate_no_box.py's box-corner metric, which is the only evaluation in that file that cannot
be gamed. LOOK AT THE RENDER; do not read a number off the residual alone, because a
7-parameter fit to one smooth arc can trace the ball from a completely wrong camera (that
failure is recorded in ablate_no_box.py: 17 ft of X scatter, railing on 21 of 59 clips).

MULTI-START, because the basin is not known a priori. An MLB centerfield seed is nowhere
near a side-view park, and a single start would sit wherever it was put. The grid spans
both sides of the field and a wide range of depth and height; the best final cost wins and
the spread across starts is reported, because starts landing in different places is itself
the finding that the pose is not identified.

    python scripts/fit_park_pose.py --game 822515
    python scripts/fit_park_pose.py --game 822515 --render
"""
import argparse
import itertools
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

# Home plate, exactly specified by rule: a 17 in square with two corners cut to a point,
# the point at the origin, the 17 in edge facing the pitcher. Coordinates in feet, y toward
# the mound. The pitching rubber is 24 x 6 in with its front edge at y = 60.5.
IN = 1.0 / 12.0
PLATE = np.array([[0.0, 0.0, 0.0],              # tip, at the origin
                  [-8.5 * IN, 8.5 * IN, 0.0],   # left of the tip
                  [-8.5 * IN, 17.0 * IN, 0.0],  # back left
                  [8.5 * IN, 17.0 * IN, 0.0],   # back right
                  [8.5 * IN, 8.5 * IN, 0.0]])   # right of the tip
RUBBER = np.array([[-12.0 * IN, 60.5, 0.0], [12.0 * IN, 60.5, 0.0],
                   [12.0 * IN, 60.5 + 6.0 * IN, 0.0], [-12.0 * IN, 60.5 + 6.0 * IN, 0.0]])

TRAJ = ["traj_x0", "traj_x1", "traj_x2", "traj_y0", "traj_y1", "traj_y2",
        "traj_z0", "traj_z1", "traj_z2"]


def traj_fn(row):
    """The manifest's 9-param polynomial, t seconds FROM RELEASE: axis = c0 + c1 t + c2 t^2.

    Verified against the manifest's own plate_x / plate_z: evaluating at zone_time
    reproduces both to 4 decimal places. Note zone_time corresponds to the PLATE FRONT
    (y = 17/12), not ph.zone_y(2026) = 8.5/12, which is the plane the broadcast DRAWS the
    zone box on. Two different planes; do not conflate them.
    """
    c = [float(row[k]) for k in TRAJ]

    def f(t):
        t = np.asarray(t, float)
        return np.stack([c[0] + c[1] * t + c[2] * t ** 2,
                         c[3] + c[4] * t + c[5] * t ** 2,
                         c[6] + c[7] * t + c[8] * t ** 2], axis=-1)
    return f


def build(game, seeds, manifest, fps_default=59.69):
    m = pd.read_csv(manifest).set_index("stem")
    d = pd.read_csv(seeds)
    d = d[d.game_pk == game]
    clips = []
    for stem, g in d.groupby("stem"):
        if stem not in m.index:
            continue
        row = m.loc[stem]
        if any(pd.isna(row[k]) for k in TRAJ):
            continue
        g = g.sort_values("frame")
        clips.append(dict(stem=stem, traj=traj_fn(row), fps=fps_default,
                          fr=g.frame.to_numpy(float),
                          obs=g[["x", "y"]].to_numpy(float),
                          zone_time=float(row["zone_time"])))
    return clips


def residual(x, clips):
    C = x[:3]
    pan, tilt, roll, f = x[3:7]
    out = []
    for k, c in enumerate(clips):
        t = x[7 + k] + c["fr"] / c["fps"]
        out.append((ph.project(C, pan, tilt, roll, f, c["traj"](t)) - c["obs"]).ravel())
    return np.concatenate(out)


def fit(clips, c0, f0):
    n = len(clips)
    # t0 = trajectory time at frame 0. The ball is in frame for the LAST part of the flight,
    # so t0 is about -(first tracked frame)/fps. Seed there, bound generously.
    t0s = [-float(c["fr"][0]) / c["fps"] for c in clips]
    x0 = list(c0) + [0.0, 0.0, 0.0, f0] + t0s
    lo = [-400, -120, 3, -1.2, -1.2, -0.6, f0 / 4] + [t - 1.5 for t in t0s]
    hi = [400, 600, 200, 1.2, 1.2, 0.6, f0 * 4] + [t + 1.5 for t in t0s]
    xs = [40, 60, 20, 0.1, 0.1, 0.05, f0 * 0.4] + [0.05] * n
    x0 = np.clip(x0, np.array(lo) + 1e-9, np.array(hi) - 1e-9)
    r = least_squares(residual, x0, args=(clips,), bounds=(lo, hi), x_scale=xs,
                      loss="soft_l1", f_scale=3.0, xtol=1e-12, ftol=1e-12, gtol=1e-12,
                      max_nfev=4000)
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", type=int, default=822515)
    ap.add_argument("--dir", default=os.path.join(ROOT, "data", "clips", "ogando_2026"))
    ap.add_argument("--seeds", default=None)
    ap.add_argument("--render", action="store_true")
    a = ap.parse_args()
    seeds = a.seeds or os.path.join(ROOT, "output", f"seed_tracks_{a.game}.csv")
    manifest = os.path.join(a.dir, os.path.basename(a.dir) + "_manifest.csv")
    clips = build(a.game, seeds, manifest)
    assert len(clips) >= 3, f"need >=3 seeded clips, have {len(clips)}"
    npts = sum(len(c["fr"]) for c in clips)
    print(f"game {a.game}: {len(clips)} clips, {npts} ball points, "
          f"{2 * npts} residuals vs {7 + len(clips)} params")

    best, tried = None, []
    grid = itertools.product([-260, -170, -110, 110, 170, 260],   # Cx
                             [-40, 10, 60, 130, 260],            # Cy
                             [15, 35, 70],                       # Cz
                             [1400, 2600])                       # f seed, px
    for cx, cy, cz, f0 in grid:
        try:
            r = fit(clips, (cx, cy, cz), f0)
        except Exception:
            continue
        rms = float(np.sqrt(np.mean(r.fun ** 2)))
        tried.append((rms, cx, cy, cz, f0, r.x))
        if best is None or rms < best[0]:
            best = (rms, cx, cy, cz, f0, r.x)
    assert best, "every start failed"
    tried.sort(key=lambda t: t[0])

    rms, cx, cy, cz, f0, x = best
    C, pan, tilt, roll, fpx = x[:3], x[3], x[4], x[5], x[6]
    print(f"\nbest of {len(tried)} starts   seed C=({cx},{cy},{cz}) f={f0}")
    print(f"  ball reprojection RMS  {rms:7.3f} px")
    print(f"  camera   Cx {C[0]:8.2f}  Cy {C[1]:8.2f}  Cz {C[2]:8.2f}   f {fpx:8.1f} px")
    print(f"  angles   pan {pan:7.4f}  tilt {tilt:7.4f}  roll {roll:7.4f}")
    print("\n  top 5 starts (rms, where they landed) -- a spread here means NOT identified:")
    for t in tried[:5]:
        print(f"    rms {t[0]:7.3f}   C=({t[5][0]:8.2f},{t[5][1]:8.2f},{t[5][2]:8.2f})"
              f"  f={t[5][6]:8.1f}")

    print("\n  per-clip reprojection:")
    for k, c in enumerate(clips):
        t = x[7 + k] + c["fr"] / c["fps"]
        pr = ph.project(C, pan, tilt, roll, fpx, c["traj"](t))
        e = np.hypot(*(pr - c["obs"]).T)
        print(f"    {c['stem']:16} n={len(e):2d}  rms {np.sqrt((e**2).mean()):6.2f} px  "
              f"max {e.max():6.2f}   t0 {x[7+k]:7.3f} s   release at frame "
              f"{-x[7+k]*c['fps']:6.1f}")

    print("\n  HELD OUT -- home plate and the rubber, never seen by the fit:")
    for name, pts in (("home plate", PLATE), ("rubber", RUBBER)):
        px = ph.project(C, pan, tilt, roll, fpx, pts)
        print(f"    {name:11} " + "  ".join(f"({p[0]:7.1f},{p[1]:7.1f})" for p in px))
    print("\n  RENDER IT AND LOOK. A low ball RMS from a wrong camera is a documented\n"
          "  failure mode here (ablate_no_box.py). The plate landing on the plate is the\n"
          "  evidence; the ball RMS is not.")

    if a.render:
        c0 = clips[0]
        cap = cv2.VideoCapture(os.path.join(a.dir, c0["stem"] + ".mp4"))
        i, frame = 0, None
        tgt = int(c0["fr"][0])
        while True:
            ok, f = cap.read()
            if not ok or i > tgt:
                break
            if i == tgt:
                frame = f.copy()
            i += 1
        cap.release()
        if frame is not None:
            for name, pts, col in (("plate", PLATE, (0, 0, 255)), ("rubber", RUBBER, (255, 0, 0))):
                p = ph.project(C, pan, tilt, roll, fpx, pts).astype(int)
                cv2.polylines(frame, [p.reshape(-1, 1, 2)], True, col, 2)
                cv2.putText(frame, name, tuple(p[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
            for k, c in enumerate(clips[:1]):
                t = x[7 + k] + c["fr"] / c["fps"]
                for p in ph.project(C, pan, tilt, roll, fpx, c["traj"](t)).astype(int):
                    cv2.circle(frame, tuple(p), 4, (0, 255, 255), -1)
                for p in c["obs"].astype(int):
                    cv2.circle(frame, tuple(p), 6, (0, 255, 0), 1)
            out = os.path.join(ROOT, "output", f"park_pose_{a.game}.png")
            cv2.imwrite(out, frame)
            print(f"\n{out}   yellow = projected ball, green = observed, "
                  f"red/blue = HELD-OUT plate and rubber")


if __name__ == "__main__":
    main()
