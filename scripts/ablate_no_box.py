"""What does losing the drawn strike-zone box cost? Measured at MLB, where truth exists.

This is the question that decides A+. OpenCommand locks the camera onto the zone box the
broadcast draws. Our Asheville feed draws none (step 1), and neither does the MLB video MLB's
own API serves us (step 2a), so the box is a crutch we cannot keep. Rather than guess how much
it matters, remove it here, where the answer is known.

Same clips, same ball detections (OpenCommand's, so the DETECTOR is never the variable):

  BOX          as published: position from ball + 4 box corners (weight 3), per-clip
               pan/tilt/roll/f refit from the box alone
  NOBOX        per clip, nothing but the ball: 7 params (Cx, Cz, pan, tilt, roll, f, t0), Cy pinned
  NOBOX_C      game position from the box-assisted votes, then per-clip pan/tilt/roll/f from the
               clip's own ball run with C frozen
  NOBOX_JOINT  no box anywhere: ONE shared (Cx, Cz) fitted across every clip at once, with each
               clip keeping its own pan/tilt/roll/f/t0. If the box can be replaced by "many balls
               from one camera", this is the variant that shows it

EVALUATION, and why the obvious one is useless here. The first version scored every variant on
the last HOLDOUT_FR frames of its own ball run. NOBOX "won" that (0.167 in vs 0.387 in) while its
camera position scattered by 17 ft in X and 37 ft in Z, railing at the solver's height bounds on
21 of 59 clips: a 7-parameter fit to ~21 points can put the camera anywhere and still trace the
ball, and extrapolating six frames further along a path it just fit is nearly free. A test a
wrong camera passes is not a test.

So four quantities are reported and only the last three can fail:
  ball@plate   the old number, kept so the trap stays visible, NOT as evidence
  box corners  the 4 zone-box corners, unprojected at the zone plane and compared in inches to
               where they truly are. The no-box variants never see the box, so for them this is a
               held-out object at a known world position
  glove plane  how far this pose disagrees with the BOX pose about one fixed pixel unprojected at
               SETUP_DEPTH_FT, the depth the catcher sets up at. That is where the miss is read
  camera       |Cx| and |Cz| against the box-assisted game solve

    python scripts/ablate_no_box.py [--game 824821] [--limit 60]
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from src import pinhole as ph  # noqa: E402
import parity_opencommand_pose as par  # noqa: E402

HOLDOUT_FR = 6
MIN_FIT_PTS = 8
SETUP_DEPTH_FT = -1.75          # OpenCommand's catcher setup plane, behind the plate tip


# --- per-clip fits ---------------------------------------------------------------------

def _res(x, c, seg, use_box):
    t = x[6] + seg.frame_idx.to_numpy() / c["fps"]
    pts = c["traj"](t)
    obs = seg[["baseball_center_x", "baseball_center_y"]].to_numpy()
    w = np.ones(len(seg))
    if use_box:
        pts = np.vstack([pts, c["box_world"]])
        obs = np.vstack([obs, c["box_obs"]])
        w = np.r_[w, np.full(4, 3.0)]
    C = np.array([x[0], ph.CY_FT, x[1]])
    return ((ph.project(C, x[2], x[3], x[4], x[5], pts) - obs) * w[:, None]).ravel()


def _bounds(c):
    f0, rel = c["f0"], c["rel"]
    t_lo, t_hi = -rel - par.RELEASE_HALF_S, -rel + par.RELEASE_HALF_S
    return ([-200, 5, -0.15, -0.15, -0.15, f0 / 2, t_lo],
            [200, 120, 0.15, 0.15, 0.15, f0 * 2, t_hi],
            [30, 15, 0.02, 0.02, 0.02, f0 * 0.3, 0.05])


def _fit(c, seg, use_box, cx0=0.0, cz0=20.0):
    lo, hi, xs = _bounds(c)
    t0 = float(np.clip(c["T_plate"] - seg.frame_idx.max() / c["fps"], lo[6], hi[6]))
    x0 = np.clip([cx0, cz0, 0, 0, 0, c["f0"], t0], np.array(lo) + 1e-9, np.array(hi) - 1e-9)
    r = least_squares(_res, x0, args=(c, seg, use_box), bounds=(lo, hi), loss="soft_l1", f_scale=2.0,
                      x_scale=xs, xtol=1e-12, ftol=1e-12, gtol=1e-12, max_nfev=2000)
    return r.x


def _ptz_from_ball(c, C, seg, t0):
    f0 = c["f0"]

    def res(q):
        t = t0 + seg.frame_idx.to_numpy() / c["fps"]
        return (ph.project(C, q[0], q[1], q[2], q[3], c["traj"](t))
                - seg[["baseball_center_x", "baseball_center_y"]].to_numpy()).ravel()
    r = least_squares(res, [0, 0, 0, f0],
                      bounds=([-0.15, -0.15, -0.15, f0 / 2], [0.15, 0.15, 0.15, f0 * 2]),
                      x_scale=[0.02, 0.02, 0.02, f0 * 0.3], xtol=1e-12, ftol=1e-12, gtol=1e-12)
    return r.x


# --- the joint solve: one camera position, many clips, ball only -----------------------

def fit_joint(prep, cx0, cz0):
    """[Cx, Cz] shared + [pan, tilt, roll, f, t0] per clip, fitted to ball pixels alone."""
    n = len(prep)
    x0 = [cx0, cz0]
    lo, hi, xs = [-200, 5], [200, 120], [30, 15]
    for c, fit_seg, _, seed in prep:
        b_lo, b_hi, b_xs = _bounds(c)
        x0 += [0, 0, 0, c["f0"], float(np.clip(seed, b_lo[6], b_hi[6]))]
        lo += b_lo[2:]; hi += b_hi[2:]; xs += b_xs[2:]
    x0 = np.clip(x0, np.array(lo) + 1e-9, np.array(hi) - 1e-9)

    def res(x):
        C = np.array([x[0], ph.CY_FT, x[1]])
        out = []
        for k, (c, seg, _, _) in enumerate(prep):
            pan, tilt, roll, f, t0 = x[2 + 5 * k: 7 + 5 * k]
            t = t0 + seg.frame_idx.to_numpy() / c["fps"]
            out.append((ph.project(C, pan, tilt, roll, f, c["traj"](t))
                        - seg[["baseball_center_x", "baseball_center_y"]].to_numpy()).ravel())
        return np.concatenate(out)

    sizes = [2 * len(s) for _, s, _, _ in prep]
    S = lil_matrix((sum(sizes), len(x0)), dtype=int)
    r0 = 0
    for k, m in enumerate(sizes):
        S[r0:r0 + m, 0:2] = 1
        S[r0:r0 + m, 2 + 5 * k: 7 + 5 * k] = 1
        r0 += m
    fit = least_squares(res, x0, bounds=(lo, hi), x_scale=xs, jac_sparsity=S,
                        loss="soft_l1", f_scale=2.0, xtol=1e-10, ftol=1e-10, gtol=1e-10,
                        max_nfev=300, verbose=0)
    return fit.x


# --- evaluations -----------------------------------------------------------------------

def box_corner_err_in(c, pose):
    """Unproject the 4 observed box-corner pixels at the zone plane; compare to where the
    corners truly are, in inches. The no-box fits never saw these."""
    zy = c["box_world"][0][1]
    xz = ph.unproject(pose[:3], pose[3], pose[4], pose[5], pose[6], c["box_obs"], zy)
    truth = c["box_world"][:, [0, 2]]
    d = (xz - truth) * 12
    return float(np.median(np.abs(d[:, 0]))), float(np.median(np.abs(d[:, 1])))


def glove_plane_gap_in(c, pose, pose_ref):
    """Disagreement between this pose and the BOX pose about one fixed pixel at the catcher's
    setup depth: the plane where the miss is measured."""
    pix = np.array([c["box_obs"][:, 0].mean(), c["box_obs"][:, 1].mean()])
    a = ph.unproject(pose[:3], pose[3], pose[4], pose[5], pose[6], pix, SETUP_DEPTH_FT)
    b = ph.unproject(pose_ref[:3], pose_ref[3], pose_ref[4], pose_ref[5], pose_ref[6], pix, SETUP_DEPTH_FT)
    return float(np.hypot(*(np.array(a) - np.array(b))) * 12)


def err_at_plate(c, pose, t0, held):
    px = held[["baseball_center_x", "baseball_center_y"]].to_numpy()
    t = t0 + held.frame_idx.to_numpy() / c["fps"]
    j = int(np.abs(t - c["T_plate"]).argmin())
    truth = c["traj"](t[j])
    bx, bz = ph.unproject(pose[:3], pose[3], pose[4], pose[5], pose[6], px[j], truth[1])
    return (bx - truth[0]) * 12, (bz - truth[2]) * 12


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", type=int, default=824821)
    ap.add_argument("--limit", type=int, default=60)
    a = ap.parse_args()
    clips = par.load_clips(a.game)[:a.limit]
    prep, votes = [], []
    for c in clips:
        runs = par.candidate_runs(c)
        if not runs or len(runs[0]) < MIN_FIT_PTS + HOLDOUT_FR:
            continue
        seg = runs[0]
        fit_seg, held = seg.iloc[:-HOLDOUT_FR], seg.iloc[-HOLDOUT_FR:]
        x_box = _fit(c, fit_seg, True)
        prep.append((c, fit_seg, held, x_box[6]))
        votes.append((x_box[:2], x_box))
    assert prep, "no clips had a long enough ball run"
    med = np.median(np.array([v[0] for v in votes]), axis=0)
    C_game = np.array([med[0], ph.CY_FT, med[1]])
    print(f"game {a.game}: {len(prep)} clips, fit points median "
          f"{np.median([len(s) for _, s, _, _ in prep]):.0f}, holdout {HOLDOUT_FR} frames")
    print(f"joint solve: {2 + 5 * len(prep)} parameters over "
          f"{sum(2 * len(s) for _, s, _, _ in prep)} residuals ...", flush=True)
    xj = fit_joint(prep, med[0], med[1])
    C_joint = np.array([xj[0], ph.CY_FT, xj[1]])
    rows = []
    for k, (c, fit_seg, held, t0_box) in enumerate(prep):
        pose_box = np.concatenate([C_game, par.fit_ptz(c, C_game)])
        x_nob = _fit(c, fit_seg, False)
        pose_nob = np.array([x_nob[0], ph.CY_FT, x_nob[1], x_nob[2], x_nob[3], x_nob[4], x_nob[5]])
        pose_cg = np.concatenate([C_game, _ptz_from_ball(c, C_game, fit_seg, t0_box)])
        pose_j = np.concatenate([C_joint, xj[2 + 5 * k: 6 + 5 * k]])
        row = dict(play_id=c["play_id"], n_fit=len(fit_seg), nobox_Cx=x_nob[0], nobox_Cz=x_nob[1])
        for name, pose, t0 in (("BOX", pose_box, t0_box), ("NOBOX", pose_nob, x_nob[6]),
                               ("NOBOX_C", pose_cg, t0_box), ("NOBOX_JOINT", pose_j, xj[6 + 5 * k])):
            ex, ez = err_at_plate(c, pose, t0, held)
            bx, bz = box_corner_err_in(c, pose)
            row |= {f"{name}_plate_x": ex, f"{name}_plate_z": ez,
                    f"{name}_box_x": bx, f"{name}_box_z": bz,
                    f"{name}_glove_gap": glove_plane_gap_in(c, pose, pose_box)}
        rows.append(row)
    d = pd.DataFrame(rows)
    print(f"\n{'variant':12} {'ball@plate |x|,|z|':>20} {'BOX CORNERS |x|,|z| in':>24} {'glove-plane gap in':>20}")
    for name in ("BOX", "NOBOX", "NOBOX_C", "NOBOX_JOINT"):
        print(f"  {name:10} "
              f"{d[f'{name}_plate_x'].abs().median():9.3f} {d[f'{name}_plate_z'].abs().median():9.3f} "
              f"{d[f'{name}_box_x'].abs().median():12.3f} {d[f'{name}_box_z'].abs().median():11.3f} "
              f"{d[f'{name}_glove_gap'].median():19.3f}")
    print(f"\ncamera position:")
    print(f"  BOX game solve      Cx {C_game[0]:8.3f}  Cz {C_game[2]:8.3f}")
    print(f"  JOINT ball-only     Cx {C_joint[0]:8.3f}  Cz {C_joint[2]:8.3f}   "
          f"(off by {abs(C_joint[0] - C_game[0]):.2f} ft, {abs(C_joint[2] - C_game[2]):.2f} ft)")
    print(f"  per-clip ball-only  Cx sd {d.nobox_Cx.std():.1f} ft  Cz sd {d.nobox_Cz.std():.1f} ft  "
          f"(railed at a bound on {int(((d.nobox_Cz < 5.5) | (d.nobox_Cz > 119)).sum())}/{len(d)} clips)")
    out = os.path.join(os.path.dirname(__file__), "..", "output", f"ablate_no_box_{a.game}.csv")
    d.to_csv(out, index=False)
    print(out)


if __name__ == "__main__":
    main()
