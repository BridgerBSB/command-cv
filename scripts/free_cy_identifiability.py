"""Is the camera still identifiable when Cy stops being pinned at 400 ft?

WHY THIS EXISTS. src/pinhole.py pins CY_FT = 400.0 -- the camera sits 400 ft behind
the plate -- and ablate_no_box.fit_joint bakes that in, fitting only (Cx, Cz). That is
right for an MLB centerfield camera and WRONG for MiLB, where the angle is not even
constant across parks: output/aplus_angle_sheet.png shows Asheville (822515) and 821780
shot from the side, up the line, while Greensboro (822246) sits behind the pitcher. A
side camera has a Cy near the plate, not 400 ft beyond it.

The tempting move is to free Cy and carry on. But a 3-param position fitted to a single
ball arc is exactly the failure ablate_no_box.py's docstring records: the 7-param
per-clip fit traced the ball beautifully while scattering the camera by 17 ft in X and
37 in Z, railing at bounds on 21 of 59 clips. A wrong camera that still traces the ball
passes any test built on that ball.

So ask the question where the answer is KNOWN. Same MLB clips, same OpenCommand ball
detections (the detector is never the variable here), same solver, same evaluation. The
only change is whether Cy is a parameter. If freeing it cannot recover a camera we
already know, no amount of MiLB data will rescue it, and the A+ plan needs a different
shape.

READ THE OUTPUT IN THIS ORDER:
  camera        |Cx|,|Cy|,|Cz| against the box-assisted game solve. Cy's own error is
                the new number, and the one the whole question turns on.
  box corners   the 4 drawn-zone corners unprojected at the zone plane, in inches. No
                arm here fits the box, so for every one of them it is a held-out object
                at a known world position -- the only evaluation that cannot be gamed.
  glove plane   disagreement with the BOX pose about one pixel at the catcher's setup
                depth, which is where the miss is actually read.

    python scripts/free_cy_identifiability.py --game 824821 --limit 60
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from src import pinhole as ph  # noqa: E402
import parity_opencommand_pose as par  # noqa: E402
import ablate_no_box as abn  # noqa: E402

# Cy bounds wide enough to contain BOTH a centerfield camera (~400 ft) and a side
# camera up the line (tens of feet), so the same solve can be pointed at either.
CY_LO, CY_HI = -50.0, 600.0


def fit_joint_freecy(prep, cx0, cy0, cz0):
    """[Cx, Cy, Cz] shared + [pan, tilt, roll, f, t0] per clip, from ball pixels alone.

    Deliberately a near-copy of ablate_no_box.fit_joint with ONE structural change, so
    the two are comparable line for line. Everything else -- bounds, scaling, robust
    loss, sparsity -- is taken from that function rather than re-chosen here.
    """
    x0 = [cx0, cy0, cz0]
    lo, hi, xs = [-400, CY_LO, 5], [400, CY_HI, 120], [30, 50, 15]
    for c, fit_seg, _, seed in prep:
        b_lo, b_hi, b_xs = abn._bounds(c)
        x0 += [0, 0, 0, c["f0"], float(np.clip(seed, b_lo[6], b_hi[6]))]
        lo += b_lo[2:]; hi += b_hi[2:]; xs += b_xs[2:]
    x0 = np.clip(x0, np.array(lo) + 1e-9, np.array(hi) - 1e-9)

    def res(x):
        C = np.array([x[0], x[1], x[2]])
        out = []
        for k, (c, seg, _, _) in enumerate(prep):
            pan, tilt, roll, f, t0 = x[3 + 5 * k: 8 + 5 * k]
            t = t0 + seg.frame_idx.to_numpy() / c["fps"]
            out.append((ph.project(C, pan, tilt, roll, f, c["traj"](t))
                        - seg[["baseball_center_x", "baseball_center_y"]].to_numpy()).ravel())
        return np.concatenate(out)

    sizes = [2 * len(s) for _, s, _, _ in prep]
    S = lil_matrix((sum(sizes), len(x0)), dtype=int)
    r0 = 0
    for k, m in enumerate(sizes):
        S[r0:r0 + m, 0:3] = 1
        S[r0:r0 + m, 3 + 5 * k: 8 + 5 * k] = 1
        r0 += m
    fit = least_squares(res, x0, bounds=(lo, hi), x_scale=xs, jac_sparsity=S,
                        loss="soft_l1", f_scale=2.0, xtol=1e-10, ftol=1e-10, gtol=1e-10,
                        max_nfev=300, verbose=0)
    return fit.x, fit


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", type=int, default=824821)
    ap.add_argument("--limit", type=int, default=60)
    ap.add_argument("--cy-seed", type=float, default=None,
                    help="start Cy here instead of the pinned 400; use to probe whether "
                         "the solve is finding a basin or just sitting where it started")
    a = ap.parse_args()

    clips = par.load_clips(a.game)[:a.limit]
    prep, votes = [], []
    for c in clips:
        runs = par.candidate_runs(c)
        if not runs or len(runs[0]) < abn.MIN_FIT_PTS + abn.HOLDOUT_FR:
            continue
        seg = runs[0]
        fit_seg, held = seg.iloc[:-abn.HOLDOUT_FR], seg.iloc[-abn.HOLDOUT_FR:]
        x_box = abn._fit(c, fit_seg, True)
        prep.append((c, fit_seg, held, x_box[6]))
        votes.append(x_box[:2])
    assert prep, "no clips had a long enough ball run"
    med = np.median(np.array(votes), axis=0)
    C_game = np.array([med[0], ph.CY_FT, med[1]])          # box-assisted answer, Cy pinned

    cy0 = ph.CY_FT if a.cy_seed is None else a.cy_seed
    print(f"game {a.game}: {len(prep)} clips, "
          f"{sum(2 * len(s) for _, s, _, _ in prep)} residuals")
    print(f"PINNED   solve: {2 + 5 * len(prep)} params (Cy fixed at {ph.CY_FT})")
    print(f"FREE-CY  solve: {3 + 5 * len(prep)} params (Cy seeded at {cy0}, "
          f"bounds [{CY_LO}, {CY_HI}])", flush=True)

    xp = abn.fit_joint(prep, med[0], med[1])
    C_pin = np.array([xp[0], ph.CY_FT, xp[1]])
    xf, fit = fit_joint_freecy(prep, med[0], cy0, med[1])
    C_free = np.array([xf[0], xf[1], xf[2]])

    rows = []
    for k, (c, fit_seg, held, t0_box) in enumerate(prep):
        pose_box = np.concatenate([C_game, par.fit_ptz(c, C_game)])
        pose_p = np.concatenate([C_pin, xp[2 + 5 * k: 6 + 5 * k]])
        pose_f = np.concatenate([C_free, xf[3 + 5 * k: 7 + 5 * k]])
        row = {"play_id": c["play_id"], "n_fit": len(fit_seg)}
        for name, pose, t0 in (("BOX", pose_box, t0_box),
                               ("PINNED", pose_p, xp[6 + 5 * k]),
                               ("FREE_CY", pose_f, xf[7 + 5 * k])):
            bx, bz = abn.box_corner_err_in(c, pose)
            row |= {f"{name}_box_x": bx, f"{name}_box_z": bz,
                    f"{name}_glove_gap": abn.glove_plane_gap_in(c, pose, pose_box)}
        rows.append(row)
    d = pd.DataFrame(rows)

    print(f"\n{'variant':10} {'BOX CORNERS |x|,|z| in':>24} {'glove-plane gap in':>20}")
    for name in ("BOX", "PINNED", "FREE_CY"):
        print(f"  {name:8} {d[f'{name}_box_x'].abs().median():12.3f} "
              f"{d[f'{name}_box_z'].abs().median():11.3f} "
              f"{d[f'{name}_glove_gap'].median():19.3f}")

    print("\ncamera position (ft):")
    print(f"  BOX game solve   Cx {C_game[0]:8.3f}  Cy {C_game[1]:8.3f}  Cz {C_game[2]:8.3f}")
    print(f"  PINNED joint     Cx {C_pin[0]:8.3f}  Cy {C_pin[1]:8.3f}  Cz {C_pin[2]:8.3f}"
          f"   off {abs(C_pin[0] - C_game[0]):5.2f} /     -- / {abs(C_pin[2] - C_game[2]):5.2f}")
    print(f"  FREE_CY joint    Cx {C_free[0]:8.3f}  Cy {C_free[1]:8.3f}  Cz {C_free[2]:8.3f}"
          f"   off {abs(C_free[0] - C_game[0]):5.2f} / {abs(C_free[1] - C_game[1]):6.2f} / "
          f"{abs(C_free[2] - C_game[2]):5.2f}")
    railed = (xf[1] <= CY_LO + 1) or (xf[1] >= CY_HI - 1)
    print(f"  FREE_CY railed at a Cy bound: {railed}   "
          f"final cost {fit.cost:.1f}  nfev {fit.nfev}  status {fit.status}")

    out = os.path.join(ROOT, "output", f"free_cy_{a.game}.csv")
    d.to_csv(out, index=False)
    print(out)


if __name__ == "__main__":
    main()
