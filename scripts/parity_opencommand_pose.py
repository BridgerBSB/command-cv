"""Parity test: does OUR pinhole solve reproduce OpenCommand's published camera poses
when given THEIR inputs for one game?

If it does not, nothing we later build on src/pinhole.py can be trusted, so this runs
before any of our own detections are swapped in.

Inputs (public, data/opencommand/, gitignored; CC BY-NC-SA, read not shipped):
  pbp_info_2026_824821.csv            9-param trajectory + reference zone per pitch
  strikezone_tracking_2026.csv.gz     zone box pixels per clip
  gloveball_tracks_2026_824821.csv.gz ball pixels per frame
  camera_poses_2026.csv.gz            their answer

Method, step for step against solve_camera_pose.py @ 984d26e1:
  phase 1  every plausible ball run: 7-param fit (Cx, Cz, pan, tilt, roll, f, t0),
           Cy pinned 400, 4 box corners at weight 3, soft_l1 loss f_scale 2.
           Runs failing 3 px RMSE retry from the median position; keep lower cost.
           Best run per clip votes if RMSE < 3 px; C = median of votes (min 2).
  phase 2  per clip: pan, tilt, roll, f from the 4 box corners alone, C frozen.
  check    reproj_px + ball-at-plate error in inches, same formulas.
Their optimizer is a hand-written batched Levenberg-Marquardt; ours is scipy's
least_squares with the same loss, bounds and scaling, so tiny differences on
nonconvex clips are expected and are reported, not hidden.

    python scripts/parity_opencommand_pose.py [--game 824821]
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src import pinhole as ph  # noqa: E402

OC = os.path.join(os.path.dirname(__file__), "..", "data", "opencommand")
VOTE_RMSE_PX, MIN_VOTES, RELEASE_HALF_S, MAX_RUN_GAP_FR = 3.0, 2, 0.60, 4


def load_clips(game, season=2026):
    pbp = pd.read_csv(os.path.join(OC, f"pbp_info_{season}_{game}.csv")).set_index("play_id")
    zones = pd.read_csv(os.path.join(OC, f"strikezone_tracking_{season}.csv.gz"))
    zones = zones[zones.game_pk == game].dropna(subset=["x_left", "y_top", "x_right", "y_bottom"])
    tracks = dict(tuple(pd.read_csv(os.path.join(OC, f"gloveball_tracks_{season}_{game}.csv.gz")).groupby("play_id")))
    zy = ph.zone_y(season)
    clips = []
    for _, b in zones.iterrows():
        pid = b.play_id
        if pid not in pbp.index or pid not in tracks:
            continue
        p, df = pbp.loc[pid], tracks[pid]
        fps = float(df.fps.iloc[0])
        roots = np.roots([0.5 * p.ay, p.vy0, p.y0 - zy])
        T_plate = float(min(r.real for r in roots if np.isreal(r) and r.real > 0))
        box_world = np.array([[x, zy, z] for z in (p.sz_top_ref, p.sz_bot_ref) for x in (-17 / 24, 17 / 24)])
        box_obs = np.array([[b.x_right, b.y_top], [b.x_left, b.y_top],
                            [b.x_right, b.y_bottom], [b.x_left, b.y_bottom]])
        rel = df.release_s_hint.dropna()
        clips.append(dict(play_id=pid, p=p, df=df, fps=fps, T_plate=T_plate, box_world=box_world,
                          box_obs=box_obs, f0=(b.x_right - b.x_left) * (ph.CY_FT * 12) / 17,
                          rel=float(rel.iloc[0]) if len(rel) else np.nan, traj=ph.statcast_traj(p)))
    return clips


def candidate_runs(c):
    ball = c["df"][c["df"].baseball_center_x.notna()][["frame_idx", "baseball_center_x", "baseball_center_y"]]
    ball = ball.sort_values("frame_idx")
    if len(ball) < 8 or np.isnan(c["rel"]):
        return []
    fr = ball.frame_idx.to_numpy()
    runs = np.split(np.arange(len(fr)), np.where(np.diff(fr) > MAX_RUN_GAP_FR)[0] + 1)
    lo, hi = c["rel"] - RELEASE_HALF_S, c["rel"] + RELEASE_HALF_S
    end_lo, end_hi = (lo + c["T_plate"] - 0.15) * c["fps"], (hi + c["T_plate"] + 0.35) * c["fps"]
    out = []
    for r in runs:
        seg = ball.iloc[r]
        if not end_lo <= seg.frame_idx.max() <= end_hi:
            continue
        seg = seg[seg.frame_idx >= seg.frame_idx.max() - int((c["T_plate"] + 0.10) * c["fps"])]
        if len(seg) >= 8:
            out.append(seg)
    return sorted(out, key=len, reverse=True)[:4]


def res_position(x, c, seg):
    t = x[6] + seg.frame_idx.to_numpy() / c["fps"]
    pts = np.vstack([c["traj"](t), c["box_world"]])
    obs = np.vstack([seg[["baseball_center_x", "baseball_center_y"]].to_numpy(), c["box_obs"]])
    w = np.r_[np.ones(len(seg)), np.full(4, 3.0)]
    C = np.array([x[0], ph.CY_FT, x[1]])
    return ((ph.project(C, x[2], x[3], x[4], x[5], pts) - obs) * w[:, None]).ravel()


def ball_rmse(x, c, seg):
    r = res_position(x, c, seg).reshape(-1, 2)[:len(seg)]
    return float(np.sqrt((r ** 2).sum() / (2 * len(seg))))


def fit_position(c, seg, cx0=0.0, cz0=20.0):
    f0, rel = c["f0"], c["rel"]
    t_lo, t_hi = -rel - RELEASE_HALF_S, -rel + RELEASE_HALF_S
    t0 = float(np.clip(c["T_plate"] - seg.frame_idx.max() / c["fps"], t_lo, t_hi))
    lo = [-200, 5, -0.15, -0.15, -0.15, f0 / 2, t_lo]
    hi = [200, 120, 0.15, 0.15, 0.15, f0 * 2, t_hi]
    x0 = np.clip([cx0, cz0, 0, 0, 0, f0, t0], np.array(lo) + 1e-9, np.array(hi) - 1e-9)
    r = least_squares(res_position, x0, args=(c, seg), bounds=(lo, hi), loss="soft_l1", f_scale=2.0,
                      x_scale=[30, 15, 0.02, 0.02, 0.02, f0 * 0.3, 0.05], xtol=1e-12, ftol=1e-12, gtol=1e-12,
                      max_nfev=2000)
    return r.x, r.cost


def fit_ptz(c, C):
    f0 = c["f0"]

    def res(q):
        return (ph.project(C, q[0], q[1], q[2], q[3], c["box_world"]) - c["box_obs"]).ravel()
    r = least_squares(res, [0, 0, 0, f0], bounds=([-0.15, -0.15, -0.15, f0 / 2], [0.15, 0.15, 0.15, f0 * 2]),
                      x_scale=[0.02, 0.02, 0.02, f0 * 0.3], xtol=1e-12, ftol=1e-12, gtol=1e-12)
    return r.x


def shipped_check(c, pose, t0, seg):
    px = seg[["baseball_center_x", "baseball_center_y"]].to_numpy()
    t = t0 + seg.frame_idx.to_numpy() / c["fps"]
    C, (pan, tilt, roll, f) = pose[:3], pose[3:]
    d = ph.project(C, pan, tilt, roll, f, c["traj"](t)) - px
    j = int(np.abs(t - c["T_plate"]).argmin())
    truth = c["traj"](t[j])
    bx, bz = ph.unproject(C, pan, tilt, roll, f, px[j], truth[1])
    return float(np.sqrt((d ** 2).mean())), (bx - truth[0]) * 12, (bz - truth[2]) * 12


def solve_game(game):
    clips = load_clips(game)
    pairs = [(i, seg) for i, c in enumerate(clips) for seg in candidate_runs(c)]
    fits = [fit_position(clips[i], seg) for i, seg in pairs]
    rm = np.array([ball_rmse(x, clips[i], seg) for (i, seg), (x, _) in zip(pairs, fits)])
    med = np.median([x[:2] for x, _ in fits], axis=0)
    for k, ((i, seg), (x, cost)) in enumerate(zip(pairs, fits)):     # restart pass
        if rm[k] >= VOTE_RMSE_PX:
            x2, cost2 = fit_position(clips[i], seg, med[0], med[1])
            if cost2 < cost:
                fits[k], rm[k] = (x2, cost2), ball_rmse(x2, clips[i], seg)
    best = {}
    for k, (i, seg) in enumerate(pairs):
        if i not in best or rm[k] < best[i][1]:
            best[i] = (fits[k][0], rm[k], seg)
    votes = [b[0][:2] for b in best.values() if b[1] < VOTE_RMSE_PX]
    assert len(votes) >= MIN_VOTES, f"only {len(votes)} votes"
    v = np.median(votes, axis=0)
    C = np.array([v[0], ph.CY_FT, v[1]])
    rows = []
    for i, c in enumerate(clips):
        q = fit_ptz(c, C)
        pose = np.concatenate([C, q])
        chk = shipped_check(c, pose, best[i][0][6], best[i][2]) if i in best else (np.nan,) * 3
        rows.append(dict(play_id=c["play_id"], Cx=C[0], Cz=C[2], pan=q[0], tilt=q[1], roll=q[2], f_px=q[3],
                         ball_rmse_px=best[i][1] if i in best else np.nan,
                         reproj_px=chk[0], balltp_err_x_in=chk[1], balltp_err_z_in=chk[2]))
    return pd.DataFrame(rows), len(pairs), len(votes)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", type=int, default=824821)
    a = ap.parse_args()
    ours, n_runs, n_votes = solve_game(a.game)
    theirs = pd.read_csv(os.path.join(OC, "camera_poses_2026.csv.gz"))
    theirs = theirs[theirs.game_pk == a.game]
    m = ours.merge(theirs, on="play_id", suffixes=("_us", "_oc"))
    print(f"game {a.game}: clips ours {len(ours)} | theirs {len(theirs)} | matched {len(m)} | "
          f"candidate runs {n_runs} | votes {n_votes}")
    assert len(m) > 0, "no clips matched - parity is vacuous"
    print(f"camera position  ours Cx {ours.Cx.iloc[0]:.4f} Cz {ours.Cz.iloc[0]:.4f}   "
          f"theirs Cx {theirs.Cx.iloc[0]:.4f} Cz {theirs.Cz.iloc[0]:.4f}")
    print("\nper-clip |ours - theirs|            median        p90         max")
    for col, unit, scale in (("pan", "rad", 1), ("tilt", "rad", 1), ("roll", "rad", 1), ("f_px", "px", 1),
                             ("reproj_px", "px", 1), ("balltp_err_x_in", "in", 1), ("balltp_err_z_in", "in", 1)):
        d = (m[f"{col}_us"] - m[f"{col}_oc"]).abs().dropna() * scale
        if len(d):
            print(f"  {col:16} {unit:4} {d.median():12.6f} {d.quantile(.9):12.6f} {d.max():12.6f}   (n={len(d)})")
    for side in ("us", "oc"):
        ex, ez = m[f"balltp_err_x_in_{side}"].abs(), m[f"balltp_err_z_in_{side}"].abs()
        print(f"  ball-at-plate |err| median ({'ours' if side == 'us' else 'theirs'}): "
              f"X {ex.median():.3f} in  Z {ez.median():.3f} in  (n={ex.notna().sum()})")
    out = os.path.join(os.path.dirname(__file__), "..", "output", f"parity_opencommand_{a.game}.csv")
    m.to_csv(out, index=False)
    print(out)


if __name__ == "__main__":
    main()
