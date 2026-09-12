"""Find the ball in broadcast video WITHOUT a ball detector, and emit free training labels.

The camera pose and the Statcast 9-param trajectory are both known, so the ball's pixel path
is known up to ONE unknown: which frame of the clip the pitch crosses the plate. Slide that
offset and score how strongly a small bright moving object sits on the predicted path.

Why the score is shaped the way it is (both earlier versions failed):
  v0  raw frame-difference along the path. REJECTED: it rewards any path crossing busy pixels,
      so the catcher and batter win. A different pitch's trajectory scored as well as the
      right one (median 8.55 vs 12.16, and on one clip the wrong one won 18.75 to 4.12).
  v1  white top-hat (keeps objects smaller than the kernel) + local contrast against a ring
      around the point, and the MEDIAN over the run rather than the mean, so most of the
      flight has to agree. This is what ships.

Controls, run with --controls: the same path MIRRORED left-right and the same path REVERSED in
time. Both are physically impossible, both keep the speed, length and pixel scale, so only the
path differs. "Another pitch from the same game" is NOT a control here: pitches trace nearly the
same screen path, and it does not separate.

Labels (--labels): at the winning offset, the projected ball pixel for every flight frame, which
is a labeled ball position per frame with no hand annotation. That is the training set for a
real ball detector, which is what A+ needs, where there is no trajectory to lean on.

    python scripts/ball_clock_align.py --game 824821 --limit 40 [--controls] [--render out.png]
                                       [--labels output/ball_labels_824821.csv]
"""
import argparse
import os
import sys

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src import pinhole as ph  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
OC = os.path.join(ROOT, "data", "opencommand")
TOPHAT_K = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
FLIGHT_S = 0.32          # of flight to score, ending at the plate
WIN_LO, WIN_HI = 120, 300
MIN_PTS = 20
PLATE_FRONT = 17 / 12


def motion_stack(clip, lo=WIN_LO, hi=WIN_HI):
    """Frame-to-frame difference, top-hatted so only small bright objects survive.

    Streams: only the previous frame and the diffs inside [lo, hi) are held. The first version
    kept every decoded frame AND the diffs, about 500 MB per clip, which is what got a
    concurrent training run OOM-killed (Sep 12 2026). Diffs alone are ~170 MB."""
    cap = cv2.VideoCapture(clip)
    fps = cap.get(cv2.CAP_PROP_FPS) or 59.94
    m, prev, i, n = {}, None, 0, 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        if lo <= i < hi and prev is not None:
            m[i] = cv2.morphologyEx(cv2.absdiff(g, prev), cv2.MORPH_TOPHAT, TOPHAT_K)
        prev = g          # one frame, 0.9 MB; the diffs are what cost memory
        i = n = i + 1
    cap.release()
    return m, fps, lo, min(hi, n - 1), n


def local_contrast(img, x, y, r_in=4, r_out=18):
    """Brightest pixel within r_in of (x, y), minus the median of the ring at r_out.
    A point inside a large moving region scores ~0; a lone small object scores high."""
    x, y = int(round(x)), int(round(y))
    if not (r_out <= x < ph.W - r_out and r_out <= y < ph.H - r_out):
        return None
    p = img[y - r_out:y + r_out + 1, x - r_out:x + r_out + 1].astype(np.float32)
    yy, xx = np.mgrid[-r_out:r_out + 1, -r_out:r_out + 1]
    d = np.hypot(xx, yy)
    return float(p[d <= r_in].max() - np.median(p[(d >= r_out - 6) & (d <= r_out)]))


def plate_time(p, y_plane=PLATE_FRONT):
    roots = np.roots([0.5 * p.ay, p.vy0, p.y0 - y_plane])
    return float(min(t.real for t in roots if np.isreal(t) and t.real > 0))


def scan(pose_row, points_at, m, fps, lo, hi):
    """(best score, best plate frame, baseline) sliding the path over every plate frame."""
    C = np.array([pose_row.Cx, ph.CY_FT, pose_row.Cz])
    args = (C, pose_row.pan, pose_row.tilt, pose_row.roll, pose_row.f_px)
    idx = np.arange(-int(FLIGHT_S * fps), 1)
    out = []
    for f0 in np.arange(lo + 5, hi - 5, 1.0):
        px = ph.project(*args, points_at(idx / fps))
        fs = (f0 + idx).astype(int)
        v = [local_contrast(m[fi], px[k, 0], px[k, 1]) for k, fi in enumerate(fs) if fi in m]
        v = [q for q in v if q is not None]
        if len(v) >= MIN_PTS:
            out.append((float(np.median(v)), float(f0)))
    if not out:
        return np.nan, np.nan, np.nan
    best = max(out)
    return best[0], best[1], float(np.median([o[0] for o in out]))


BALL_DIAMETER_FT = 2.9 / 12


def ball_pixels(pose_row, p, f0, fps):
    """(frame_idx, x, y, diameter_px) for the flight at the winning offset: the free labels.

    A detector needs a BOX, not a point, so the ball's apparent size comes from the same
    geometry: a 2.9 in ball at depth d along the optical axis subtends f * (2.9/12) / d
    pixels. It grows as the ball approaches, which a fixed-size box would get wrong."""
    C = np.array([pose_row.Cx, ph.CY_FT, pose_row.Cz])
    args = (C, pose_row.pan, pose_row.tilt, pose_row.roll, pose_row.f_px)
    traj = ph.statcast_traj(p)
    idx = np.arange(-int(FLIGHT_S * fps), 1)
    world = traj(plate_time(p) + idx / fps)
    px = ph.project(*args, world)
    fwd = ph.camera(C, pose_row.pan, pose_row.tilt, pose_row.roll)[2]
    depth = (world - C) @ fwd
    dia = pose_row.f_px * BALL_DIAMETER_FT / depth
    return [(int(f0 + i), float(x), float(y), float(d))
            for i, (x, y), d in zip(idx, px, dia)]


def load_inputs(game, poses_csv):
    poses = pd.read_csv(poses_csv)
    ren = {c: c.replace("_us", "") for c in poses.columns if c.endswith("_us")}
    poses = poses.rename(columns=ren).set_index("play_id")          # our side of the parity file
    pbp = pd.read_csv(os.path.join(OC, f"pbp_info_2026_{game}.csv")).set_index("play_id")
    return poses, pbp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", type=int, default=824821)
    ap.add_argument("--limit", type=int, default=40)
    ap.add_argument("--controls", action="store_true", help="also score mirrored + time-reversed paths")
    ap.add_argument("--render", default=None, help="PNG: predicted ball circled on real frames")
    ap.add_argument("--labels", default=None, help="CSV of per-frame ball pixels at the winning offset")
    ap.add_argument("--poses", default=os.path.join(ROOT, "output", "parity_opencommand_824821.csv"))
    a = ap.parse_args()
    poses, pbp = load_inputs(a.game, a.poses)
    ids = [i for i in poses.index if i in pbp.index][:a.limit]
    clips = os.path.join(ROOT, "data", "clips", f"mlb_{a.game}")
    rows, labels = [], []
    for pid in ids:
        clip = os.path.join(clips, f"{pid}_BROADCAST.mp4")
        if not os.path.exists(clip):
            continue
        p, r = pbp.loc[pid], poses.loc[pid]
        m, fps, lo, hi, nfr = motion_stack(clip)
        traj, T = ph.statcast_traj(p), plate_time(pbp.loc[pid])
        own, f0, base = scan(r, lambda d: traj(T + d), m, fps, lo, hi)
        row = dict(play_id=pid, pitch=p.pitch_type, frames=nfr, score=own, baseline=base, plate_frame=f0)
        if a.controls:
            pm = p.copy()
            pm["x0"], pm["vx0"], pm["ax"] = -p.x0, -p.vx0, -p.ax
            tm = ph.statcast_traj(pm)
            row["mirrored"] = scan(r, lambda d: tm(T + d), m, fps, lo, hi)[0]
            row["reversed"] = scan(r, lambda d: traj(T - d), m, fps, lo, hi)[0]
        rows.append(row)
        if a.labels and np.isfinite(f0):
            labels.append((pid, f0, ball_pixels(r, p, f0, fps)))
        print(row, flush=True)
    d = pd.DataFrame(rows)
    print(f"\nclips {len(d)} | score median {d.score.median():.1f} | baseline median {d.baseline.median():.1f} "
          f"| score/baseline median {(d.score / d.baseline.clip(lower=1e-6)).median():.1f}x")
    print(f"plate frame: median {d.plate_frame.median():.0f}, range {d.plate_frame.min():.0f}-{d.plate_frame.max():.0f}")
    if a.controls:
        print(f"controls: mirrored median {d.mirrored.median():.1f}, reversed median {d['reversed'].median():.1f}")
        # A control can be UNSCORABLE: mirrored/reversed paths can leave the frame, so fewer
        # than MIN_PTS samples land and scan() returns NaN. Counting that as a loss understates
        # the separation (it read "39/40" on a run where nothing lost). Report the denominator.
        for name in ("mirrored", "reversed"):
            ok = d[name].notna()
            wins = int((d.score[ok] > d[name][ok]).sum())
            print(f"  own beats {name} {wins}/{int(ok.sum())} scored"
                  + (f"  ({int((~ok).sum())} not scorable: path leaves the frame)" if (~ok).any() else ""))
            assert ok.sum() == 0 or wins >= 0.9 * ok.sum(), f"{name} control not separated - score is not specific"
    if a.labels:
        # Clips from one game are cut the same way, so the plate crossing lands in a tight frame
        # band: 277 of 280 inside 198-215 on game 824821. The three strays (231, 224, 290) are
        # alignments that locked onto the wrong thing, and their labels would be boxes drawn on
        # empty grass. Gate on distance from the game's OWN median rather than a hardcoded band,
        # so a differently-cut camera calibrates itself.
        PLATE_FRAME_TOL = 10
        med = float(np.median([f0 for _, f0, _ in labels]))
        keep = [(pid, f0, px) for pid, f0, px in labels if abs(f0 - med) <= PLATE_FRAME_TOL]
        dropped = [(pid, f0) for pid, f0, _ in labels if abs(f0 - med) > PLATE_FRAME_TOL]
        flat = [dict(play_id=pid, frame_idx=fi, x=x, y=y, diameter_px=dia)
                for pid, _, px in keep for fi, x, y, dia in px]
        pd.DataFrame(flat).to_csv(a.labels, index=False)
        print(f"{len(flat)} ball-pixel labels over {len(keep)} clips -> {a.labels}")
        if dropped:
            print(f"  dropped {len(dropped)} clips whose plate frame is >{PLATE_FRAME_TOL} frames "
                  f"from the game median {med:.0f}: "
                  + ", ".join(f"{p[:8]}@{f:.0f}" for p, f in dropped))
    if a.render:
        tiles = []
        for pid in [r["play_id"] for r in rows[:3]]:
            p, r = pbp.loc[pid], poses.loc[pid]
            f0 = int(d.set_index("play_id").loc[pid, "plate_frame"])
            cap = cv2.VideoCapture(os.path.join(clips, f"{pid}_BROADCAST.mp4"))
            fps = cap.get(cv2.CAP_PROP_FPS) or 59.94
            fr = []
            while True:
                ok, f = cap.read()
                if not ok:
                    break
                fr.append(f)
            row = []
            for off in (-16, -12, -8, -4, 0):
                fi = f0 + off
                if fi >= len(fr):
                    continue
                C = np.array([r.Cx, ph.CY_FT, r.Cz])
                x, y = ph.project(C, r.pan, r.tilt, r.roll, r.f_px,
                                  ph.statcast_traj(p)(plate_time(p) + off / fps))
                im = fr[fi].copy()
                cv2.circle(im, (int(round(x)), int(round(y))), 13, (0, 255, 0), 2, cv2.LINE_AA)
                x0 = max(0, min(int(x - 150), ph.W - 300)); y0 = max(0, min(int(y - 110), ph.H - 220))
                t = cv2.resize(im[y0:y0 + 220, x0:x0 + 300], (600, 440), interpolation=cv2.INTER_NEAREST)
                cv2.rectangle(t, (0, 0), (600, 24), (0, 0, 0), -1)
                cv2.putText(t, f"{pid[:8]} {p.pitch_type} f{fi} (plate{off:+d})", (6, 17), 0, 0.5,
                            (0, 255, 255), 1, cv2.LINE_AA)
                row.append(t)
            if row:
                tiles.append(np.hstack(row[:5]))
        cv2.imwrite(a.render, np.vstack(tiles))
        print(a.render)


if __name__ == "__main__":
    main()
