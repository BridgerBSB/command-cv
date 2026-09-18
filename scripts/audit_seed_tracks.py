"""Render every seed track against the park's pose so a human can see which ones are balls.

THE GAP THIS CLOSES. onboard_park.py explicitly refuses to decide that a track is real -- it
renders 8 zoom tiles and stops. fit_park_pose.py stage 3 then REJECTS tracks whose
reprojection sits above 3x the park median, and that filter is good (Durham's batting-helmet
track came back at 60.8 px against 2-10 px for real ones). But nothing rendered the tracks it
rejected, or the ones it kept, at full frame. So the filter was trusted on its own arithmetic,
which is the exact shape of defect this project has been bitten by five times.

WHAT A PANEL SHOWS, and how to read it:
  GREEN circles  the observed seed track -- what the differencing found
  YELLOW dots    the trajectory projected through the park pose at this clip's best t0
  A REAL track   has green and yellow lying on each other along a smooth arc, and the green
                 circles sit on a baseball when you zoom
  A JUNK track   has green somewhere the projected arc never goes -- on the wall padding, a
                 helmet, a scoreboard digit -- and the yellow arc runs off on its own
The zoom strip beside each frame is the same points at 6x, because a 1280-wide frame
downscaled to fit a sheet cannot show a 10 px ball.

WHY t0 IS REFIT PER CLIP HERE. The saved pose carries the shared camera only; t0 (trajectory
time at sequential frame 0) is per-clip and is not stored. It is recovered by a 1-parameter
solve with the camera FROZEN, which is what refine_park_pose.py does for its clock polish.
That makes the test more forgiving ALONG the path and no more forgiving across it -- a junk
track has the wrong SHAPE and no t0 rescues it, which is precisely why the separation stays
6x wide.

THIS SCRIPT DOES NOT DECIDE ANYTHING. It prints the separation and writes a sheet. Read the
sheet before the labels built on these tracks go anywhere near a training set.

    python scripts/fit_park_pose.py --game 815419 --dir data/clips/aaa_815419   # writes pose
    python scripts/audit_seed_tracks.py --game 815419 --dir data/clips/aaa_815419
"""
import argparse
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
TILE = 132          # zoom tile edge, px
ZOOM = 6            # magnification inside a tile
FRAME_W = 640       # full frame is downscaled to this for the sheet


def fit_t0(clip, C, pan, tilt, roll, f):
    """Best trajectory-time offset for this clip with the camera frozen.

    Seeded from -(first tracked frame)/fps: the ball is in frame for the last part of the
    flight, so t0 is about minus that. Bounded +-1.5 s, the same window fit_park_pose uses,
    so a junk track cannot be rescued by sliding the clock into a different pitch.
    """
    fr, obs = clip["fr"], clip["obs"]
    t0 = -float(fr[0]) / clip["fps"]

    def res(z):
        t = z[0] + fr / clip["fps"]
        return (ph.project(C, pan, tilt, roll, f, clip["traj"](t)) - obs).ravel()

    r = least_squares(res, [t0], bounds=([t0 - 1.5], [t0 + 1.5]), xtol=1e-12, ftol=1e-12)
    e = np.hypot(*res(r.x).reshape(-1, 2).T)
    return float(r.x[0]), float(np.sqrt((e ** 2).mean())), e


def read_frames(path, wanted):
    """Sequential decode only -- never cap.set(POS_FRAMES).

    A seek lands up to 2 frames early on this footage, and mixing the two conventions is a
    defect class this project has hit three times (.claude/rules/index-and-time-origins.md).
    """
    cap = cv2.VideoCapture(path)
    out, i, top = {}, 0, max(wanted)
    while True:
        ok, f = cap.read()
        if not ok or i > top:
            break
        if i in wanted:
            out[i] = f
        i += 1
    cap.release()
    return out


def panel(path, clip, proj, e, verdict):
    """One row: the full frame with both tracks drawn, then 4 zoom tiles."""
    fr = clip["fr"].astype(int)
    mid = int(fr[len(fr) // 2])
    picks = [int(v) for v in np.linspace(fr[0], fr[-1], min(4, len(fr)))]
    frames = read_frames(path, set(picks) | {mid})
    if mid not in frames:
        return None
    big = frames[mid].copy()

    for p in proj.astype(int):
        cv2.circle(big, tuple(p), 3, (0, 255, 255), -1)
    for p in clip["obs"].astype(int):
        cv2.circle(big, tuple(p), 9, (0, 255, 0), 2)
    col = (0, 0, 255) if verdict.startswith("FLAG") else (0, 200, 0)
    cv2.putText(big, f"{clip['stem']}  f{fr[0]}-{fr[-1]}  n={len(fr)}  "
                     f"reproj {np.sqrt((e ** 2).mean()):.1f}px  {verdict}",
                (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.85, col, 2)
    h = int(big.shape[0] * FRAME_W / big.shape[1])
    big = cv2.resize(big, (FRAME_W, h))

    tiles = []
    for f_i in picks:
        t = np.zeros((TILE, TILE, 3), np.uint8)
        if f_i in frames:
            k = int(np.argmin(np.abs(fr - f_i)))
            x, y = clip["obs"][k]
            r = TILE // (2 * ZOOM)
            y0, y1 = max(0, int(y) - r), int(y) + r
            x0, x1 = max(0, int(x) - r), int(x) + r
            crop = frames[f_i][y0:y1, x0:x1]
            if crop.size:
                t = cv2.resize(crop, (TILE, TILE), interpolation=cv2.INTER_NEAREST)
                cv2.circle(t, (TILE // 2, TILE // 2), 14, (0, 255, 0), 2)
                cv2.putText(t, f"f{f_i}", (4, 16), cv2.FONT_HERSHEY_SIMPLEX,
                            0.45, (0, 255, 255), 1)
        tiles.append(t)
    while len(tiles) < 4:
        tiles.append(np.zeros((TILE, TILE, 3), np.uint8))
    # 2x2, not a 1x4 strip: a strip leaves most of the row's height black, and on a sheet
    # of a dozen rows that is most of the image.
    strip = np.vstack([np.hstack(tiles[:2]), np.hstack(tiles[2:4])])
    pad = np.zeros((max(h, strip.shape[0]), FRAME_W + strip.shape[1], 3), np.uint8)
    pad[:h, :FRAME_W] = big
    pad[:strip.shape[0], FRAME_W:] = strip
    return pad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", type=int, required=True)
    ap.add_argument("--dir", required=True)
    ap.add_argument("--pose", default=None, help="npz; default output/park_pose_<game>.npz")
    ap.add_argument("--seeds", default=None)
    ap.add_argument("--flag", type=int, default=6, help="how many worst clips to render")
    ap.add_argument("--keep", type=int, default=6, help="how many best clips to render")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()

    d = a.dir if os.path.isabs(a.dir) else os.path.join(ROOT, a.dir)
    pose = a.pose or os.path.join(ROOT, "output", f"park_pose_{a.game}.npz")
    assert os.path.exists(pose), (
        f"no pose at {pose} -- run fit_park_pose.py --game {a.game} --dir {a.dir} first")
    z = np.load(pose)
    C, (pan, tilt, roll), f = z["C"], z["ang"], float(z["f"])

    seeds = a.seeds or os.path.join(ROOT, "output", f"seed_tracks_{a.game}.csv")
    manifest = os.path.join(d, os.path.basename(d) + "_manifest.csv")
    clips = fpp.build(a.game, seeds, manifest, fps_default=FPS)
    assert clips, "no clips built -- check seeds and manifest"

    rows = []
    for c in clips:
        t0, rms, e = fit_t0(c, C, pan, tilt, roll, f)
        t = t0 + c["fr"] / c["fps"]
        rows.append(dict(stem=c["stem"], n=len(c["fr"]), f0=int(c["fr"][0]),
                         f1=int(c["fr"][-1]), rms=rms, mx=float(e.max()), t0=t0,
                         proj=ph.project(C, pan, tilt, roll, f, c["traj"](t)), e=e, clip=c))
    rows.sort(key=lambda r: r["rms"])
    med = float(np.median([r["rms"] for r in rows]))
    cut = max(3.0 * med, 12.0)

    print(f"game {a.game}  {len(rows)} seeded clips   pose C=({C[0]:.1f},{C[1]:.1f},"
          f"{C[2]:.1f}) f={f:.0f}")
    print(f"park median reprojection {med:.2f} px   flag cut {cut:.1f} px "
          f"(3x median, floor 12)\n")
    print(f"  {'clip':24} {'n':>3} {'frames':>11} {'reproj':>8} {'max':>8}  verdict")
    flagged = []
    for r in rows:
        v = "FLAG" if r["rms"] > cut else "ok"
        if v == "FLAG":
            flagged.append(r)
        print(f"  {r['stem']:24} {r['n']:3d} {r['f0']:5d}-{r['f1']:<5d} "
              f"{r['rms']:8.2f} {r['mx']:8.2f}  {v}")

    good = [r for r in rows if r["rms"] <= cut]
    if flagged and good:
        sep = min(x["rms"] for x in flagged) / max(1e-6, np.median([x["rms"] for x in good]))
        print(f"\nseparation: worst kept {max(x['rms'] for x in good):.2f} px, "
              f"best flagged {min(x['rms'] for x in flagged):.2f} px  "
              f"({sep:.1f}x the kept median)")
        print("A narrow separation means the cut is guessing. Durham's known-junk helmet "
              "track sat\nat 6x. Read the sheet, not this number.")
    else:
        print(f"\n{len(flagged)} flagged, {len(good)} kept -- no separation to report.")

    # Render the WORST (should be junk) and the BEST (should plainly be baseballs). Both
    # halves matter: a filter is only trustworthy if what it drops is bad AND what it keeps
    # is good, and only the second half can catch a cut that is throwing away real tracks.
    #
    # THE BANNER STATES THE CUT'S VERDICT, NOT THE ROW'S POSITION. The first version of
    # this stamped "FLAG" on all `--flag` worst rows, so an Isotopes sheet with ONE track
    # over the cut showed six panels reading FLAG -- contradicting its own table two lines
    # up. A label that says the same thing for every row in a group is not reporting the
    # test, it is reporting the sort order.
    picks = ([("worst", r) for r in rows[::-1][:a.flag]]
             + [("best", r) for r in rows[:a.keep]])
    sheets = []
    for group, r in picks:
        verdict = f"{'FLAG' if r['rms'] > cut else 'ok'}  ({group} of {len(rows)})"
        p = panel(os.path.join(d, r["stem"] + ".mp4"), r["clip"], r["proj"], r["e"], verdict)
        if p is not None:
            sheets.append(p)
    if sheets:
        w = max(s.shape[1] for s in sheets)
        sheets = [cv2.copyMakeBorder(s, 2, 2, 0, w - s.shape[1], cv2.BORDER_CONSTANT,
                                     value=(40, 40, 40)) for s in sheets]
        out = os.path.join(ROOT, "output", f"audit_tracks_{a.game}{a.tag}.png")
        cv2.imwrite(out, np.vstack(sheets))
        print(f"\n{out}")
        print("Top rows are the WORST-reprojecting tracks and should be junk; bottom rows "
              "are the\nBEST and should be baseballs. Green = observed, yellow = projected.")

    pd.DataFrame([{k: r[k] for k in ("stem", "n", "f0", "f1", "rms", "mx", "t0")}
                  for r in rows]).to_csv(
        os.path.join(ROOT, "output", f"audit_tracks_{a.game}{a.tag}.csv"), index=False)


if __name__ == "__main__":
    main()
