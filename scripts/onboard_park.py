"""Onboard a new park: recover ball tracks, correct the half-frame lag, render for eye-check.

THE REPEATABLE STEP. Every park needs seed ball tracks before its camera can be fitted, and
until now that was a hand process done twice (Asheville by eye, then its hard tail). This is
that process as a tool, so a new level or park is one command plus one look.

WHAT IT DOES NOT DO: decide that the tracks are real. It renders a verification sheet and
stops. LOOK AT IT before running the pose fit. Three separate times in this project a
good-looking aggregate was overturned by rendering it -- a "track" that was the batter's
uniform, a 21/24 harvest at 10.9x baseline with every label on grass, and a 2.23 px accuracy
measured on a selection-biased sample. The sheet is not a formality.

THE HALF-FRAME CORRECTION IS APPLIED HERE AND MATTERS MORE THAN THE DETECTOR SETTINGS.
absdiff(frame_i, frame_i-1) leaves TWO lobes for a moving ball, so the centroid lags the true
position by half the inter-frame step -- measured +4.71 px against a predicted 4.96 px on 106
hand-verified points, with no across-track component. Uncorrected it is a systematic bias
LARGER than the random error, and because ground truth was extracted the same way it was
invisible to every accuracy check. See .claude/rules/index-and-time-origins.md.

WHAT TRANSFERS AND WHAT DOES NOT. Asheville (A+) and Durham (AAA) are both SIDE views --
pitcher left, batter and catcher right -- and Durham needed zero tuning: same lane, same
threshold, same gates, ball verified in 6/6 rendered clips. Greensboro (822246) is a
different archetype, shot from BEHIND the pitcher with the ball moving AWAY from camera and
shrinking against a brick wall, and these gates (rightward, mostly-horizontal) do NOT apply
there. Pass --direction -1 or widen --lane for a mirrored park; a behind-the-pitcher park
needs its own gates and is not solved.

FRAME INDICES ARE SEQUENTIAL DECODE, never cap.set(POS_FRAMES) -- a seek lands up to 2
frames early and mixing the two conventions is a defect class this project has hit three
times.

    python scripts/onboard_park.py --dir data/clips/aaa_816698 --game 816698
    python scripts/onboard_park.py --dir data/clips/aa_817407 --game 817407 --lo 120 --hi 300
"""
import argparse
import os
import sys

import cv2
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from aplus_seed_tracks import TOPHAT_K, THRESH, MIN_AREA, MAX_AREA, MAX_SIDE  # noqa: E402

STEP_LO, STEP_HI = 3.0, 22.0
MIN_RUN = 5
MAX_QUAD_RMS = 2.5


def best_run(rows, direction):
    best, cur = [], []
    for r in rows:
        if cur:
            dx, dy = r[1] - cur[-1][1], r[2] - cur[-1][2]
            dt = r[0] - cur[-1][0]
            step = np.hypot(dx, dy) / max(dt, 1)
            ok = (dt == 1 and direction * dx > 0 and STEP_LO <= step <= STEP_HI
                  and abs(dy) <= abs(dx) * 1.8)
            if not ok:
                if len(cur) > len(best):
                    best = cur
                cur = []
        cur.append(r)
    return cur if len(cur) > len(best) else best


def quad_rms(run):
    t = np.array([p[0] for p in run], float)
    t -= t.mean()
    res = []
    for ax in (1, 2):
        v = np.array([p[ax] for p in run], float)
        res.append(v - np.polyval(np.polyfit(t, v, 2), t))
    return float(np.sqrt(np.mean(np.square(np.concatenate(res)))))


def half_frame_correct(run):
    """Shift each point forward by half the local displacement. See module docstring."""
    if len(run) < 2:
        return run
    out = []
    for k, (f, x, y) in enumerate(run):
        j = min(k + 1, len(run) - 1)
        i0 = k if j != k else max(k - 1, 0)
        gap = max(1.0, float(run[j][0] - run[i0][0]))
        out.append((f, x + 0.5 * (run[j][1] - run[i0][1]) / gap,
                    y + 0.5 * (run[j][2] - run[i0][2]) / gap))
    return out


def track_clip(path, lane, lo, hi, direction):
    x0, x1, y0, y1 = lane
    cap = cv2.VideoCapture(path)
    prev, i, rows = None, 0, []
    while True:
        ok, f = cap.read()
        if not ok or i > hi:
            break
        roi = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)[y0:y1, x0:x1]
        if i >= lo and prev is not None:
            th = cv2.morphologyEx(cv2.absdiff(roi, prev), cv2.MORPH_TOPHAT, TOPHAT_K)
            _, bw = cv2.threshold(th, THRESH, 255, cv2.THRESH_BINARY)
            n, lab, st, ce = cv2.connectedComponentsWithStats(bw, 8)
            c = []
            for k in range(1, n):
                bx, by, bw_, bh, ar = st[k]
                if (MIN_AREA <= ar <= MAX_AREA and 2 <= bw_ <= MAX_SIDE
                        and 2 <= bh <= MAX_SIDE and 0.35 <= bw_ / max(bh, 1) <= 2.8):
                    c.append((float(th[lab == k].mean()),
                              float(ce[k][0] + x0), float(ce[k][1] + y0)))
            c.sort(reverse=True)
            if c:
                rows.append((i, c[0][1], c[0][2]))
        prev = roi
        i += 1
    cap.release()
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--game", type=int, required=True)
    ap.add_argument("--lane", default="420,980,240,470")
    ap.add_argument("--lo", type=int, default=140)
    ap.add_argument("--hi", type=int, default=280)
    ap.add_argument("--direction", type=int, default=1, help="+1 = ball moves right")
    ap.add_argument("--limit", type=int, default=60)
    a = ap.parse_args()

    d = a.dir if os.path.isabs(a.dir) else os.path.join(ROOT, a.dir)
    lane = tuple(int(v) for v in a.lane.split(","))
    man = os.path.join(d, os.path.basename(d) + "_manifest.csv")
    m = pd.read_csv(man)
    stems = [s for s in m.stem if os.path.exists(os.path.join(d, s + ".mp4"))][:a.limit]

    rows, kept, panels = [], 0, []
    for j, s in enumerate(stems, 1):
        raw = track_clip(os.path.join(d, s + ".mp4"), lane, a.lo, a.hi, a.direction)
        run = best_run(raw, a.direction)
        if len(run) < MIN_RUN:
            continue
        rms = quad_rms(run)
        if rms > MAX_QUAD_RMS:
            continue
        run = half_frame_correct(run)
        kept += 1
        for f, x, y in run:
            rows.append(dict(stem=s, game_pk=a.game, frame=int(f), x=x, y=y))
        if len(panels) < 8:
            panels.append((s, run))
        print(f"  [{j}/{len(stems)}] {s:22} f{run[0][0]:3d}-{run[-1][0]:3d} n={len(run):2d} "
              f"rms={rms:4.2f}", flush=True)

    assert rows, "no clip produced a run -- check --lane / --lo / --hi / --direction"
    df = pd.DataFrame(rows)

    # CONSENSUS FLIGHT WINDOW. A park's clips share a broadcast cut convention, so real
    # flights cluster tightly in frame number -- Durham's land f190-215, Asheville's
    # f191-209. A run far outside that is not a pitch (a scoreboard digit at f142, a blob
    # on grass at f268). This is the one filter here that is SELF-CALIBRATING: derived from
    # the park's own distribution, never a threshold carried between parks.
    #
    # BRIGHTNESS IS DELIBERATELY NOT USED, having been measured and rejected: Asheville
    # (night) medians 246, Durham (day) 188, a known-junk scoreboard digit scores 123 and a
    # REAL ball scores 178. The distributions overlap, so any absolute cut drops true tracks
    # at dim parks. A discriminator that does not separate on the data is not a discriminator.
    mid = df.groupby("stem").frame.median()
    centre = float(mid.median())
    spread = float((mid - centre).abs().median()) or 1.0
    tol = max(12.0, 4.0 * spread)
    keep = set(mid[(mid - centre).abs() <= tol].index)
    dropped = sorted(set(mid.index) - keep)
    df = df[df.stem.isin(keep)]
    print(f"\nconsensus flight window: centre f{centre:.0f}, keeping within +-{tol:.0f} frames")
    if dropped:
        print("  dropped outside it: " + ", ".join(f"{s}@f{mid[s]:.0f}" for s in dropped))
    print("  This does NOT catch an in-window false positive (a helmet, a blob on the wall).\n"
          "  Those are left to the robust loss in the pose fit and to the render below.")

    out = os.path.join(ROOT, "output", f"seed_tracks_{a.game}.csv")
    df.to_csv(out, index=False)
    print(f"\n{df.stem.nunique()}/{len(stems)} clips kept, {len(df)} points (half-frame corrected)")
    print(out)

    tiles = []
    for s, run in panels:
        want = {int(f): (x, y) for f, x, y in run[:4]}
        cap = cv2.VideoCapture(os.path.join(d, s + ".mp4"))
        i, row = 0, []
        while True:
            ok, f = cap.read()
            if not ok or (want and i > max(want)):
                break
            if i in want:
                x, y = want[i]
                c = f[max(0, int(y) - 22):int(y) + 22, max(0, int(x) - 22):int(x) + 22]
                if c.size:
                    c = cv2.resize(c, (140, 140), interpolation=cv2.INTER_NEAREST)
                    cv2.circle(c, (70, 70), 18, (0, 0, 255), 2)
                    cv2.putText(c, f"{s[-6:]} f{i}", (3, 14), cv2.FONT_HERSHEY_SIMPLEX,
                                0.4, (0, 255, 255), 1)
                    row.append(c)
            i += 1
        cap.release()
        while len(row) < 4:
            row.append(np.zeros((140, 140, 3), np.uint8))
        tiles.append(np.hstack(row[:4]))
    png = os.path.join(ROOT, "output", f"onboard_{a.game}.png")
    cv2.imwrite(png, np.vstack(tiles))
    print(f"{png}\n\nLOOK AT IT. Every circle must contain a baseball before you fit a pose "
          "to these.")


if __name__ == "__main__":
    main()
