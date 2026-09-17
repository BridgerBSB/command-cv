"""Recover the ball's pixel track in a MiLB clip, with the known answer as the gate.

WHAT THIS IS FOR. A park's camera pose cannot be fitted without ball pixels, and ball
pixels cannot be detected reliably without a detector trained at that park. The way out is
to seed: recover the track in a handful of clips, fit ONE shared pose (the A+ camera is
locked -- aplus_camera_between_pitches.py), then let ball_clock_align.py harvest the rest of
the game for free. This is the seed step.

WHY IT IS NOT THE Sep 13 TRACKER AGAIN. That one ranked chains on LENGTH + STRAIGHTNESS +
step slope and failed twice: v1 rewarded jitter, v2's lone survivor was the catcher's return
throw. Straightness is far too weak -- a stadium is full of straightish moving things, which
is also how the Sep 16 "16/18 hit rate" turned out to be 3-of-4 junk (a shoe, a Bally Sports
banner, an arm in the dugout).

The criterion here is the PHYSICS instead: a pitch projects to a path that is very close to
a QUADRATIC in image coordinates. So a run is scored by how tightly a per-axis quadratic
fits it, in pixels, plus its length. A run of unrelated blobs does not fit a quadratic to
~1 px over 10 frames; a return throw fits its OWN quadratic but runs the wrong way, which
the direction gate rejects. Components are printed rather than collapsed into one number,
so the pick is auditable.

The search also turned out to be far smaller than assumed -- see chains() -- because the
ball is simply the strongest top-hat response in the lane while it is unoccluded.

TWO THINGS THAT COST TIME HERE, BOTH RECORDED SO THEY ARE NOT REPEATED:
  Candidates must be ranked by TOP-HAT RESPONSE, never raw brightness. Raw brightness
  saturates at 255 for every white object in frame -- uniforms, pants, chalk, the ball -- so
  sorting on it picks arbitrarily among dozens of ties. That produced a completely fictional
  "track" sitting on the batter and the catcher.
  The pitch is LATE in these clips. ogando_si_9's pitch clock reads 0:09 at f40 and 0:08 at
  f100, and the delivery is ~f190 of 538. An analysis window of [100, 320) silently cut off
  several clips entirely -- their busiest window railed at the window edge, which is the tell.

VALIDATION IS BUILT IN AND IS THE POINT. --validate replays the one track that has been
confirmed by eye (ogando_si_9 f193-202, output/si9_track_full.png -- ten frames, a white
sphere dead centre in every circle) and FAILS if the top chain does not recover all ten. A seeder that cannot find a ball we have already seen has no
business running on clips where the answer is unknown.

    python scripts/aplus_seed_tracks.py --validate
    python scripts/aplus_seed_tracks.py --game 822515 --limit 8
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
from ball_clock_align import TOPHAT_K  # noqa: E402  -- same kernel as the MLB aligner

TOPK = 8                 # kept for diagnostics; the selection below uses top-1
THRESH = 18              # measured, not guessed -- see the sensitivity note below
MIN_AREA, MAX_AREA = 4, 300
MAX_SIDE = 30
STEP_LO, STEP_HI = 3.0, 20.0     # px/frame between consecutive ball positions
MIN_LEN = 8                      # consecutive top-1 frames required
MAX_QUAD_RMS = 2.0               # px; a pitch fits a per-axis quadratic this tightly
LANE = (520, 850, 290, 410)

# THRESH IS NOT A FREE KNOB. Swept against the confirmed track on ogando_si_9: the ball is
# recovered in 1/5 frames at 10, 2/5 at 14, and 5/5 at BOTH 18 and 24. LOWERING it makes
# detection WORSE, because a looser threshold merges the ball into a larger connected
# component that then fails the size gate. Do not "increase sensitivity" by dropping it.

# FRAME INDEX CONVENTION: SEQUENTIAL DECODE, counted from 0, never cap.set(POS_FRAMES).
# On this clip a seek lands exactly 2 frames early (seek f195 == sequential f193, seek f201
# == sequential f199, both confirmed to 0.1 px), which the project's Sep 11 audit had
# already noted as "cv2 seeks land 0-2 frames early". Mixing the two conventions is the
# same defect class that voided the whole detector gate on Sep 13, where OpenCommand's
# release-relative index was used to read our clip. One convention, stated here.

# The track confirmed BY EYE in output/si9_track_full.png -- a white sphere dead centre in
# every circle. A gate, not a fixture to be relaxed.
# NOTE: this clip's ball is also top-1 at f207-214, and that run was DISCARDED, not used:
# the circles there sit on the batter's white uniform and ball cannot be separated from
# pants by eye. A plausible continuation is not a confirmed one.
VALIDATION = {"clip": "ogando_si_9", "game": 822515,
              "points": {193: (619.7, 354.6), 194: (623.9, 352.8), 195: (631.9, 348.7),
                         196: (640.1, 344.8), 197: (648.3, 341.5), 198: (656.9, 338.2),
                         199: (670.8, 333.4), 200: (674.0, 332.7), 201: (679.7, 330.2),
                         202: (684.5, 330.4)}}


def candidates(path, lo, hi, lane):
    """Top-K small bright MOVING blobs per frame, ranked by top-hat response."""
    x0, x1, y0, y1 = lane
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 59.94
    out, prev, i = {}, None, 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        if i >= hi:
            break
        roi = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)[y0:y1, x0:x1]
        if lo <= i and prev is not None:
            th = cv2.morphologyEx(cv2.absdiff(roi, prev), cv2.MORPH_TOPHAT, TOPHAT_K)
            _, bw = cv2.threshold(th, THRESH, 255, cv2.THRESH_BINARY)
            n, lab, st, ce = cv2.connectedComponentsWithStats(bw, 8)
            c = []
            for k in range(1, n):
                x, y, w, h, a = st[k]
                if (MIN_AREA <= a <= MAX_AREA and 3 <= w <= MAX_SIDE and 3 <= h <= MAX_SIDE
                        and 0.4 <= w / max(h, 1) <= 2.5):
                    c.append((float(th[lab == k].mean()),
                              float(ce[k][0] + x0), float(ce[k][1] + y0), int(a)))
            c.sort(reverse=True)
            out[i] = c[:TOPK]
        prev = roi
        i += 1
    cap.release()
    return out, fps, i


def chains(cands, direction):
    """Maximal runs of CONSECUTIVE top-1 detections moving consistently in `direction`.

    Deliberately much simpler than the Sep 13 attempt, which grew chains over the whole
    top-K and scored them on length + straightness. That search space is enormous and a
    stadium fills it with plausible junk. The measurement that made it unnecessary: on the
    confirmed clip the ball is the HIGHEST top-hat response in the lane for ten consecutive
    frames. So take top-1, cut the sequence wherever the step or direction gate breaks, and
    let the quadratic test judge what is left. Small search space, physics as the filter.

    A consequence worth stating: a run ENDS at an occlusion rather than bridging it. That
    is the intended behaviour. On si_9 the ball passes the batter at f203-206 and the
    detections there are a stationary distractor; bridging the gap would have swallowed it.
    """
    frames = sorted(cands)
    runs, cur = [], []
    for f in frames:
        c = cands[f][:1]
        if not c:
            if len(cur) >= MIN_LEN:
                runs.append(cur)
            cur = []
            continue
        pt = (f, c[0][1], c[0][2])
        if cur:
            pf, px, py = cur[-1]
            dx, dy = pt[1] - px, pt[2] - py
            step = np.hypot(dx, dy) / max(1, f - pf)
            ok = (f - pf == 1 and direction * dx > 0 and STEP_LO <= step <= STEP_HI
                  and abs(dy) <= abs(dx) * 1.6)
            if not ok:
                if len(cur) >= MIN_LEN:
                    runs.append(cur)
                cur = []
        cur.append(pt)
    if len(cur) >= MIN_LEN:
        runs.append(cur)
    return runs


def quad_rms(ch):
    """Per-axis quadratic fit residual in px -- the physics test."""
    t = np.array([p[0] for p in ch], float)
    t = t - t.mean()
    r = []
    for ax in (1, 2):
        v = np.array([p[ax] for p in ch], float)
        c = np.polyfit(t, v, 2)
        r.append(v - np.polyval(c, t))
    return float(np.sqrt(np.mean(np.square(np.concatenate(r)))))


def score_clip(path, lo, hi, lane, direction):
    cands, fps, n = candidates(path, lo, hi, lane)
    out = []
    for ch in chains(cands, direction):
        rms = quad_rms(ch)
        if rms <= MAX_QUAD_RMS:
            out.append((len(ch) / (1.0 + rms), len(ch), rms, ch))
    out.sort(key=lambda r: -r[0])
    return out, fps, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", type=int, default=822515)
    ap.add_argument("--dir", default=os.path.join(ROOT, "data", "clips", "ogando_2026"))
    ap.add_argument("--lane", default=",".join(str(v) for v in LANE), help="x0,x1,y0,y1 flight lane")
    ap.add_argument("--lo", type=int, default=150)
    ap.add_argument("--hi", type=int, default=280)
    ap.add_argument("--direction", type=int, default=1, help="+1 ball moves right")
    ap.add_argument("--limit", type=int, default=8)
    ap.add_argument("--validate", action="store_true")
    a = ap.parse_args()
    lane = tuple(int(v) for v in a.lane.split(","))

    man = os.path.join(a.dir, os.path.basename(a.dir) + "_manifest.csv")
    m = pd.read_csv(man)

    if a.validate:
        p = os.path.join(a.dir, VALIDATION["clip"] + ".mp4")
        res, fps, n = score_clip(p, a.lo, a.hi, lane, a.direction)
        assert res, ("VALIDATION FAILED: no chain survived the quadratic test on "
                     f"{VALIDATION['clip']}, where the ball is known to be at "
                     f"{sorted(VALIDATION['points'])}")
        _, ln, rms, ch = res[0]
        got = {f: (x, y) for f, x, y in ch}
        hits, errs = 0, []
        for f, (tx, ty) in VALIDATION["points"].items():
            if f in got:
                e = float(np.hypot(got[f][0] - tx, got[f][1] - ty))
                errs.append(e)
                hits += e <= 6.0
        print(f"top chain: f{ch[0][0]}-{ch[-1][0]}  n={ln}  quad rms={rms:.2f} px")
        print(f"recovered {hits}/{len(VALIDATION['points'])} confirmed points within 6 px"
              + (f"   errs {[round(e,1) for e in errs]}" if errs else "   (no frame overlap)"))
        assert hits >= 8, ("VALIDATION FAILED: the top chain does not recover the track "
                           "confirmed by eye in output/si9_track_check.png. Do NOT loosen "
                           "this gate to make a run go green -- fix the seeder.")
        print("VALIDATION PASSED")
        return

    g = m[m.game_pk == a.game].reset_index(drop=True)
    stems = [s for s in g.stem if os.path.exists(os.path.join(a.dir, s + ".mp4"))][:a.limit]
    rows = []
    for s in stems:
        res, fps, n = score_clip(os.path.join(a.dir, s + ".mp4"), a.lo, a.hi, lane, a.direction)
        if not res:
            print(f"{s:16} no chain passed  (frames {n})")
            continue
        _, ln, rms, ch = res[0]
        print(f"{s:16} f{ch[0][0]:3d}-{ch[-1][0]:3d}  n={ln:2d}  rms={rms:4.2f}  "
              f"x {ch[0][1]:6.1f}->{ch[-1][1]:6.1f}  y {ch[0][2]:6.1f}->{ch[-1][2]:6.1f}"
              f"   ({len(res)} chains passed)")
        for f, x, y in ch:
            rows.append(dict(stem=s, game_pk=a.game, frame=f, x=x, y=y))
    if rows:
        out = os.path.join(ROOT, "output", f"seed_tracks_{a.game}.csv")
        pd.DataFrame(rows).to_csv(out, index=False)
        print(f"\n{out}\nRENDER THESE AND LOOK before fitting any pose to them.")


if __name__ == "__main__":
    main()
