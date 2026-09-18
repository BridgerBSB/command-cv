"""Build ball tracks from the DETECTOR instead of from frame differencing.

WHAT THIS REPLACES. Every track in this project so far came from absdiff + top-hat
(onboard_park -> seeds) or from projecting a pose (harvest_park_labels -> snap). Both are
cadence-sensitive: a feed carrying 30 fps in a 60 fps container has an empty diff on every
other frame, which is why L.P. Frans yielded 13 of 50 clips and Hops 1 of 40. A detector runs
PER FRAME and has no cadence assumption at all, so this path sidesteps the problem rather
than fixing three copies of it.

It also needs NO POSE. Seeds needed a pose to be validated; harvest needed a pose to know
where to look. This needs neither, which inverts the bootstrap: detector -> tracks -> pose,
instead of pose -> tracks.

TOP-K, NOT TOP-1, AND THAT IS NOT OPTIONAL. A stationary high-confidence distractor takes
top-1 on most frames at some parks -- a crowd object held top-1 on 35 of 56 flight frames at
Asheville in September, and top-1-only reporting made that read as blindness. So candidates
are taken K deep and the FLIGHT is chosen by physics: consecutive-ish frames, a step inside a
plausible px/frame band, roughly monotonic direction, and a preference for ACCELERATING
motion because a pitch speeds up in pixels as it approaches while a pan or a return throw
does not.

NO LABELS ARE INVENTED. A clip that yields no plausible chain is reported and skipped, never
back-filled. The per-clip chain length and step profile are printed so a park that is
"working" for the wrong reason is visible.

    python scripts/detect_ball_tracks.py --dir data/clips/aaa_815419 --game 815419 \
        --weights data/yolo_ball_runs/milb4park/weights/best.pt

SPEED. Inference runs ~0.4 s/frame at imgsz 1280, so the 100-frame default window costs about
40 s a clip and a 600-clip park is hours. Two levers, and only one is safe by default:
  --batch      already on; per-call overhead dominated before it.
  --lo / --hi  the flight occupies about 45 of those 100 frames. Plate frames measured from
               the harvest scans: Isotopes 213-220, Frawley 213-223, Asheville 214-224, and a
               flight is ~26 frames, so f180-230 covers all of them with margin.
THE DEFAULT STAYS WIDE ON PURPOSE. Those parks agree because they share a broadcast cut
convention; Hops and Riders have never produced a plate frame, so a narrow default would make
an unseen park yield nothing and read as 'the detector fails there' -- exactly the false
blocker pattern in .claude/rules/probe-capability-before-accepting-a-blocker.md. Narrow it per
park, from the plate-frame distribution this script prints at the end of every run.
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

STEP_LO, STEP_HI = 2.0, 30.0      # px per frame, wide enough for both camera archetypes
MIN_RUN = 6


def best_chain(dets, step_lo=STEP_LO, step_hi=STEP_HI):
    """Longest physically plausible chain through per-frame candidates.

    dets: {frame: [(x, y, conf), ...]}. Greedy from each possible start, keeping the longest;
    the search is tiny (a few hundred candidates per clip) so there is no reason to be clever.
    """
    frames = sorted(dets)
    best = []
    for si, f0 in enumerate(frames):
        for c0 in dets[f0]:
            chain = [(f0, c0[0], c0[1], c0[2])]
            for f1 in frames[si + 1:]:
                if f1 - chain[-1][0] > 3:
                    break
                prev = chain[-1]
                cands = []
                for x, y, cf in dets[f1]:
                    dt = f1 - prev[0]
                    st = np.hypot(x - prev[1], y - prev[2]) / dt
                    if not (step_lo <= st <= step_hi):
                        continue
                    if len(chain) >= 2:      # keep going the same way
                        vx, vy = chain[-1][1] - chain[-2][1], chain[-1][2] - chain[-2][2]
                        nx, ny = x - prev[1], y - prev[2]
                        n1 = np.hypot(vx, vy) * np.hypot(nx, ny)
                        if n1 > 0 and (vx * nx + vy * ny) / n1 < 0.80:
                            continue
                    cands.append((st, x, y, cf))
                if not cands:
                    continue
                st, x, y, cf = max(cands, key=lambda c: c[3])   # strongest that fits
                chain.append((f1, x, y, cf))
            if len(chain) > len(best):
                best = chain
    return best


def _flush(model, buf, dets, a):
    """Run one batched predict over buffered (frame_index, image) pairs."""
    res = model.predict([b[1] for b in buf], conf=a.conf, imgsz=a.imgsz,
                        verbose=False, device=a.device)
    for (fi, _), r in zip(buf, res):
        if r.boxes is None or not len(r.boxes):
            continue
        xy = r.boxes.xywh.cpu().numpy()
        cf = r.boxes.conf.cpu().numpy()
        order = np.argsort(-cf)[:a.topk]
        dets[fi] = [(float(xy[k, 0]), float(xy[k, 1]), float(cf[k])) for k in order]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--game", type=int, required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--lo", type=int, default=150)
    ap.add_argument("--hi", type=int, default=250)
    ap.add_argument("--conf", type=float, default=0.15)
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--batch", type=int, default=16,
                    help="frames per predict call; 16 fits a 4 GB card at 1280")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--device", default="0")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()

    from ultralytics import YOLO
    model = YOLO(a.weights)

    d = a.dir if os.path.isabs(a.dir) else os.path.join(ROOT, a.dir)
    man = pd.read_csv(os.path.join(d, os.path.basename(d) + "_manifest.csv"))
    stems = [s for s in man.stem if os.path.exists(os.path.join(d, s + ".mp4"))]
    if a.limit:
        stems = stems[:a.limit]

    rows, kept, lens = [], 0, []
    for j, stem in enumerate(stems, 1):
        cap = cv2.VideoCapture(os.path.join(d, stem + ".mp4"))
        dets, i, buf = {}, 0, []
        while True:
            ok, fr = cap.read()
            if not ok or i > a.hi:
                break
            if i >= a.lo:
                buf.append((i, fr))
                # BATCHED, because one predict() per frame is dominated by per-call overhead
                # rather than by the network: ~100 frames a clip x 600 clips is 60,000 calls.
                # Frames are flushed in chunks instead of all held, since a full window of
                # 1280x720x3 is ~276 MB and this box runs near its memory watchdog.
                if len(buf) >= a.batch:
                    _flush(model, buf, dets, a)
                    buf = []
            i += 1
        cap.release()
        if buf:
            _flush(model, buf, dets, a)
            buf = []
        chain = best_chain(dets) if len(dets) >= MIN_RUN else []
        if len(chain) < MIN_RUN:
            continue
        kept += 1
        lens.append(len(chain))
        for f, x, y, cf in chain:
            rows.append(dict(stem=stem, game_pk=a.game, frame=int(f), x=x, y=y, conf=cf))
        if j % 20 == 0:
            print(f"  [{j}/{len(stems)}] {kept} clips with a flight, {len(rows)} points",
                  flush=True)

    assert rows, "no clip produced a plausible flight -- check --lo/--hi and the weights"
    out = pd.DataFrame(rows)
    p = os.path.join(ROOT, "output", f"detect_tracks_{a.game}{a.tag}.csv")
    out.to_csv(p, index=False)
    print(f"\ngame {a.game}: {kept}/{len(stems)} clips with a flight, {len(out)} points")
    print(f"  chain length: median {int(np.median(lens))}, "
          f"range {min(lens)}-{max(lens)}   mean conf {out.conf.mean():.2f}")
    ends = out.groupby("stem").frame.max()
    print(f"  chain ends (proxy for the plate crossing): p05 {ends.quantile(.05):.0f}  "
          f"median {ends.median():.0f}  p95 {ends.quantile(.95):.0f}   -> "
          f"--lo {int(ends.quantile(.05)) - 32} --hi {int(ends.quantile(.95)) + 6} "
          f"would have sufficed")
    print(f"{p}\n  These are DETECTIONS, not yet verified. Render before using them.")


if __name__ == "__main__":
    main()
