"""How good are OUR ball detections, in pixels, before any camera solve touches them?

The real gate for this project is: swap our detections into the joint camera solve and
see whether it still lands within an inch. But that gate is slow and it confounds two
things - detector accuracy and solve robustness. This is the cheap gate that runs first:
compare our detected ball pixel to OpenCommand's on THE SAME frames. If we are 10 px out
here, no camera solve can land within an inch, and we know in minutes.

THE SCALE PROBLEM, which is the reason this script is not a one-liner.
build_ball_dataset.py trains on 320 px CROPS at NATIVE resolution, deliberately: the ball
is ~10-20 px in a 1280x720 broadcast frame, and a whole frame resized to 320 makes it ~4 px
and mostly gone. So inference cannot just hand a full frame to a 320 px model. Two ways out,
and which one wins is an empirical question:

  tile  cut the frame into overlapping 320 px windows and detect in each. Matches training
        conditions exactly. ~15-20 windows per frame.
  full  run at imgsz=1280 so nothing is downscaled and the ball keeps its true size. One
        pass, much faster, but the model never saw a 1280 px input.

Both are implemented. The script reports each one's pixel error against OpenCommand and
its detection rate, so the choice is made on numbers.

REPORTED, and read them in this order:
  found%      frames where we detected any ball at all. A tiny median error on 20% of
              frames is worse than a modest error on 95% - the solve needs points.
  err px      euclidean distance to OpenCommand's detection, median and p90.
  err in      the same at ~0.30 in/px, so it is comparable to every other number here.

WHICH FRAMES THIS READS (fixed Sep 13 2026 - every number this script printed before
that date is VOID). OpenCommand's frame_idx is RELATIVE TO RELEASE, spanning about
-28..+30. Our clip is indexed from 0 and the pitch crosses the plate around frame 205.
This script used to read our video at THEIR number, so it scored the detector on frames
0-30 - the pitcher still standing on the mound, three seconds before release - against a
mid-flight ball position. That, and not the detector, produced the 44-308 px errors.
scripts/oc_frame_map.py holds the mapping and verifies it two independent ways.

    python scripts/detect_ball_eval.py --weights data/yolo_ball_runs/ball_3park_gpu/weights/best.pt
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
import parity_opencommand_pose as par  # noqa: E402
import oc_frame_map as ofm  # noqa: E402

CROP, OVERLAP = 320, 80
IN_PER_PX = 0.30          # a 17 in zone spans ~57 px in this feed
TOPK = 5                  # see THE TOP-1 TRAP below

# THE TOP-1 TRAP. The first version of this script scored only the highest-confidence
# detection per frame, and reported 167-308 px median error - which reads as "the
# detector cannot see the ball". detect_diagnose.py then showed the ball IS detected in
# 2 of 3 frames, ranked #3, at 40.0 px and 15.5 px from truth. A ball found within 16 px
# but ranked third produced the SAME table row as a 167 px miss, so the metric turned a
# RANKING problem (fixable with epochs) into what looked like a BLINDNESS problem (not).
# Report best-of-top-K beside top-1 or the number lies in a way that changes the decision.


def tiles(W, H):
    step = CROP - OVERLAP
    xs = list(range(0, max(1, W - CROP + 1), step)) + [W - CROP]
    ys = list(range(0, max(1, H - CROP + 1), step)) + [H - CROP]
    return sorted(set((x, y) for x in xs for y in ys))


def detect_tiled(model, frame, conf):
    """Best detection across overlapping native-resolution windows."""
    H, W = frame.shape[:2]
    best = None
    wins = tiles(W, H)
    batch = [frame[y:y + CROP, x:x + CROP] for x, y in wins]
    res = model.predict(batch, imgsz=CROP, conf=conf, verbose=False)
    for (x0, y0), r in zip(wins, res):
        if r.boxes is None or len(r.boxes) == 0:
            continue
        c = r.boxes.conf.cpu().numpy()
        b = r.boxes.xywh.cpu().numpy()
        j = int(c.argmax())
        if best is None or c[j] > best[2]:
            best = (b[j][0] + x0, b[j][1] + y0, float(c[j]))
    return best


def detect_tiled_topk(model, frame, conf, k=TOPK):
    """Top-k across overlapping native-resolution windows, ranked by confidence.

    Tiling's failure is specifically a RANKING one: the true ball competes against a
    false positive from every OTHER window, so top-1 across ~15-20 windows is far more
    exposed than top-1 on a single pass. That is the stated reason tile mode scored
    29% against full-frame's 48%. Reporting top-k here is what actually tests it.
    """
    H, W = frame.shape[:2]
    wins = tiles(W, H)
    batch = [frame[y:y + CROP, x:x + CROP] for x, y in wins]
    res = model.predict(batch, imgsz=CROP, conf=conf, verbose=False)
    cand = []
    for (x0, y0), r in zip(wins, res):
        if r.boxes is None or len(r.boxes) == 0:
            continue
        c = r.boxes.conf.cpu().numpy()
        b = r.boxes.xywh.cpu().numpy()
        for j in range(len(c)):
            cand.append((b[j][0] + x0, b[j][1] + y0, float(c[j])))
    cand.sort(key=lambda t: -t[2])
    return cand[:k]


def detect_full(model, frame, conf):
    r = model.predict(frame, imgsz=1280, conf=conf, verbose=False)[0]
    if r.boxes is None or len(r.boxes) == 0:
        return None
    c = r.boxes.conf.cpu().numpy()
    b = r.boxes.xywh.cpu().numpy()
    j = int(c.argmax())
    return (b[j][0], b[j][1], float(c[j]))


def detect_full_topk(model, frame, conf, k=TOPK):
    """Top-k detections, highest confidence first. See the TOP-1 TRAP below."""
    r = model.predict(frame, imgsz=1280, conf=conf, verbose=False)[0]
    if r.boxes is None or len(r.boxes) == 0:
        return []
    c = r.boxes.conf.cpu().numpy()
    b = r.boxes.xywh.cpu().numpy()
    return [(b[j][0], b[j][1], float(c[j])) for j in np.argsort(-c)[:k]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", type=int, default=824821)
    ap.add_argument("--weights", default=os.path.join(
        ROOT, "data", "yolo_ball_runs", "ball_3park_gpu", "weights", "best.pt"))
    ap.add_argument("--mode", default="both", choices=["tile", "full", "both"])
    ap.add_argument("--clips", type=int, default=8, help="clips to evaluate")
    ap.add_argument("--conf", type=float, default=0.10)
    ap.add_argument("--device", default="0")
    a = ap.parse_args()

    assert os.path.exists(a.weights), f"no weights: {a.weights} (has training finished?)"
    clips_dir = os.path.join(ROOT, "data", "clips", f"mlb_{a.game}")
    assert os.path.isdir(clips_dir), clips_dir

    from ultralytics import YOLO
    model = YOLO(a.weights)
    # ultralytics takes device="0"; torch's .to() does NOT - it wants "cuda:0" and raises
    # RuntimeError: Invalid device string: '0'. Normalise before handing it to torch.
    dev = f"cuda:{a.device}" if str(a.device).isdigit() else a.device
    model.to(dev)
    print(f"device: {dev}")

    clips = par.load_clips(a.game)
    # Place OC's release-relative frames in our clip, and prove the placement before
    # trusting it. A play with no shift is SKIPPED rather than defaulted to 0 - a
    # silent fallback here is exactly the defect this map exists to fix.
    shifts = ofm.build_shifts(a.game, clips)
    ofm.verify_shifts(a.game, clips, shifts)
    todo = []
    for c in clips:
        runs = par.candidate_runs(c)
        if runs and len(runs[0]) >= 10 and c["play_id"] in shifts and os.path.exists(
                os.path.join(clips_dir, f"{c['play_id']}_BROADCAST.mp4")):
            todo.append((c, runs[0]))
        if len(todo) >= a.clips:
            break
    assert todo, "no clips with a usable ball run"
    print(f"game {a.game}: {len(todo)} clips, weights {os.path.basename(a.weights)}\n")

    modes = ["tile", "full"] if a.mode == "both" else [a.mode]
    out = {m: [] for m in modes}
    for c, seg in todo:
        pid = c["play_id"]
        cap = cv2.VideoCapture(os.path.join(clips_dir, f"{pid}_BROADCAST.mp4"))
        s = shifts[pid]      # release-relative -> our clip's own index
        want = {int(r.frame_idx) + s: (r.baseball_center_x, r.baseball_center_y)
                for r in seg.itertuples()}
        frames, i, hi = {}, 0, max(want)
        while i <= hi:
            ok, f = cap.read()
            if not ok:
                break
            if i in want:
                frames[i] = f
            i += 1
        cap.release()

        for m in modes:
            fn = detect_tiled_topk if m == "tile" else detect_full_topk
            for fi, fr in frames.items():
                ds = fn(model, fr, a.conf)
                ox, oy = want[fi]
                if not ds:
                    out[m].append((pid, fi, np.nan, np.nan, np.nan, np.nan))
                    continue
                dist = [float(np.hypot(x - ox, y - oy)) for x, y, _ in ds]
                # top-1 is what the old version reported; best-of-K and the RANK of the
                # closest detection are what reveal a found-but-outranked ball
                best_i = int(np.argmin(dist))
                out[m].append((pid, fi, dist[0], dist[best_i], float(best_i + 1), ds[0][2]))
        print(f"  {pid[:8]}  {len(frames)} frames", flush=True)

    print(f"\n{'mode':6} {'frames':>7} {'found%':>7} {'top1 px':>9} {'bestK px':>9} "
          f"{'bestK in':>9} {'med rank':>9} {'conf med':>9}")
    for m in modes:
        d = pd.DataFrame(out[m], columns=["play", "frame", "err1", "errk", "rank", "conf"])
        f = d.err1.notna()
        if not f.any():
            print(f"{m:6} {len(d):7} {0.0:6.1f}%   NOTHING DETECTED")
            continue
        # d['rank'] not d.rank - DataFrame.rank is a METHOD, and d.rank would silently
        # hand back the bound method rather than the column
        print(f"{m:6} {len(d):7} {100*f.mean():6.1f}% {d.err1[f].median():9.2f} "
              f"{d.errk[f].median():9.2f} {d.errk[f].median()*IN_PER_PX:9.3f} "
              f"{d['rank'][f].median():9.1f} {d.conf[f].median():9.3f}")
    print(f"\ntop1 vs bestK is the whole point (K={TOPK}). If bestK is far smaller than top1,")
    print("the detector SEES the ball and ranks it badly - a ranking problem, which more")
    print("epochs fix. If they are close and both large, it genuinely cannot find the ball.")
    print("med rank says where the closest detection sat: 1.0 = correctly ranked first.")

    print("\nRead found% FIRST: a small error on few frames is worse than a modest error on")
    print("most of them - the joint solve needs >=8 points per clip to fit anything.")
    print("OpenCommand's own ball-at-plate error is ~0.4 in, so anything near that is parity.")


if __name__ == "__main__":
    main()
