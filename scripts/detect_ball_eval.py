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

CROP, OVERLAP = 320, 80
IN_PER_PX = 0.30          # a 17 in zone spans ~57 px in this feed


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


def detect_full(model, frame, conf):
    r = model.predict(frame, imgsz=1280, conf=conf, verbose=False)[0]
    if r.boxes is None or len(r.boxes) == 0:
        return None
    c = r.boxes.conf.cpu().numpy()
    b = r.boxes.xywh.cpu().numpy()
    j = int(c.argmax())
    return (b[j][0], b[j][1], float(c[j]))


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
    model.to(a.device)

    clips = par.load_clips(a.game)
    todo = []
    for c in clips:
        runs = par.candidate_runs(c)
        if runs and len(runs[0]) >= 10 and os.path.exists(
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
        want = {int(r.frame_idx): (r.baseball_center_x, r.baseball_center_y)
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
            fn = detect_tiled if m == "tile" else detect_full
            for fi, fr in frames.items():
                d = fn(model, fr, a.conf)
                ox, oy = want[fi]
                if d is None:
                    out[m].append((pid, fi, np.nan, np.nan))
                else:
                    out[m].append((pid, fi, float(np.hypot(d[0] - ox, d[1] - oy)), d[2]))
        print(f"  {pid[:8]}  {len(frames)} frames", flush=True)

    print(f"\n{'mode':6} {'frames':>7} {'found%':>7} {'err px med':>11} {'err px p90':>11} "
          f"{'err in med':>11} {'conf med':>9}")
    for m in modes:
        d = pd.DataFrame(out[m], columns=["play", "frame", "err", "conf"])
        f = d.err.notna()
        if not f.any():
            print(f"{m:6} {len(d):7} {0.0:6.1f}%   NOTHING DETECTED")
            continue
        print(f"{m:6} {len(d):7} {100*f.mean():6.1f}% {d.err[f].median():11.2f} "
              f"{d.err[f].quantile(.9):11.2f} {d.err[f].median()*IN_PER_PX:11.3f} "
              f"{d.conf[f].median():9.3f}")

    print("\nRead found% FIRST: a small error on few frames is worse than a modest error on")
    print("most of them - the joint solve needs >=8 points per clip to fit anything.")
    print("OpenCommand's own ball-at-plate error is ~0.4 in, so anything near that is parity.")


if __name__ == "__main__":
    main()
