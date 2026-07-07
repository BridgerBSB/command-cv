"""Ball-at-rest CATCH detector + calibration validation. RUN ON THE WORK LAPTOP
(or any GPU box) — the ball model is ~21s/frame on the CPU-only personal laptop,
so this is infeasible there; on GPU it is milliseconds.

Why ball-at-rest: a caught pitch STOPS the ball in the glove for ~10-15 frames.
That is a tight multi-frame stable cluster of ball detections — unambiguous catch
timing, and the ACTUAL ball location (a better calibration anchor than the glove
box center, which sits off the pocket). Per-frame ball detection is noisy, but a
stable cluster is trivially separable from scattered false positives.

Pipeline per clip:
  1. Fast mitt model (data/yolo/runs/mitt/weights/best.pt) -> glove trajectory ->
     setup + the pitch-flight detection gap (src.auto_glove; setup is SOLVED, 2.3px).
  2. Run BaseballCV `glove_tracking` ball class ONLY in the post-flight window
     (small -> cheap even at imgsz=1280).
  3. Ball-at-rest = densest stable cluster of ball detections in that window.
  4. Calibrate: fit per-game affine ball-rest-pixel -> Trackman plate_x/plate_z,
     leave-one-out. Report the inch error. THIS is the trust metric — NOT the
     ball-vs-glove-label pixel gap (ball rests off the mitt center by design).

Compare the pooled LOO inches here to the hand-catch-glove calibration
(median 4.2in on the sampled set): with ~30 auto ball-rest points per full game,
the goal is sub-4in.

Run (work laptop):
  python scripts/validate_catch_ball.py --pitcher ogando --min-clips 5
"""
import argparse
import csv
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.auto_glove import detect_setup_catch
from src.calibration import solve_affine

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _flight_window(clip, mitt_model):
    """Use the fast glove detector to bracket the post-flight (receive) window."""
    r = detect_setup_catch(clip, mitt_model)
    if not r.get("ok"):
        return None
    cframe = r["catch"][2]                 # motion-based catch frame (rough)
    return max(0, cframe - 20), cframe + 25


def _ball_dets(clip, f0, f1, ball_model, imgsz=1280, conf=0.20):
    import cv2
    cap = cv2.VideoCapture(clip)
    out = []
    for fi in range(f0, f1):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ok, fr = cap.read()
        if not ok:
            break
        res = ball_model.predict(fr, imgsz=imgsz, conf=conf, verbose=False)[0]
        for i in range(len(res.boxes)):
            if ball_model.names[int(res.boxes.cls[i])] == "baseball":
                x, y = res.boxes.xywh[i][:2].cpu().numpy()
                out.append((float(x), float(y)))
    cap.release()
    return out


def _rest_cluster(dets, radius=18.0, min_pts=3):
    """Densest neighborhood of ball detections = ball parked in the glove."""
    if len(dets) < min_pts:
        return None
    P = np.array(dets)
    best, bestn = None, 0
    for p in P:
        nb = P[np.hypot(P[:, 0] - p[0], P[:, 1] - p[1]) <= radius]
        if len(nb) > bestn:
            bestn, best = len(nb), nb
    if bestn < min_pts:
        return None
    return float(np.median(best[:, 0])), float(np.median(best[:, 1]))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pitcher", default="ogando")
    p.add_argument("--min-clips", type=int, default=5)
    a = p.parse_args()

    from ultralytics import YOLO
    from baseballcv.functions import LoadTools
    mitt = YOLO(os.path.join(ROOT, "data", "yolo", "runs", "mitt", "weights", "best.pt"))
    ball = YOLO(LoadTools().load_model("glove_tracking"))

    man = {r["stem"]: r for r in csv.DictReader(
        open(os.path.join(ROOT, "data", "clips", f"{a.pitcher}_2026",
                          f"{a.pitcher}_2026_manifest.csv")))}

    pts = []  # (game_pk, ball_x, ball_y, plate_x, plate_z)
    for f in sorted(glob.glob(os.path.join(ROOT, "data", "annotations", f"{a.pitcher}_*.json"))):
        stem = json.load(open(f))["stem"]
        r = man.get(stem)
        clip = os.path.join(ROOT, "data", "clips", f"{a.pitcher}_2026", f"{stem}.mp4")
        if not r or not os.path.exists(clip):
            continue
        try:
            px, pz = float(r["plate_x"]), float(r["plate_z"])
        except (TypeError, ValueError):
            continue
        win = _flight_window(clip, mitt)
        if not win:
            print(f"{stem}: no flight window"); continue
        c = _rest_cluster(_ball_dets(clip, win[0], win[1], ball))
        if c is None:
            print(f"{stem}: no ball-rest cluster"); continue
        pts.append((r["game_pk"], c[0], c[1], px, pz))
        print(f"{stem}: ball-rest ({c[0]:.0f},{c[1]:.0f})")

    # Per-game LOO calibration on the ball-rest points.
    by_game = {}
    for gp, bx, by, px, pz in pts:
        by_game.setdefault(gp, []).append((bx, by, px, pz))
    errs = []
    for gp, rs in by_game.items():
        if len(rs) < a.min_clips:
            continue
        img = np.array([(r[0], r[1]) for r in rs]); wld = np.array([(r[2], r[3]) for r in rs])
        for i in range(len(rs)):
            keep = [j for j in range(len(rs)) if j != i]
            if len(keep) < 3:
                continue
            aff = solve_affine(img[keep], wld[keep])
            d = (np.array(aff.pixel_to_world(*img[i])) - wld[i]) * 12.0
            errs.append(float(np.hypot(*d)))
    if errs:
        e = np.array(errs)
        print(f"\nBALL-REST calibration LOO: median {np.median(e):.1f}in  "
              f"RMSE {np.sqrt(np.mean(e**2)):.1f}in  n={len(e)}")
        print("Compare to hand-catch-glove LOO median 4.2in. Lower = ball-rest wins.")
    else:
        print("\nnot enough per-game ball-rest points to validate")


if __name__ == "__main__":
    main()
