"""Score a ball detector in PIXELS on a built dataset's val split, separating two failures.

WHY NOT mAP. mAP at a fixed IoU is a poor proxy for pixel distance to a ~16 px ball -- a
detection half a ball off scores near zero IoU while being perfectly useful for a camera fit,
and the 09-12 run watched mAP sit flat at 0.08-0.12 while the actual pixel error fell by 4x.
The quantity this project needs is: how far, in pixels, is the detection from the ball.

WHY top-1 ALONE IS NOT ENOUGH, which cost a wrong diagnosis once already
(.claude/rules/metric-must-separate-failure-modes.md). A large top-1 error has TWO causes
that demand OPPOSITE actions:
    RANKING failure     - finds the ball, scores a distractor higher  -> train more
    PERCEPTION failure  - cannot find the ball at all                 -> the data is wrong
Top-1 reports the same number for both. So this prints top-1 error, best-of-K error, AND the
RANK of the closest detection. med rank 1 with a large bestK error is perception; med rank 3
with a small bestK error is ranking.

The val split is a PARK the model never trained on, which is the honest test and also the
reason the numbers here are worse than a random split would give.

    python scripts/eval_ball_detector.py --data data/yolo_milb --weights .../best.pt
"""
import argparse
import glob
import os

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--conf", type=float, default=0.01,
                    help="deliberately LOW: this measures localisation, not a shipped "
                         "operating point, and a high floor would hide ranking failures")
    ap.add_argument("--device", default="0")
    ap.add_argument("--hit-px", type=float, default=8.0,
                    help="a 'find' is within this many px -- about half a ball")
    a = ap.parse_args()

    from ultralytics import YOLO
    model = YOLO(a.weights)

    imgs = sorted(glob.glob(os.path.join(a.data, "images", a.split, "*.png")))
    assert imgs, f"no images in {a.data}/images/{a.split}"

    top1, bestk, ranks, missing = [], [], [], 0
    for ip in imgs:
        lp = os.path.join(a.data, "labels", a.split, os.path.basename(ip)[:-4] + ".txt")
        txt = open(lp).read().strip() if os.path.exists(lp) else ""
        if not txt:
            continue                                   # negatives are not scored here
        r = model.predict(ip, conf=a.conf, verbose=False, device=a.device)[0]
        b = r.boxes
        h, w = r.orig_shape
        _, cx, cy, _, _ = [float(v) for v in txt.split()[:5]]
        tx, ty = cx * w, cy * h
        if b is None or len(b) == 0:
            missing += 1
            continue
        xy = b.xywh.cpu().numpy()[:, :2]
        conf = b.conf.cpu().numpy()
        order = np.argsort(-conf)[:a.topk]             # detector's own ranking
        d = np.hypot(xy[order, 0] - tx, xy[order, 1] - ty)
        top1.append(d[0])
        bestk.append(d.min())
        ranks.append(int(np.argmin(d)) + 1)

    t1, bk, rk = np.array(top1), np.array(bestk), np.array(ranks)
    n = len(t1)
    print(f"{a.data} [{a.split}]  {n} labelled images scored, "
          f"{missing} with NO detection at conf>={a.conf}")
    print(f"  weights: {a.weights}\n")
    print(f"  {'':22} {'median':>8} {'p90':>8} {'<8px':>7} {'<16px':>7}")
    for nm, v in (("top-1 error px", t1), (f"best-of-{a.topk} error px", bk)):
        print(f"  {nm:22} {np.median(v):8.2f} {np.percentile(v, 90):8.2f} "
              f"{100 * (v < a.hit_px).mean():6.0f}% {100 * (v < 2 * a.hit_px).mean():6.0f}%")
    print(f"  {'rank of closest':22} {np.median(rk):8.1f} {np.percentile(rk, 90):8.1f}")
    print()
    if np.median(rk) <= 1.5 and np.median(bk) > a.hit_px:
        print("  READ: ball ranked FIRST and still far -> PERCEPTION. More epochs will not "
              "fix this;\n  suspect the labels, the crop scale or the park mix.")
    elif np.median(rk) > 1.5:
        print("  READ: the ball is found but OUTRANKED by a distractor -> RANKING. More "
              "training or\n  harder negatives is the lever, and top-1 alone would have "
              "called this blindness.")
    else:
        print("  READ: found and ranked first. Localisation is the remaining error.")
    print("\n  This is still not the FINAL test. That is feeding these detections into the "
          "camera\n  solve and checking the plate crossing in inches "
          "(scripts/plate_scale_check.py).")


if __name__ == "__main__":
    main()
