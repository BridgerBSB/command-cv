"""WHAT is the detector locking onto? Undertrained, or structurally broken?

detect_ball_eval.py returned a median error of 167-308 px against a ~16 px ball in a
1280 px frame, at confidence 0.13-0.16. That number says the detector is wrong; it
does not say HOW, and the two possibilities call for opposite decisions:

  undertrained      it fires on plausible ball-like things - a white logo, the plate,
                    a helmet highlight - just not the ball. More epochs fix this.
  structurally bad  it fires on arbitrary texture with no ball-like character at all.
                    More epochs will not fix that; the dataset or the box scale is wrong.

A metric cannot tell those apart. Looking can. So: draw OpenCommand's known ball
position (green, ground truth) and our top-5 detections (red, ranked, with
confidence) on the SAME full frame - full frame deliberately, because errors of
167-660 px put the two nowhere near each other and a crop would hide the answer.

Also prints, per frame, the distance from each of our detections to truth, so a
detection that is nearly right but ranked 4th is visible - that would mean the
detector CAN see the ball and is merely scoring it poorly, which is a different
and much more fixable problem than not seeing it.

WHICH FRAMES THIS READS (fixed Sep 13 2026). OpenCommand's frame_idx is RELATIVE TO
RELEASE; our clip is indexed from 0. This script used to read our video at their number
and so drew "truth" on a frame three seconds before the pitch, where there is no ball to
find. Every conclusion drawn from its earlier output - including "the ball is detected
and ranked #3" - was read off the wrong picture. scripts/oc_frame_map.py does the
mapping and verifies it.

    python scripts/detect_diagnose.py --weights data/yolo_ball_runs/ball_3park_b8/weights/best.pt
"""
import argparse
import os
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import parity_opencommand_pose as par  # noqa: E402
import oc_frame_map as ofm  # noqa: E402

TOPK = 5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", type=int, default=824821)
    ap.add_argument("--weights", default=os.path.join(
        ROOT, "data", "yolo_ball_runs", "ball_3park_b8", "weights", "best.pt"))
    ap.add_argument("--clips", type=int, default=3)
    ap.add_argument("--per-clip", type=int, default=1, help="frames to show per clip")
    ap.add_argument("--conf", type=float, default=0.02, help="low, so we see what it ranks")
    ap.add_argument("--device", default="0")
    ap.add_argument("--out", default=os.path.join(ROOT, "output", "detect_diagnose.png"))
    a = ap.parse_args()

    assert os.path.exists(a.weights), f"no weights: {a.weights}"
    clips_dir = os.path.join(ROOT, "data", "clips", f"mlb_{a.game}")
    assert os.path.isdir(clips_dir), clips_dir

    from ultralytics import YOLO
    dev = f"cuda:{a.device}" if str(a.device).isdigit() else a.device
    model = YOLO(a.weights)
    model.to(dev)
    print(f"device: {dev}  weights: {os.path.basename(a.weights)}")

    clips = par.load_clips(a.game)
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
    assert todo, "no usable clips"

    panels = []
    for c, seg in todo:
        pid = c["play_id"]
        # take frames from the MIDDLE of the flight, where the ball is clearly airborne
        mid = seg.iloc[len(seg) // 2: len(seg) // 2 + a.per_clip]
        cap = cv2.VideoCapture(os.path.join(clips_dir, f"{pid}_BROADCAST.mp4"))
        s = shifts[pid]      # release-relative -> our clip's own index
        want = {int(r.frame_idx) + s: (r.baseball_center_x, r.baseball_center_y)
                for r in mid.itertuples()}
        frames, i, hi = {}, 0, max(want)
        while i <= hi:
            ok, f = cap.read()
            if not ok:
                break
            if i in want:
                frames[i] = f
            i += 1
        cap.release()

        for fi, fr in frames.items():
            ox, oy = want[fi]
            r = model.predict(fr, imgsz=1280, conf=a.conf, verbose=False)[0]
            im = fr.copy()
            print(f"\n{pid[:8]} f{fi}   truth ({ox:.0f}, {oy:.0f})")
            if r.boxes is None or len(r.boxes) == 0:
                print("   NO DETECTIONS at all")
            else:
                cf = r.boxes.conf.cpu().numpy()
                xy = r.boxes.xywh.cpu().numpy()
                order = np.argsort(-cf)[:TOPK]
                for rank, j in enumerate(order, 1):
                    x, y, w, h = xy[j]
                    d = float(np.hypot(x - ox, y - oy))
                    col = (0, 165, 255) if rank == 1 else (0, 0, 255)
                    cv2.rectangle(im, (int(x - w / 2), int(y - h / 2)),
                                  (int(x + w / 2), int(y + h / 2)), col, 2)
                    cv2.putText(im, f"{rank} {cf[j]:.2f}", (int(x - w / 2), int(y - h / 2) - 4),
                                0, 0.5, col, 2, cv2.LINE_AA)
                    print(f"   #{rank} conf {cf[j]:.3f}  at ({x:6.0f},{y:6.0f})  "
                          f"{w:4.0f}x{h:<4.0f}px  dist_to_truth {d:7.1f} px")
            # ground truth last so it is never hidden under a box
            cv2.circle(im, (int(ox), int(oy)), 16, (0, 255, 0), 3, cv2.LINE_AA)
            cv2.putText(im, "TRUTH", (int(ox) - 26, int(oy) - 22), 0, 0.55,
                        (0, 255, 0), 2, cv2.LINE_AA)
            cv2.rectangle(im, (0, 0), (im.shape[1], 28), (0, 0, 0), -1)
            cv2.putText(im, f"{pid[:8]} f{fi}   green=truth  amber=our #1  red=our #2-5",
                        (8, 20), 0, 0.6, (0, 255, 255), 1, cv2.LINE_AA)
            panels.append(im)

    assert panels, "nothing rendered"
    cv2.imwrite(a.out, np.vstack(panels))
    print("\nwrote", a.out)
    print("\nREAD IT THIS WAY: boxes on white/round things (logos, helmet, plate) = it learned")
    print("'small bright blob' and needs more epochs. Boxes on arbitrary grass/dirt texture =")
    print("something structural is wrong and more epochs will not fix it.")


if __name__ == "__main__":
    main()
