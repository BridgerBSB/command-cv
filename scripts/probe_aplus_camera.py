"""Step 1 probe: does the A+ broadcast camera move while the ball is in the air?

Decides which OpenCommand pieces port to A+ (docs/2026-09-01-opencommand-v12-
and-cross-camera-accuracy.md section 4). Their phase 2 refits pan/tilt/roll/f
per clip from the drawn zone box and excludes the ball because "camera often
moves mid-ball flight". If our camera is static during the flight, a ball-only
per-clip refit is open to us even without a box.

Window: hand-labeled catch frame minus 30 (~0.5 s at 60 fps; flight ~0.42 s)
through the catch. Motion: LK optical flow on background corners in the CENTER
band of the crowd/wall (x 30-70%, y 5-35%), which excludes every corner
scorebug, then a RANSAC similarity fit per frame against the window's first
frame -> worst translation (px) and worst zoom change (%).

Do NOT swap this for whole-frame phase correlation: on the same 55 clips it
read up to 307 px because the pitcher and batter moving dominate it, while the
background did not move (verified by rendering the worst clips, Sep 11 2026).

    python scripts/probe_aplus_camera.py [--sheet out.png]
"""
import argparse
import glob
import json
import os

import cv2
import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ANN = os.path.join(HERE, "data", "annotations")
CLIPS = os.path.join(HERE, "data", "clips", "ogando_2026")


def flight_motion(clip, catch_frame, lead=30):
    c = cv2.VideoCapture(clip)
    s0 = catch_frame - lead
    c.set(cv2.CAP_PROP_POS_FRAMES, s0)
    ok, f0 = c.read()
    if not ok:
        return None
    g0 = cv2.cvtColor(f0, cv2.COLOR_BGR2GRAY)
    h, w = g0.shape
    mask = np.zeros_like(g0)
    mask[int(.05 * h):int(.35 * h), int(.30 * w):int(.70 * w)] = 255
    p0 = cv2.goodFeaturesToTrack(g0, 300, 0.01, 8, mask=mask)
    if p0 is None or len(p0) < 15:
        return None
    pts, prev, keep = p0.copy(), g0, np.ones(len(p0), bool)
    worst_t = worst_z = 0.0
    for _ in range(lead):
        ok, f = c.read()
        if not ok:
            return None
        g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        p1, st, _ = cv2.calcOpticalFlowPyrLK(prev, g, pts, None,
                                             winSize=(21, 21), maxLevel=3)
        keep &= st.ravel() == 1
        if keep.sum() < 15:
            return None
        m, _ = cv2.estimateAffinePartial2D(p0[keep], p1[keep], method=cv2.RANSAC,
                                           ransacReprojThreshold=2.0)
        if m is None:
            return None
        worst_t = max(worst_t, float(np.hypot(m[0, 2], m[1, 2])))
        worst_z = max(worst_z, abs(float(np.hypot(m[0, 0], m[1, 0])) - 1) * 100)
        prev, pts = g, p1
    return worst_t, worst_z, len(p0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sheet", help="write catch-30..catch frames of the 6 most-moving clips")
    a = ap.parse_args()
    assert os.path.isdir(ANN) and os.path.isdir(CLIPS), (ANN, CLIPS)
    files = sorted(glob.glob(os.path.join(ANN, "*.json")))
    rows, skipped = [], []
    for fp in files:
        d = json.load(open(fp))
        clip = os.path.join(CLIPS, d["stem"] + ".mp4")
        if not os.path.exists(clip) or not d.get("catch"):
            skipped.append(d["stem"])
            continue
        r = flight_motion(clip, d["catch"]["frame"])
        if r is None:
            skipped.append(d["stem"])
            continue
        gw = (d["setup"]["w"] + d["catch"]["w"]) / 2
        rows.append((d["stem"], r[0], r[1], r[0] * 12.0 / gw, d["catch"]["frame"]))
    print(f"annotations {len(files)} -> measured {len(rows)}, skipped {len(skipped)} {skipped}")
    t = np.array([r[1] for r in rows]); z = np.array([r[2] for r in rows])
    i = np.array([r[3] for r in rows])
    print(f"translation px   median {np.median(t):.2f}  p90 {np.percentile(t, 90):.2f}  max {t.max():.2f}")
    print(f"~inches @ glove  median {np.median(i):.2f}  p90 {np.percentile(i, 90):.2f}  max {i.max():.2f}")
    print(f"zoom change %    median {np.median(z):.3f}  p90 {np.percentile(z, 90):.3f}  max {z.max():.3f}")
    worst = sorted(rows, key=lambda r: -r[1])[:6]
    for s, tp, zp, ip, _ in worst:
        print(f"  {s:16} {tp:.2f}px  {zp:.3f}%  ~{ip:.2f}in")
    if a.sheet:
        tiles = []
        for s, *_, cf in worst:
            c = cv2.VideoCapture(os.path.join(CLIPS, s + ".mp4"))
            row = []
            for k in (cf - 30, cf - 15, cf - 5, cf):
                c.set(cv2.CAP_PROP_POS_FRAMES, k)
                ok, f = c.read()
                f = cv2.resize(f, (480, 270))
                cv2.putText(f, f"{s} catch{k - cf:+d}", (6, 18), 0, 0.5, (0, 255, 255), 1)
                row.append(f)
            tiles.append(np.hstack(row))
        cv2.imwrite(a.sheet, np.vstack(tiles))
        print(a.sheet)


if __name__ == "__main__":
    main()
