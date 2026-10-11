"""Eye-check camera-projected ball labels BEFORE they train anything.

project_labels.py writes where a validated park camera says the ball is on every flight frame.
A label that misses the ball teaches the detector to fire on grass, and no loss curve will show
it: the 2026-09-17 harvest reported 21/24 clips "kept" at 10.9x over baseline while every label
sat on empty grass (index-and-time-origins.md #3). Only a render caught that. So this is a
standing step, not a debug print.

WHAT IT SAMPLES. The labels worth checking are the ones the detector did NOT already have:
  unconfirmed     no detector candidate near the projected point (--all-frames rows). The ball
                  may be over a low-contrast background (useful) or hidden by the batter (harmful)
  missed-by-chain a candidate was there, but the detector's chain did not pick it
  control         already in the detector's chain - should be a clean hit every time
Each tile is a crop around the label at 2x, the projected ball drawn as a circle of the
projected diameter. Frames are read by SEQUENTIAL decode, the convention project_labels uses.

    python scripts/render_proj_labels.py --game 815419 --clips data/clips/aaa_815419
"""
import argparse
import os

import cv2
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HALF = 48          # crop half-width in source pixels; tile shows 96x96 at 2x
SCALE = 2
GROUPS = (("unconfirmed", (180, 60, 230)), ("missed-by-chain", (40, 160, 255)),
          ("control", (60, 200, 60)))


def pick(lab, n_each, rng):
    lab = lab.copy()
    lab["group"] = np.where(~lab.confirmed, "unconfirmed",
                            np.where(~lab.in_chain, "missed-by-chain", "control"))
    out = []
    for name, _ in GROUPS:
        g = lab[lab.group == name]
        k = n_each // 2 if name == "control" else n_each
        if len(g):
            out.append(g.sample(min(k, len(g)), random_state=rng))
    return pd.concat(out) if out else lab.iloc[:0]


def tile(frame, r, color):
    H, W = frame.shape[:2]
    x, y = float(r.x), float(r.y)
    x0, y0 = int(round(x)) - HALF, int(round(y)) - HALF
    pad = cv2.copyMakeBorder(frame, HALF, HALF, HALF, HALF, cv2.BORDER_CONSTANT, value=(30, 30, 30))
    crop = pad[y0 + HALF:y0 + 3 * HALF, x0 + HALF:x0 + 3 * HALF]
    crop = cv2.resize(crop, (4 * HALF * SCALE // 2, 4 * HALF * SCALE // 2),
                      interpolation=cv2.INTER_NEAREST)
    c = ((x - x0) * SCALE, (y - y0) * SCALE)
    rad = max(3, int(round(float(r.diameter_px) * SCALE / 2)))
    cv2.circle(crop, (int(round(c[0])), int(round(c[1]))), rad + 3, color, 1, cv2.LINE_AA)
    cap = "%s f%d" % (str(r.stem)[-10:], int(r.frame))
    cv2.putText(crop, cap, (3, 11), cv2.FONT_HERSHEY_SIMPLEX, 0.33, (255, 255, 255), 1, cv2.LINE_AA)
    sub = "" if r.det_dist != r.det_dist else "det %.0fpx" % float(r.det_dist)
    cv2.putText(crop, sub, (3, crop.shape[0] - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.33,
                (255, 255, 255), 1, cv2.LINE_AA)
    cv2.rectangle(crop, (0, 0), (crop.shape[1] - 1, crop.shape[0] - 1), color, 2)
    return crop


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", type=int, required=True)
    ap.add_argument("--clips", required=True)
    ap.add_argument("--labels", default=None, help="default output/proj_labels_<game>.csv")
    ap.add_argument("--n", type=int, default=12, help="tiles per group (control gets half)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cols", type=int, default=10)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    clips = a.clips if os.path.isabs(a.clips) else os.path.join(ROOT, a.clips)
    assert os.path.isdir(clips), clips
    lp = a.labels or os.path.join(ROOT, "output", "proj_labels_%d.csv" % a.game)
    lab = pd.read_csv(lp)
    assert len(lab), "no labels in %s" % lp
    s = pick(lab, a.n, a.seed)
    colors = dict(GROUPS)
    tiles = {name: [] for name, _ in GROUPS}
    for stem, g in s.groupby("stem"):
        want = {int(f): r for f, r in zip(g.frame, g.itertuples())}
        cap = cv2.VideoCapture(os.path.join(clips, "%s.mp4" % stem))
        i, hi = 0, max(want)
        while i <= hi:
            ok, fr = cap.read()
            if not ok:
                break
            r = want.get(i)
            i += 1
            if r is not None:
                tiles[r.group].append(tile(fr, r, colors[r.group]))
        cap.release()
    rows = []
    tw = 4 * HALF * SCALE // 2
    for name, color in GROUPS:
        t = tiles[name]
        if not t:
            continue
        head = np.full((22, a.cols * tw, 3), 255, np.uint8)
        n_all = int((s.group == name).sum())
        cv2.putText(head, "%s  (%d shown)" % (name, len(t)), (4, 16), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, color, 1, cv2.LINE_AA)
        rows.append(head)
        for k in range(0, len(t), a.cols):
            chunk = t[k:k + a.cols]
            chunk += [np.full((tw, tw, 3), 255, np.uint8)] * (a.cols - len(chunk))
            rows.append(np.hstack(chunk))
        assert n_all >= len(t)
    sheet = np.vstack(rows)
    out = a.out or os.path.join(ROOT, "output", "proj_labels_%d_sheet.png" % a.game)
    cv2.imwrite(out, sheet)
    counts = lab.assign(group=np.where(~lab.confirmed, "unconfirmed",
                                       np.where(~lab.in_chain, "missed-by-chain", "control")))
    print("game %d: %s" % (a.game, counts.group.value_counts().to_dict()))
    print("->", out)


if __name__ == "__main__":
    main()
