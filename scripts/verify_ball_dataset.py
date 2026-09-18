"""Check a built YOLO dataset by reading what SHIPPED, not what produced it.

WHY THIS READS THE DATASET AND NOT THE LABEL CSV. Everything upstream -- the seed tracks, the
pose, the audit, the diameter -- can be right while the dataset is still wrong, because
build_ball_dataset crops, jitters, converts to normalised xywh and writes negatives in
between. Verifying the CSV verifies the CSV. This opens the .png and its .txt and asks
whether that box is on a baseball.

TWO CHECKS, because one of them cannot fail on its own.

  1. CONTRAST AT THE BOX vs AT A DELIBERATELY-OFFSET CONTROL. A baseball is a small bright
     compact object, so the disc at the box centre should sit well above the ring around it.
     Measuring only at the label proves nothing -- a number needs something to beat. The
     control samples the same statistic at a point far enough away that it CANNOT be the
     ball, so the gap between them is the evidence. (This is the check the Sep 12 run did on
     yolo_ball5 and got +21.7 at the label against +2.0 at the control; it lived in a
     scratchpad and did not survive it, which is why it is a committed script now.)

  2. A ZOOM SHEET. The contrast check can be satisfied by any bright blob -- a helmet logo,
     a shoe, a scoreboard digit. Only looking separates a baseball from those, and this
     project has had three good-looking aggregates overturned by a render.

NEGATIVES ARE COUNTED, NOT VERIFIED. An empty label file asserts "no ball in this crop" and
nothing here can confirm that cheaply. The count is printed so a missing or runaway negative
rate is visible.

    python scripts/verify_ball_dataset.py --data data/yolo_milb3 --n 400 --render 12
"""
import argparse
import glob
import os
import random

import cv2
import numpy as np


def disc_ring(img, cx, cy, r_disc, r_ring):
    """(mean inside a disc) - (mean in the ring around it), in grey levels.

    The same local-contrast idea the seeder's top-hat uses: a compact bright object beats its
    immediate surroundings, while a bright BACKGROUND (sky, a white wall, a lit panel) does
    not, because the ring is just as bright.
    """
    h, w = img.shape[:2]
    yy, xx = np.ogrid[:h, :w]
    d2 = (xx - cx) ** 2 + (yy - cy) ** 2
    disc = d2 <= r_disc ** 2
    ring = (d2 > r_disc ** 2) & (d2 <= r_ring ** 2)
    if disc.sum() < 4 or ring.sum() < 8:
        return np.nan
    return float(img[disc].mean() - img[ring].mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="dataset dir containing data.yaml")
    ap.add_argument("--n", type=int, default=400, help="labelled crops to measure")
    ap.add_argument("--render", type=int, default=12, help="crops to draw on the sheet")
    ap.add_argument("--offset", type=int, default=90, help="control offset in px")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    rng = random.Random(a.seed)
    rows = []
    for split in ("train", "val"):
        for ip in sorted(glob.glob(os.path.join(a.data, "images", split, "*.png"))):
            lp = os.path.join(a.data, "labels", split,
                              os.path.basename(ip)[:-4] + ".txt")
            txt = open(lp).read().strip() if os.path.exists(lp) else ""
            rows.append((split, ip, txt))

    pos = [r for r in rows if r[2]]
    neg = [r for r in rows if not r[2]]
    print(f"{a.data}: {len(rows)} images -- {len(pos)} labelled, {len(neg)} negatives "
          f"({100 * len(neg) / max(1, len(rows)):.0f}%)")
    n_tr = sum(1 for r in rows if r[0] == "train")
    print(f"  split: {n_tr} train / {len(rows) - n_tr} val")
    assert pos, "no labelled images at all -- nothing to verify"

    sample = rng.sample(pos, min(a.n, len(pos)))
    at_label, at_control, sides = [], [], []
    for split, ip, txt in sample:
        img = cv2.imread(ip, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        h, w = img.shape[:2]
        _, cx, cy, bw, bh = [float(v) for v in txt.split()[:5]]
        cx, cy, bw = cx * w, cy * h, bw * w
        sides.append(bw)
        r_disc = max(2.0, bw / 2.0)
        v = disc_ring(img, cx, cy, r_disc, r_disc * 3)
        # the control must land INSIDE the crop and far enough out that the ball cannot be
        # in it; mirror the offset when the first direction would leave the image
        ox = cx + a.offset if cx + a.offset + r_disc * 3 < w else cx - a.offset
        oy = cy + a.offset if cy + a.offset + r_disc * 3 < h else cy - a.offset
        c = disc_ring(img, ox, oy, r_disc, r_disc * 3)
        if np.isfinite(v):
            at_label.append(v)
        if np.isfinite(c):
            at_control.append(c)

    al, ac = np.array(at_label), np.array(at_control)
    print(f"\ncontrast (disc minus surrounding ring, grey levels), n={len(al)}")
    print(f"  AT THE BOX        median {np.median(al):7.2f}   "
          f"IQR {np.percentile(al, 25):6.2f} .. {np.percentile(al, 75):6.2f}")
    print(f"  AT +{a.offset}px CONTROL  median {np.median(ac):7.2f}   "
          f"IQR {np.percentile(ac, 25):6.2f} .. {np.percentile(ac, 75):6.2f}")
    print(f"  gap {np.median(al) - np.median(ac):+.2f}   "
          f"{100 * float((al > np.percentile(ac, 95)).mean()):.0f}% of boxes beat the "
          f"control's 95th percentile")
    print(f"  box side: median {np.median(sides):.1f} px, "
          f"range {min(sides):.1f}-{max(sides):.1f}")
    print("\n  A gap near zero means the boxes are NOT on compact bright objects. A large "
          "gap means\n  they are on SOMETHING compact and bright -- which a helmet logo or "
          "a shoe also satisfies.\n  That is what the sheet is for.")

    if a.render:
        picks = rng.sample(sample, min(a.render, len(sample)))
        tiles, Z, T = [], 4, 180
        for split, ip, txt in picks:
            img = cv2.imread(ip)
            h, w = img.shape[:2]
            _, cx, cy, bw, bh = [float(v) for v in txt.split()[:5]]
            cx, cy, bw, bh = cx * w, cy * h, bw * w, bh * h
            r = T // (2 * Z)
            y0, x0 = max(0, int(cy) - r), max(0, int(cx) - r)
            crop = img[y0:y0 + 2 * r, x0:x0 + 2 * r]
            if crop.size == 0:
                continue
            t = cv2.resize(crop, (T, T), interpolation=cv2.INTER_NEAREST)
            sx, sy = T / (2 * r), T / (2 * r)
            bx, by = (cx - x0) * sx, (cy - y0) * sy
            cv2.rectangle(t, (int(bx - bw * sx / 2), int(by - bh * sy / 2)),
                          (int(bx + bw * sx / 2), int(by + bh * sy / 2)), (0, 255, 0), 2)
            cv2.putText(t, f"{split} {os.path.basename(ip)[:14]}", (3, 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 255, 255), 1)
            tiles.append(t)
        if tiles:
            per = 4
            while len(tiles) % per:
                tiles.append(np.zeros((T, T, 3), np.uint8))
            sheet = np.vstack([np.hstack(tiles[i:i + per])
                               for i in range(0, len(tiles), per)])
            out = os.path.join(os.path.dirname(os.path.abspath(a.data)) or ".",
                               f"verify_{os.path.basename(a.data)}.png")
            out = os.path.join("output", f"verify_{os.path.basename(a.data)}.png")
            os.makedirs("output", exist_ok=True)
            cv2.imwrite(out, sheet)
            print(f"\n{out}\n  Every green box must contain a baseball. The box is drawn at "
                  f"the size the\n  detector will be trained on, so check it is not visibly "
                  f"too small or too large\n  for the ball inside it.")


if __name__ == "__main__":
    main()
