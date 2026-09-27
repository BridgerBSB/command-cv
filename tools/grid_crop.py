"""Zoomed crop of one clip frame with a labelled pixel grid, for reading landmark pixels by eye.

Frames are read by SEQUENTIAL decode (never cap.set(POS_FRAMES): 0-2 frame seek error,
index-and-time-origins.md #2). Grid labels are FULL-FRAME pixel coordinates.

    python tools/grid_crop.py CLIP.mp4 FRAME x0 y0 x1 y1 [--scale 6] [--step 5] --out p.png
"""
import argparse

import cv2


def read_frame(path, idx):
    cap = cv2.VideoCapture(path)
    i, fr = 0, None
    while True:
        ok, f = cap.read()
        if not ok:
            break
        if i == idx:
            fr = f
            break
        i += 1
    cap.release()
    assert fr is not None, (path, idx)
    return fr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("clip"); ap.add_argument("frame", type=int)
    ap.add_argument("box", type=int, nargs=4)
    ap.add_argument("--scale", type=int, default=6)
    ap.add_argument("--step", type=int, default=5)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    x0, y0, x1, y1 = a.box
    fr = read_frame(a.clip, a.frame)[y0:y1, x0:x1]
    s = a.scale
    z = cv2.resize(fr, None, fx=s, fy=s, interpolation=cv2.INTER_NEAREST)
    for x in range(x0 - x0 % a.step, x1, a.step):
        if x < x0:
            continue
        major = x % (a.step * 2) == 0
        cv2.line(z, ((x - x0) * s, 0), ((x - x0) * s, z.shape[0]), (0, 255, 255) if major else (0, 140, 140), 1)
        if major:
            cv2.putText(z, str(x), ((x - x0) * s + 2, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
    for y in range(y0 - y0 % a.step, y1, a.step):
        if y < y0:
            continue
        major = y % (a.step * 2) == 0
        cv2.line(z, (0, (y - y0) * s), (z.shape[1], (y - y0) * s), (255, 0, 255) if major else (140, 0, 140), 1)
        if major:
            cv2.putText(z, str(y), (2, (y - y0) * s - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1)
    cv2.imwrite(a.out, z)
    print(a.out, z.shape)


if __name__ == "__main__":
    main()
