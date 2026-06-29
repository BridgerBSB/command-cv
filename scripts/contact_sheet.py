"""Dump a labeled montage of frames from a clip to pick the setup + catch frames.

Run: python scripts/contact_sheet.py --clip mypitch.mp4 [--start 80 --end 125 --step 3]
Writes output/<clip>_contact.png — each tile labeled with its frame index.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main() -> None:
    import cv2
    import numpy as np

    p = argparse.ArgumentParser()
    p.add_argument("--clip", required=True)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--end", type=int, default=-1)
    p.add_argument("--step", type=int, default=4)
    p.add_argument("--cols", type=int, default=6)
    p.add_argument("--crop", action="store_true",
                   help="crop to the catcher region (right-center) and zoom")
    a = p.parse_args()

    cap = cv2.VideoCapture(a.clip)
    frames = []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        frames.append(fr)
    cap.release()
    n = len(frames)
    end = a.end if a.end >= 0 else n - 1
    idxs = list(range(a.start, min(end, n - 1) + 1, a.step))

    tiles = []
    for i in idxs:
        fr = frames[i]
        if a.crop:
            h, w = fr.shape[:2]
            fr = fr[int(h * 0.40):int(h * 0.92), int(w * 0.52):int(w * 0.84)]
            fr = cv2.resize(fr, None, fx=1.6, fy=1.6)
        else:
            fr = cv2.resize(fr, (320, 180))
        cv2.putText(fr, f"f{i}", (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 255, 0), 2)
        tiles.append(fr)

    th, tw = tiles[0].shape[:2]
    tiles = [cv2.resize(t, (tw, th)) for t in tiles]
    cols = a.cols
    rows = (len(tiles) + cols - 1) // cols
    sheet = np.zeros((rows * th, cols * tw, 3), np.uint8)
    for k, t in enumerate(tiles):
        r, c = divmod(k, cols)
        sheet[r * th:(r + 1) * th, c * tw:(c + 1) * tw] = t

    out = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output",
        os.path.splitext(os.path.basename(a.clip))[0] + "_contact.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    cv2.imwrite(out, sheet)
    print("frames", idxs[0], "..", idxs[-1], "step", a.step, "->", out)


if __name__ == "__main__":
    main()
