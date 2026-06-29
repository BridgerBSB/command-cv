"""Synthesize a short behind-plate test clip with a 'glove flash'.

Writes output/test_clip.mp4 (glove drifts in, then HOLDS at the target = the
glove-flash frame) plus output/test_clip_coords.json with the plate / zone /
glove pixel coords at the flash frame. Lets us exercise the full provided-video
path (decode -> annotate -> report) without a real broadcast clip.

Run: python scripts/make_test_clip.py
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.synth_frame import IMG_H, IMG_W, build

FLASH_FRAME = 20
N_FRAMES = 30
FPS = 30.0


def main() -> None:
    import cv2

    sf = build()  # gives consistent plate/zone/glove pixel coords
    det = sf.detections
    gx, gy = det.glove_px

    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "output")
    os.makedirs(out_dir, exist_ok=True)
    clip_path = os.path.join(out_dir, "test_clip.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(clip_path, fourcc, FPS, (IMG_W, IMG_H))

    for i in range(N_FRAMES):
        img = np.full((IMG_H, IMG_W, 3), (52, 93, 63), np.uint8)  # BGR field green
        # strike-zone quad (yellow)
        pts = np.array(det.zone_pts, np.int32).reshape(-1, 1, 2)
        cv2.polylines(img, [pts], True, (0, 255, 255), 2)
        # plate line (white)
        (lx, ly), (rx, ry) = det.plate_pts
        cv2.line(img, (int(lx), int(ly)), (int(rx), int(ry)), (255, 255, 255), 3)
        # glove: drift in for frames < FLASH, then hold
        if i < FLASH_FRAME:
            t = i / FLASH_FRAME
            cx = gx + (1 - t) * 120
            cy = gy + (1 - t) * -80
        else:
            cx, cy = gx, gy
        cv2.circle(img, (int(cx), int(cy)), 22, (42, 42, 42), -1)
        cv2.circle(img, (int(cx), int(cy)), 22, (140, 180, 200), 3)
        vw.write(img)
    vw.release()

    coords = {
        "flash_frame": FLASH_FRAME,
        "fps": FPS,
        "plate_pts": [list(p) for p in det.plate_pts],
        "zone_pts": [list(p) for p in det.zone_pts],
        "glove_px": list(det.glove_px),
    }
    coords_path = os.path.join(out_dir, "test_clip_coords.json")
    with open(coords_path, "w") as f:
        json.dump(coords, f, indent=2)

    print("wrote", clip_path)
    print("wrote", coords_path, "(flash frame", FLASH_FRAME, ")")


if __name__ == "__main__":
    main()
