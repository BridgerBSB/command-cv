"""Click-to-annotate a clip's glove-flash frame -> coords.json.

Usable today (before the detector is trained): open the chosen frame, click the
reference points in order, and write the coords JSON that clip_to_report.py
consumes. Doubles as the labeling tool that seeds the YOLO training set.

Click ORDER (9 clicks):
  1-2  home-plate front corners: LEFT then RIGHT
  3-6  strike-zone corners: TOP-LEFT, TOP-RIGHT, BOTTOM-RIGHT, BOTTOM-LEFT
  7    catcher's glove center (the intended target)

Run: python scripts/annotate_clip.py --clip mypitch.mp4 --frame 120 --out coords.json
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.video import frame_at_index, frame_at_time

_LABELS = [
    "PLATE left corner", "PLATE right corner",
    "ZONE top-left", "ZONE top-right", "ZONE bottom-right", "ZONE bottom-left",
    "GLOVE center (intended target)",
]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--clip", required=True)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--frame", type=int)
    g.add_argument("--time", type=float)
    p.add_argument("--out", required=True)
    a = p.parse_args()

    frame = frame_at_index(a.clip, a.frame) if a.frame is not None \
        else frame_at_time(a.clip, a.time)

    import matplotlib
    matplotlib.use("TkAgg")  # interactive backend for clicking
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(frame)
    ax.set_title("Click in order — see console for the current point")
    pts = []
    for label in _LABELS:
        ax.set_xlabel(f"NEXT: {label}")
        fig.canvas.draw()
        print(f"click: {label}")
        (xy,) = plt.ginput(1, timeout=0)
        pts.append([float(xy[0]), float(xy[1])])
        ax.scatter([xy[0]], [xy[1]], c="orange", s=60)
        fig.canvas.draw()
    plt.close(fig)

    coords = {
        "frame": a.frame, "time": a.time,
        "plate_pts": pts[0:2],
        "zone_pts": pts[2:6],
        "glove_px": pts[6],
    }
    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(coords, f, indent=2)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
