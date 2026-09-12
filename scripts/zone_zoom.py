"""Is the zone box IN the MLB feed, and does our projection land on it?

zone_demo.png raised both questions and answered neither cleanly: the raw panel
seems to show a faint white rectangle near the catcher, and the overlay panel drew
magenta (OpenCommand's observed box) ON TOP of amber (ours), so agreement between
them was invisible.

Four panels per pitch, same crop, zoomed, layers separated so nothing hides anything:

  1 RAW        not one pixel drawn. If a box is visible here it is in the SOURCE,
               which matters because our own docstrings say the MLB feed draws none.
  2 +OURS      amber: the 4 zone corners at +/-17/24 ft, sz_bot..sz_top, on the zone
               plane, through this clip's solved pose. Never sees the drawn box.
  3 +OC        magenta: OpenCommand's OBSERVED box pixels from strikezone_tracking.
  4 BOTH       the comparison. If 2 and 3 coincide, our camera reproduces a thing it
               was never shown.

    python scripts/zone_zoom.py [--game 824821] [--n 2]
"""
import argparse
import os
import sys

import cv2
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src import pinhole as ph  # noqa: E402

OC = os.path.join(ROOT, "data", "opencommand")
SEASON = 2026
PAD, ZOOM = 110, 4


def label(im, txt, colour=(0, 255, 255)):
    cv2.rectangle(im, (0, 0), (im.shape[1], 30), (0, 0, 0), -1)
    cv2.putText(im, txt, (8, 21), 0, 0.62, colour, 2, cv2.LINE_AA)
    return im


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", type=int, default=824821)
    ap.add_argument("--n", type=int, default=2)
    ap.add_argument("--out", default=os.path.join(ROOT, "output", "zone_zoom.png"))
    a = ap.parse_args()

    clips_dir = os.path.join(ROOT, "data", "clips", f"mlb_{a.game}")
    assert os.path.isdir(clips_dir), clips_dir

    pbp = pd.read_csv(os.path.join(OC, f"pbp_info_{SEASON}_{a.game}.csv")).set_index("play_id")
    poses = pd.read_csv(os.path.join(ROOT, "output", f"parity_opencommand_{a.game}.csv")).set_index("play_id")
    labels = pd.read_csv(os.path.join(ROOT, "output", f"ball_labels_{a.game}.csv"))
    zt = pd.read_csv(os.path.join(OC, f"strikezone_tracking_{SEASON}.csv.gz"))
    zt = zt[zt.game_pk == a.game].set_index("play_id")
    zy = ph.zone_y(SEASON)

    rows_out = []
    done = 0
    for pid in poses.index:
        if done >= a.n or pid not in pbp.index or pid not in zt.index:
            continue
        b = zt.loc[pid]
        if any(pd.isna([b.x_left, b.y_top, b.x_right, b.y_bottom])):
            continue
        clip = os.path.join(clips_dir, f"{pid}_BROADCAST.mp4")
        lab = labels[labels.play_id == pid]
        if not os.path.exists(clip) or lab.empty:
            continue
        fi = int(lab.frame_idx.max())
        cap = cv2.VideoCapture(clip)
        frame, i = None, 0
        while True:
            ok, f = cap.read()
            if not ok:
                break
            if i == fi:
                frame = f
                break
            i += 1
        cap.release()
        if frame is None:
            continue

        p, r = pbp.loc[pid], poses.loc[pid]
        top, bot = float(p.sz_top_ref), float(p.sz_bot_ref)
        corners = np.array([[-17 / 24, zy, top], [17 / 24, zy, top],
                            [17 / 24, zy, bot], [-17 / 24, zy, bot]])
        C = np.array([r.Cx_us, ph.CY_FT, r.Cz_us])
        q = ph.project(C, r.pan_us, r.tilt_us, r.roll_us, r.f_px_us, corners)

        cx, cy = int(q[:, 0].mean()), int(q[:, 1].mean())
        x0 = max(0, min(cx - PAD, ph.W - 2 * PAD))
        y0 = max(0, min(cy - PAD, ph.H - 2 * PAD))

        def crop(img):
            c = img[y0:y0 + 2 * PAD, x0:x0 + 2 * PAD]
            return cv2.resize(c, (2 * PAD * ZOOM, 2 * PAD * ZOOM), interpolation=cv2.INTER_NEAREST)

        def ours(img):
            cv2.polylines(img, [((q - [x0, y0]) * ZOOM).round().astype(int)], True,
                          (0, 200, 255), 3, cv2.LINE_AA)

        def oc(img):
            p1 = ((np.array([b.x_left, b.y_top]) - [x0, y0]) * ZOOM).round().astype(int)
            p2 = ((np.array([b.x_right, b.y_bottom]) - [x0, y0]) * ZOOM).round().astype(int)
            cv2.rectangle(img, tuple(p1), tuple(p2), (255, 0, 255), 3)

        panels = []
        for tag, fns in (("1 RAW - nothing drawn", []),
                         ("2 +OURS (amber, projected)", [ours]),
                         ("3 +OPENCOMMAND (magenta, observed)", [oc]),
                         ("4 BOTH", [ours, oc])):
            im = crop(frame)
            for fn in fns:
                fn(im)
            panels.append(label(im, f"{tag}   {pid[:8]} f{fi}"))
        rows_out.append(np.hstack(panels))

        d = np.abs(np.array([q[:, 0].min(), q[:, 1].min(), q[:, 0].max(), q[:, 1].max()])
                   - np.array([b.x_left, b.y_top, b.x_right, b.y_bottom]))
        print(f"{pid[:8]} f{fi}  ours L{q[:,0].min():.1f} T{q[:,1].min():.1f} "
              f"R{q[:,0].max():.1f} B{q[:,1].max():.1f} | "
              f"OC L{b.x_left:.1f} T{b.y_top:.1f} R{b.x_right:.1f} B{b.y_bottom:.1f} | "
              f"|diff| px L{d[0]:.1f} T{d[1]:.1f} R{d[2]:.1f} B{d[3]:.1f}")
        done += 1

    assert rows_out, "nothing rendered"
    cv2.imwrite(a.out, np.vstack(rows_out))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
