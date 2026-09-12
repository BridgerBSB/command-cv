"""Draw the strike zone on real broadcast video, through OUR solved camera.

Two jobs in one picture, per pitch:

  LEFT   the broadcast frame EXACTLY as served, nothing drawn. This is the control.
         `ball_path_check_824821.png` shows a white rectangle near the catcher that
         NOTHING in our render code draws (ball_clock_align.py draws only a green
         circle, a black caption bar and yellow text). Either it is in the source
         feed - which would contradict "every MLB feed is clean" - or it comes from
         somewhere we have not looked. An untouched frame settles it.

  RIGHT  the same frame with the zone we PROJECT ourselves: the four corners at
         +/-17/24 ft, sz_bot_ref..sz_top_ref high, on the zone plane, pushed through
         ph.project() with this clip's solved pose. Plus the ball, and - where
         OpenCommand recorded one - their observed box corners in a second colour so
         the two can be compared directly.

We never harvested a drawn box. We solve the camera, so we can draw the zone
ourselves, which is the stronger artifact: it works at A+ where no box exists.

    python scripts/render_zone_demo.py [--game 824821] [--n 3]
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", type=int, default=824821)
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--out", default=os.path.join(ROOT, "output", "zone_demo.png"))
    a = ap.parse_args()

    clips_dir = os.path.join(ROOT, "data", "clips", f"mlb_{a.game}")
    assert os.path.isdir(clips_dir), clips_dir          # a bad root renders nothing, silently

    pbp = pd.read_csv(os.path.join(OC, f"pbp_info_{SEASON}_{a.game}.csv")).set_index("play_id")
    poses = pd.read_csv(os.path.join(ROOT, "output", f"parity_opencommand_{a.game}.csv")).set_index("play_id")
    labels = pd.read_csv(os.path.join(ROOT, "output", f"ball_labels_{a.game}.csv"))

    # OpenCommand's OBSERVED zone box, if they recorded one for this park
    zt = pd.read_csv(os.path.join(OC, f"strikezone_tracking_{SEASON}.csv.gz"))
    zt = zt[zt.game_pk == a.game].set_index("play_id")
    print(f"strikezone_tracking rows for game {a.game}: {len(zt)}")

    zy = ph.zone_y(SEASON)
    print(f"zone plane y = {zy:.4f} ft ({'mid-plate, ABS' if SEASON >= 2026 else 'plate front'})")

    panels = []
    for pid in [p for p in poses.index if p in pbp.index][: a.n]:
        clip = os.path.join(clips_dir, f"{pid}_BROADCAST.mp4")
        if not os.path.exists(clip):
            continue
        lab = labels[labels.play_id == pid]
        if lab.empty:
            continue
        fi = int(lab.frame_idx.max())            # last labelled frame ~ the crossing

        cap = cv2.VideoCapture(clip)
        frame = None
        i = 0
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
        raw = frame.copy()
        drawn = frame.copy()

        # our projected zone: 4 corners on the zone plane
        top, bot = float(p.sz_top_ref), float(p.sz_bot_ref)
        corners = np.array([[-17 / 24, zy, top], [17 / 24, zy, top],
                            [17 / 24, zy, bot], [-17 / 24, zy, bot]])
        C = np.array([r.Cx_us, ph.CY_FT, r.Cz_us])
        q = ph.project(C, r.pan_us, r.tilt_us, r.roll_us, r.f_px_us, corners)
        cv2.polylines(drawn, [q.round().astype(int)], True, (0, 200, 255), 2, cv2.LINE_AA)

        # the ball at this frame, from the label
        row = lab[lab.frame_idx == fi].iloc[0]
        cv2.circle(drawn, (int(round(row.x)), int(round(row.y))), 13, (0, 255, 0), 2, cv2.LINE_AA)

        # OpenCommand's observed box, for comparison
        note = "no OC box recorded"
        if pid in zt.index:
            b = zt.loc[pid]
            if not any(pd.isna([b.x_left, b.y_top, b.x_right, b.y_bottom])):
                cv2.rectangle(drawn, (int(b.x_left), int(b.y_top)),
                              (int(b.x_right), int(b.y_bottom)), (255, 0, 255), 2)
                note = "magenta = OC observed box"

        for im, tag in ((raw, f"RAW as served  {pid[:8]} f{fi}"),
                        (drawn, f"OURS projected  amber=zone green=ball  {note}")):
            cv2.rectangle(im, (0, 0), (ph.W, 26), (0, 0, 0), -1)
            cv2.putText(im, tag, (8, 19), 0, 0.6, (0, 255, 255), 1, cv2.LINE_AA)
        panels.append(np.hstack([raw, drawn]))
        print(f"{pid[:8]} f{fi} {p.pitch_type}  zone top {top:.2f} bot {bot:.2f} ft  "
              f"corners px {q.round(0).tolist()}")

    assert panels, "nothing rendered"
    cv2.imwrite(a.out, np.vstack(panels))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
