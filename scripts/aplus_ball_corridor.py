"""Find the ball's image corridor at a park by stacking many pitches, with no camera and no trajectory.

WHY THIS CAN WORK NOW AND COULD NOT BEFORE. aplus_camera_between_pitches.py measured the A+
camera as LOCKED across a whole game (Asheville 822515: median 0.13 px over 59 clips). One
pitcher throwing 60 pitches from one release point to one catcher, filmed by a camera that
never moves, puts every ball on nearly the SAME narrow band of pixels. A stationary
distractor -- the Greensboro crowd object at (826.5, 236.5), the catcher's gear at
(800, 462) that took top-1 on ~35 of 56 flight frames on Sep 13 -- is by definition a POINT
in that stack, not a band. Stacking separates them on shape alone.

This is the instrument the Sep 13 top-K tracker needed and did not have. That attempt tried
to pick the ball's chain inside ONE clip, where a crowd object and a pitch are both just
"some detections", and it failed twice (v1 ranked jitter highest; v2's lone survivor was the
catcher's return throw). Across 60 clips they stop being confusable.

WHAT IT IS NOT. This finds WHERE the ball goes, not WHEN. It is the seed for a pose, which
then lets ball_clock_align.py do the per-clip clock search it already does at MLB. Nothing
here is a label yet, and nothing here should be reported as a detection rate.

READ THE OUTPUT BY EYE, NOT BY SCORE. A corridor running from the release side of the frame
to the mitt is the result. Bright isolated blobs are fixed distractors and are the OTHER
useful output -- they are exactly what to mask. The script prints the top recurring fixed
points so they can be named, but the picture is the evidence.

MEMORY. ball_clock_align.motion_stack holds every diff in its window (~170 MB/clip) because
it needs random access by frame later. This one only accumulates, so it streams and holds two
frames. That matters: this box has been running near 1.8 GB free and an OOM kill on Sep 12
took out a training run.

    python scripts/aplus_ball_corridor.py --game 822515 --limit 12
    python scripts/aplus_ball_corridor.py --game 822246 --limit 12 --lo 100 --hi 320
"""
import argparse
import os
import sys
from collections import Counter

import cv2
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from ball_clock_align import TOPHAT_K  # noqa: E402  -- same kernel, deliberately

# A baseball is ~16 px wide in these 1280x720 feeds (MLB measurement, Sep 12). Allow well
# either side of that; the point is to exclude limbs and bat barrels, not to be precise.
MIN_AREA, MAX_AREA = 3, 160
MAX_SIDE = 26
THRESH = 28                      # top-hat response; a ball is bright against its ring


def blobs_in_clip(path, lo, hi):
    """Streaming: (frame, x, y, area, peak) for every small bright moving object in [lo, hi)."""
    cap = cv2.VideoCapture(path)
    out, prev, i = [], None, 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        if lo <= i < hi and prev is not None:
            th = cv2.morphologyEx(cv2.absdiff(g, prev), cv2.MORPH_TOPHAT, TOPHAT_K)
            _, bw = cv2.threshold(th, THRESH, 255, cv2.THRESH_BINARY)
            n, _, stats, cent = cv2.connectedComponentsWithStats(bw, 8)
            for k in range(1, n):
                x, y, w, h, area = stats[k]
                if MIN_AREA <= area <= MAX_AREA and w <= MAX_SIDE and h <= MAX_SIDE:
                    cx, cy = cent[k]
                    out.append((i, float(cx), float(cy), int(area),
                                int(th[y:y + h, x:x + w].max())))
        prev = g
        i += 1
    cap.release()
    return out, i


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", type=int, default=822515)
    ap.add_argument("--dir", default=os.path.join(ROOT, "data", "clips", "ogando_2026"))
    ap.add_argument("--lo", type=int, default=100)
    ap.add_argument("--hi", type=int, default=320)
    ap.add_argument("--limit", type=int, default=12)
    a = ap.parse_args()

    man = os.path.join(a.dir, os.path.basename(a.dir) + "_manifest.csv")
    m = pd.read_csv(man)
    g = m[m.game_pk == a.game].reset_index(drop=True)
    stems = [s for s in g.stem if os.path.exists(os.path.join(a.dir, s + ".mp4"))][:a.limit]
    assert stems, f"no clips on disk for game {a.game}"

    rows = []
    for j, s in enumerate(stems, 1):
        b, n = blobs_in_clip(os.path.join(a.dir, s + ".mp4"), a.lo, a.hi)
        for fr, x, y, area, peak in b:
            rows.append(dict(stem=s, frame=fr, x=x, y=y, area=area, peak=peak))
        print(f"  [{j}/{len(stems)}] {s:16} {n:5} frames  {len(b):6} blobs in "
              f"[{a.lo},{a.hi})", flush=True)
    d = pd.DataFrame(rows)
    assert len(d), "no blobs found at all -- check --lo/--hi and THRESH"

    # A fixed distractor recurs at the same pixel in MANY DIFFERENT clips. Count clips, not
    # blobs: one clip firing 200 times at one spot is one clip, not 200 pieces of evidence.
    d["cell"] = list(zip((d.x // 8).astype(int), (d.y // 8).astype(int)))
    per_cell_clips = d.groupby("cell").stem.nunique()
    fixed = per_cell_clips[per_cell_clips >= max(3, int(0.6 * len(stems)))].sort_values(ascending=False)

    print(f"\ngame {a.game}: {len(stems)} clips, {len(d)} blobs, "
          f"{d.cell.nunique()} occupied 8px cells")
    print(f"cells firing in >= {max(3, int(0.6 * len(stems)))} of {len(stems)} clips "
          f"(fixed-distractor candidates): {len(fixed)}")
    for cell, k in list(fixed.items())[:12]:
        print(f"    px ~({cell[0] * 8 + 4:4d},{cell[1] * 8 + 4:4d})  in {k:3d} clips")

    # heatmap: how many DISTINCT clips put a blob in each cell
    H, W = 720, 1280
    heat = np.zeros((H // 8, W // 8), np.float32)
    for (cx, cy), k in per_cell_clips.items():
        if 0 <= cy < heat.shape[0] and 0 <= cx < heat.shape[1]:
            heat[cy, cx] = k
    hm = cv2.resize(heat, (W, H), interpolation=cv2.INTER_NEAREST)
    hm = (255 * hm / max(1e-6, hm.max())).astype(np.uint8)
    hm = cv2.applyColorMap(hm, cv2.COLORMAP_INFERNO)

    cap = cv2.VideoCapture(os.path.join(a.dir, stems[0] + ".mp4"))
    cap.set(cv2.CAP_PROP_POS_FRAMES, a.lo)
    ok, bg = cap.read(); cap.release()
    bg = bg if ok else np.zeros((H, W, 3), np.uint8)
    blend = cv2.addWeighted(bg, 0.45, hm, 0.55, 0)
    cv2.putText(blend, f"game {a.game}  {len(stems)} clips  frames [{a.lo},{a.hi})  "
                       f"colour = # distinct clips with a blob in that cell",
                (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    out_png = os.path.join(ROOT, "output", f"ball_corridor_{a.game}.png")
    cv2.imwrite(out_png, np.vstack([blend, cv2.addWeighted(bg, 0.15, hm, 0.85, 0)]))
    out_csv = os.path.join(ROOT, "output", f"ball_blobs_{a.game}.csv")
    d.drop(columns=["cell"]).to_csv(out_csv, index=False)
    print(f"\n{out_png}\n{out_csv}")
    print("\nLOOK AT THE PNG. A band from the release side to the mitt is the corridor; "
          "isolated\nbright spots are fixed distractors to mask. Do not read a number off this.")


if __name__ == "__main__":
    main()
