"""Is the zone graphic in OUR MLB clips? Sweep every frame, do not pick one.

docs/2026-09-11-step2-pinhole-parity.md says our copy of the video does NOT carry
the drawn zone box, and that no angle MLB serves us carries it. That conclusion
rests on ONE play (c4958213, frames 0-240), while OpenCommand's own README - quoted
two lines above it - says "sometimes broadcasts don't draw a strikezone box".

zone_zoom.png then showed a sharp white rectangle sitting exactly on the projected
zone in 3 of 3 plays I happened to pick. Three hand-picked frames do not overturn a
written finding, and one hand-picked play did not establish it either. So: score
EVERY frame of each clip, including the exact play the doc used.

SCORE. The projected box outline is known, so build three masks from it - the border
ring itself, the same ring eroded inward, the same ring dilated outward - and take
    mean(border) - mean(inner and outer neighbours)
A drawn bright line makes the border brighter than BOTH its neighbours, which no
player, wall or uniform edge does at exactly that rectangle.

Reports, per clip: how many frames score high, and WHERE in the clip they sit. A
graphic that flashes only near the crossing looks completely different from one
that is always on, and both differ from absent.

    python scripts/box_presence_sweep.py [--game 824821] [--n 6] [--strip]
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
DOC_PLAY = "c4958213"          # the play the doc concluded "no graphic" from
HI = 8.0                       # border brighter than neighbours by this = drawn line
STRIP_KEEP = 400               # --strip: frames retained at once. The long clips here run
                               # 1602-1678 frames, and holding all of them as 1280x720x3
                               # is ~4.4 GB - an instant watchdog kill on this box.


def box_masks(q, shape):
    """border ring of the projected quad, plus its inner and outer neighbours."""
    poly = [q.round().astype(np.int32)]
    ring = np.zeros(shape[:2], np.uint8)
    cv2.polylines(ring, poly, True, 255, 2, cv2.LINE_8)
    thick = np.zeros(shape[:2], np.uint8)
    cv2.polylines(thick, poly, True, 255, 10, cv2.LINE_8)
    neigh = cv2.subtract(thick, cv2.dilate(ring, np.ones((5, 5), np.uint8)))
    return ring > 0, neigh > 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", type=int, default=824821)
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--strip", action="store_true", help="also render a filmstrip")
    ap.add_argument("--out", default=os.path.join(ROOT, "output", "box_presence_strip.png"))
    a = ap.parse_args()

    clips_dir = os.path.join(ROOT, "data", "clips", f"mlb_{a.game}")
    assert os.path.isdir(clips_dir), clips_dir

    pbp = pd.read_csv(os.path.join(OC, f"pbp_info_{SEASON}_{a.game}.csv")).set_index("play_id")

    # POSE SOURCE DIFFERS BY GAME. 824821 has output/parity_opencommand_<g>.csv - our
    # solve merged against OpenCommand's, so every column is suffixed _us / _oc. The
    # other four games have only data/opencommand/poses_<g>.csv, with BARE names.
    # Normalise to bare so everything downstream is uniform.
    par_csv = os.path.join(ROOT, "output", f"parity_opencommand_{a.game}.csv")
    oc_csv = os.path.join(OC, f"poses_{a.game}.csv")
    if os.path.exists(par_csv):
        poses = pd.read_csv(par_csv).set_index("play_id").rename(
            columns={f"{c}_us": c for c in ("Cx", "Cz", "pan", "tilt", "roll", "f_px")})
        src = "parity (our own solve)"
    elif os.path.exists(oc_csv):
        poses = pd.read_csv(oc_csv).set_index("play_id")
        src = "opencommand published poses"
    else:
        raise SystemExit(f"no pose file for game {a.game}: tried\n  {par_csv}\n  {oc_csv}")
    missing = [c for c in ("Cx", "Cz", "pan", "tilt", "roll", "f_px") if c not in poses.columns]
    assert not missing, f"pose file missing {missing}; has {list(poses.columns)}"

    # Ball labels exist ONLY for games we ran ball_clock_align on (824821, 824259,
    # 824341). Miami 823850 and Anaheim 824018 were never labelled - and this sweep does
    # not need them, since plate_fr is reporting only. An empty frame with the right
    # columns keeps the downstream plate_fr logic working unchanged.
    lab_csv = os.path.join(ROOT, "output", f"ball_labels_{a.game}.csv")
    labels = (pd.read_csv(lab_csv) if os.path.exists(lab_csv)
              else pd.DataFrame(columns=["play_id", "frame_idx"]))
    zy = ph.zone_y(SEASON)
    park = poses.park.iloc[0] if "park" in poses.columns and len(poses) else "?"
    print(f"game {a.game} ({park}): {len(poses)} poses from {src}; "
          f"labels {'present' if len(labels) else 'NONE - plate_fr reads -1'}")

    # always include the doc's play first, if we hold a pose for it
    ids = [p for p in poses.index if str(p).startswith(DOC_PLAY)]
    ids += [p for p in poses.index if p not in ids][: max(0, a.n - len(ids))]

    strip = []
    print(f"{'play':10} {'frames':>7} {'hi':>5} {'hi%':>6} {'first':>6} {'last':>6} "
          f"{'peak':>7} {'@frame':>7}  plate_fr")
    for pid in ids:
        if pid not in pbp.index:
            continue
        clip = os.path.join(clips_dir, f"{pid}_BROADCAST.mp4")
        if not os.path.exists(clip):
            print(f"{str(pid)[:8]:10} NO CLIP ON DISK")
            continue
        p, r = pbp.loc[pid], poses.loc[pid]
        top, bot = float(p.sz_top_ref), float(p.sz_bot_ref)
        corners = np.array([[-17 / 24, zy, top], [17 / 24, zy, top],
                            [17 / 24, zy, bot], [-17 / 24, zy, bot]])
        # bare names: the loader above renames the parity CSV's _us suffixes away, so
        # both pose sources look identical here
        C = np.array([r.Cx, ph.CY_FT, r.Cz])
        q = ph.project(C, r.pan, r.tilt, r.roll, r.f_px, corners)

        cap = cv2.VideoCapture(clip)
        scores, frames = [], []
        ring = neigh = None
        while True:
            ok, f = cap.read()
            if not ok:
                break
            if ring is None:
                ring, neigh = box_masks(q, f.shape)
            g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32)
            scores.append(float(g[ring].mean() - g[neigh].mean()))
            # MEMORY: only the handful of indices in `picks` below are ever rendered, so
            # retaining every decoded frame is pure waste - and on the long clips in this
            # set (1602 and 1678 frames) it is ~4.4 GB of 1280x720x3, which on this box
            # means an instant watchdog kill. Keep a bounded ring and index into it.
            if a.strip:
                frames.append(f)
                if len(frames) > STRIP_KEEP:
                    frames[len(frames) - 1 - STRIP_KEEP] = None   # drop the old pixels
            else:
                frames.append(None)
        cap.release()
        if not scores:
            continue
        s = np.array(scores)
        hi = np.where(s >= HI)[0]
        lab = labels[labels.play_id == pid]
        pf = int(lab.frame_idx.max()) if not lab.empty else -1
        print(f"{str(pid)[:8]:10} {len(s):7} {len(hi):5} {100*len(hi)/len(s):5.1f}% "
              f"{(hi.min() if len(hi) else -1):6} {(hi.max() if len(hi) else -1):6} "
              f"{s.max():7.2f} {int(s.argmax()):7}  {pf}")

        if a.strip and frames:
            cx, cy = int(q[:, 0].mean()), int(q[:, 1].mean())
            x0 = max(0, min(cx - 90, ph.W - 180)); y0 = max(0, min(cy - 90, ph.H - 180))
            picks = [0, len(s) // 4, len(s) // 2, max(0, pf - 4), pf if pf >= 0 else len(s) - 1]
            row = []
            for fi in picks:
                if fi >= len(frames) or frames[fi] is None:
                    continue
                c = cv2.resize(frames[fi][y0:y0 + 180, x0:x0 + 180], (360, 360),
                               interpolation=cv2.INTER_NEAREST)
                cv2.rectangle(c, (0, 0), (360, 22), (0, 0, 0), -1)
                cv2.putText(c, f"{str(pid)[:8]} f{fi} s={s[fi]:.1f}", (5, 16), 0, 0.45,
                            (0, 255, 255), 1, cv2.LINE_AA)
                row.append(c)
            if row:
                strip.append(np.hstack(row))

    print(f"\nscore = mean(border ring) - mean(neighbours), grey levels. "
          f"'hi' counts frames >= {HI}.")
    print("all frames hi  => graphic always on.  a burst near plate_fr => it flashes.  "
          "none => genuinely absent.")
    if strip:
        cv2.imwrite(a.out, np.vstack(strip))
        print("wrote", a.out)


if __name__ == "__main__":
    main()
