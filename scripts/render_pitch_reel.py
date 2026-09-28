"""Watch the model: one reel of every pitch in a game, ball + target drawn on the real video.

Per pitch it plays the broadcast clip from just before release to just after the catch and
draws, on the ORIGINAL frames (no stabilization, no crop):
  - the glove TARGET at set-up (orange reticle)            <- targets_<game>_*.csv raw_px/raw_py
  - the detector's ball (cyan ring) with a fading trail      <- detect_tracks_<game>_*.csv
  - a caption: pitch type, call, and the miss in inches both ways (video / data)
                                                            <- command_miss_<game>_*.csv
then holds the last frame for ~1 s with the whole flight drawn.

Frames are decoded IN ORDER (never cv2 seek: it lands 0-2 frames early on these clips, see
.claude/rules/index-and-time-origins.md), so a track's `frame` is the same frame drawn here.

A pitch with no detector track still plays, captioned "no ball track" -- nothing is invented.

    python scripts/render_pitch_reel.py --game 822515 --dir data/clips/ogando_2026 \
        --tracks output/detect_tracks_822515_v4.csv \
        --targets output/targets_822515_stab2c.csv \
        --miss output/command_miss_822515_s2c.csv --out output/reel_822515.mp4
"""
import argparse
import os

import cv2
import imageio_ffmpeg
import numpy as np
import pandas as pd

ORANGE = (31, 110, 235)   # BGR, Astros orange
NAVY = (98, 45, 0)
CYAN = (255, 230, 0)
WHITE = (255, 255, 255)
PRE, POST, HOLD = 25, 12, 60   # frames before first track pt, after last, freeze length


def _reticle(img, x, y, r=16):
    x, y = int(round(x)), int(round(y))
    cv2.circle(img, (x, y), r, ORANGE, 2, cv2.LINE_AA)
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        cv2.line(img, (x + dx * (r - 6), y + dy * (r - 6)), (x + dx * (r + 8), y + dy * (r + 8)),
                 ORANGE, 2, cv2.LINE_AA)


def _caption(img, lines, n, total):
    h = 34 * len(lines) + 16
    ov = img.copy()
    cv2.rectangle(ov, (0, 0), (660, h), NAVY, -1)
    cv2.addWeighted(ov, 0.78, img, 0.22, 0, img)
    for i, t in enumerate(lines):
        cv2.putText(img, t, (14, 36 + 34 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.8 if i else 0.9,
                    WHITE, 2, cv2.LINE_AA)
    cv2.putText(img, f"{n}/{total}", (img.shape[1] - 110, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                WHITE, 2, cv2.LINE_AA)
    # legend
    y = img.shape[0] - 24
    _reticle(img, 30, y - 4, r=9)
    cv2.putText(img, "catcher target", (50, y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 2,
                cv2.LINE_AA)
    cv2.circle(img, (230, y - 4), 9, CYAN, 2, cv2.LINE_AA)
    cv2.putText(img, "ball (model)", (248, y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 2,
                cv2.LINE_AA)


def _fmt_miss(dx, dz):
    if pd.isna(dx):
        return "-"
    lr = "catcher's R" if dx > 0 else "catcher's L"   # plate_x convention: + = catcher's right
    ud = "up" if dz > 0 else "down"
    return f'{np.hypot(dx, dz):.1f}" ({abs(dx):.1f} {lr}, {abs(dz):.1f} {ud})'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", required=True, nargs="+")
    ap.add_argument("--dir", required=True)
    ap.add_argument("--tracks", required=True, nargs="+")
    ap.add_argument("--targets", default="", help="omit at a park with no camera: no target, no inches")
    ap.add_argument("--miss", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--sheet", default=None, help="optional contact sheet PNG of end frames")
    ap.add_argument("--pitcher", default="")
    a = ap.parse_args()

    tr = pd.concat([pd.read_csv(f) for f in a.tracks], ignore_index=True)
    games = [str(g) for g in a.game]
    tr = tr[tr.game_pk.astype(str).isin(games)]
    tg = pd.read_csv(a.targets).set_index("stem") if a.targets else pd.DataFrame()
    ms = pd.read_csv(a.miss).set_index("stem") if a.miss else pd.DataFrame()
    man_p = [f for f in os.listdir(a.dir) if f.endswith("_manifest.csv")]
    man = pd.read_csv(os.path.join(a.dir, man_p[0]))
    man = man[man.game_pk.astype(str).isin(games)].copy()
    stems = [s for s in man.stem if os.path.exists(os.path.join(a.dir, s + ".mp4"))]
    assert stems, f"no clips for games {games} in {a.dir}"
    # play in game order = manifest order
    print(f"games {games}: {len(stems)} clips, {tr.stem.nunique()} with a ball track, "
          f"{ms.index.nunique()} with a miss row")

    w = imageio_ffmpeg.write_frames(a.out, (1280, 720), fps=30, codec="libx264",
                                    pix_fmt_out="yuv420p", quality=7, macro_block_size=8)
    w.send(None)
    ends = []
    for n, s in enumerate(stems, 1):
        t = tr[tr.stem == s].sort_values("frame")
        mrow = man[man.stem == s].iloc[0]
        f_lo = int(t.frame.min()) - PRE if len(t) else 160
        f_hi = int(t.frame.max()) + POST if len(t) else 235
        pts = {int(r.frame): (r.x, r.y) for r in t.itertuples()}
        tgt = None
        if s in tg.index:
            tgt = (tg.loc[s, "raw_px"], tg.loc[s, "raw_py"])
        lines = [f"{a.pitcher + ' - ' if a.pitcher else ''}{mrow.pitch_type} - {mrow.call}"]
        if s in ms.index:
            r = ms.loc[s]
            lines.append("miss, video: " + _fmt_miss(r.B_dx, r.B_dz))
            lines.append("miss, data:  " + _fmt_miss(r.A_dx, r.A_dz))
        else:
            lines.append("ball only - park camera not solved yet" if not a.miss
                         else "no target / no miss row")
        if not len(t):
            lines.append("no ball track")

        cap = cv2.VideoCapture(os.path.join(a.dir, s + ".mp4"))
        fi, last = -1, None
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            fi += 1
            if fi < f_lo:
                continue
            if fi > f_hi:
                break
            if fr.shape[:2] != (720, 1280):
                fr = cv2.resize(fr, (1280, 720))
            if tgt is not None:
                _reticle(fr, *tgt)
            trail = [pts[k] for k in sorted(pts) if k <= fi]
            for i in range(1, len(trail)):
                cv2.line(fr, tuple(map(int, trail[i - 1])), tuple(map(int, trail[i])), CYAN, 2,
                         cv2.LINE_AA)
            if fi in pts:
                cv2.circle(fr, tuple(map(int, pts[fi])), 13, CYAN, 2, cv2.LINE_AA)
            _caption(fr, lines, n, len(stems))
            w.send(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB).tobytes())
            last = fr
        cap.release()
        if last is None:
            print("  unreadable:", s)
            continue
        for _ in range(HOLD):
            w.send(cv2.cvtColor(last, cv2.COLOR_BGR2RGB).tobytes())
        ends.append(last)
    w.close()
    print("wrote", a.out, f"({len(ends)} pitches)")

    if a.sheet and ends:
        cols = 6
        th = [cv2.resize(e, (426, 240)) for e in ends]
        while len(th) % cols:
            th.append(np.zeros_like(th[0]))
        rows = [np.hstack(th[i:i + cols]) for i in range(0, len(th), cols)]
        cv2.imwrite(a.sheet, np.vstack(rows))
        print("wrote", a.sheet)


if __name__ == "__main__":
    main()
