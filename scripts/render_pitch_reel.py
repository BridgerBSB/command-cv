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
GREEN = (80, 230, 80)
PRE, POST, HOLD = 25, 12, 60   # frames before first track pt, after last, freeze length


def _reticle(img, x, y, r=16):
    x, y = int(round(x)), int(round(y))
    cv2.circle(img, (x, y), r, ORANGE, 2, cv2.LINE_AA)
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        cv2.line(img, (x + dx * (r - 6), y + dy * (r - 6)), (x + dx * (r + 8), y + dy * (r + 8)),
                 ORANGE, 2, cv2.LINE_AA)


def _caption(img, lines, n, total, headline=None):
    h = 34 * len(lines) + 16 + (40 if headline else 0)
    ov = img.copy()
    cv2.rectangle(ov, (0, 0), (660, h), NAVY, -1)
    cv2.addWeighted(ov, 0.78, img, 0.22, 0, img)
    y0 = 36
    for i, t in enumerate(lines):
        cv2.putText(img, t, (14, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.8 if i else 0.9,
                    WHITE, 2, cv2.LINE_AA)
        y0 += 34
        if i == 0 and headline:
            cv2.putText(img, headline, (14, y0 + 8), cv2.FONT_HERSHEY_DUPLEX, 1.0, ORANGE, 2,
                        cv2.LINE_AA)
            y0 += 40
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
    cv2.circle(img, (400, y - 4), 9, GREEN, 2, cv2.LINE_AA)
    cv2.putText(img, "glove (model)", (418, y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 2,
                cv2.LINE_AA)


def track_gloves(clips, weights, cache, conf=0.05, imgsz=640, batch=32, topk=5):
    """Per-frame mitt CANDIDATES (top-k, low threshold) over each clip's reel window, cached.

    clips = [(stem, path, f_lo, f_hi)]. Sequential decode (never seek). The threshold is low
    on purpose: at 0.25 the mitt is found on ~20% of frames; at 0.05 on ~95%, but top-1 then
    sits on the umpire's mask through the windup (ogando_fo_3). So this returns candidates and
    `glove_path` picks among them by continuity.
    """
    have = pd.read_csv(cache) if cache and os.path.exists(cache) else pd.DataFrame(
        columns=["stem", "frame", "x", "y", "w", "h", "conf"])
    todo = [c for c in clips if c[0] not in set(have.stem)]
    if not todo:
        return have
    from ultralytics import YOLO
    model = YOLO(weights)
    rows = []
    for k, (s, path, lo, hi) in enumerate(todo, 1):
        cap = cv2.VideoCapture(path)
        fi, frames, idx = -1, [], []
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            fi += 1
            if fi < lo:
                continue
            if fi > hi:
                break
            frames.append(fr)
            idx.append(fi)
        cap.release()
        for i in range(0, len(frames), batch):
            res = model.predict(frames[i:i + batch], imgsz=imgsz, conf=conf, verbose=False,
                                device=0)
            for f_, r in zip(idx[i:i + batch], res):
                c = r.boxes.conf.cpu().numpy()
                xywh = r.boxes.xywh.cpu().numpy()
                for j in np.argsort(-c)[:topk]:
                    x, y, w, h = xywh[j]
                    rows.append((s, f_, float(x), float(y), float(w), float(h), float(c[j])))
        if k % 20 == 0:
            print(f"  glove: {k}/{len(todo)} clips")
    new = pd.DataFrame(rows, columns=["stem", "frame", "x", "y", "w", "h", "conf"])
    out = pd.concat([have, new], ignore_index=True)
    if cache:
        out.to_csv(cache, index=False)
    print(f"glove: {new.stem.nunique()}/{len(todo)} new clips with a mitt, {len(new)} frames")
    return out


def glove_path(cands, seed=None, anchor_conf=0.35, base=25.0, per_gap=2.0, cap=45.0):
    """Pick ONE glove position per frame from low-threshold candidates, by continuity.

    Anchors are trustworthy points: every frame whose best candidate clears `anchor_conf`
    (in practice the glove after the catch), plus `seed` = (frame, x, y), the set-up target
    where one was measured. From each anchor walk outward frame by frame, accepting the
    candidate nearest the last accepted point only if it lies within
    base + per_gap * frames_since_last (capped). A frame with nothing in the gate stays
    EMPTY -- nothing is interpolated. Returns {frame: (x, y)}.
    """
    if not len(cands):
        return {}
    by_f = {f: g[["x", "y", "conf"]].to_numpy() for f, g in cands.groupby("frame")}
    frames = sorted(by_f)
    path = {}
    anchors = [(f, *by_f[f][np.argmax(by_f[f][:, 2])][:2]) for f in frames
               if by_f[f][:, 2].max() >= anchor_conf]
    if seed is not None:
        anchors.append(seed)
    for f0, x0, y0 in anchors:
        path.setdefault(int(f0), (x0, y0))
        for step in (1, -1):
            lx, ly, lf = x0, y0, f0
            f = f0 + step
            while frames and frames[0] <= f <= frames[-1]:
                if abs(f - lf) * per_gap + base > cap * 1.5:
                    break
                if f in by_f:
                    d = np.hypot(by_f[f][:, 0] - lx, by_f[f][:, 1] - ly)
                    j = int(np.argmin(d))
                    if d[j] <= min(base + per_gap * (abs(f - lf) - 1), cap):
                        lx, ly, lf = by_f[f][j, 0], by_f[f][j, 1], f
                        path.setdefault(int(f), (lx, ly))
                f += step
    return path


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
    ap.add_argument("--glove-weights", default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data", "yolo", "runs", "mitt", "weights", "best.pt"),
        help="mitt detector; '' to skip glove tracking")
    ap.add_argument("--glove-cache", default="", help="CSV to reuse/extend glove detections")
    ap.add_argument("--glove-games", nargs="*", default=["822515"],
                    help="games where the glove circle is drawn (validated parks only)")
    ap.add_argument("--glove-all", action="store_true", help="draw the glove at every park")
    a = ap.parse_args()

    tr = pd.concat([pd.read_csv(f) for f in a.tracks], ignore_index=True)
    games = [str(g) for g in a.game]
    tr = tr[tr.game_pk.astype(str).isin(games)]
    tg = pd.read_csv(a.targets).set_index("stem") if a.targets else pd.DataFrame()
    ms = pd.read_csv(a.miss).set_index("stem") if a.miss else pd.DataFrame()
    man_p = [f for f in os.listdir(a.dir) if f.endswith("_manifest.csv")]
    man = pd.read_csv(os.path.join(a.dir, man_p[0]))
    man = man[man.game_pk.astype(str).isin(games)].copy()
    if "date" in man:
        man = man.sort_values("date", kind="stable")   # season order across games
    stems = [s for s in man.stem if os.path.exists(os.path.join(a.dir, s + ".mp4"))]
    assert stems, f"no clips for games {games} in {a.dir}"
    # play in game order = manifest order
    print(f"games {games}: {len(stems)} clips, {tr.stem.nunique()} with a ball track, "
          f"{ms.index.nunique()} with a miss row")

    def _window(s):
        t = tr[tr.stem == s]
        return ((int(t.frame.min()) - PRE, int(t.frame.max()) + POST) if len(t) else (160, 235))

    gl = pd.DataFrame(columns=["stem", "frame", "x", "y"])
    if a.glove_weights:
        gstems = [s for s in stems if a.glove_all or
                  str(man.loc[man.stem == s, "game_pk"].iloc[0]) in a.glove_games]
        gl = track_gloves([(s, os.path.join(a.dir, s + ".mp4"), *_window(s)) for s in gstems],
                          a.glove_weights, a.glove_cache)

    w = imageio_ffmpeg.write_frames(a.out, (1280, 720), fps=30, codec="libx264",
                                    pix_fmt_out="yuv420p", quality=7, macro_block_size=8)
    w.send(None)
    ends = []
    for n, s in enumerate(stems, 1):
        t = tr[tr.stem == s].sort_values("frame")
        mrow = man[man.stem == s].iloc[0]
        f_lo, f_hi = _window(s)
        # Glove only where it has been validated: the mitt detector was trained and checked on
        # Asheville, and at other parks it wanders onto the chest protector (render 2026-09-28).
        g = gl[gl.stem == s] if (a.glove_all or str(mrow.game_pk) in a.glove_games) else gl.iloc[:0]
        seed = (f_lo, *tg.loc[s, ["raw_px", "raw_py"]].to_numpy()) if s in tg.index else None
        gpts = glove_path(g, seed)
        f_arrive = int(t.frame.max()) if len(t) else None
        pts = {int(r.frame): (r.x, r.y) for r in t.itertuples()}
        tgt = None
        if s in tg.index:
            tgt = (tg.loc[s, "raw_px"], tg.loc[s, "raw_py"])
        lines = [f"{a.pitcher + ' - ' if a.pitcher else ''}{mrow.pitch_type} - {mrow.call}"]
        headline = None
        if s in ms.index:
            r = ms.loc[s]
            headline = f'MISS {r.final_miss:.1f} in ' + ("(video)" if r.final_src == "B" else "(data)")
            lines.append("video: " + _fmt_miss(r.B_dx, r.B_dz))
            lines.append("data:  " + _fmt_miss(r.A_dx, r.A_dz))
        else:
            # a game with no miss rows at all = no camera at that park; otherwise a per-pitch gap
            solved = len(ms) and man.loc[man.stem.isin(ms.index), "game_pk"].eq(mrow.game_pk).any()
            lines.append("no target / no miss row" if solved
                         else "ball only - park camera not solved yet")
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
            gtrail = [gpts[k] for k in sorted(gpts) if f_lo <= k <= fi]
            for i in range(1, len(gtrail)):
                cv2.line(fr, tuple(map(int, gtrail[i - 1])), tuple(map(int, gtrail[i])),
                         GREEN, 1, cv2.LINE_AA)
            if fi in gpts:
                cv2.circle(fr, tuple(map(int, gpts[fi])), 22, GREEN, 2, cv2.LINE_AA)
            trail = [pts[k] for k in sorted(pts) if k <= fi]
            for i in range(1, len(trail)):
                cv2.line(fr, tuple(map(int, trail[i - 1])), tuple(map(int, trail[i])), CYAN, 2,
                         cv2.LINE_AA)
            if fi in pts:
                cv2.circle(fr, tuple(map(int, pts[fi])), 13, CYAN, 2, cv2.LINE_AA)
            show = headline if (headline and (f_arrive is None or fi >= f_arrive)) else (
                "MISS ..." if headline else None)
            _caption(fr, lines, n, len(stems), show)
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
