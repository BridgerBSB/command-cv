"""Map every clip at a park into ONE reference camera view, so one pose fits them all.

WHY. The command miss (command_miss.py) holds up at Asheville and Frawley and scatters 4-5 in
per pitch at Durham and Isotopes (PROGRESS-LOG 85). Broadcast cameras there change framing
between pitches, so a single park pose is wrong for many clips, for BOTH roots: the glove
target goes through the same pose as the video ball.

WHAT. A broadcast camera pans and zooms about a fixed point, and a pure rotation + zoom maps
the whole image by a homography regardless of scene depth. So each frame is related to a
reference frame by one 3x3 H, estimated from the STATIC background with RANSAC.

THE OVERLAY BUG (fixed 2026-09-27, v2 outputs). The first version used ORB over the whole
frame. The SCOREBUG is the one thing guaranteed identical in every clip, so on any clip whose
framing differed from the reference, ORB's 300+ "inliers" were the scorebug and RANSAC
returned ~identity. Isotopes wide clips (0000 / 0050 / 0120) mapped at zoom ~0.92-1.00 when
the truth, found with the overlay masked, is 1.26-1.28 relative to the reference, i.e.
those clips were stabilized with a 27% scale error. Masked ORB found only 7-64 inliers on
the same pairs, so the fix is two parts:
  1. MASK every broadcast overlay (per-park boxes in data/landmarks/<game>.json
     `overlay_boxes`, or --overlay). Features inside a mask never enter the match.
  2. SIFT + Lowe ratio test instead of ORB: it matches across the zoom change ORB missed.
Outputs are written with a `2` suffix (targets_<g>_stab2.csv, detect_tracks_<g>_v4stab2.csv)
so the broken v1 files stay distinguishable. Do not use the v1 files.

A MATCH MUST ALSO BE SPREAD OUT. Inliers clustered in one corner (a residual overlay, a
single ad board) fit a homography that is right there and wrong everywhere else, so a frame
is kept only if its inliers span at least MIN_SPREAD of the frame width and height.

PER CLIP, H is computed at the set-up frame and at the first and last tracked ball frames;
each ball point uses the H of the nearest of those. The camera does not re-frame during a
pitch, and this is ~8x fewer SIFT passes than one per frame. Every H is written to
stab2_H_<game>.csv so a later change of reference costs one extra homography, not a rerun.

SET-UPS are cached to output/setups_<game>.csv (stem, x, y, frame); a rerun reuses them and
skips the mitt-detector pass, which is most of the runtime (~1.5 h for 327 clips).

FRAMES are decoded SEQUENTIALLY, never by seek (index-and-time-origins.md, instance 2).

    python scripts/stabilize_park.py --game 815419 --dir data/clips/aaa_815419 \\
        --tracks output/detect_tracks_815419_v4.csv
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

MIN_INLIERS = 30
MIN_SPREAD = 0.25
RATIO = 0.75


class Matcher:
    def __init__(self, ref_gray, mask):
        self.sift = cv2.SIFT_create(nfeatures=6000)
        self.mask = mask
        self.bf = cv2.BFMatcher()
        self.rk, self.rd = self.sift.detectAndCompute(ref_gray, mask)
        self.shape = ref_gray.shape

    def to_ref(self, gray):
        """(H mapping THIS frame into the reference, inlier count, spread ok?)."""
        k, d = self.sift.detectAndCompute(gray, self.mask)
        if d is None or len(k) < 10:
            return None, 0, False
        m = self.bf.knnMatch(d, self.rd, k=2)
        good = [p[0] for p in m if len(p) == 2 and p[0].distance < RATIO * p[1].distance]
        if len(good) < 10:
            return None, 0, False
        src = np.float32([k[x.queryIdx].pt for x in good])
        dst = np.float32([self.rk[x.trainIdx].pt for x in good])
        H, inl = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)
        if H is None:
            return None, 0, False
        inl = inl.ravel().astype(bool)
        p = dst[inl]
        h, w = self.shape
        spread = (np.ptp(p[:, 0]) >= MIN_SPREAD * w) and (np.ptp(p[:, 1]) >= MIN_SPREAD * h) if len(p) else False
        return H, int(inl.sum()), bool(spread)


# kept for scripts that import the old helpers (fit/diagnostics); same maths
def apply(H, x, y):
    p = cv2.perspectiveTransform(np.float32([[[x, y]]]), H)[0, 0]
    return float(p[0]), float(p[1])


def zoom(H):
    return float(np.sqrt(abs(np.linalg.det(H[:2, :2] / H[2, 2]))))


def overlay_mask(shape, boxes):
    m = np.full(shape, 255, np.uint8)
    for x0, y0, x1, y1 in boxes:
        m[int(y0):int(y1), int(x0):int(x1)] = 0
    return m


def read_frames(path, want):
    cap = cv2.VideoCapture(path)
    i, got, top = 0, {}, max(want)
    while i <= top:
        ok, fr = cap.read()
        if not ok:
            break
        if i in want:
            got[i] = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        i += 1
    cap.release()
    return got


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", type=int, required=True)
    ap.add_argument("--dir", required=True)
    ap.add_argument("--tracks", required=True)
    ap.add_argument("--mitt-weights", default="data/yolo/runs/mitt/weights/best.pt")
    ap.add_argument("--overlay", action="append", default=[],
                    help="x0,y0,x1,y1 box to mask (repeatable); default from data/landmarks/<game>.json")
    ap.add_argument("--limit", type=int, default=0, help="first N clips only (smoke test)")
    a = ap.parse_args()
    d = a.dir if os.path.isabs(a.dir) else os.path.join(ROOT, a.dir)
    assert os.path.isdir(d), d
    man = pd.read_csv(os.path.join(d, os.path.basename(d) + "_manifest.csv"))
    man = man[man.game_pk == a.game]
    stems = [s for s in man.stem if os.path.exists(os.path.join(d, s + ".mp4"))]
    if a.limit:
        stems = stems[:a.limit]
    assert stems, "no clips on disk"
    trk = pd.read_csv(os.path.join(ROOT, a.tracks), float_precision="round_trip")

    lm_p = os.path.join(ROOT, "data", "landmarks", "%d.json" % a.game)
    lm = json.load(open(lm_p, encoding="utf-8")) if os.path.exists(lm_p) else {}
    boxes = [tuple(float(v) for v in b.split(",")) for b in a.overlay] or lm.get("overlay_boxes", [])
    assert boxes, "no overlay boxes: pass --overlay or add overlay_boxes to %s (the scorebug " \
                  "is what broke v1)" % lm_p
    print("overlay mask boxes:", boxes)

    # SET-UPS, cached
    cache = os.path.join(ROOT, "output", "setups_%d.csv" % a.game)
    setups = {}
    if os.path.exists(cache):
        c = pd.read_csv(cache)
        setups = {r.stem: (r.x, r.y, int(r.frame)) for r in c.itertuples()}
        print("set-ups from cache: %d" % len(setups))
    todo = [s for s in stems if s not in setups]
    if todo:
        from ultralytics import YOLO
        from src.auto_glove import detect_setup_catch
        mitt = YOLO(os.path.join(ROOT, a.mitt_weights))
        for i, s in enumerate(todo, 1):
            r = detect_setup_catch(os.path.join(d, s + ".mp4"), mitt)
            if r.get("ok"):
                setups[s] = r["setup"]
            if i % 20 == 0:
                print("  setup %d/%d" % (i, len(todo)), flush=True)
                pd.DataFrame([dict(stem=k, x=v[0], y=v[1], frame=int(v[2])) for k, v in setups.items()]
                             ).to_csv(cache, index=False)
        pd.DataFrame([dict(stem=k, x=v[0], y=v[1], frame=int(v[2])) for k, v in setups.items()]
                     ).to_csv(cache, index=False)
    assert setups, "no set-up targets found"

    # REFERENCE: the landmark json's, else the set-up nearest the median (v1 rule)
    if lm.get("ref_stem"):
        ref_stem, ref_frame = lm["ref_stem"], int(lm["ref_frame"])
    else:
        sx = np.median([v[0] for v in setups.values()]); sy = np.median([v[1] for v in setups.values()])
        ref_stem = min(setups, key=lambda s: np.hypot(setups[s][0] - sx, setups[s][1] - sy))
        ref_frame = int(setups[ref_stem][2])
    ref = read_frames(os.path.join(d, ref_stem + ".mp4"), {ref_frame})[ref_frame]
    M = Matcher(ref, overlay_mask(ref.shape, boxes))
    print("reference view: %s frame %d" % (ref_stem, ref_frame), flush=True)

    tgt_rows, trk_rows, clip_rows, h_rows = [], [], [], []
    for i, s in enumerate(stems, 1):
        g = trk[trk.stem == s].sort_values("frame")
        key = set()
        if s in setups:
            key.add(int(setups[s][2]))
        if len(g):
            key |= {int(g.frame.iloc[0]), int(g.frame.iloc[-1])}
        if not key:
            continue
        got = read_frames(os.path.join(d, s + ".mp4"), key)
        Hs = {}
        for f_, im in got.items():
            H, n, spread = M.to_ref(im)
            Hs[f_] = (H, n, spread)
            h_rows.append(dict(stem=s, frame=f_, inliers=n, spread=spread,
                               **({"h%d" % k: float(v) for k, v in enumerate((H / H[2, 2]).ravel())}
                                  if H is not None else {})))
        good = {f_: v for f_, v in Hs.items() if v[0] is not None and v[1] >= MIN_INLIERS and v[2]}
        if s in setups and int(setups[s][2]) in good:
            H = good[int(setups[s][2])][0]
            px, py = apply(H, setups[s][0], setups[s][1])
            tgt_rows.append(dict(stem=s, px=px, py=py, raw_px=setups[s][0], raw_py=setups[s][1],
                                 inliers=good[int(setups[s][2])][1], zoom=zoom(H)))
        kept = 0
        if good:
            gf = np.array(sorted(good))
            for r in g.itertuples():
                f_ = int(gf[np.argmin(np.abs(gf - r.frame))])
                x, y = apply(good[f_][0], r.x, r.y)
                trk_rows.append(dict(stem=s, game_pk=a.game, frame=int(r.frame), x=x, y=y,
                                     conf=r.conf, inliers=good[f_][1], h_frame=f_))
                kept += 1
        zs = [zoom(v[0]) for v in good.values()]
        clip_rows.append(dict(stem=s, track_pts=len(g), kept=kept, n_h=len(Hs), n_h_good=len(good),
                              zoom_med=float(np.median(zs)) if zs else np.nan))
        if i % 20 == 0:
            print("  stabilized %d/%d" % (i, len(stems)), flush=True)

    sfx = "" if not a.limit else "_smoke"
    out_t = os.path.join(ROOT, "output", "targets_%d_stab2%s.csv" % (a.game, sfx))
    out_k = os.path.join(ROOT, "output", "detect_tracks_%d_v4stab2%s.csv" % (a.game, sfx))
    out_c = os.path.join(ROOT, "output", "stab2_clips_%d%s.csv" % (a.game, sfx))
    out_h = os.path.join(ROOT, "output", "stab2_H_%d%s.csv" % (a.game, sfx))
    pd.DataFrame(tgt_rows).to_csv(out_t, index=False)
    pd.DataFrame(trk_rows).to_csv(out_k, index=False, float_format="%.17g")
    c = pd.DataFrame(clip_rows)
    c.to_csv(out_c, index=False)
    pd.DataFrame(h_rows).to_csv(out_h, index=False, float_format="%.17g")
    print("\ngame %d: %d clips, reference %s f%d" % (a.game, len(stems), ref_stem, ref_frame))
    print("  targets mapped %d of %d found; track points kept %d of %d"
          % (len(tgt_rows), len(setups), int(c.kept.sum()), int(c.track_pts.sum())))
    print("  clips with NO usable homography: %d" % int((c.n_h_good == 0).sum()))
    z = c.zoom_med.dropna()
    print("  zoom vs reference, per clip: p05 %.3f  median %.3f  p95 %.3f"
          % (z.quantile(.05), z.median(), z.quantile(.95)))
    print("  zoom histogram:", np.histogram(z, bins=[0, .8, .9, .97, 1.03, 1.1, 1.2, 1.3, 1.5, 9])[0].tolist())
    for p in (out_t, out_k, out_c, out_h):
        print("  ->", p)


if __name__ == "__main__":
    main()
