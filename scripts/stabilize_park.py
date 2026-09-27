"""Map every clip at a park into ONE reference camera view, so one pose fits them all.

WHY. The command miss (command_miss.py) holds up at Asheville and Frawley and scatters 4-5 in
per pitch at Durham and Isotopes (PROGRESS-LOG 85). Durham's broadcast camera zooms between
pitches, 0.89-1.17 (row 54), so a single park pose is wrong for most clips, and it is wrong
for BOTH roots: the glove target goes through the same pose as the video ball.

WHAT. A broadcast camera pans and zooms about a fixed point, and a pure rotation + zoom maps
the whole image by a homography regardless of scene depth. So each frame is related to a
reference frame by one 3x3 H, estimated from the STATIC background with ORB + RANSAC -- the
instrument aplus_camera_between_pitches.py already uses, chosen because it fits the static
majority and throws moving players out as outliers.

Per clip:
  target  : the mitt detector's set-up glove (auto_glove.detect_setup_catch), mapped by the
            H of its set-up frame
  track   : every v4 ball point, mapped by the H of ITS OWN frame (the camera can move
            during a clip; one H per clip would re-create the problem inside the clip)
Frames whose H has fewer than MIN_INLIERS inliers are DROPPED, never guessed: a low count
means a replay or cutaway, not a camera move.

THEN fit_park_pose on the stabilized tracks (one shared pose for the reference view), and
command_miss on the stabilized tracks + targets. The A-vs-B gap stays an honest check
because the pose is still shared across clips -- nothing is fitted per clip.

FRAMES are decoded SEQUENTIALLY, never by seek (index-and-time-origins.md, instance 2).
Detector track frames come from sequential decode, so they index the same frames here.

    python scripts/stabilize_park.py --game 816698 --dir data/clips/aaa_816698 \\
        --tracks output/detect_tracks_816698_v4.csv
"""
import argparse
import os
import sys

import cv2
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

MIN_INLIERS = 40


def feats(orb, gray):
    kp, des = orb.detectAndCompute(gray, None)
    return kp, des


def to_ref(ref, orb, bf, gray):
    """H mapping THIS frame's pixels into the reference frame, and its inlier count."""
    rkp, rdes = ref
    kp, des = feats(orb, gray)
    if des is None or len(kp) < 10:
        return None, 0
    m = bf.match(des, rdes)
    if len(m) < 10:
        return None, 0
    src = np.float32([kp[x.queryIdx].pt for x in m]).reshape(-1, 1, 2)
    dst = np.float32([rkp[x.trainIdx].pt for x in m]).reshape(-1, 1, 2)
    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)
    return H, int(mask.sum()) if mask is not None else 0


def apply(H, x, y):
    p = cv2.perspectiveTransform(np.float32([[[x, y]]]), H)[0, 0]
    return float(p[0]), float(p[1])


def zoom(H):
    return float(np.sqrt(abs(np.linalg.det(H[:2, :2] / H[2, 2]))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", type=int, required=True)
    ap.add_argument("--dir", required=True)
    ap.add_argument("--tracks", required=True)
    ap.add_argument("--mitt-weights", default="data/yolo/runs/mitt/weights/best.pt")
    a = ap.parse_args()
    d = a.dir if os.path.isabs(a.dir) else os.path.join(ROOT, a.dir)
    assert os.path.isdir(d), d
    man = pd.read_csv(os.path.join(d, os.path.basename(d) + "_manifest.csv"))
    man = man[man.game_pk == a.game]
    stems = [s for s in man.stem if os.path.exists(os.path.join(d, s + ".mp4"))]
    assert stems, "no clips on disk"
    trk = pd.read_csv(os.path.join(ROOT, a.tracks), float_precision="round_trip")

    from ultralytics import YOLO
    from src.auto_glove import detect_setup_catch
    mitt = YOLO(os.path.join(ROOT, a.mitt_weights))
    orb = cv2.ORB_create(nfeatures=4000)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # set-up targets first; the REFERENCE is the set-up frame of the clip whose target is
    # the median one -- a typical framing, so most clips map with a small H
    setups = {}
    for i, s in enumerate(stems, 1):
        r = detect_setup_catch(os.path.join(d, s + ".mp4"), mitt)
        if r.get("ok"):
            setups[s] = r["setup"]
        if i % 20 == 0:
            print("  setup %d/%d" % (i, len(stems)), flush=True)
    assert setups, "no set-up targets found"
    sx = np.median([v[0] for v in setups.values()]); sy = np.median([v[1] for v in setups.values()])
    ref_stem = min(setups, key=lambda s: np.hypot(setups[s][0] - sx, setups[s][1] - sy))
    ref_frame = int(setups[ref_stem][2])

    def frames(stem, want):
        cap = cv2.VideoCapture(os.path.join(d, stem + ".mp4")); i = 0; got = {}
        top = max(want)
        while i <= top:
            ok, fr = cap.read()
            if not ok:
                break
            if i in want:
                got[i] = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
            i += 1
        cap.release()
        return got

    ref = feats(orb, frames(ref_stem, {ref_frame})[ref_frame])
    print("reference view: %s frame %d" % (ref_stem, ref_frame), flush=True)

    tgt_rows, trk_rows, clip_rows = [], [], []
    for i, s in enumerate(stems, 1):
        g = trk[trk.stem == s]
        want = set(int(f) for f in g.frame)
        if s in setups:
            want.add(int(setups[s][2]))
        if not want:
            continue
        got = frames(s, want)
        Hs = {f: to_ref(ref, orb, bf, im) for f, im in got.items()}
        if s in setups:
            H, n = Hs.get(int(setups[s][2]), (None, 0))
            if H is not None and n >= MIN_INLIERS:
                px, py = apply(H, setups[s][0], setups[s][1])
                tgt_rows.append(dict(stem=s, px=px, py=py, raw_px=setups[s][0], raw_py=setups[s][1],
                                     inliers=n, zoom=zoom(H)))
        kept = 0
        for r in g.itertuples():
            H, n = Hs.get(int(r.frame), (None, 0))
            if H is None or n < MIN_INLIERS:
                continue
            x, y = apply(H, r.x, r.y)
            trk_rows.append(dict(stem=s, game_pk=a.game, frame=int(r.frame), x=x, y=y, conf=r.conf,
                                 inliers=n))
            kept += 1
        zs = [zoom(H) for H, n in Hs.values() if H is not None and n >= MIN_INLIERS]
        clip_rows.append(dict(stem=s, track_pts=len(g), kept=kept,
                              zoom_med=float(np.median(zs)) if zs else np.nan))
        if i % 20 == 0:
            print("  stabilized %d/%d" % (i, len(stems)), flush=True)

    out_t = os.path.join(ROOT, "output", "targets_%d_stab.csv" % a.game)
    out_k = os.path.join(ROOT, "output", "detect_tracks_%d_v4stab.csv" % a.game)
    pd.DataFrame(tgt_rows).to_csv(out_t, index=False)
    pd.DataFrame(trk_rows).to_csv(out_k, index=False, float_format="%.17g")
    c = pd.DataFrame(clip_rows)
    print("\ngame %d: %d clips, reference %s f%d" % (a.game, len(stems), ref_stem, ref_frame))
    print("  targets mapped %d of %d found; track points kept %d of %d"
          % (len(tgt_rows), len(setups), int(c.kept.sum()), int(c.track_pts.sum())))
    print("  zoom vs reference, per clip: p05 %.3f  median %.3f  p95 %.3f"
          % (c.zoom_med.quantile(.05), c.zoom_med.median(), c.zoom_med.quantile(.95)))
    print("  ->", out_t, "\n  ->", out_k)


if __name__ == "__main__":
    main()
