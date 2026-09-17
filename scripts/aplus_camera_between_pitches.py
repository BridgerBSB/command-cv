"""Does the A+ camera move BETWEEN pitches, or is one pose good for the whole game?

This decides the shape of the MiLB label pipeline, so measure it before building one.

At MLB every clip gets its own pan/tilt/roll/f because the camera pans and zooms between
pitches; ablate_no_box.fit_joint carries 5 nuisance parameters per clip for exactly that.
If the A+ camera is instead LOCKED game to game, one 7-param pose serves every clip at the
park and each clip reduces to a single unknown -- which frame the pitch starts on -- and
that is a 1-D search the existing ball_clock_align.py already knows how to do.

Step 1 (Sep 11) measured motion DURING flight: max 1.46 px over 55 clips. Motion BETWEEN
pitches was left open. This closes it.

INSTRUMENT, and why not the obvious ones. This project already rejected phase correlation
for camera motion because players dominate the frame (docs/2026-09-11-step2-pinhole-parity.md),
and whole-frame absdiff failed the same way on Sep 13. Both measure the moving majority.
ORB + RANSAC homography is the opposite: it fits the STATIC majority and throws the movers
out as outliers, which is the right polarity for "did the camera move". Reported per clip:

  dx, dy     translation the homography applies at frame centre, px
  scale      sqrt(det) of the linear part -- zoom. 1.000 is no zoom
  inliers    how many ORB matches survived RANSAC. A LOW count means the match itself is
             untrustworthy (a replay, a cutaway, a dugout shot), NOT that the camera moved,
             so it is printed rather than folded into the summary

A clip whose frame lands on a replay is a legitimate outlier and is reported as one. The
answer is the DISTRIBUTION, not a mean: 55 locked clips and 5 wild ones is a locked camera
with 5 replays, and a mean would hide both facts.

    python scripts/aplus_camera_between_pitches.py --game 822515
    python scripts/aplus_camera_between_pitches.py --game 822246 --frac 0.15
"""
import argparse
import os
import sys

import cv2
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

MIN_INLIERS = 25          # below this the homography is not evidence either way


def frame_at(path, frac):
    cap = cv2.VideoCapture(path)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if n <= 1:
        cap.release()
        return None, 0
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, min(n - 1, int(n * frac))))
    ok, f = cap.read()
    cap.release()
    return (cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) if ok else None), n


def homography(ref_kp, ref_des, orb, bf, img):
    kp, des = orb.detectAndCompute(img, None)
    if des is None or len(kp) < 10:
        return None, 0
    matches = bf.match(ref_des, des)
    if len(matches) < 10:
        return None, 0
    src = np.float32([ref_kp[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst = np.float32([kp[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)
    return H, int(mask.sum()) if mask is not None else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", type=int, default=822515)
    ap.add_argument("--dir", default=os.path.join(ROOT, "data", "clips", "ogando_2026"))
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--frac", type=float, default=0.10,
                    help="where in each clip to sample, as a fraction of its length")
    ap.add_argument("--limit", type=int, default=100)
    a = ap.parse_args()

    man = a.manifest or os.path.join(a.dir, os.path.basename(a.dir) + "_manifest.csv")
    m = pd.read_csv(man)
    g = m[m.game_pk == a.game].reset_index(drop=True)
    stems = [s for s in g.stem if os.path.exists(os.path.join(a.dir, s + ".mp4"))][:a.limit]
    assert stems, f"no clips on disk for game {a.game} in {a.dir}"

    orb = cv2.ORB_create(nfeatures=4000)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    ref_img, ref_n = frame_at(os.path.join(a.dir, stems[0] + ".mp4"), a.frac)
    assert ref_img is not None, f"could not read reference frame from {stems[0]}"
    ref_kp, ref_des = orb.detectAndCompute(ref_img, None)
    h, w = ref_img.shape
    ctr = np.float32([[[w / 2, h / 2]]])

    rows = []
    for s in stems:
        img, n = frame_at(os.path.join(a.dir, s + ".mp4"), a.frac)
        if img is None:
            rows.append(dict(stem=s, n=n, dx=np.nan, dy=np.nan, scale=np.nan, inliers=0))
            continue
        H, inl = homography(ref_kp, ref_des, orb, bf, img)
        if H is None:
            rows.append(dict(stem=s, n=n, dx=np.nan, dy=np.nan, scale=np.nan, inliers=inl))
            continue
        moved = cv2.perspectiveTransform(ctr, H)[0, 0]
        rows.append(dict(stem=s, n=n, dx=float(moved[0] - w / 2), dy=float(moved[1] - h / 2),
                         scale=float(np.sqrt(abs(np.linalg.det(H[:2, :2])))), inliers=inl))

    d = pd.DataFrame(rows)
    good = d[d.inliers >= MIN_INLIERS].copy()
    good["shift"] = np.hypot(good.dx, good.dy)

    print(f"game {a.game}: {len(d)} clips, reference = {stems[0]} at frac {a.frac}")
    print(f"  usable matches (inliers >= {MIN_INLIERS}): {len(good)} / {len(d)}   "
          f"-- the rest are replays/cutaways or a failed match, NOT evidence of motion\n")
    print(d.to_string(index=False, float_format=lambda v: f"{v:8.2f}"))

    if len(good):
        q = good["shift"].quantile([0.5, 0.9, 1.0])
        print(f"\n  centre shift px : median {q[0.5]:7.2f}   p90 {q[0.9]:7.2f}   max {q[1.0]:7.2f}")
        print(f"  zoom scale      : median {good.scale.median():7.4f}   "
              f"min {good.scale.min():7.4f}   max {good.scale.max():7.4f}")
        locked = int((good["shift"] < 5).sum())
        print(f"  within 5 px of the reference: {locked} / {len(good)} usable clips")
        print("\n  READ IT AS: a median shift of a few px means ONE pose serves the park and "
              "each clip\n  reduces to a 1-D clock search. Tens of px means per-clip "
              "pan/tilt/roll/f stays,\n  exactly as at MLB.")
    out = os.path.join(ROOT, "output", f"aplus_between_pitches_{a.game}.csv")
    d.to_csv(out, index=False)
    print(f"\n{out}")


if __name__ == "__main__":
    main()
