"""Tighten a park's pose by alternating: find ball pixels -> refit pose -> repeat.

WHY A LOOP. The pieces each solve half the problem and neither is enough alone:
  aplus_seed_tracks.py  finds ACCURATE pixels (top-hat, the ball is top-1 while unoccluded)
                        but only on ~25% of clips, because it scans a fixed window and needs
                        8 consecutive top-1 frames.
  harvest_park_labels.py finds the RIGHT FRAMES on every clip (24/24, plate frame clustered
                        at f214-231 exactly where physics says) but its pixels come from
                        projecting a rough pose, and measured against hand-verified tracks
                        they sit 8.60 px median out -- about one ball width.
So: use the harvest's plate frame to WINDOW a top-hat search, take the accurate pixels that
fall near the projected path, refit the shared pose on those, and go again. Standard ICP.

THE CIRCULARITY, AND HOW IT IS BOUNDED. Selecting blobs near the current projected path and
then fitting to them biases toward the current (wrong) pose -- ICP drift. Two guards:
  the association gate is GENEROUS (default 30 px) relative to the ~8.6 px pose error, so
  the true ball is inside it and the fit can be pulled off the current answer; and
  the five HAND-VERIFIED tracks are held out of the fit entirely and scored every pass.
That held-out error is the only number worth reading. If it does not fall, the loop is
drifting, not converging, and the run says so.

READ THE OUTPUT AS: held-out median px, per pass. Everything else is diagnostics. A falling
train RMS with a flat or rising held-out error is the signature of drift.

THE HELD-OUT SET IS TOO SMALL AND IT IS SELECTION-BIASED. READ THIS BEFORE QUOTING A NUMBER.
With gate=13 this loop reports a held-out median of 2.23 px, which equals our MLB label
quality (2.15 px vs OpenCommand's independent detections). That number is TRUE for the five
verified clips and DOES NOT GENERALISE. Rendering the refined labels zoomed on unverified
clips (output/refined_label_zoom.png) shows them 10-20 px off on ogando_si_12 and ogando_si_33
and landing on empty grass on ogando_fc_7 while the ball is plainly elsewhere in the tile.

The cause is the validation set, not the loop. The five verified clips are the only ones
aplus_seed_tracks could recover, and it recovers a track exactly where the ball is MOST
separable -- unoccluded, high contrast, ten consecutive frames as top-1. So the held-out set
is drawn from the easy tail of the distribution it is supposed to be validating. 50 points
from one park, one pitcher, one game.

Fixing this needs MORE HAND-VERIFIED TRACKS, chosen without regard to whether the seeder
found them -- specifically including clips it failed on. Until then, treat every number this
script prints as a lower bound on the error, and LOOK at refined_label_zoom.png for any park
before using its labels to train anything. Same lesson as render-and-look.md: an aggregate
computed over a biased sample is honest about what it measured and silent about what it did
not cover.

    python scripts/refine_park_pose.py --game 822515 --passes 3
"""
import argparse
import itertools
import os
import sys

import cv2
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from src import pinhole as ph  # noqa: E402
import fit_park_pose as fpp  # noqa: E402
import aplus_seed_tracks as ast  # noqa: E402

GATE_PX = 30.0           # association radius; generous vs the ~8.6 px starting error
MIN_TRACK = 6            # accepted pixels per clip before it joins the fit
PRE, POST = 34, 3        # frames before / after the harvested plate frame to search


def ball_pixels_near_path(clip_path, plate_frame, path_xy, path_frames, gate):
    """Top-hat candidates in a window, kept when close to the projected path.

    The pixels are the SEEDER's (accurate); the path only says where to look and which
    frames. Frame indices are SEQUENTIAL DECODE throughout -- see aplus_seed_tracks.py for
    why that convention is pinned.
    """
    want = {int(f): p for f, p in zip(path_frames, path_xy)}
    lo, hi = int(plate_frame) - PRE, int(plate_frame) + POST
    x0, x1, y0, y1 = 400, 980, 250, 440
    cap = cv2.VideoCapture(clip_path)
    out, prev, i = [], None, 0
    while True:
        ok, f = cap.read()
        if not ok or i > hi:
            break
        roi = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)[y0:y1, x0:x1]
        if lo <= i and prev is not None and i in want:
            th = cv2.morphologyEx(cv2.absdiff(roi, prev), cv2.MORPH_TOPHAT, ast.TOPHAT_K)
            _, bw = cv2.threshold(th, ast.THRESH, 255, cv2.THRESH_BINARY)
            n, lab, st, ce = cv2.connectedComponentsWithStats(bw, 8)
            px, py = want[i]
            best = None
            for k in range(1, n):
                bx, by, bw_, bh, area = st[k]
                if not (ast.MIN_AREA <= area <= ast.MAX_AREA
                        and 3 <= bw_ <= ast.MAX_SIDE and 3 <= bh <= ast.MAX_SIDE):
                    continue
                cx, cy = float(ce[k][0] + x0), float(ce[k][1] + y0)
                d = np.hypot(cx - px, cy - py)
                if d > gate:
                    continue
                resp = float(th[lab == k].mean())
                # nearest-strong: prefer response, but only among things inside the gate
                if best is None or resp > best[0]:
                    best = (resp, cx, cy, d)
            if best:
                out.append((i, best[1], best[2], best[3]))
        prev = roi
        i += 1
    cap.release()
    return half_frame_correct(out)


def half_frame_correct(pts):
    """Undo the half-frame lag that frame-differencing builds into every centroid.

    A moving ball leaves TWO bright lobes in absdiff(frame_i, frame_i-1) -- where it is now
    and where it was -- so the blob centroid sits BETWEEN them and lags the true position at
    frame i by half the inter-frame displacement. Measured over 106 hand-verified points on
    822515: offset ALONG motion +4.71 px against a half-step of 4.96 px, offset ACROSS
    motion -0.86 px. Exactly the predicted lag, in the predicted direction, with no
    across-track component.

    This matters more than it looks. The bias is SYSTEMATIC and was ~4.7 px against a 2.71 px
    random error -- and because the "hand-verified" tracks were extracted the same way, both
    sides of the accuracy check carried the SAME offset, so the agreement metric read clean
    while the rendered labels sat visibly behind the ball. A shared bias is invisible to a
    comparison of the two things that share it; only looking at the raw frame exposed it.

    The correction is exact for a two-lobe difference blob: shift forward by half the local
    displacement. Verified by eye afterwards -- the ball goes from sitting right of the
    circle to dead centre (output/halfframe_corrected.png).
    """
    if len(pts) < 2:
        return pts
    out = []
    for k, (f, x, y, d) in enumerate(pts):
        j = min(k + 1, len(pts) - 1)
        i0 = max(k - 1, 0) if j == k else k
        gap = max(1.0, float(pts[j][0] - pts[i0][0]))
        dx = (pts[j][1] - pts[i0][1]) / gap
        dy = (pts[j][2] - pts[i0][2]) / gap
        out.append((f, x + 0.5 * dx, y + 0.5 * dy, d))
    return out


def fit_best(clips):
    best = None
    for cx, cy, cz, f0 in itertools.product([-260, -110, 110, 260], [-40, 60, 260],
                                            [15, 35, 70], [1400, 2600]):
        try:
            r = fpp.fit(clips, (cx, cy, cz), f0)
        except Exception:
            continue
        rms = float(np.sqrt(np.mean(r.fun ** 2)))
        if best is None or rms < best[0]:
            best = (rms, r.x)
    return best


def heldout_error(C, pan, tilt, roll, fpx, clips_by_stem, verified):
    """Median px between the projected path and the HAND-VERIFIED pixels, per clip.

    The verified tracks never enter the fit. Their t0 is refitted alone, because t0 is a
    per-clip nuisance the camera does not determine -- refitting the camera here would
    make this not held out.
    """
    from scipy.optimize import least_squares
    errs = []
    for stem, g in verified.groupby("stem"):
        c = clips_by_stem.get(stem)
        if c is None:
            continue
        fr = g.frame.to_numpy(float)
        obs = g[["x", "y"]].to_numpy(float)

        def res(t0):
            t = t0[0] + fr / c["fps"]
            return (ph.project(C, pan, tilt, roll, fpx, c["traj"](t)) - obs).ravel()
        t00 = -float(fr[0]) / c["fps"]
        r = least_squares(res, [t00], bounds=([t00 - 1.5], [t00 + 1.5]))
        errs += list(np.hypot(*res(r.x).reshape(-1, 2).T))
    return (np.median(errs), np.percentile(errs, 90), len(errs)) if errs else (np.nan,) * 3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", type=int, default=822515)
    ap.add_argument("--dir", default=os.path.join(ROOT, "data", "clips", "ogando_2026"))
    ap.add_argument("--passes", type=int, default=3)
    ap.add_argument("--gate", type=float, default=GATE_PX)
    a = ap.parse_args()

    manifest = os.path.join(a.dir, os.path.basename(a.dir) + "_manifest.csv")
    man = pd.read_csv(manifest).set_index("stem")
    harvest = pd.read_csv(os.path.join(ROOT, "output", f"harvest_{a.game}.csv"))
    verified = pd.read_csv(os.path.join(ROOT, "output", f"seed_tracks_{a.game}.csv"))
    verified = verified[verified.game_pk == a.game]

    # a clip dict per harvested stem, for projection and for the held-out scorer
    by_stem = {}
    for stem in harvest.stem.unique():
        if stem not in man.index:
            continue
        row = man.loc[stem]
        if any(pd.isna(row[k]) for k in fpp.TRAJ):
            continue
        by_stem[stem] = dict(stem=stem, traj=fpp.traj_fn(row), fps=59.69,
                             zone_time=float(row["zone_time"]))

    # pass 0 pose: the hand-verified seed tracks only
    seed_clips = fpp.build(a.game, os.path.join(ROOT, "output", f"seed_tracks_{a.game}.csv"),
                           manifest)
    rms, x = fit_best(seed_clips)
    C, pan, tilt, roll, fpx = x[:3], x[3], x[4], x[5], x[6]
    med, p90, n = heldout_error(C, pan, tilt, roll, fpx, by_stem, verified)
    print(f"pass 0  (fit on {len(seed_clips)} hand-verified clips)")
    print(f"        train rms {rms:6.2f} px   IN-SAMPLE verified median {med:6.2f} px "
          f"(p90 {p90:5.2f}, n={n})")
    print(f"        camera ({C[0]:8.2f},{C[1]:8.2f},{C[2]:7.2f})  f {fpx:9.1f}")
    print("        NOTE pass 0's verified error is IN sample -- it is a floor, not a score.\n")

    for p in range(1, a.passes + 1):
        built, kept = [], 0
        for stem, c in by_stem.items():
            g = harvest[harvest.stem == stem].sort_values("frame")
            if not len(g):
                continue
            pts = ball_pixels_near_path(os.path.join(a.dir, stem + ".mp4"),
                                        g.frame.max(), g[["x", "y"]].to_numpy(),
                                        g.frame.to_numpy(), a.gate)
            if len(pts) < MIN_TRACK:
                continue
            kept += 1
            built.append(dict(stem=stem, traj=c["traj"], fps=c["fps"],
                              fr=np.array([q[0] for q in pts], float),
                              obs=np.array([[q[1], q[2]] for q in pts], float),
                              zone_time=c["zone_time"]))
        # the verified clips are EXCLUDED from the fit so the score stays held out
        train = [b for b in built if b["stem"] not in set(verified.stem)]
        if len(train) < 3:
            print(f"pass {p}: only {len(train)} trainable clips, stopping")
            break
        rms, x = fit_best(train)
        C, pan, tilt, roll, fpx = x[:3], x[3], x[4], x[5], x[6]
        med, p90, n = heldout_error(C, pan, tilt, roll, fpx, by_stem, verified)
        npts = sum(len(b["fr"]) for b in train)
        print(f"pass {p}  associated {kept} clips, fit on {len(train)} "
              f"({npts} px, verified clips excluded)")
        print(f"        train rms {rms:6.2f} px   HELD-OUT verified median {med:6.2f} px "
              f"(p90 {p90:5.2f}, n={n})")
        print(f"        camera ({C[0]:8.2f},{C[1]:8.2f},{C[2]:7.2f})  f {fpx:9.1f}")
        np.savez(os.path.join(ROOT, "output", f"park_pose_{a.game}_pass{p}.npz"),
                 C=C, ang=np.array([pan, tilt, roll]), f=fpx)
        # refresh the projected paths for the next association step
        for b in train + [q for q in built if q["stem"] in set(verified.stem)]:
            pass
        # PER-CLIP SUB-FRAME CLOCK. The first version wrote labels as
        # t = zone_time + (frame - last_frame)/fps, i.e. it ASSUMED the last harvested
        # frame is exactly the plate crossing. That frame comes from an INTEGER scan, so
        # it is up to a frame out, and at ~8 px/frame a 1-2 frame clock error displaces
        # every label 8-16 px ALONG THE PATH. Rendering showed exactly that on unverified
        # clips (fc_7 ~20-25 px; ff_1 and fc_8 catching only the ball's left edge) -- a
        # consistent ALONG-track shift, which is the signature of a clock error, because a
        # wrong CAMERA displaces perpendicular to the path as well.
        # So refit t0 ALONE per clip against that clip's own detected pixels with the
        # camera frozen. Clips with too few detections keep the integer estimate and are
        # COUNTED in the output rather than silently passed off as refined.
        from scipy.optimize import least_squares as _ls
        newrows, polished, kept_integer = [], 0, 0
        for stem, c in by_stem.items():
            g = harvest[harvest.stem == stem].sort_values("frame")
            if not len(g):
                continue
            fr = g.frame.to_numpy(float)
            t0 = c["zone_time"] - fr.max() / c["fps"]          # integer starting estimate
            pts = ball_pixels_near_path(os.path.join(a.dir, stem + ".mp4"),
                                        fr.max(), g[["x", "y"]].to_numpy(), fr, a.gate)
            if len(pts) >= MIN_TRACK:
                ofr = np.array([q[0] for q in pts], float)
                oobs = np.array([[q[1], q[2]] for q in pts], float)

                def _res(z):
                    tt = z[0] + ofr / c["fps"]
                    return (ph.project(C, pan, tilt, roll, fpx, c["traj"](tt)) - oobs).ravel()
                r = _ls(_res, [t0], bounds=([t0 - 0.12], [t0 + 0.12]))
                t0 = float(r.x[0])
                polished += 1
            else:
                kept_integer += 1
            tt = t0 + fr / c["fps"]
            px = ph.project(C, pan, tilt, roll, fpx, c["traj"](tt))
            for k, f in enumerate(fr):
                newrows.append(dict(stem=stem, game_pk=a.game, frame=int(f),
                                    x=float(px[k, 0]), y=float(px[k, 1])))
        print(f"        clock: {polished} clips refit to sub-frame, "
              f"{kept_integer} kept the integer estimate")
        harvest = pd.DataFrame(newrows)

    out = os.path.join(ROOT, "output", f"refined_labels_{a.game}.csv")
    harvest.to_csv(out, index=False)
    print(f"\n{out}")
    print("The HELD-OUT verified median is the only number here worth reporting. "
          "A falling\ntrain rms with a flat held-out error means the loop is drifting.")


if __name__ == "__main__":
    main()
