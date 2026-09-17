"""Harvest ball labels for every clip at a park from a rough pose, with no detector.

THE BOOTSTRAP THIS CLOSES. fit_park_pose.py turns a handful of hand-verified seed tracks
into one shared pose, but only ~25% of clips yield a seed track, and 5 noisy tracks give a
pose whose leave-one-out reprojection is 3-7 px. This step uses that rough pose to find the
ball in EVERY clip -- the camera and the 3D trajectory are both known, so the ball's pixel
path is known up to ONE unknown, which frame the pitch starts on. Slide it and score.

That is exactly ball_clock_align.py's method, which works at MLB at 17x over baseline with
no detector at all. What is re-implemented here, and ONLY this, is that the camera carries a
free Cy: ball_clock_align.scan() reads ph.CY_FT, the 400 ft centerfield pin, which is wrong
at a park shot from up the line (55bfba19). The SCORE is imported, not re-chosen -- the
top-hat + local-contrast + median-over-the-run shaping in that file is the product of two
earlier failed versions and must not be silently re-litigated here.

THE LOOP THIS IS THE SECOND HALF OF:
    seed tracks (verified by eye)  ->  rough pose  ->  harvest  ->  better pose  ->  ...
Each pass has more and longer tracks, so the pose tightens. Convergence is reported as the
change in camera position and in held-out reprojection between passes; if neither moves,
stop. A pass that makes reprojection WORSE means the harvest let junk in, and the run says
so rather than quietly accepting it.

WHAT IS NOT CLAIMED. These labels are good to a few pixels, not to the 2.15 px (0.65 in)
our MLB labels achieve against OpenCommand's independent detections. There is no
independent truth at a MiLB park to check them against, so the only honest checks are
(a) held-out reprojection and (b) LOOKING at rendered overlays. Both are produced. Do not
report a harvest count as an accuracy.

    python scripts/harvest_park_labels.py --game 822515 --limit 60
    python scripts/harvest_park_labels.py --game 822515 --render 12
"""
import argparse
import os
import sys

import cv2
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from src import pinhole as ph  # noqa: E402
import ball_clock_align as bca  # noqa: E402  -- the SCORE comes from here, unchanged
import fit_park_pose as fpp  # noqa: E402

FLIGHT_S = 0.32          # seconds of flight scored, ending at the plate (bca's choice)
MIN_PTS = 14             # scored points required before an offset is considered at all
MIN_SCORE = 12.0         # median local contrast; below this there is no ball on that path


def scan_free_cy(C, pan, tilt, roll, fpx, traj, zone_time, m, fps, lo, hi):
    """Slide the plate frame over the window; return (best score, best f0, baseline).

    TIME ORIGIN -- THE DEFECT THIS SIGNATURE EXISTS TO PREVENT. bca.scan evaluates its path
    at t in [-FLIGHT_S, 0] where t = 0 is the PLATE CROSSING. The manifest's polynomial is
    RELEASE-relative (t = 0 is release, the plate is at zone_time). Passing one into the
    other evaluates the trajectory from 0.32 s BEFORE RELEASE -- the pitcher's hand -- so
    every label lands near the mound and the scan locks onto the delivery commotion instead
    of the ball. That shipped once here: 21 of 24 clips "kept" at a median 10.9x baseline,
    every label wrong, caught only by rendering them. `zone_time` is therefore REQUIRED and
    explicit, never defaulted. Third instance of this class in this project, after
    OpenCommand's release-relative frame_idx (Sep 13) and cv2 seek-vs-sequential indexing.

    Otherwise a near-copy of ball_clock_align.scan with C passed in whole instead of
    (Cx, CY_FT, Cz).
    The baseline -- the MEDIAN score over every offset -- is returned because a peak is only
    meaningful against it: at MLB the true offset beats baseline by 17x, and an absolute
    score alone cannot tell a found ball from a generally busy clip.
    """
    idx = np.arange(-int(FLIGHT_S * fps), 1)
    out = []
    for f0 in np.arange(lo + 5, hi - 5, 1.0):
        px = ph.project(C, pan, tilt, roll, fpx, traj(zone_time + idx / fps))
        fs = (f0 + idx).astype(int)
        v = [bca.local_contrast(m[fi], px[k, 0], px[k, 1])
             for k, fi in enumerate(fs) if fi in m]
        v = [q for q in v if q is not None]
        if len(v) >= MIN_PTS:
            out.append((float(np.median(v)), float(f0)))
    if not out:
        return np.nan, np.nan, np.nan
    best = max(out)
    return best[0], best[1], float(np.median([o[0] for o in out]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", type=int, default=822515)
    ap.add_argument("--dir", default=os.path.join(ROOT, "data", "clips", "ogando_2026"))
    ap.add_argument("--pose", default=None, help="npz from a previous fit; else refit now")
    ap.add_argument("--lo", type=int, default=150)
    ap.add_argument("--hi", type=int, default=290)
    ap.add_argument("--limit", type=int, default=60)
    ap.add_argument("--render", type=int, default=0, help="render N harvested clips")
    a = ap.parse_args()

    manifest = os.path.join(a.dir, os.path.basename(a.dir) + "_manifest.csv")
    seeds = os.path.join(ROOT, "output", f"seed_tracks_{a.game}.csv")

    if a.pose and os.path.exists(a.pose):
        z = np.load(a.pose)
        C, pan, tilt, roll, fpx = z["C"], *z["ang"], float(z["f"])
    else:
        import itertools
        clips = fpp.build(a.game, seeds, manifest)
        assert len(clips) >= 3, f"need >=3 seed clips, have {len(clips)}"
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
        rms, x = best
        C, pan, tilt, roll, fpx = x[:3], x[3], x[4], x[5], x[6]
        print(f"pose refitted from {len(clips)} seed clips, ball rms {rms:.2f} px")
    print(f"camera Cx {C[0]:.2f}  Cy {C[1]:.2f}  Cz {C[2]:.2f}  f {fpx:.1f}\n")

    m_man = pd.read_csv(manifest).set_index("stem")
    g = m_man[m_man.game_pk == a.game]
    stems = [s for s in g.index if os.path.exists(os.path.join(a.dir, s + ".mp4"))][:a.limit]

    rows, stats = [], []
    for j, stem in enumerate(stems, 1):
        row = m_man.loc[stem]
        if any(pd.isna(row[k]) for k in fpp.TRAJ):
            continue
        traj = fpp.traj_fn(row)
        path = os.path.join(a.dir, stem + ".mp4")
        try:
            mstack, fps, lo, hi, n = bca.motion_stack(path, a.lo, a.hi)
        except Exception as e:
            print(f"  [{j}/{len(stems)}] {stem:16} motion_stack failed: {e}")
            continue
        zt = float(row['zone_time'])
        score, f0, base = scan_free_cy(C, pan, tilt, roll, fpx, traj, zt,
                                       mstack, fps, lo, hi)
        del mstack
        ratio = score / base if base and base > 0 else np.nan
        ok = (not np.isnan(score)) and score >= MIN_SCORE and ratio >= 1.8
        stats.append(dict(stem=stem, score=score, baseline=base, ratio=ratio,
                          plate_frame=f0, kept=bool(ok)))
        print(f"  [{j}/{len(stems)}] {stem:16} score {score:6.1f}  base {base:6.1f}  "
              f"x{ratio:5.2f}  plate f{f0:6.1f}  {'KEEP' if ok else 'drop'}", flush=True)
        if ok:
            idx = np.arange(-int(FLIGHT_S * fps), 1)
            px = ph.project(C, pan, tilt, roll, fpx, traj(zt + idx / fps))
            for k, fi in enumerate((f0 + idx).astype(int)):
                rows.append(dict(stem=stem, game_pk=a.game, frame=int(fi),
                                 x=float(px[k, 0]), y=float(px[k, 1])))

    st = pd.DataFrame(stats)
    kept = st.kept.sum() if len(st) else 0
    print(f"\n{kept}/{len(st)} clips kept   "
          f"(a KEEP needs score >= {MIN_SCORE} AND >= 1.8x its own baseline)")
    if len(st):
        print(f"  score ratio: median {st.ratio.median():.2f}  "
              f"p90 {st.ratio.quantile(0.9):.2f}  max {st.ratio.max():.2f}")
        print("  For scale: at MLB with a TRUE pose this method beats baseline 17x. "
              "A median\n  near 1 means the pose is not good enough to find the ball, "
              "not that the ball is absent.")
    out = os.path.join(ROOT, "output", f"harvest_{a.game}.csv")
    pd.DataFrame(rows).to_csv(out, index=False)
    st.to_csv(os.path.join(ROOT, "output", f"harvest_stats_{a.game}.csv"), index=False)
    print(f"\n{out}  ({len(rows)} label points)")
    print("LOOK at the render before using these. A harvest count is not an accuracy.")

    if a.render and rows:
        d = pd.DataFrame(rows)
        panels = []
        for stem in list(d.stem.unique())[:a.render]:
            gg = d[d.stem == stem].sort_values("frame")
            mid = int(gg.frame.iloc[len(gg) // 2])
            cap = cv2.VideoCapture(os.path.join(a.dir, stem + ".mp4"))
            i, base = 0, None
            while True:
                ok2, f = cap.read()
                if not ok2 or i > mid:
                    break
                if i == mid:
                    base = f.copy()
                i += 1
            cap.release()
            if base is None:
                continue
            for x, y in zip(gg.x, gg.y):
                cv2.circle(base, (int(x), int(y)), 4, (0, 0, 255), -1)
            c = cv2.resize(base[260:420, 520:900], None, fx=1.9, fy=1.9)
            cv2.putText(c, stem, (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            panels.append(c)
        if panels:
            p = os.path.join(ROOT, "output", f"harvest_check_{a.game}.png")
            cv2.imwrite(p, np.vstack(panels))
            print(p)


if __name__ == "__main__":
    main()
