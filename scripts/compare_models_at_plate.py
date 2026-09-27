"""Grade several ball detectors against the TRACKING SYSTEM, not against pixel labels.

WHY. The 119-image Asheville exam scores a detector against labels made by frame
differencing -- the same method that made v1's training labels. v2 and v3 learned from
DETECTOR-made labels, so part of what that exam measures is agreement with a labelling
method, and all three medians sit under the labels' own ~2.8 px noise. The plate crossing
in inches (plate_x / plate_z from the manifest) never touched any model or any label, so
it can rank them without that bias.

METHOD. Identical to scripts/plate_scale_check.py: fit the clock to the OBSERVED ball
pixels, interpolate the pixel path to zone_time, unproject onto the plate-front plane,
compare to the tracking system. The camera pose is FIXED per park, so the only thing that
varies between rows is the detector.

LAST-K, DEFAULT 10. A park's pose is fitted from late-flight tracks, so it projects the
ball well near the plate and worse near release. Grading a WHOLE chain therefore penalises
a detector for following MORE of the flight: at Isotopes (2026-09-24) v2's pitch-matched
chains started ~10 frames earlier than v1's and scored 39 clips over 12 in against v1's 14
-- graded on the last 10 frames the same chains scored 16 vs 8, medians 2.75 vs 2.68 in.
The whole-chain grade was measuring the pose's reach, not the detector. --last-k 0 restores
it; plate_scale_check.py still grades the whole chain.

GRAIN. Each model tracks a different set of clips. Averages over different clips are not
comparable, so the headline table uses ONLY the clips every model tracked, and each model's
own yield is printed beside it.

    python scripts/compare_models_at_plate.py --game 816698 --dir data/clips/aaa_816698 --tags _v1,_v2,_v3
"""
import argparse, os, sys
import numpy as np, pandas as pd
from scipy.optimize import least_squares

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "scripts"))
from src import pinhole as ph  # noqa: E402
import fit_park_pose as fpp  # noqa: E402

FPS = 59.69
PLATE_FRONT_Y = 17.0 / 12.0
MIN_PTS = 5


def clip_error(g, row, C, pan, tilt, roll, f, with_gap=False):
    """(ex, ez) inches for one clip's track, or None."""
    g = g.sort_values("frame")
    fr = g.frame.to_numpy(float); obs = g[["x", "y"]].to_numpy()
    traj = fpp.traj_fn(row); zt = float(row["zone_time"])
    t0 = zt - fr.max() / FPS

    def res(zz):
        return (ph.project(C, pan, tilt, roll, f, traj(zz[0] + fr / FPS)) - obs).ravel()
    t0 = float(least_squares(res, [t0], bounds=([t0 - 0.30], [t0 + 0.30])).x[0])
    t = t0 + fr / FPS
    px = np.polyval(np.polyfit(t, obs[:, 0], 2), zt)
    py = np.polyval(np.polyfit(t, obs[:, 1], 2), zt)
    wx, wz = ph.unproject(C, pan, tilt, roll, f, (px, py), PLATE_FRONT_Y)
    err = ((wx - float(row["plate_x"])) * 12.0, (wz - float(row["plate_z"])) * 12.0)
    if with_gap:   # frames between the last tracked point and the plate crossing
        return err + ((zt - t[-1]) * FPS,)
    return err


def summarise(e):
    ex, ez = e[:, 0], e[:, 1]
    r = np.hypot(ex - np.median(ex), ez - np.median(ez))
    return (np.median(ex), np.median(ez), np.median(np.abs(ex - np.median(ex))),
            np.median(np.abs(ez - np.median(ez))), np.median(r), np.percentile(r, 90))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", type=int, required=True)
    ap.add_argument("--dir", required=True)
    ap.add_argument("--tags", required=True, help="comma list; reads output/detect_tracks_<game><tag>.csv")
    ap.add_argument("--pose", default=None)
    ap.add_argument("--last-k", type=int, default=10,
                    help="grade each chain on its LAST k frames only (0 = whole chain)")
    a = ap.parse_args()
    d = a.dir if os.path.isabs(a.dir) else os.path.join(ROOT, a.dir)
    assert os.path.isdir(d), d
    pose = a.pose or os.path.join(ROOT, "output", f"park_pose_{a.game}.npz")
    z = np.load(pose); C, (pan, tilt, roll), f = z["C"], z["ang"], float(z["f"])
    man = pd.read_csv(os.path.join(d, os.path.basename(d) + "_manifest.csv")).set_index("stem")
    man = man[man.game_pk == a.game]
    usable = {s for s, r in man.iterrows()
              if not any(pd.isna(r[k]) for k in fpp.TRAJ) and not pd.isna(r.get("plate_x"))}

    tags = a.tags.split(",")
    per = {}
    for tg in tags:
        p = os.path.join(ROOT, "output", f"detect_tracks_{a.game}{tg}.csv")
        assert os.path.exists(p), p
        lab = pd.read_csv(p)
        errs = {}
        for stem, g in lab.groupby("stem"):
            if a.last_k:
                g = g.sort_values("frame").tail(a.last_k)
            if stem in usable and len(g) >= MIN_PTS:
                errs[stem] = clip_error(g, man.loc[stem], C, pan, tilt, roll, f)
        per[tg] = errs

    common = set.intersection(*(set(v) for v in per.values()))
    print(f"game {a.game}   pose {os.path.basename(pose)}   clips with truth: {len(usable)}")
    print(f"{'model':6} {'tracked':>8}   (every model: {len(common)} shared clips)")
    for tg in tags:
        print(f"{tg:6} {len(per[tg]):>8}")
    if len(common) < 5:
        print("FEWER THAN 5 SHARED CLIPS -- no verdict."); return
    print("\nON THE SHARED CLIPS ONLY (inches vs tracking):")
    print(f"{'model':6} {'bias x':>7} {'bias z':>7} {'MAD x':>6} {'MAD z':>6} {'spread med':>10} {'spread p90':>10}")
    for tg in tags:
        e = np.array([per[tg][s] for s in sorted(common)])
        bx, bz, mx, mz, rm, r9 = summarise(e)
        print(f"{tg:6} {bx:+7.2f} {bz:+7.2f} {mx:6.2f} {mz:6.2f} {rm:10.2f} {r9:10.2f}")
    print("\nSpread = distance from the model's own median miss; the bias is a constant the camera")
    print("pose contributes equally to every model. SPREAD is the detector's resolution in inches.")


if __name__ == "__main__":
    main()
