"""Is a MiLB park's camera METRIC? Unproject the observed ball to the plate and compare inches.

THE CLAIM THIS TESTS. fit_park_pose's docstring says these poses are good for labels but NOT
for miss distance in inches, because (Cy, f) rail and a metric camera "needs a known world
object in frame". OpenCommand's own 2026 pose file argues the opposite: Cy is PINNED at
400 ft on all 499,800 rows -- an assumption, and a false one at any park not shot from dead
centre -- and their median ball-at-plate error is still 0.04-0.13 in. So the absolute camera
distance cannot be what matters; f absorbs it, and only the RATIO f/depth does. That ratio is
exactly what the ball track determines, which is why our diameters came out right at parks
whose cameras are wrong.

THE TEST, and it needs no landmark and no drawn zone. Take the OBSERVED ball pixels (a
measurement, not a projection), fit the clock, interpolate the pixel path to zone_time, and
unproject that pixel onto the plate-front plane. Compare to the manifest's plate_x / plate_z,
which come from the tracking system and never entered our fit. If the camera's scale is
wrong, this is wrong, in inches.

y = 17/12 ft is the PLATE FRONT, which is what zone_time refers to -- NOT ph.zone_y(2026) =
8.5/12, which is the plane the broadcast draws its box on. Two different planes.
"""
import argparse, os, sys
import numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "scripts"))
from src import pinhole as ph
import fit_park_pose as fpp
from scipy.optimize import least_squares

FPS = 59.69
PLATE_FRONT_Y = 17.0 / 12.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", type=int, required=True)
    ap.add_argument("--dir", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--pose", default=None)
    a = ap.parse_args()
    d = a.dir if os.path.isabs(a.dir) else os.path.join(ROOT, a.dir)
    pose = a.pose or os.path.join(ROOT, "output", f"park_pose_{a.game}.npz")
    z = np.load(pose); C, (pan, tilt, roll), f = z["C"], z["ang"], float(z["f"])
    man = pd.read_csv(os.path.join(d, os.path.basename(d) + "_manifest.csv")).set_index("stem")
    lab = pd.read_csv(a.labels)

    ex, ez, ns = [], [], 0
    for stem, g in lab.groupby("stem"):
        if stem not in man.index or len(g) < 5:
            continue
        row = man.loc[stem]
        if any(pd.isna(row[k]) for k in fpp.TRAJ) or pd.isna(row.get("plate_x")):
            continue
        g = g.sort_values("frame")
        fr = g.frame.to_numpy(float); obs = g[["x", "y"]].to_numpy()
        traj = fpp.traj_fn(row); zt = float(row["zone_time"])
        t0 = zt - fr.max() / FPS

        def res(zz):
            return (ph.project(C, pan, tilt, roll, f, traj(zz[0] + fr / FPS)) - obs).ravel()
        t0 = float(least_squares(res, [t0], bounds=([t0 - 0.30], [t0 + 0.30])).x[0])

        # the OBSERVED pixel path as a function of trajectory time, evaluated AT zone_time.
        # quadratic in t: the image of a parabola under perspective is not a parabola, but
        # over the ~0.2 s these tracks span it is an excellent local fit, and this only ever
        # interpolates or extrapolates a few frames.
        t = t0 + fr / FPS
        px = np.polyval(np.polyfit(t, obs[:, 0], 2), zt)
        py = np.polyval(np.polyfit(t, obs[:, 1], 2), zt)
        wx, wz = ph.unproject(C, pan, tilt, roll, f, (px, py), PLATE_FRONT_Y)
        ex.append((wx - float(row["plate_x"])) * 12.0)
        ez.append((wz - float(row["plate_z"])) * 12.0)
        ns += 1

    ex, ez = np.array(ex), np.array(ez)
    print(f"game {a.game}  n={ns} clips   pose C=({C[0]:.0f},{C[1]:.0f},{C[2]:.0f}) f={f:.0f}"
          f"   f/Cy {f/C[1]:.1f} px/ft")
    for nm, e in (("plate_x", ex), ("plate_z", ez)):
        print(f"  {nm} error inches: median {np.median(e):+6.2f}  "
              f"MAD {np.median(np.abs(e - np.median(e))):5.2f}  "
              f"p90|e| {np.percentile(np.abs(e), 90):5.2f}")
    r = np.hypot(ex - np.median(ex), ez - np.median(ez))
    print(f"  radial spread about the median: median {np.median(r):.2f} in, "
          f"p90 {np.percentile(r, 90):.2f} in")
    print("  OpenCommand at MLB, same quantity: median 0.04 / 0.13 in, p90 1.37 / 1.59 in.")
    print("  A MEDIAN offset is a constant bias (fixable); the SPREAD is the real resolution.")


if __name__ == "__main__":
    main()
