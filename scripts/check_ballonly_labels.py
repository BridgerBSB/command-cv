"""Are labels from a BALL-ONLY camera as good as labels from the LANDMARK camera? (PROGRESS-LOG 115)

Per solved park: fit_park_pose on the fit-half seeds in the stabilized reference view (--stab),
project_labels through that pose, and compare to the landmark camera's labels on (stem, frame).
Outputs go to output/ballonly_check/; the park's own pose and labels are never touched.

Result 2026-10-11, five parks, 7,931 shared labels: median 0.47-1.10 px apart, p90 1.17-2.17,
worst 5.5 px, 1 label more than half a ball apart (Durham). That is why label_new_park.py skips
the landmark fit.

    python scripts/check_ballonly_labels.py            # all five
    python scripts/check_ballonly_labels.py 821922     # one park
"""
import os
import subprocess
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "output")
DST = os.path.join(OUT, "ballonly_check")
PARKS = [  # game, clip dir, landmark pose, raw-dets file if not dets_<g>_v4.csv
    ("822515", "data/clips/ogando_2026", "park_pose_822515_s2lm1.npz", None),
    ("821922", "data/clips/ogando_2026", "park_pose_821922_s2lm2.npz", "dets_821922_ogando_v4.csv"),
    ("816698", "data/clips/aaa_816698", "park_pose_816698_s2lm1.npz", None),
    ("815419", "data/clips/aaa_815419", "park_pose_815419_s2lm1.npz", None),
    ("821819", "data/clips/aplus_821819", "park_pose_821819_s2lm1.npz", None),
]


def run(args):
    r = subprocess.run([sys.executable, "-u"] + args, cwd=ROOT, capture_output=True, text=True)
    if r.returncode:
        sys.exit("FAILED: %s\n%s" % (" ".join(args), (r.stdout + r.stderr)[-2000:]))
    return r.stdout


def labels(g, d, pose, dets, out):
    if not os.path.exists(out):
        run(["scripts/project_labels.py", "--game", g, "--dir", d, "--pose", pose, "--stab-tag", "c",
             "--all-frames", "--out", out] + (["--dets", os.path.join(OUT, dets)] if dets else []))
    return pd.read_csv(out)


def main():
    os.makedirs(DST, exist_ok=True)
    only = set(sys.argv[1:])
    for g, d, lm, dets in PARKS:
        if only and g not in only:
            continue
        pose = os.path.join(DST, "pose_%s_ballonly.npz" % g)
        if not os.path.exists(pose):
            log = run(["scripts/fit_park_pose.py", "--game", g, "--dir", d,
                       "--seeds", "output/seeds_%s_v4fit.csv" % g,
                       "--stab", "output/detect_tracks_%s_v4stab2c.csv" % g, "--out", pose])
            open(os.path.join(DST, "fit_%s.log" % g), "w").write(log)
        A = labels(g, d, os.path.join("output", lm), dets, os.path.join(DST, "pl_%s_landmark.csv" % g))
        B = labels(g, d, pose, dets, os.path.join(DST, "pl_%s_ballonly.csv" % g))
        m = A.merge(B, on=["stem", "frame"], suffixes=("_lm", "_bo"))
        assert len(m), "no shared (stem, frame) labels at %s" % g
        dd = np.hypot(m.x_lm - m.x_bo, m.y_lm - m.y_bo)
        half = int((dd > 0.5 * m.diameter_px_lm).sum())
        print("%s  labels lm %d / ball-only %d / shared %d   apart px: median %.2f  p90 %.2f  max %.2f"
              "   > half a ball: %d   det_dist (confirmed) lm %.2f / ball-only %.2f"
              % (g, len(A), len(B), len(m), dd.median(), dd.quantile(.9), dd.max(), half,
                 A[A.confirmed].det_dist.median(), B[B.confirmed].det_dist.median()), flush=True)


if __name__ == "__main__":
    main()
