"""Grade command_miss outputs on a park's HELD-OUT clips (never used in any pose fit).

B-A gap = |video actual - data actual| per pitch, inches; both roots share one target, so
this is the video's plate-crossing error under that pose. Signed means are printed too: a
mean far from 0 is a BIAS (camera or tracking offset), which a median of |gap| hides.

    python scripts/grade_holdout.py --game 816698 --tags _final _v4fit _lm0 _lm1
"""
import argparse
import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", type=int, required=True)
    ap.add_argument("--tags", nargs="+", required=True)
    ap.add_argument("--src", default="auto")
    a = ap.parse_args()
    h = set(pd.read_csv(os.path.join(ROOT, "output", "holdout_%d.csv" % a.game)).stem)
    print("game %d: %d held-out stems" % (a.game, len(h)))
    print("  %-8s %4s %6s %6s %6s %8s %8s %6s %6s" % ("tag", "n", "gapMed", "gapP90", "plaus",
                                                     "dx mean", "dz mean", "B_ok", "Bokgap"))
    for t in a.tags:
        d = pd.read_csv(os.path.join(ROOT, "output", "command_miss_%d%s.csv" % (a.game, t)))
        d = d[(d.target_src == a.src) & d.stem.isin(h)]
        g = d.B_minus_A_in.dropna()
        ok = d[d.B_ok].B_minus_A_in.dropna()
        print("  %-8s %4d %6.2f %6.2f %6d %8.2f %8.2f %6d %6.2f" % (
            t, len(d), g.median(), g.quantile(.9), int(d.plaus.sum()),
            (d.B_dx - d.A_dx).mean(), (d.B_dz - d.A_dz).mean(), int(d.B_ok.sum()),
            ok.median() if len(ok) else np.nan))


if __name__ == "__main__":
    main()
