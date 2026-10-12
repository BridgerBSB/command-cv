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


def grade(game, tag, src="auto"):
    """One row of the table below, as a dict. miss_season.py gates on this, so the season
    ledger and the printed table cannot disagree about what a game's camera scored."""
    h = set(pd.read_csv(os.path.join(ROOT, "output", "holdout_%d.csv" % game)).stem)
    d = pd.read_csv(os.path.join(ROOT, "output", "command_miss_%d%s.csv" % (game, tag)))
    d = d[(d.target_src == src) & d.stem.isin(h)]
    g = d.B_minus_A_in.dropna()
    ok = d[d.B_ok].B_minus_A_in.dropna()
    return dict(n_holdout=len(h), n=len(d), gap_med=g.median(), gap_p90=g.quantile(.9),
                plaus=int(d.plaus.sum()), dx_mean=(d.B_dx - d.A_dx).mean(),
                dz_mean=(d.B_dz - d.A_dz).mean(), b_ok=int(d.B_ok.sum()),
                bok_gap=ok.median() if len(ok) else np.nan)


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
        r = grade(a.game, t, a.src)
        print("  %-8s %4d %6.2f %6.2f %6d %8.2f %8.2f %6d %6.2f" % (
            t, r["n"], r["gap_med"], r["gap_p90"], r["plaus"], r["dx_mean"], r["dz_mean"],
            r["b_ok"], r["bok_gap"]))


if __name__ == "__main__":
    main()
