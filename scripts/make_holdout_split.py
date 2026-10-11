"""Held-out split + fit-half seeds for a park, by the rule every solved park already used.

WHY. Step 5 of docs/2026-10-11-park-camera-solve-recipe.md was never scripted: each park's
holdout_<g>.csv and seeds_<g>_v4fit.csv were made by hand. Every later step keys off those two
files (fit_park_pose / fit_pose_landmarks fit on the seeds, grade_holdout / project_labels /
v6_run grade or label against the held-out half), so a park cannot be solved without them.

THE RULE (recovered 2026-10-11 from all five solved parks, it holds at every one):
  pool     = the sorted unique stems of the park's RAW v4 tracks
  held-out = pool[1::2]   (odd positions)
  seeds    = every raw track row of pool[0::2]   (even positions), coordinates untouched
At Asheville 822515, AdventHealth 821922 and Frawley 821819 the pool was every tracked stem and
--check reproduces both files exactly. At Isotopes 815419 and Durham 816698 the pool was a
hand-picked subset (165 of 242, 25 of 46) whose filter was not recorded - the alternation inside
it still holds - so --check cannot reproduce those two and says so.

NEVER OVERWRITES. A held-out half that changes after a fit has seen the other half is no longer
held out. To redo a split, delete both files by hand first.

    python scripts/make_holdout_split.py --game 821922 --tracks output/detect_tracks_821922_ogando_v4.csv --check
    python scripts/make_holdout_split.py --game 817407 --tracks output/detect_tracks_817407_v4.csv
"""
import argparse
import os
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "output")


def split(tracks):
    """(held-out stems, seed rows) from a raw track frame."""
    pool = sorted(tracks.stem.unique())
    hold = pool[1::2]
    seeds = tracks[tracks.stem.isin(set(pool[0::2]))]
    return hold, seeds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", type=int, required=True)
    ap.add_argument("--tracks", required=True, help="RAW (unstabilized) v4 tracks, detect_tracks_<g>_v4.csv")
    ap.add_argument("--check", action="store_true",
                    help="compare against the existing files instead of writing")
    a = ap.parse_args()

    t = pd.read_csv(a.tracks if os.path.isabs(a.tracks) else os.path.join(ROOT, a.tracks))
    t = t[t.game_pk == a.game] if "game_pk" in t else t
    assert len(t), "no track rows for game %d in %s" % (a.game, a.tracks)
    hold, seeds = split(t)
    hp = os.path.join(OUT, "holdout_%d.csv" % a.game)
    sp = os.path.join(OUT, "seeds_%d_v4fit.csv" % a.game)
    print("game %d: %d tracked clips -> %d held out, %d seed clips (%d points)"
          % (a.game, t.stem.nunique(), len(hold), seeds.stem.nunique(), len(seeds)))

    if a.check:
        old_h = set(pd.read_csv(hp).stem)
        old_s = pd.read_csv(sp)
        same_h = old_h == set(hold)
        key = ["stem", "frame"]
        m = old_s.merge(seeds, on=key, how="outer", indicator=True)
        same_s = (m._merge == "both").all()
        print("  held-out stems identical: %s (existing %d)" % (same_h, len(old_h)))
        print("  seed (stem, frame) rows identical: %s (existing %d)" % (same_s, len(old_s)))
        if not (same_h and same_s):
            print("  NOT REPRODUCED - the existing split used a different pool")
            sys.exit(1)
        return

    for p in (hp, sp):
        if os.path.exists(p):
            sys.exit("REFUSING: %s exists. A held-out half must never change after a fit; "
                     "delete both files by hand to redo the split." % p)
    pd.DataFrame({"stem": hold}).to_csv(hp, index=False)
    seeds.to_csv(sp, index=False, float_format="%.17g")
    print("  -> %s\n  -> %s" % (hp, sp))


if __name__ == "__main__":
    main()
