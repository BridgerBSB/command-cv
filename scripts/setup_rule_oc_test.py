"""Score OpenCommand set-up rule VARIANTS against Zac's clip-review grades (2026-10-16), offline.

Inputs: the 134 graded pitches (docs/target_review/clips_key + clips_results), and each one's
every-frame glove track (scripts/review_glove_tracks.py, which proved the track reproduces both
graded picks). A variant's pick inherits a grade when it lands within SAME_PX of a graded ring
(right or wrong); anywhere else it is UNKNOWN - never counted right or wrong.

Variants are the two shortcuts the season rerun would like to take:
  step 3      read every 3rd frame (the hold rule's step) instead of every frame: ~3x cheaper
  fixed ppi   one px-per-inch for every park instead of the game's command_miss slope, so a
              NEW game can be targeted before its camera exists (the slope needs the camera)
The baseline (step 1, game ppi) must reproduce the graded OpenCommand pick on every item - that
is the check that src.auto_glove.select_setup_opencommand is the rule that was graded.

    python scripts/setup_rule_oc_test.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from src.auto_glove import select_setup_opencommand  # noqa: E402
import target_review_clips as rc  # noqa: E402

DOCS = os.path.join(ROOT, "docs", "target_review")
TRK = os.path.join(rc.OUT, "tracks")
SAME_PX = rc.SAME_PX


def graded_points(r):
    """[(x, y, right)] for the rings Zac graded on this item; [] when unclear."""
    v = r.verdict
    ours, oc = (r.ours_x, r.ours_y), (r.oc_x, r.oc_y)
    if v == "unclear" or not v:
        return []
    if pd.isna(r.oc_x):                                   # only our ring was shown
        return [(*ours, v == "both")]
    if r.same or v in ("both", "neither"):
        return [(*ours, v == "both"), (*oc, v == "both")]
    right_rule = r.A if v == "a" else r.B
    return [(*ours, right_rule == "ours"), (*oc, right_rule == "opencommand")]


def grade(pick, pts):
    if pick is None:
        return "no target"
    best = min(pts, key=lambda p: np.hypot(pick[0] - p[0], pick[1] - p[1]), default=None)
    if best is None or np.hypot(pick[0] - best[0], pick[1] - best[1]) > SAME_PX:
        return "unknown"
    return "right" if best[2] else "wrong"


def main():
    k = pd.read_csv(os.path.join(DOCS, "clips_key_2026-10-16.csv"))
    g = pd.DataFrame(json.load(open(os.path.join(DOCS, "clips_results_2026-10-16.json"), encoding="utf-8")))
    k = k.merge(g[["rid", "verdict"]], on="rid")
    ppi_game = {gp: rc.px_per_inch(gp) for gp in k.game_pk.unique()}
    fixed = float(np.median(list(ppi_game.values())))
    tracks = {r: [tuple(t) for t in pd.read_csv(os.path.join(TRK, "%d.csv" % r)).itertuples(index=False)]
              for r in k.rid}
    tracks = {r: [(int(f), x, y, c) for f, x, y, c in t] for r, t in tracks.items()}
    print("%d graded pitches; game ppi %.2f-%.2f, fixed ppi = their median %.2f"
          % (len(k), min(ppi_game.values()), max(ppi_game.values()), fixed))

    variants = {
        "step 1, game ppi (graded)": (1, lambda r: ppi_game[r.game_pk]),
        "step 3, game ppi": (3, lambda r: ppi_game[r.game_pk]),
        "step 1, fixed ppi": (1, lambda r: fixed),
        "step 3, fixed ppi": (3, lambda r: fixed),
        "step 1, game ppi x0.75": (1, lambda r: 0.75 * ppi_game[r.game_pk]),
        "step 1, game ppi x1.33": (1, lambda r: 1.33 * ppi_game[r.game_pk]),
    }
    base, rows = {}, []
    for name, (step, ppi) in variants.items():
        res, moved = [], 0
        for r in k.itertuples():
            tr = [p for p in tracks[r.rid] if p[0] % step == 0]
            reason, pick = select_setup_opencommand(tr, int(r.anchor), float(r.fps), ppi(r))
            if name.endswith("(graded)"):
                base[r.rid] = pick
                if pd.isna(r.oc_x):
                    assert pick is None, r.rid
                else:
                    assert pick and np.hypot(pick[0] - r.oc_x, pick[1] - r.oc_y) <= 0.5, r.rid
            b = base[r.rid]
            if (pick is None) != (b is None) or (pick and np.hypot(pick[0] - b[0], pick[1] - b[1]) > SAME_PX):
                moved += 1
            res.append(grade(pick, graded_points(r)))
        c = pd.Series(res).value_counts()
        known = c.get("right", 0) + c.get("wrong", 0)
        rows.append(dict(variant=name, right=c.get("right", 0), wrong=c.get("wrong", 0),
                         unknown=c.get("unknown", 0), no_target=c.get("no target", 0),
                         right_of_known="%d/%d" % (c.get("right", 0), known),
                         moved_vs_graded=moved))
    out = pd.DataFrame(rows)
    print(out.to_string(index=False))
    out.to_csv(os.path.join(rc.OUT, "setup_rule_oc_test.csv"), index=False)


if __name__ == "__main__":
    main()
