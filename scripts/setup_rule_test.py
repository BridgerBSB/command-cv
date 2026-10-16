"""Test a set-up rule against Zac's target review, with no GPU (reads saved glove tracks).

Inputs: output/target_review/key.csv (what the old rule picked, per review item), the review
answers (Downloads/target_review_results*.json, newest by default), the raw glove tracks from
scripts/glove_track_dump.py, and each game's ball tracks (output/detect_tracks_<g>_v4.csv).

Order matters and every step prints its denominator:
  1. PROVENANCE. The longest-hold rule re-derived from the saved track must land on key.csv's
     pick. If it does not, the track is not the one the pick came from and nothing below
     means anything (index-and-time-origins.md).
  2. The release-anchored rule (src/auto_glove._select_setup_anchored), anchor = the clip's
     first ball frame in flight, else the game's median first ball frame. Pass marks, fixed
     BEFORE the first run:
       ok items         stay put      (new pick within KEEP_PX of the old one)
       wrong_time items move          (new pick more than MOVE_PX from the old one)
     Moving is necessary, not sufficient: whether the NEW spot is right needs eyes, which is
     what --sheet is for.
  3. Two sensitivity variants (a shorter and a longer end window). A rule whose verdict flips
     under either is not settled. Items rid 1-18 were used to design the rule and are
     reported apart; the TEST rows are the ones that count.

    python scripts/setup_rule_test.py [--results PATH] [--out CSV]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src.auto_glove import (_run_center, _select_setup, _select_setup_anchored,  # noqa: E402
                            anchored_setup_point)

KEY = os.path.join(ROOT, "output", "target_review", "key.csv")
TRACKS = os.path.join(ROOT, "output", "target_review", "glove_tracks")
RADIUS, MAX_GAP, MIN_LEN = 22.0, 8, 4        # detect_setup_catch defaults, as the season runs
KEEP_PX, MOVE_PX = 5.0, 10.0
# (name, end_window frames). Fixed 2026-10-16 from the FIRST 18 dumped items only (DESIGN_RIDS);
# every other item is the test. The last-hold-before-the-ball rule failed on those 18 (see
# _select_setup_anchored) and is not re-run.
# "+ lead 30" added 2026-10-16 after rids 26-34 (all implausible) and BEFORE any plausible
# wrong-moment item was dumped: no target when the hold starts < 30 frames before the ball.
VARIANTS = [("primary", 60, 0), ("window 40", 40, 0), ("window 90", 90, 0), ("+ lead 30", 60, 30)]
DESIGN_RIDS = set(range(1, 18)) | {26}     # the 18 tracks on disk at the first run


def newest_results():
    c = glob.glob(os.path.join(os.path.expanduser("~"), "Downloads", "target_review_results*.json"))
    assert c, "no target_review_results*.json in Downloads"
    return max(c, key=os.path.getmtime)


def ball_starts(games):
    """{game: {stem: first ball frame}}, and {game: median of those}."""
    per, med = {}, {}
    for g in games:
        p = os.path.join(ROOT, "output", "detect_tracks_%d_v4.csv" % g)
        if not os.path.exists(p):
            continue
        s = pd.read_csv(p, usecols=["stem", "frame"]).groupby("stem").frame.min()
        per[g], med[g] = s.to_dict(), float(s.median())
    return per, med


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="")
    ap.add_argument("--out", default=os.path.join(ROOT, "output", "target_review", "rule_test.csv"))
    a = ap.parse_args()
    res_p = a.results or newest_results()
    answers = {int(x["rid"]): x for x in json.load(open(res_p, encoding="utf-8"))}
    key = pd.read_csv(KEY)
    print("answers: %s (%d items, %d answered)" % (res_p, len(answers),
                                                  sum(bool(x["verdict"]) for x in answers.values())))

    per, med = ball_starts(sorted(set(key.game_pk.astype(int))))
    rows, missing = [], 0
    for r in key.itertuples():
        meta_p = os.path.join(TRACKS, "%d.json" % r.rid)
        if not os.path.exists(meta_p):
            missing += 1
            continue
        meta = json.load(open(meta_p))
        t = pd.read_csv(os.path.join(TRACKS, "%d.csv" % r.rid))
        track = list(t.itertuples(index=False, name=None))
        g = int(r.game_pk)
        own = per.get(g, {}).get(r.stem)
        anchor, src = (own, "clip") if own is not None else (med.get(g), "game median")
        reason, run, _ = _select_setup(track, meta["n_frames"], RADIUS, MAX_GAP, MIN_LEN)
        old = _run_center(run) if run else None
        row = dict(rid=r.rid, game=g, stem=r.stem, stratum=r.stratum,
                   verdict=answers.get(r.rid, {}).get("verdict") or "blank",
                   key_x=r.x, key_y=r.y, key_f=r.frame, anchor=anchor, anchor_src=src,
                   redo_ok=old is not None and np.hypot(old[0] - r.x, old[1] - r.y) <= 1.0
                   and old[2] == r.frame)
        for name, win, lead in VARIANTS:
            if anchor is None:
                row[name + "_reason"] = "no ball track in this game"
                continue
            why, nrun = _select_setup_anchored(track, anchor, RADIUS, MAX_GAP, MIN_LEN, win, lead)
            row[name + "_reason"] = why or "ok"
            if nrun:
                nx, ny, nf = anchored_setup_point(nrun)
                row.update({name + "_x": nx, name + "_y": ny, name + "_f": nf,
                            name + "_px": float(np.hypot(nx - r.x, ny - r.y))})
        fx = answers.get(r.rid, {}).get("fx")
        if fx is not None:                   # "wrong spot" items carry Zac's corrected click
            row.update(fix_x=fx, fix_y=answers[r.rid]["fy"])
        rows.append(row)
    d = pd.DataFrame(rows)
    d.to_csv(a.out, index=False)
    print("glove tracks: %d of %d review items (%d not dumped yet)" % (len(d), len(key), missing))
    print("anchor source: %s" % d.anchor_src.value_counts().to_dict())

    bad = d[~d.redo_ok]
    print("\n1. PROVENANCE  old rule re-derived = the graded pick: %d of %d" % (len(d) - len(bad), len(d)))
    if len(bad):
        print(bad[["rid", "stem", "key_x", "key_y", "key_f"]].to_string(index=False))
        print("   -> STOP: these tracks are not the ones the picks came from.")

    print("\n2-3. RULES  (stay = within %.0f px of the old pick, moved = more than %.0f px)"
          % (KEEP_PX, MOVE_PX))
    groups = [("plausible", "ok"), ("plausible", "wrong_time"), ("implausible", "wrong_time"),
              ("plausible", "wrong")]
    d["part"] = np.where(d.rid.isin(DESIGN_RIDS), "design", "TEST")
    for (name, _, _), part in [(v, p) for p in ("TEST", "design") for v in VARIANTS]:
        print("\n  %s  [%s items]" % (name, part))
        for st, v in groups:
            s = d[(d.stratum == st) & (d.verdict == v) & (d.part == part)]
            px = s.get(name + "_px", pd.Series(dtype=float))
            nopick = int(px.isna().sum())
            print("    %-11s %-10s n=%3d  stay %3d  moved %3d  between %2d  no pick %2d   median move %5.1f px"
                  % (st, v, len(s), (px <= KEEP_PX).sum(), (px > MOVE_PX).sum(),
                     ((px > KEEP_PX) & (px <= MOVE_PX)).sum(), nopick,
                     px.median() if px.notna().any() else float("nan")))
    w = d[d.verdict == "wrong"]
    if len(w) and "primary_x" in w:
        print("\n  wrong-spot items vs Zac's click (px): old %s -> new %s" % (
            np.round(np.hypot(w.key_x - w.fix_x, w.key_y - w.fix_y), 1).tolist(),
            np.round(np.hypot(w.primary_x - w.fix_x, w.primary_y - w.fix_y), 1).tolist()))
    print("\nper-item results: %s" % a.out)


if __name__ == "__main__":
    main()
