"""One image: every park's yield, its numbers, and actual ball crops from its labels.

WHY A SINGLE SHEET. The project now has labels from a dozen ballparks and the per-park
renders are a dozen separate files, so the only way to answer "which parks are good" is to
open them one at a time and hold the numbers in your head. That is exactly the review a
person skips. This puts every park on one page with its own pixels beside its own yield, so
"Greensboro is 9% and here is what those 9% look like" is one glance instead of twelve.

THE CROPS ARE DRAWN AT RANDOM, NOT PICKED. A sheet built from the best-looking examples is a
brochure. Seeded, so the same sheet regenerates identically.

IT SHOWS WHAT IS MISSING. A park with labels but no clips left on disk (the rolling loop
deletes the video once the labels exist) draws its row with the numbers and an explicit
"clips pruned" note rather than being silently dropped -- an absent row reads as a park that
failed, and that is a different fact (.claude/rules/tautological-display.md).

    python scripts/park_scoreboard.py
    python scripts/park_scoreboard.py --out output/parks.png --crops 4
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import cv2
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

TILE = 132
ZOOM = 6
ROW_H = TILE + 26
NAME_W = 470


def read_frame_seq(clip, idx):
    """Frame `idx` by decoding IN ORDER. The project's pinned convention; see
    ingest_annotation._read_frame for the original statement of it."""
    cap = cv2.VideoCapture(clip)
    fr, i = None, 0
    while i <= idx:
        ok, f = cap.read()
        if not ok:
            break
        fr, i = f, i + 1
    cap.release()
    return fr if (fr is not None and i > idx) else None


def clip_dir_for(pk):
    """Where this game's clips live, whichever naming era produced them."""
    base = os.path.join(ROOT, "data", "clips")
    dirs = [d for d in sorted(os.listdir(base)) if os.path.isdir(os.path.join(base, d))]

    # EXACT SUFFIX FIRST, across ALL directories, before falling back to a manifest search.
    # Interleaving the two let a PITCHER dir win: langford_2026 contains one clip from game
    # 821924 and sorts before pull_821924, so AdventHealth reported "1 clip, 159 used" and a
    # yield of 15900%. A percentage over 100 is the tell; the denominator came from a
    # different directory than the numerator.
    for d in dirs:
        if d.endswith("_%d" % pk):
            return os.path.join(base, d)
    for d in dirs:
        p = os.path.join(base, d)
        man = os.path.join(p, d + "_manifest.csv")
        if os.path.exists(man):
            try:
                m = pd.read_csv(man, usecols=["game_pk"])
            except Exception:                                # noqa: BLE001
                continue
            if (m.game_pk.astype(str) == str(pk)).any():
                return p
    return None


_VENUE_CACHE = {}


def venue_from_api(pk):
    """Venue + date from the PUBLIC schedule endpoint. No auth, one call, cached.

    The pitcher-dir manifests predate the venue column, so two real ballparks were printing
    as "game 821922" and "game 822246" -- unreadable exactly where the sheet is supposed to
    replace reading twelve files. Looking it up beats leaving a row nobody can identify.
    """
    if pk in _VENUE_CACHE:
        return _VENUE_CACHE[pk]
    import json
    import urllib.request
    try:
        u = "https://statsapi.mlb.com/api/v1/schedule?gamePk=%d" % pk
        req = urllib.request.Request(u, headers={"User-Agent": "command-cv/0.1"})
        with urllib.request.urlopen(req, timeout=20) as r:
            g = json.loads(r.read().decode())["dates"][0]["games"][0]
        got = (g["venue"]["name"], g.get("officialDate", ""))
    except Exception:                                        # noqa: BLE001
        got = ("", "")
    _VENUE_CACHE[pk] = got
    return got


def meta_for(pk, d):
    """(venue, date, n_clips_in_game) from the manifest, or blanks."""
    if not d:
        return "", "", 0
    man = os.path.join(d, os.path.basename(d) + "_manifest.csv")
    if not os.path.exists(man):
        return "", "", 0
    try:
        m = pd.read_csv(man)
    except Exception:                                        # noqa: BLE001
        return "", "", 0
    if "game_pk" in m.columns:
        m = m[m.game_pk.astype(str) == str(pk)]
    if not len(m):
        return "", "", 0
    return (str(m.iloc[0].get("venue", "")), str(m.iloc[0].get("date", "")), len(m))


def crops_for(lab, d, k, rng):
    """k zoomed crops centred on randomly drawn labels. Empty if the video is gone."""
    out = []
    if not d:
        return out
    stems = [s for s in lab.stem.unique()
             if os.path.exists(os.path.join(d, str(s) + ".mp4"))]
    if not stems:
        return out
    rng.shuffle(stems)
    for stem in stems[: k * 3]:
        g = lab[lab.stem == stem]
        r = g.iloc[len(g) // 2]
        # SEQUENTIAL DECODE, NEVER cap.set(CAP_PROP_POS_FRAMES). Measured on these clips a
        # seek lands 0-2 frames EARLY (.claude/rules/index-and-time-origins.md, instance #2),
        # which puts the ball a few pixels back along its path from where the label is. On a
        # sheet whose whole job is "does the circle sit on the ball" that reads as the LABELS
        # being wrong. The first version of this file used a seek and did exactly that.
        fr = read_frame_seq(os.path.join(d, str(stem) + ".mp4"), int(r["frame"]))
        if fr is None:
            continue
        h = TILE // (2 * ZOOM)
        y0, x0 = max(0, int(r["y"]) - h), max(0, int(r["x"]) - h)
        c = fr[y0:y0 + 2 * h, x0:x0 + 2 * h]
        if c.size == 0:
            continue
        c = cv2.resize(c, (TILE, TILE), interpolation=cv2.INTER_NEAREST)
        cv2.circle(c, (TILE // 2, TILE // 2), 13, (0, 255, 0), 2)
        out.append(c)
        if len(out) >= k:
            break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "output", "park_scoreboard.png"))
    ap.add_argument("--crops", type=int, default=4)
    ap.add_argument("--seed", type=int, default=3)
    a = ap.parse_args()

    rng = np.random.default_rng(a.seed)
    import random as _r
    pyr = _r.Random(a.seed)

    rows = []
    for f in sorted(glob.glob(os.path.join(ROOT, "output", "det_labels_*.csv"))):
        base = os.path.basename(f)
        if any(ch.isalpha() for ch in base[11:-4].replace("_", "")):
            continue                                          # tagged variants, not a park
        try:
            pk = int(base[len("det_labels_"):-len(".csv")])
        except ValueError:
            continue
        lab = pd.read_csv(f)
        d = clip_dir_for(pk)
        venue, date, n_clips = meta_for(pk, d)
        if not venue or not date:
            v2, d2 = venue_from_api(pk)
            venue, date = venue or v2, date or d2
        rows.append({"pk": pk, "venue": venue or "game %d" % pk, "date": date,
                     "clips_in_game": n_clips, "clips_labelled": lab.stem.nunique(),
                     "points": len(lab), "lab": lab, "dir": d})

    if not rows:
        print("no det_labels_*.csv found")
        return 1
    rows.sort(key=lambda r: -(r["clips_labelled"] / max(1, r["clips_in_game"])))

    print("%-30s %-11s %7s %7s %7s  %s" %
          ("venue", "date", "clips", "used", "points", "yield"))
    panels = []
    for r in rows:
        y = r["clips_labelled"] / max(1, r["clips_in_game"])
        print("%-30s %-11s %7d %7d %7d  %5.0f%%" %
              (r["venue"][:30], r["date"], r["clips_in_game"], r["clips_labelled"],
               r["points"], 100 * y))
        tiles = crops_for(r["lab"], r["dir"], a.crops, pyr)
        row = np.full((ROW_H, NAME_W + a.crops * (TILE + 4), 3), 28, np.uint8)
        cv2.putText(row, r["venue"][:30], (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.72,
                    (255, 255, 255), 2)
        # Colour the yield, because the whole point of the sheet is to sort by it at a glance.
        col = (90, 220, 90) if y >= 0.6 else (60, 200, 255) if y >= 0.3 else (90, 90, 245)
        cv2.putText(row, "%.0f%% of clips   %d points" % (100 * y, r["points"]),
                    (10, 66), cv2.FONT_HERSHEY_SIMPLEX, 0.62, col, 2)
        cv2.putText(row, "%s   game %d   %d clips" % (r["date"], r["pk"], r["clips_in_game"]),
                    (10, 98), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (170, 170, 170), 1)
        if not tiles:
            cv2.putText(row, "clips pruned -- labels kept, video deleted",
                        (NAME_W + 8, ROW_H // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.52,
                        (140, 140, 140), 1)
        for i, t in enumerate(tiles):
            x = NAME_W + i * (TILE + 4)
            row[12:12 + TILE, x:x + TILE] = t
        panels.append(row)

    w = max(p.shape[1] for p in panels)
    panels = [cv2.copyMakeBorder(p, 0, 3, 0, w - p.shape[1], cv2.BORDER_CONSTANT,
                                 value=(28, 28, 28)) for p in panels]
    head = np.full((44, w, 3), 18, np.uint8)
    cv2.putText(head, "command-cv  park scoreboard   green >=60%%  amber >=30%%  red <30%%",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (235, 235, 235), 1)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    cv2.imwrite(a.out, np.vstack([head] + panels))
    print("\n" + a.out)
    print("Crops are a RANDOM draw from each park's own labels, not the best ones.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
