"""Per-pitch context the clip manifests do not carry: pitcher, batter side, count, inning.

Joined to our clips on play_id (the GUID in every manifest row). Pulled from the public Stats
API game feed (no key, stdlib only -- same endpoint src/ingest/statsapi.py uses) and cached to
output/pitch_context_<game>.csv so the board never re-fetches.

The match count against the manifest is printed: a join that silently matches nothing is the
failure mode in index-and-time-origins.md.

    python scripts/pitch_context.py --game 822515 --dir data/clips/ogando_2026
"""
import argparse
import os
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src.ingest.statsapi import FEED_URL, _get_json  # noqa: E402


def fetch(game):
    feed = _get_json(FEED_URL.format(game_pk=game))
    rows = []
    for ab in feed.get("liveData", {}).get("plays", {}).get("allPlays", []):
        m = ab.get("matchup", {})
        about = ab.get("about", {})
        for ev in ab.get("playEvents", []):
            if not ev.get("isPitch") or not ev.get("playId"):
                continue
            c = ev.get("count", {})
            rows.append(dict(play_id=ev["playId"], game_pk=game,
                             pitcher_id=m.get("pitcher", {}).get("id"),
                             pitcher_name=m.get("pitcher", {}).get("fullName"),
                             p_throws=(m.get("pitchHand") or {}).get("code"),
                             batter_id=m.get("batter", {}).get("id"),
                             batter_name=m.get("batter", {}).get("fullName"),
                             bat_side=(m.get("batSide") or {}).get("code"),
                             inning=about.get("inning"), half=about.get("halfInning"),
                             # count AFTER the pitch in the feed; balls/strikes BEFORE it
                             # are one step back, which the board does not need yet
                             balls_after=c.get("balls"), strikes_after=c.get("strikes"),
                             pitch_number=ev.get("pitchNumber")))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", type=int, required=True)
    ap.add_argument("--dir", required=True)
    a = ap.parse_args()
    d = a.dir if os.path.isabs(a.dir) else os.path.join(ROOT, a.dir)
    assert os.path.isdir(d), d
    man = pd.read_csv(os.path.join(d, os.path.basename(d) + "_manifest.csv"))
    man = man[man.game_pk == a.game]
    ctx = fetch(a.game)
    j = man[["stem", "play_id"]].merge(ctx, on="play_id", how="left")
    out = os.path.join(ROOT, "output", "pitch_context_%d.csv" % a.game)
    j.to_csv(out, index=False)
    print("game %d: feed pitches %d, manifest clips %d, matched %d"
          % (a.game, len(ctx), len(man), int(j.pitcher_id.notna().sum())))
    print(j.groupby("pitcher_name", dropna=False).size().sort_values(ascending=False).to_string())
    print("->", out)


if __name__ == "__main__":
    main()
