"""Every venue an Astros affiliate played in, at every level, from the public Stats API.

WHY THIS EXISTS. The detector's error is a PARK property, not a level property -- the
09-18 measurements killed "one pose per park" as a level/archetype rule and showed
Isotopes locking 30/30 clips while Durham zooms 1.166x within the same clip set. So the
lever on generalisation is CAMERA VARIETY, and the cheapest source of camera variety we
have a claim on is the road schedule: every away game is a different broadcast truck, a
different camera position, and a different backdrop, at no extra cost in entitlement.

WHAT IT DOES NOT DO. It does not fetch clips. MiLB clip URLs come from the Okta-gated
analytics endpoint (docs/2026-07-03-statsapi-video-auth.md), which is why the browser does
that half (browser/cv_pull.js). This script answers the question that comes FIRST: which
gamePks are worth spending that browser session on, ordered so no two consecutive pulls
land in the same ballpark.

THE SCHEDULE ENDPOINT IS PUBLIC AND UNAUTHED. sportId + season + teamId + gameType=R
returns gamePk, venue.id, venue.name, dayNight and the home/away split for every game. One
call per team, five teams, about two seconds total. Nothing here needs a token.

COST (rule 18b). --plan alone is 5 HTTP calls. --probe N adds N feed/live calls PER LEVEL,
and a feed is 1-6 MB, so --probe 3 is 15 downloads and roughly a minute. The probe exists
because a venue with no tracked pitchData in the public feed is still a candidate for
CLIPS (the detector needs pixels, not trajectories) but cannot be used for the labels-from-
pose path, and that distinction should be visible before a pull is planned, not after.

    python scripts/affiliate_schedule.py --season 2026
    python scripts/affiliate_schedule.py --season 2026 --probe 2
    python scripts/affiliate_schedule.py --season 2026 --games-per-venue 2 --out output/pull_plan.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import urllib.request
from collections import defaultdict

_TIMEOUT = 40
_UA = {"User-Agent": "command-cv/0.1"}

# parentOrgId 117. Verified against /api/v1/teams?sportId=<id>&season=2026 on 2026-09-18;
# re-derive rather than edit by hand if an affiliate moves.
AFFILIATES = [
    ("mlb", 1, 117, "Houston Astros"),
    ("aaa", 11, 5434, "Sugar Land Space Cowboys"),
    ("aa", 12, 482, "Corpus Christi Hooks"),
    ("aplus", 13, 573, "Asheville Tourists"),
    ("a", 14, 3712, "Fayetteville Woodpeckers"),
]
# FCL (1998) and DSL (601/5005) are deliberately absent: those levels return 404 for video
# at every sampled game (09-17), so scheduling them would plan a pull that cannot happen.

SCHEDULE = ("https://statsapi.mlb.com/api/v1/schedule"
            "?sportId={sport}&season={season}&teamId={team}&gameType=R")
FEED = "https://statsapi.mlb.com/api/v1.1/game/{pk}/feed/live"


def _get(url):
    with urllib.request.urlopen(urllib.request.Request(url, headers=_UA), timeout=_TIMEOUT) as r:
        return json.loads(r.read().decode("utf-8"))


def season_games(sport, team, season):
    """Flatten the schedule into one row per completed regular-season game."""
    d = _get(SCHEDULE.format(sport=sport, season=season, team=team))
    out = []
    for day in d.get("dates", []):
        for g in day.get("games", []):
            if g.get("status", {}).get("codedGameState") != "F":
                continue
            home = g["teams"]["home"]["team"]
            away = g["teams"]["away"]["team"]
            side = "home" if home["id"] == team else "away"
            opp = away if side == "home" else home
            out.append({
                "game_pk": g["gamePk"],
                "date": g.get("officialDate"),
                "venue_id": g.get("venue", {}).get("id"),
                "venue": g.get("venue", {}).get("name"),
                "side": side,
                "opponent": opp["name"],
                "day_night": g.get("dayNight"),
            })
    return out


def probe_tracked(game_pk):
    """(pitches, pitches carrying pitchData.coordinates) in the PUBLIC feed.

    A zero here means the public feed has no trajectory for that level, which is the
    measured state at AA/A+/A. It does NOT mean the game is useless -- the detector path
    needs pixels only -- so this is reported, never used as a filter.
    """
    try:
        d = _get(FEED.format(pk=game_pk))
    except Exception:
        return (-1, -1)
    n = k = 0
    for p in d.get("liveData", {}).get("plays", {}).get("allPlays", []):
        for ev in p.get("playEvents", []):
            if not ev.get("isPitch"):
                continue
            n += 1
            c = ev.get("pitchData", {}).get("coordinates", {})
            if c.get("pX") is not None and c.get("pZ") is not None:
                k += 1
    return (n, k)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, default=2026)
    ap.add_argument("--probe", type=int, default=0,
                    help="sample this many games PER LEVEL for tracked-pitch counts")
    ap.add_argument("--games-per-venue", type=int, default=2,
                    help="how many games the emitted plan takes from each venue")
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    have = set()
    clips = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "data", "clips")
    if os.path.isdir(clips):
        for d in os.listdir(clips):
            tail = d.rsplit("_", 1)[-1]
            if tail.isdigit():
                have.add(int(tail))

    rows = []
    for level, sport, team, name in AFFILIATES:
        try:
            gs = season_games(sport, team, a.season)
        except Exception as e:
            print("  {:6s} SCHEDULE FAILED {}".format(level, e))
            continue
        for g in gs:
            g["level"] = level
            g["team"] = name
            g["pulled"] = int(g["game_pk"] in have)
        rows += gs
        venues = set(g["venue"] for g in gs)
        away = sum(1 for g in gs if g["side"] == "away")
        print("  {:6s} {:28s} {:4d} games  {:2d} venues  ({} away)".format(
            level, name, len(gs), len(venues), away))

    if not rows:
        print("no games; the schedule endpoint returned nothing")
        return 1

    by_venue = defaultdict(list)
    for g in rows:
        by_venue[(g["level"], g["venue"])].append(g)

    print("\nTOTAL {} games across {} level-venues; {} already pulled\n".format(
        len(rows), len(by_venue), sum(r["pulled"] for r in rows)))

    if a.probe:
        print("TRACKED-PITCH PROBE (public feed; 0 tracked is expected below AAA)")
        for level, _sport, _team, _name in AFFILIATES:
            lv = [g for g in rows if g["level"] == level]
            if not lv:
                continue
            step = max(1, len(lv) // max(1, a.probe))
            for g in lv[::step][:a.probe]:
                n, k = probe_tracked(g["game_pk"])
                print("  {:6s} {} {:34s} pitches {:4d} tracked {:4d}".format(
                    level, g["game_pk"], (g["venue"] or "")[:34], n, k))
        print()

    plan = []
    for (level, venue), gs in sorted(by_venue.items()):
        gs.sort(key=lambda g: (g["pulled"], g["date"]))
        plan += gs[:a.games_per_venue]
    # Interleave levels so an interrupted pull still spans every level rather than
    # finishing MLB and leaving A+ empty -- the same cheap-first ordering rule 18b asks
    # for on queries, applied to a download queue.
    order = dict((lv, i) for i, (lv, _s, _t, _n) in enumerate(AFFILIATES))
    plan.sort(key=lambda g: (g["date"], order[g["level"]]))
    print("PLAN: {} game(s) per venue = {} games".format(a.games_per_venue, len(plan)))
    for level, _s, _t, _n in AFFILIATES:
        k = [g for g in plan if g["level"] == level]
        print("  {:6s} {:3d} games  {:2d} venues  {} already on disk".format(
            level, len(k), len(set(g["venue"] for g in k)),
            sum(g["pulled"] for g in k)))

    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        cols = ["level", "team", "game_pk", "date", "venue_id", "venue", "side",
                "opponent", "day_night", "pulled"]
        with open(a.out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for g in plan:
                w.writerow(dict((c, g.get(c)) for c in cols))
        print("\nwrote {}".format(a.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
