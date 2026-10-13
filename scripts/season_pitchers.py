"""Every Houston pitcher in the 2026 season list, what the season run has measured of him, and
whether he is still in the organisation.

Zac, 2026-10-13: the goal is the miss distances + chart for the WHOLE MiLB season for each
Houston pitcher still on eBIS. That needs a denominator the board does not have: the board
only knows the games that passed, so a pitcher whose home park is blocked looks complete.

  pitched   public box score of every game in output/season_pull_2026.csv: Houston's side
            (parentOrgId 117), every pitcher with a pitch, his pitch count. Cached per game.
  measured  output/command_board_org_<board>_pitches.csv, the per-pitch rows the board drew.
  in org    Stats API people?hydrate=rosterEntries: an ACTIVE entry at a club whose parent is
            117. This is MLB's transaction record, the same source eBIS is fed from, read on the
            day it runs. (Not currentTeam - see current_orgs.)
            It is NOT a PP_MASTER read; the CSV says which it was in `org_source`.

    python scripts/season_pitchers.py                     # -> output/season_pitchers_2026.csv
    python scripts/season_pitchers.py --board season2026
"""
import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src.ingest.statsapi import _get_json  # noqa: E402

HOU = 117
SEASON = os.path.join(ROOT, "output", "season_pull_2026.csv")
CACHE = os.path.join(ROOT, "output", "boxscores_2026")
BOX_URL = "https://statsapi.mlb.com/api/v1/game/{g}/boxscore"
PEOPLE_URL = "https://statsapi.mlb.com/api/v1/people?personIds={ids}&hydrate=rosterEntries"
TEAM_URL = "https://statsapi.mlb.com/api/v1/teams/{t}"
LEVEL = {"aaa": "AAA", "aa": "AA", "aplus": "A+", "a": "A"}


def box(g):
    p = os.path.join(CACHE, "%d.json" % g)
    if os.path.exists(p):
        return json.load(open(p, encoding="utf-8"))
    j = _get_json(BOX_URL.format(g=g))
    json.dump(j, open(p, "w", encoding="utf-8"))
    return j


def houston_pitchers(g, level):
    j = box(g)
    rows = []
    for side in ("home", "away"):
        t = j["teams"][side]
        if (t.get("team") or {}).get("parentOrgId") != HOU:
            continue
        for pid in t.get("pitchers", []):
            pl = t["players"].get("ID%d" % pid, {})
            st = (pl.get("stats") or {}).get("pitching") or {}
            rows.append(dict(game_pk=g, level=LEVEL.get(level, level), pitcher_id=pid,
                             pitcher_name=(pl.get("person") or {}).get("fullName"),
                             pitches=st.get("numberOfPitches") or st.get("pitchesThrown") or 0,
                             gs=st.get("gamesStarted", 0)))
    return rows


def current_orgs(ids):
    """The org = the club of each ACTIVE roster entry (isActive, no endDate), mapped to its
    parent org. NOT currentTeam: that is where he is PLAYING, which in October is an AFL or
    winter club (Scottsdale Scorpions, Tigres de Aragua) whose parent is not his org - read
    that way, 18 pitchers left the org who had not (2026-10-13). A player assigned to the AFL
    keeps an active entry at his own affiliate beside the Scottsdale one."""
    out, team_org = {}, {HOU: HOU}
    ids = sorted(ids)
    for k in range(0, len(ids), 50):
        j = _get_json(PEOPLE_URL.format(ids=",".join(map(str, ids[k:k + 50]))))
        for p in j.get("people", []):
            act = [e for e in p.get("rosterEntries", [])
                   if e.get("isActive") and not e.get("endDate")]
            out[p["id"]] = dict(active=p.get("active"),
                                active_teams=[(e["team"]["id"], e["team"].get("name")) for e in act],
                                throws=(p.get("pitchHand") or {}).get("code"))
    for t in {tid for v in out.values() for tid, _ in v["active_teams"]} - set(team_org):
        tj = _get_json(TEAM_URL.format(t=t)).get("teams", [{}])[0]
        # a big-league club IS its org (no parentOrgId); an affiliate names its parent
        team_org[t] = t if (tj.get("sport") or {}).get("id") == 1 else tj.get("parentOrgId")
    for v in out.values():
        orgs = {team_org.get(tid) for tid, _ in v["active_teams"]}
        v["active_teams"] = "; ".join(n for _, n in v.pop("active_teams"))
        v["org_ids"] = " ".join(str(o) for o in sorted(o for o in orgs if o))
        v["current_org"] = HOU if HOU in orgs else (sorted(o for o in orgs if o) or [None])[0]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--board", default="season2026")
    ap.add_argument("--out", default=os.path.join(ROOT, "output", "season_pitchers_2026.csv"))
    a = ap.parse_args()
    os.makedirs(CACHE, exist_ok=True)
    season = pd.read_csv(SEASON)
    with ThreadPoolExecutor(8) as pool:
        parts = list(pool.map(lambda r: houston_pitchers(int(r.game_pk), r.level),
                              season.itertuples(index=False)))
    app = pd.DataFrame([r for p in parts for r in p])
    n_games = app.game_pk.nunique()
    print("box scores: %d of %d season games carry a Houston pitcher (%d appearances)"
          % (n_games, len(season), len(app)))
    assert n_games == len(season), "a season game has no Houston side in its box score"

    per = app.groupby("pitcher_id").agg(pitcher_name=("pitcher_name", "first"),
                                        games=("game_pk", "nunique"), pitches=("pitches", "sum"),
                                        gs=("gs", "sum"),
                                        levels=("level", lambda s: "/".join(sorted(set(s), key=list(LEVEL.values()).index))))
    orgs = current_orgs(per.index.tolist())
    per = per.join(pd.DataFrame.from_dict(orgs, orient="index"))
    per["in_org"] = per.current_org == HOU
    per["org_source"] = "statsapi rosterEntries " + datetime.now().strftime("%Y-%m-%d")

    bp = os.path.join(ROOT, "output", "command_board_org_%s_pitches.csv" % a.board)
    if os.path.exists(bp):
        d = pd.read_csv(bp)
        m = d.groupby("pitcher_id").agg(games_measured=("game", "nunique"),
                                        pitches_measured=("game", "size"))
        per = per.join(m)
    per[["games_measured", "pitches_measured"]] = per[["games_measured", "pitches_measured"]].fillna(0).astype(int)
    per["share_pitches"] = per.pitches_measured / per.pitches.where(per.pitches > 0)
    per = per.sort_values(["in_org", "pitches"], ascending=[False, False])
    per.to_csv(a.out)
    io = per[per.in_org]
    print("pitchers: %d Houston pitchers threw in these games; %d still in the org (%s)"
          % (len(per), len(io), per.org_source.iloc[0]))
    print("  in org: %d pitches thrown, %d measured on the board (%.0f%%); %d pitchers with none"
          % (io.pitches.sum(), io.pitches_measured.sum(),
             100 * io.pitches_measured.sum() / max(io.pitches.sum(), 1),
             (io.pitches_measured == 0).sum()))
    print(a.out)


if __name__ == "__main__":
    main()
