"""MLB / MiLB Stats API feeder.

The public Stats API (statsapi.mlb.com, no key) exposes, per game, every pitch
with Hawk-Eye/Trackman `pitchData` (pX/pZ in feet, strike-zone top/bottom) AND a
`playId` GUID that maps to the pitch's video clip. This feeder pulls that into
TrackmanPitch rows so the pipeline's "actual" location is data, not CV.

Video DOWNLOAD from playId (Film Room entitlement) is intentionally NOT here —
this returns the playId so a separate, validated clip-fetch step can use it.

No third-party deps: uses urllib from the stdlib.
"""
from __future__ import annotations

import json
import urllib.request
from dataclasses import dataclass

from ..pipeline import TrackmanPitch

FEED_URL = "https://statsapi.mlb.com/api/v1.1/game/{game_pk}/feed/live"
_TIMEOUT = 20


@dataclass
class PitchRecord:
    """A Stats API pitch: the Trackman row + the bits needed to fetch its clip."""
    trackman: TrackmanPitch
    play_id: str | None          # GUID -> video clip (Film Room / playback)
    game_pk: int
    at_bat_index: int
    pitch_number: int
    is_pitch: bool
    call: str                    # umpire call description (e.g. "Ball", "Called Strike")


def _get_json(url: str) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": "command-cv/0.1"})
    with urllib.request.urlopen(req, timeout=_TIMEOUT) as r:
        return json.loads(r.read().decode("utf-8"))


def fetch_game_pitches(game_pk: int) -> list[PitchRecord]:
    """Return every tracked pitch in a game as PitchRecord rows.

    Pitches missing `pitchData.coordinates` (pX/pZ) are skipped — those have no
    'actual' location to compare against (Hawk-Eye sparsity at non-tracked MiLB
    parks; ~50% per our tracking notes).
    """
    feed = _get_json(FEED_URL.format(game_pk=game_pk))
    plays = feed.get("liveData", {}).get("plays", {}).get("allPlays", [])

    out: list[PitchRecord] = []
    for ab in plays:
        ab_index = ab.get("atBatIndex", -1)
        matchup = ab.get("matchup", {})
        pitcher = matchup.get("pitcher", {})
        pit_hand = (matchup.get("pitchHand", {}) or {}).get("code", "R")
        pitcher_id = str(pitcher.get("id", ""))
        pitcher_name = pitcher.get("fullName", "")
        game_date = feed.get("gameData", {}).get("datetime", {}).get("officialDate", "")

        for ev in ab.get("playEvents", []):
            if not ev.get("isPitch"):
                continue
            pd = ev.get("pitchData", {})
            coords = pd.get("coordinates", {})
            pX, pZ = coords.get("pX"), coords.get("pZ")
            if pX is None or pZ is None:
                continue  # no tracked location -> no 'actual' to compare
            details = ev.get("details", {})
            ptype = (details.get("type", {}) or {}).get("description", "")
            tm = TrackmanPitch(
                pitch_id=str(ev.get("playId", "") or f"{game_pk}-{ab_index}-{ev.get('pitchNumber')}"),
                plate_x=float(pX),                 # VERIFY sign vs catcher view
                plate_z=float(pZ),
                pitcher_throws=pit_hand,
                sz_top=pd.get("strikeZoneTop"),
                sz_bot=pd.get("strikeZoneBottom"),
                pitcher_id=pitcher_id,
                pitcher_name=pitcher_name,
                pitch_type=ptype,
                session_date=game_date,
            )
            out.append(PitchRecord(
                trackman=tm,
                play_id=ev.get("playId"),
                game_pk=game_pk,
                at_bat_index=ab_index,
                pitch_number=ev.get("pitchNumber", -1),
                is_pitch=True,
                call=(details.get("description", "") or ""),
            ))
    return out


if __name__ == "__main__":
    import sys
    gp = int(sys.argv[1]) if len(sys.argv) > 1 else 814841
    recs = fetch_game_pitches(gp)
    print(f"game {gp}: {len(recs)} tracked pitches")
    for r in recs[:5]:
        t = r.trackman
        print(f"  {t.pitcher_name:22s} {t.pitch_type:14s} "
              f"pX={t.plate_x:+.2f} pZ={t.plate_z:.2f} sz=({t.sz_bot},{t.sz_top}) "
              f"play_id={r.play_id} call={r.call}")
