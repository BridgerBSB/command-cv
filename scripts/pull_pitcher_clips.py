"""Pull ALL of one pitcher's tracked CENTERFIELD clips for a season + a paired
ground-truth manifest (Model-A actual: plate_x/z + strike-zone box per pitch).

This is the training-set builder for the calibration/training plan
(docs/2026-07-03-calibration-training-plan.md). For each tracked pitch it writes:
  - data/clips/<name>_2026/<name>_<type>_<n>.mp4   (CENTERFIELD, annotator-ready)
  - data/clips/<name>_2026/<name>_2026_manifest.csv (one row/pitch = Model-A truth)

The manifest is the pairing that Model B (CV glove) gets calibrated against: each
clip's playId <-> its Trackman/Hawk-Eye plate_x, plate_z, sz_top, sz_bot.

Auth: statsapi analytics endpoint needs the Okta Bearer (see
docs/2026-07-03-statsapi-video-auth.md). The .mp4 CDN needs only a browser UA
(handled in video.download). Idempotent: re-run after a token refresh; existing
clips + manifest rows are skipped, so a mid-run 401 is safe.

Usage:
  python scripts/pull_pitcher_clips.py --name Ogando  --mlbam 808010 \
      --bearer "$(cat data/.mlb_token)"
  python scripts/pull_pitcher_clips.py --name Langford --mlbam 815308 \
      --bearer "$(cat data/.mlb_token)"
  # options: --season 2026  --sport-ids 13,12,11,14,16  --angle CENTERFIELD
  #          --games 821922,822246  --max-games N  --out <dir>
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ingest.video import (AuthError, _abbr, download, extract_video_urls,
                              fetch_play_analytics)

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_UA = {"User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                      "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36")}
_FEED = "https://statsapi.mlb.com/api/v1.1/game/{gp}/feed/live"
_MANIFEST_COLS = ["stem", "game_pk", "date", "play_id", "pitch_type", "hand",
                  "plate_x", "plate_z", "sz_top", "sz_bot", "call", "angle",
                  "clip_downloaded", "cf_url"]


def _get_json(url: str) -> dict:
    with urllib.request.urlopen(urllib.request.Request(url, headers=_UA), timeout=30) as r:
        return json.loads(r.read().decode("utf-8"))


def pitcher_pitches_from_feed(game_pk: int, mlbam: int) -> tuple[str, list[dict]]:
    """Every pitch (incl. untracked) the pitcher threw, from the PUBLIC feed.

    A+ feeds carry the playId + type/hand/call but NO pX/pZ (that comes from the
    authed analytics payload instead). Returns (game_date, [pitch dicts]).
    """
    feed = _get_json(_FEED.format(gp=game_pk))
    date = feed.get("gameData", {}).get("datetime", {}).get("officialDate", "")
    out = []
    for ab in feed.get("liveData", {}).get("plays", {}).get("allPlays", []):
        m = ab.get("matchup", {})
        if (m.get("pitcher", {}) or {}).get("id") != mlbam:
            continue
        hand = (m.get("pitchHand", {}) or {}).get("code", "R")
        for ev in ab.get("playEvents", []):
            if not ev.get("isPitch") or not ev.get("playId"):
                continue
            det = ev.get("details", {})
            out.append({
                "play_id": ev["playId"],
                "pitch_type": (det.get("type", {}) or {}).get("description", ""),
                "hand": hand,
                "call": det.get("description", "") or "",
            })
    return date, out


def pitch_truth(payload: dict) -> dict:
    """Model-A ground truth from the authed analytics payload: plate crossing
    location (x,z) + strike-zone box. Present at A+ even when feed pX/pZ is empty."""
    play = (((payload.get("metaData") or {}).get("stat") or {}).get("play") or {})
    pd = play.get("pitchData", {}) or {}
    loc = ((((payload.get("ballSegments", {}) or {}).get("pitchSegment", {}) or {})
            .get("trajectoryData", {}) or {}).get("location") or {})
    return {
        "plate_x": loc.get("x"),          # plate crossing horiz (VERIFY sign vs catcher view)
        "plate_z": loc.get("z"),          # plate crossing height
        "sz_top": pd.get("strikeZoneTop"),
        "sz_bot": pd.get("strikeZoneBottom"),
        # pitch type + hand also live in the analytics payload (feed omits them at A+)
        "pitch_type": ((play.get("details", {}) or {}).get("type", {}) or {}).get("description", ""),
        "hand": ((play.get("details", {}) or {}).get("pitchHand", {}) or {}).get("code", ""),
    }


def best_video(payload: dict, angle: str):
    """Return (mediaSourceType, url) for the preferred angle, or ('', '') if none."""
    urls = extract_video_urls(payload, angle=angle)
    if not urls:
        return "", ""
    for g in (payload.get("video", {}) or {}).get("playbackGroups", []):
        for r in (g.get("playbackRenditions") or []):
            if r.get("playbackUrl") == urls[0]:
                return g.get("mediaSourceType", ""), urls[0]
    return "", urls[0]


def game_pks_for_pitcher(mlbam: int, season: int, sport_ids) -> list[tuple[str, int]]:
    """(date, game_pk) the pitcher appeared in that season, across the given levels."""
    seen: dict[int, str] = {}
    for sid in sport_ids:
        url = (f"https://statsapi.mlb.com/api/v1/people/{mlbam}/stats"
               f"?stats=gameLog&group=pitching&season={season}&sportId={sid}")
        try:
            d = _get_json(url)
        except Exception as e:
            print(f"  WARN: gameLog sportId={sid} failed ({e}) — that level's games are SKIPPED")
            continue
        for st in d.get("stats", []):
            for sp in st.get("splits", []):
                gp = sp.get("game", {}).get("gamePk")
                if gp:
                    seen[gp] = sp.get("date", "")
    return sorted(((dt, gp) for gp, dt in seen.items()))


def _load_manifest(path: str) -> tuple[list[dict], set, dict]:
    """Return (rows, seen_play_ids, per-type counter) from an existing manifest."""
    rows, seen, counts = [], set(), {}
    if os.path.exists(path):
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                rows.append(row)
                if row.get("play_id"):
                    seen.add(row["play_id"])
                ab = row["stem"].rsplit("_", 1)[0].rsplit("_", 1)[-1]  # <name>_<ab>_<n> (name may contain _)
                counts[ab] = max(counts.get(ab, 0), int(row["stem"].rsplit("_", 1)[-1]))
    return rows, seen, counts


def main() -> None:
    p = argparse.ArgumentParser(description="Pull one pitcher's season CENTERFIELD clips + ground truth.")
    p.add_argument("--name", required=True, help="last name / label for clip stems (e.g. Ogando)")
    p.add_argument("--mlbam", type=int, required=True, help="MLBAM player id")
    p.add_argument("--season", type=int, default=2026)
    p.add_argument("--sport-ids", default="13,12,11,14,16",
                   help="MiLB levels to search game logs (A+,AA,AAA,A,Rok). comma-sep")
    p.add_argument("--angle", default="CENTERFIELD")
    p.add_argument("--games", default=None, help="comma-sep game_pks to override the game-log lookup")
    p.add_argument("--max-games", type=int, default=None)
    p.add_argument("--sample", type=int, default=None,
                   help="pull only N clips, evenly spread across the season (bootstrap set). "
                        "Omit = pull ALL.")
    p.add_argument("--out", default=None)
    p.add_argument("--bearer", default=None, help="Okta access_token (or set MLB_STATS_BEARER)")
    a = p.parse_args()

    name = a.name.strip().lower()
    out_dir = a.out or os.path.join(_ROOT, "data", "clips", f"{name}_{a.season}")
    os.makedirs(out_dir, exist_ok=True)
    manifest_path = os.path.join(out_dir, f"{name}_{a.season}_manifest.csv")

    if a.games:
        games = [("", int(g)) for g in a.games.split(",")]
    else:
        sport_ids = [int(s) for s in a.sport_ids.split(",")]
        games = game_pks_for_pitcher(a.mlbam, a.season, sport_ids)
    if a.max_games:
        games = games[:a.max_games]
    print(f"{a.name} ({a.mlbam}) {a.season}: {len(games)} games")

    rows, seen_play, counts = _load_manifest(manifest_path)

    # Enumerate all pitches from the public feed up front (cheap) so we can subsample.
    print("enumerating feeds...")
    all_p: list[tuple[int, str, dict]] = []
    for date0, gp in games:
        try:
            date, ps = pitcher_pitches_from_feed(gp, a.mlbam)
        except Exception as e:
            print(f"  game {gp}: feed error ({e}) — skip")
            continue
        for p in ps:
            all_p.append((gp, date or date0, p))
    print(f"  {len(all_p)} pitches across {len(games)} games")

    # Evenly spread the sample across the season (dates + within-game variety).
    if a.sample and len(all_p) > a.sample:
        stride = len(all_p) / a.sample
        keep = sorted({int(i * stride) for i in range(a.sample)})
        sel = [all_p[i] for i in keep]
    else:
        sel = all_p
    sel = [x for x in sel if x[2]["play_id"] not in seen_play]
    print(f"  pulling {len(sel)} clips (sample={a.sample or 'ALL'})")

    n_new, n_clips, n_auth_stop = 0, 0, False
    try:
        for gp, date, p in sel:
            if p["play_id"] in seen_play:   # dup game_pk (manual --games or gameLog overlap)
                continue
            try:
                payload = fetch_play_analytics(gp, p["play_id"], bearer=a.bearer)
            except AuthError:
                n_auth_stop = True
                raise
            truth = pitch_truth(payload)
            ptype = truth.get("pitch_type") or p["pitch_type"]
            hand = truth.get("hand") or p["hand"]
            ab = _abbr(ptype)
            counts[ab] = counts.get(ab, 0) + 1
            stem = f"{name}_{ab}_{counts[ab]}"
            dest = os.path.join(out_dir, f"{stem}.mp4")
            angle_used, cf_url = best_video(payload, a.angle)
            got = 0
            if cf_url:
                if not os.path.exists(dest):
                    download(cf_url, dest)
                got = 1
                n_clips += 1
            rows.append({
                "stem": stem, "game_pk": gp, "date": date, "play_id": p["play_id"],
                "pitch_type": ptype, "hand": hand,
                "plate_x": truth.get("plate_x"), "plate_z": truth.get("plate_z"),
                "sz_top": truth.get("sz_top"), "sz_bot": truth.get("sz_bot"),
                "call": p["call"], "angle": angle_used,
                "clip_downloaded": got, "cf_url": cf_url,
            })
            seen_play.add(p["play_id"])
            n_new += 1
    except AuthError as e:
        print(f"\nAUTH: {e}\n-> token expired. Re-grab (statsapi login redirect) into "
              f"data/.mlb_token and re-run this exact command; it resumes.")
    finally:
        with open(manifest_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=_MANIFEST_COLS)
            w.writeheader()
            w.writerows(rows)

    dl = sum(1 for r in rows if str(r.get("clip_downloaded")) == "1")
    print(f"\nmanifest: {len(rows)} tracked pitches ({dl} with CENTERFIELD clip) -> {manifest_path}")
    print(f"this run: +{n_new} new rows, +{n_clips} clips downloaded"
          + ("  [STOPPED on auth — re-run to resume]" if n_auth_stop else ""))
    print(f"clips dir: {out_dir}  (open annotate.html, load the mp4s, tag SETUP glove)")


if __name__ == "__main__":
    main()
