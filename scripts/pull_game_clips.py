"""Pull EVERY pitch of a game: clip + trajectory + a manifest row. Token optional at MLB.

`pull_pitcher_clips.py` is pitcher-scoped, which is the wrong unit for a camera solve. The
camera does not care who is pitching, and the joint solve that replaces the drawn zone box
(docs/2026-09-11-step2-pinhole-parity.md, step 2c) needs many clips from the ONE camera. Our
pitcher gives 5 to 60 clips in an A+ game; the game itself carries 280 to 362.

It is also how the training set gets its variety: multiple levels and multiple stadiums, not
one A+ pitcher's home park.

TWO ROUTES (checked Sep 12 2026):
  PUBLIC   Baseball Savant's sporty-videos page hands back the same clip with no auth. Verified
           BYTE-IDENTICAL to the authed pull (same md5, play c4958213 of game 824821), and the
           public game feed carries the 9-param trajectory at MLB. So an MLB game needs NO
           token at all: 5 of 5 plays resolved.
  AUTHED   The analytics endpoint, which needs the Okta token. Required for MiLB: 0 of 4 A+
           plays resolve through Savant, and MiLB tracking is stripped from the public feed,
           so A+ needs the token for both the clip and the trajectory.

Default is auto: use the token when one is present, fall back to public per play otherwise.

Resumable (skips a play that already has its files), threaded, and every file is written to
.part then renamed, so an expired token leaves no half-written clip behind.

    python scripts/pull_game_clips.py 824821 --public          # MLB, no token needed
    python scripts/pull_game_clips.py 822515 --angle BROADCAST # A+, uses data/.mlb_token
"""
import argparse
import csv
import json
import os
import re
import sys
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from src.ingest.video import _api_headers, _cdn_headers, extract_video_urls  # noqa: E402
from pull_pitcher_clips import _TRAJ_COLS, trajectory_truth  # noqa: E402

FEED = "https://statsapi.mlb.com/api/v1.1/game/{pk}/feed/live"
ANALYTICS = "https://statsapi.mlb.com/api/v1/game/{pk}/{pid}/analytics?hydrate=analytics(video)"
SAVANT = "https://baseballsavant.mlb.com/sporty-videos?playId={pid}"
_MP4 = re.compile(r'https://[^"\'\s]+?\.mp4')
MANIFEST_COLS = ["play_id", "game_pk", "date", "venue", "level", "pitcher", "pitcher_id",
                 "pitch_type", "hand", "call", "angle", "source", "clip_downloaded"] + list(_TRAJ_COLS)


def savant_mp4(pid):
    """Public clip URL for a play, no auth. MLB only (A+ returns a page with no mp4)."""
    req = urllib.request.Request(SAVANT.format(pid=pid), headers=_cdn_headers())
    html = urllib.request.urlopen(req, timeout=45).read().decode("utf-8", "replace")
    m = _MP4.findall(html)
    return m[0] if m else None


def _traj_from_feed(pd_):
    """The 9-param trajectory from the PUBLIC feed's pitchData.

    Statcast writes x = x0 + vx0 t + 0.5 ax t^2 while our columns hold the polynomial
    coefficients, so the quadratic term is HALF the acceleration. Only the nine coefficients
    are available here; the authed payload's timing fields (zone_time, t_start/t_end, release
    position) are not, and are left blank rather than guessed."""
    c = (pd_ or {}).get("coordinates") or {}
    need = ("x0", "y0", "z0", "vX0", "vY0", "vZ0", "aX", "aY", "aZ")
    if not all(k in c for k in need):
        return {}
    return {"traj_x0": c["x0"], "traj_x1": c["vX0"], "traj_x2": 0.5 * c["aX"],
            "traj_y0": c["y0"], "traj_y1": c["vY0"], "traj_y2": 0.5 * c["aY"],
            "traj_z0": c["z0"], "traj_z1": c["vZ0"], "traj_z2": 0.5 * c["aZ"]}


def game_pitches(pk):
    """Every pitch with a playId, plus the game's date / venue / level."""
    req = urllib.request.Request(FEED.format(pk=pk), headers={"User-Agent": "Mozilla/5.0"})
    f = json.load(urllib.request.urlopen(req, timeout=60))
    g = f["gameData"]
    meta = dict(game_pk=pk, date=g["datetime"]["officialDate"], venue=g["venue"]["name"],
                level=(g.get("teams", {}).get("home", {}).get("sport", {}) or {}).get("abbreviation", ""))
    out = []
    for play in f["liveData"]["plays"]["allPlays"]:
        for e in play["playEvents"]:
            if e.get("isPitch") and e.get("playId"):
                out.append(dict(play_id=e["playId"],
                                pitcher=play["matchup"]["pitcher"]["fullName"],
                                pitcher_id=play["matchup"]["pitcher"]["id"],
                                hand=play["matchup"]["pitchHand"]["code"],
                                pitch_type=(e.get("details", {}).get("type", {}) or {}).get("code", ""),
                                call=(e.get("details", {}) or {}).get("description", ""),
                                _feed_traj=_traj_from_feed(e.get("pitchData"))))
    return meta, out


def one(pk, rec, angle, out_dir, tok, public):
    """(status, extra manifest fields). Authed route first when a token exists, else public."""
    pid = rec["play_id"]
    mp4 = os.path.join(out_dir, f"{pid}_{angle}.mp4")
    js = os.path.join(out_dir, f"{pid}.json")
    if os.path.exists(mp4):
        return "skip", {}
    url, extra = None, {}
    if tok and not public:
        d = json.load(urllib.request.urlopen(
            urllib.request.Request(ANALYTICS.format(pk=pk, pid=pid), headers=_api_headers(None, tok)), timeout=60))
        if not os.path.exists(js):
            with open(js + ".part", "w") as f:
                json.dump(d, f)
            os.replace(js + ".part", js)
        urls = extract_video_urls(d, angle)
        url = urls[0] if urls else None
        extra = dict(trajectory_truth(d) or {}, source="analytics")
    if url is None:
        url = savant_mp4(pid)
        if url is None:
            return "no_url", extra
        extra = dict(rec.get("_feed_traj") or {}, source="savant") if not extra else extra
    data = urllib.request.urlopen(urllib.request.Request(url, headers=_cdn_headers()), timeout=180).read()
    with open(mp4 + ".part", "wb") as f:
        f.write(data)
    os.replace(mp4 + ".part", mp4)
    return "ok", extra


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("game_pk", type=int, nargs="+")
    ap.add_argument("--angle", default="BROADCAST")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=None, help="first N pitches (a cheap smoke test)")
    ap.add_argument("--public", action="store_true",
                    help="skip the authed endpoint entirely (MLB only; A+ has no public clips)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--bearer", default=None)
    a = ap.parse_args()
    tok = None
    if not a.public:
        tok = a.bearer or os.environ.get("MLB_STATS_BEARER")
        if not tok:
            p = os.path.join(ROOT, "data", ".mlb_token")
            tok = open(p).read().strip() if os.path.exists(p) else None
        if not tok:
            print("no token found -> public route (MLB only; A+ will return no clips)")
    for pk in a.game_pk:
        meta, pitches = game_pitches(pk)
        if a.limit:
            pitches = pitches[:a.limit]
        out_dir = a.out or os.path.join(ROOT, "data", "clips", f"mlb_{pk}")
        os.makedirs(out_dir, exist_ok=True)
        print(f"game {pk} {meta['date']} {meta['venue']} [{meta['level']}]: {len(pitches)} pitches "
              f"-> {out_dir}", flush=True)
        counts, rows = {}, []
        with ThreadPoolExecutor(a.workers) as ex:
            futs = {ex.submit(one, pk, r, a.angle, out_dir, tok, a.public): r for r in pitches}
            for n, fut in enumerate(as_completed(futs), 1):
                rec = futs[fut]
                try:
                    status, extra = fut.result()
                except urllib.error.HTTPError as e:
                    status, extra = f"HTTP{e.code}", {}
                    if e.code in (401, 403):
                        print("AUTH FAILED - token expired. Re-run with --public (MLB) or re-paste.", flush=True)
                except Exception as e:
                    status, extra = type(e).__name__, {}
                counts[status] = counts.get(status, 0) + 1
                row = dict({k: v for k, v in rec.items() if not k.startswith("_")}, **meta,
                           angle=a.angle, clip_downloaded=int(status in ("ok", "skip")), **extra)
                rows.append(row)
                if n % 25 == 0:
                    print(f"  {n}/{len(pitches)} {counts}", flush=True)
        man = os.path.join(out_dir, f"mlb_{pk}_manifest.csv")
        with open(man, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=MANIFEST_COLS, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
        print(f"game {pk} done {counts} -> {man}", flush=True)


if __name__ == "__main__":
    main()
