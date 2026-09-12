"""Pull EVERY pitch of a game: clip + authed payload + a manifest row with the trajectory.

`pull_pitcher_clips.py` is pitcher-scoped, which is the wrong unit for a camera solve. The
camera does not care who is pitching, and the joint solve that replaces the drawn zone box
(docs/2026-09-11-step2-pinhole-parity.md, step 2c) needs many clips from the ONE camera. Our
pitcher gives 5 to 60 clips in an A+ game; the game itself carries 280 to 362.

It is also how the training set gets its variety: multiple levels and multiple stadiums, not
one A+ pitcher's home park.

Resumable (skips a play that already has both files), threaded, and every file is written to
.part then renamed, so an expired token leaves no half-written clip behind.

    python scripts/pull_game_clips.py 824821 --angle BROADCAST --workers 6
    python scripts/pull_game_clips.py 822515 821922 --angle BROADCAST     # several games

Token: data/.mlb_token (Okta access token, 1 hour). Zac pastes it; see
docs/2026-07-03-statsapi-video-auth.md.
"""
import argparse
import csv
import json
import os
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
MANIFEST_COLS = ["play_id", "game_pk", "date", "venue", "level", "pitcher", "pitcher_id",
                 "pitch_type", "hand", "call", "angle", "clip_downloaded"] + list(_TRAJ_COLS)


def game_pitches(pk):
    """Every pitch with a playId, plus the game's date / venue / level."""
    req = urllib.request.Request(FEED.format(pk=pk), headers={"User-Agent": "Mozilla/5.0"})
    f = json.load(urllib.request.urlopen(req, timeout=60))
    g = f["gameData"]
    meta = dict(game_pk=pk, date=g["datetime"]["officialDate"], venue=g["venue"]["name"],
                level=(g.get("game", {}) or {}).get("type", "") or "")
    try:
        meta["level"] = g["teams"]["home"].get("sport", {}).get("abbreviation", meta["level"])
    except Exception:
        pass
    out = []
    for play in f["liveData"]["plays"]["allPlays"]:
        for e in play["playEvents"]:
            if e.get("isPitch") and e.get("playId"):
                out.append(dict(play_id=e["playId"],
                                pitcher=play["matchup"]["pitcher"]["fullName"],
                                pitcher_id=play["matchup"]["pitcher"]["id"],
                                hand=play["matchup"]["pitchHand"]["code"],
                                pitch_type=(e.get("details", {}).get("type", {}) or {}).get("code", ""),
                                call=(e.get("details", {}) or {}).get("description", "")))
    return meta, out


def one(pk, rec, angle, out_dir, tok):
    pid = rec["play_id"]
    mp4 = os.path.join(out_dir, f"{pid}_{angle}.mp4")
    js = os.path.join(out_dir, f"{pid}.json")
    if os.path.exists(mp4) and os.path.exists(js):
        return "skip", None
    d = json.load(urllib.request.urlopen(
        urllib.request.Request(ANALYTICS.format(pk=pk, pid=pid), headers=_api_headers(None, tok)), timeout=60))
    if not os.path.exists(js):
        with open(js + ".part", "w") as f:
            json.dump(d, f)
        os.replace(js + ".part", js)
    urls = extract_video_urls(d, angle)
    if not urls:
        return "no_url", d
    if not os.path.exists(mp4):
        data = urllib.request.urlopen(urllib.request.Request(urls[0], headers=_cdn_headers()), timeout=180).read()
        with open(mp4 + ".part", "wb") as f:
            f.write(data)
        os.replace(mp4 + ".part", mp4)
    return "ok", d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("game_pk", type=int, nargs="+")
    ap.add_argument("--angle", default="BROADCAST")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=None, help="first N pitches (a cheap smoke test)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--bearer", default=None)
    a = ap.parse_args()
    tok = a.bearer or os.environ.get("MLB_STATS_BEARER")
    if not tok:
        p = os.path.join(ROOT, "data", ".mlb_token")
        assert os.path.exists(p), "no token: pass --bearer or write data/.mlb_token"
        tok = open(p).read().strip()
    for pk in a.game_pk:
        meta, pitches = game_pitches(pk)
        if a.limit:
            pitches = pitches[:a.limit]
        out_dir = a.out or os.path.join(ROOT, "data", "clips", f"mlb_{pk}")
        os.makedirs(out_dir, exist_ok=True)
        print(f"game {pk} {meta['date']} {meta['venue']}: {len(pitches)} pitches -> {out_dir}", flush=True)
        counts, rows = {}, []
        with ThreadPoolExecutor(a.workers) as ex:
            futs = {ex.submit(one, pk, r, a.angle, out_dir, tok): r for r in pitches}
            for n, fut in enumerate(as_completed(futs), 1):
                rec = futs[fut]
                try:
                    status, payload = fut.result()
                except urllib.error.HTTPError as e:
                    status, payload = f"HTTP{e.code}", None
                    if e.code in (401, 403):
                        print("AUTH FAILED - token expired. Re-paste and re-run; it resumes.", flush=True)
                except Exception as e:
                    status, payload = type(e).__name__, None
                counts[status] = counts.get(status, 0) + 1
                row = dict(rec, **meta, angle=a.angle, clip_downloaded=int(status in ("ok", "skip")))
                if payload is not None:
                    row.update(trajectory_truth(payload) or {})
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
