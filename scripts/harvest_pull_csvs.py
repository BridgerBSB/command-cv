"""Harvest cvpull CSVs for many games with a bearer. The authed half, in Python.

WHY THIS AND NOT THE BROWSER. browser/cv_pull.js exists because the analytics endpoint is
Okta-gated and the token dies in 60 minutes, so the browser was doing the authed fetches with
the signed-in session. Two things measured 2026-09-18 collapse that:
  - the session ALONE returns 401, so the browser never removed the bearer from the loop;
  - once the bearer is in hand, Python can make the identical call.
So the browser's only remaining job is producing the token, and everything after it belongs
here, where it is scriptable, resumable and not dependent on a console paste surviving
copy-paste mangling.

THE TOKEN GATES THE JSON, NOT THE VIDEO. sporty-clips URLs need no auth at all (probed with
a Chrome UA, a plain UA and no referer -- all 200) and they do not expire: a URL harvested in
July still served in September. So the right use of a 60-minute token is to harvest as MANY
GAMES' CSVs as possible and download the mp4s afterwards, with no clock running. Downloading
inside the token window would spend the scarce resource on the part that does not need it.

RATE LIMIT IS 25 REQUESTS PER SECOND (MLB's own docs). Exceeding it returns 429, and their
docs also warn that repeatedly requesting NEW tokens can get an account blocked -- so this
throttles below the limit, retries a 429 by waiting rather than by re-authenticating, and
never touches the token beyond reading it.

RESUMABLE BY DESIGN. One CSV per game, written only when the game completes, and a game whose
CSV already exists is skipped. A token expiring mid-run costs the game in flight, not the run.
It stops cleanly on the first 401 instead of grinding through hundreds of failures, because a
401 means the clock ran out and every subsequent call will fail the same way.

    python scripts/harvest_pull_csvs.py --plan output/pull_plan_2026.csv --limit 12
    python scripts/harvest_pull_csvs.py --games 821819,817407 --out-dir output/cvpull
"""
from __future__ import annotations

import argparse
import csv as _csv
import json
import os
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import pull_pitcher_clips as ppc  # noqa: E402  (pinned trajectory + truth extractors)

FEED = "https://statsapi.mlb.com/api/v1.1/game/{pk}/feed/live"
ANALYTICS = ("https://statsapi.mlb.com/api/v1/game/{pk}/{guid}"
             "/analytics?hydrate=analytics(video)")
UA = {"User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                     "(KHTML, like Gecko) Chrome/126.0 Safari/537.36")}

# Columns are exactly what scripts/pull_from_browser_csv.py reads, in its order, so the two
# producers stay interchangeable and its provenance assert applies to both.
COLS = (["play_id", "game_pk", "date", "venue", "pitch_type", "hand", "plate_x", "plate_z",
         "sz_top", "sz_bot", "call", "angle", "cf_url"] + ppc._TRAJ_COLS)

_lock = threading.Lock()
_last = [0.0]
RATE = 18.0          # requests/sec, deliberately under MLB's documented 25


class Expired(Exception):
    """401 -- the token window closed. Distinct from a transient failure on purpose."""


def _throttle():
    with _lock:
        wait = _last[0] + 1.0 / RATE - time.time()
        if wait > 0:
            time.sleep(wait)
        _last[0] = time.time()


def _get(url, bearer=None, tries=3):
    h = dict(UA)
    if bearer:
        h["Authorization"] = "Bearer " + bearer
    for k in range(tries):
        _throttle()
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=h), timeout=30) as r:
                return json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            if e.code == 401:
                raise Expired()
            if e.code == 429:
                # Wait it out. Never re-authenticate on a 429 -- MLB's docs say repeatedly
                # requesting new tokens is what gets an account blocked.
                time.sleep(2.0 * (k + 1))
                continue
            if k == tries - 1:
                raise
            time.sleep(0.5 * (k + 1))
        except Exception:
            if k == tries - 1:
                raise
            time.sleep(0.5 * (k + 1))
    raise RuntimeError("exhausted retries: " + url)


def pick_video(payload, want="CENTERFIELD"):
    """Preferred angle, else the lowest rank actually served. MiLB serves BROADCAST only, so
    the preference is a no-op there -- recorded per row rather than assumed."""
    best = None
    for g in ((payload.get("video") or {}).get("playbackGroups") or []):
        for rd in (g.get("playbackRenditions") or []):
            u = rd.get("playbackUrl") or ""
            if ".mp4" not in u:
                continue
            exact = 0 if g.get("mediaSourceType") == want else 1
            rank = rd.get("rank") if isinstance(rd.get("rank"), int) else 999
            if best is None or (exact, rank) < (best[0], best[1]):
                best = (exact, rank, u, g.get("mediaSourceType"))
    return best


def harvest(pk, bearer, out_dir, want, workers):
    feed = _get(FEED.format(pk=pk))
    gd = feed.get("gameData", {})
    venue = (gd.get("venue") or {}).get("name", "")
    date = (gd.get("datetime") or {}).get("officialDate", "")
    pitches = []
    for ab in feed.get("liveData", {}).get("plays", {}).get("allPlays", []):
        hand = ((ab.get("matchup") or {}).get("pitchHand") or {}).get("code", "")
        for ev in ab.get("playEvents", []):
            if not ev.get("isPitch") or not ev.get("playId"):
                continue
            det = ev.get("details") or {}
            pitches.append({"play_id": ev["playId"], "hand": hand,
                            "pitch_type": ((det.get("type") or {}).get("description") or ""),
                            "call": det.get("description") or ""})
    print("  %d %-34s %s  %d pitches" % (pk, venue[:34], date, len(pitches)), flush=True)
    if not pitches:
        return None, 0, 0

    rows, fails = [], [0]

    def one(p):
        try:
            pay = _get(ANALYTICS.format(pk=pk, guid=p["play_id"]), bearer)
        except Expired:
            raise
        except Exception:
            fails[0] += 1
            return None
        t = ppc.pitch_truth(pay)
        v = pick_video(pay, want)
        row = {"play_id": p["play_id"], "game_pk": pk, "date": date, "venue": venue,
               "call": p["call"], "cf_url": v[2] if v else "", "angle": v[3] if v else ""}
        # The analytics payload fills what the feed omits below AAA (type, hand); a value the
        # feed already answered is not blanked by an empty one from the payload.
        for k, val in t.items():
            if val not in (None, "") or not row.get(k):
                row[k] = val
        for k in ("pitch_type", "hand"):
            if not row.get(k):
                row[k] = p[k]
        return row

    with ThreadPoolExecutor(workers) as ex:
        for r in ex.map(one, pitches):
            if r:
                rows.append(r)

    with_url = sum(1 for r in rows if r.get("cf_url"))
    if not with_url:
        print("    NO CLIP URLS on any row -- not writing a CSV that downloads nothing.",
              flush=True)
        return None, len(rows), fails[0]
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "cvpull_%d.csv" % pk)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = _csv.DictWriter(f, fieldnames=COLS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print("    -> %s  (%d rows, %d with a clip URL, %d failed)"
          % (path, len(rows), with_url, fails[0]), flush=True)
    return path, len(rows), fails[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", default="", help="comma-separated gamePks")
    ap.add_argument("--plan", default="", help="CSV with a game_pk column")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "output", "cvpull"))
    ap.add_argument("--token", default=os.path.join(ROOT, "data", ".mlb_token"))
    # BROADCAST, NOT CENTERFIELD, AND THIS DEFAULT WAS WRONG ON THE FIRST RUN.
    #
    # docs/2026-07-03-statsapi-video-auth.md says "We use CENTERFIELD for CV command work",
    # and that was right for the era it was written in: the pitcher-catcher axis is the best
    # view for a plate-plane homography and for glove tracking. It is the wrong default NOW,
    # because the detector is trained entirely on BROADCAST -- every MiLB park serves only
    # BROADCAST, so the whole training set is. MLB and AAA also serve CENTERFIELD, so the
    # first harvest quietly pulled Daikin Park and Sutter Health Park in a geometry the
    # detector has never seen, and Sutter returned a flight in 0 of 14 clips.
    #
    # Nothing about the trajectory columns is angle-dependent, so those CSVs are still good;
    # only `cf_url` points at the wrong video. CENTERFIELD stays a deliberate FUTURE branch
    # (it exists only at MLB/AAA, and the project's value is at MiLB where it does not), not
    # a default that silently splits the dataset in two.
    ap.add_argument("--angle", default="BROADCAST")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()

    bearer = open(a.token, encoding="utf-8").read().strip()
    import base64
    import datetime
    p = bearer.split(".")[1]
    p += "=" * (-len(p) % 4)
    exp = json.loads(base64.urlsafe_b64decode(p))["exp"]
    left = exp - time.time()
    print("token has %.1f minutes left" % (left / 60))
    if left < 60:
        print("REFUSING: under a minute left. Get a fresh token rather than start a run that "
              "will 401 on its first call.")
        return 1

    pks = [int(x) for x in a.games.split(",") if x.strip()]
    if a.plan:
        import pandas as pd
        d = pd.read_csv(a.plan)
        pks += [int(x) for x in d["game_pk"].tolist()]
    seen, order = set(), []
    for x in pks:
        if x not in seen:
            seen.add(x)
            order.append(x)
    todo = [x for x in order
            if not os.path.exists(os.path.join(a.out_dir, "cvpull_%d.csv" % x))]
    if a.limit:
        todo = todo[:a.limit]
    print("%d games requested, %d already harvested, %d to do\n"
          % (len(order), len(order) - len([x for x in order if x in todo]), len(todo)))

    done = 0
    t0 = time.time()
    for pk in todo:
        try:
            path, n, f = harvest(pk, bearer, a.out_dir, a.angle, a.workers)
            done += 1 if path else 0
        except Expired:
            # Stop on the FIRST 401. Every later call fails the same way, and grinding through
            # them turns one expiry into hundreds of confusing failures.
            print("\nTOKEN EXPIRED after %d games (%.1f min elapsed). Re-run with a fresh "
                  "token; finished games are skipped automatically."
                  % (done, (time.time() - t0) / 60))
            return 2
    print("\n%d games harvested in %.1f min -> %s"
          % (done, (time.time() - t0) / 60, a.out_dir))
    return 0


if __name__ == "__main__":
    sys.exit(main())
