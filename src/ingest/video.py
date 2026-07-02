"""Automated per-pitch clip fetch via Brodie's analytics(video) endpoint.

Builds on statsapi.fetch_game_pitches (public feed -> every playId GUID) and adds
the clip fetch Brodie pointed us at:

    https://statsapi.mlb.com/api/v1/game/{gamePK}/{guid}/analytics?hydrate=analytics(video)

That endpoint maps a single playId -> that pitch's video clip. Point this at a
game_pk and it pulls EVERY tracked pitch's clip into data/clips/, named
`<pitcher>_<type>_<n>.mp4` so they drop straight into the annotator + methodb.

--- AUTH (read this) ---------------------------------------------------------
The endpoint is gated behind MLB internal Okta SSO. Probed Jul 1 2026 -> HTTP 401
"Please Login" for anonymous requests (even with a browser UA). Brodie's example
works because his session is authenticated. To automate we need to carry that
auth. Supply ONE of:

    MLB_STATS_COOKIE   env var  (a `name=val; name2=val2` cookie string from an
                                 authenticated statsapi.mlb.com browser session)
    MLB_STATS_BEARER   env var  (an `Authorization: Bearer <token>`)

or pass --cookie / --bearer on the CLI. Ask Brodie which he can provide (a
service token / API key is best; a session cookie works but expires).

Until auth is set every fetch raises AuthError with this same guidance — nothing
here silently half-works.

The public path only exposes CURATED HIGHLIGHTS (game/{pk}/content -> ~40 HR /
condensed clips), NOT every pitch, so it can't replace this endpoint for CV.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.error
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.ingest.statsapi import fetch_game_pitches

ANALYTICS_URL = ("https://statsapi.mlb.com/api/v1/game/{game_pk}/{guid}"
                 "/analytics?hydrate=analytics(video)")
_TIMEOUT = 30
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# MLB pitch-type description -> short code used in clip names (ogando_ff_1.mp4).
_ABBR = {
    "four-seam fastball": "ff", "fastball": "ff", "sinker": "si", "two-seam fastball": "ft",
    "cutter": "fc", "slider": "sl", "curveball": "cu", "knuckle curve": "kc",
    "changeup": "ch", "split-finger": "fs", "splitter": "fs", "sweeper": "st",
    "slurve": "sv", "screwball": "sc", "knuckleball": "kn", "eephus": "ep",
}


class AuthError(RuntimeError):
    pass


_AUTH_MSG = (
    "analytics(video) needs MLB auth (Okta-gated). Set MLB_STATS_COOKIE or "
    "MLB_STATS_BEARER (or pass --cookie/--bearer). Ask Brodie for a service "
    "token/API key (best) or an authenticated statsapi.mlb.com cookie."
)


def _headers(cookie: str | None, bearer: str | None) -> dict:
    h = {"User-Agent": "Mozilla/5.0", "Accept": "application/json"}
    cookie = cookie or os.environ.get("MLB_STATS_COOKIE")
    bearer = bearer or os.environ.get("MLB_STATS_BEARER")
    if cookie:
        h["Cookie"] = cookie
    if bearer:
        h["Authorization"] = f"Bearer {bearer}"
    return h


def fetch_play_analytics(game_pk: int, guid: str, *, cookie=None, bearer=None) -> dict:
    """GET the analytics(video) payload for one playId. Raises AuthError on 401/403."""
    url = ANALYTICS_URL.format(game_pk=game_pk, guid=guid)
    req = urllib.request.Request(url, headers=_headers(cookie, bearer))
    try:
        with urllib.request.urlopen(req, timeout=_TIMEOUT) as r:
            return json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        if e.code in (401, 403):
            raise AuthError(f"HTTP {e.code} - {_AUTH_MSG}") from e
        raise


def extract_video_urls(payload: dict) -> list[str]:
    """Recursively pull .mp4 URLs out of the analytics(video) JSON.

    Schema-defensive: once we can see a real authenticated response we can pin the
    exact path (likely analytics.video[].playbacks[].url). Until then, walk the
    tree for any http(s) .mp4 and de-dupe, best-resolution-first heuristic.
    """
    urls: list[str] = []

    def walk(o):
        if isinstance(o, dict):
            for v in o.values():
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)
        elif isinstance(o, str) and re.match(r"https?://\S+\.mp4", o):
            urls.append(o)

    walk(payload)
    # de-dupe preserving order; prefer URLs that look higher-res
    seen, out = set(), []
    for u in sorted(urls, key=lambda x: (-_res_hint(x), x)):
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out


def _res_hint(url: str) -> int:
    m = re.search(r"(\d{3,4})[pP]|_(\d{3,4})[kK]", url)
    return int(next(g for g in (m.groups() if m else []) if g)) if m else 0


def _abbr(desc: str) -> str:
    return _ABBR.get((desc or "").strip().lower(), re.sub(r"[^a-z]", "", (desc or "x").lower())[:2] or "x")


def download(url: str, dest: str, *, cookie=None, bearer=None) -> str:
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    req = urllib.request.Request(url, headers=_headers(cookie, bearer))
    with urllib.request.urlopen(req, timeout=_TIMEOUT) as r, open(dest, "wb") as f:
        while True:
            chunk = r.read(1 << 16)
            if not chunk:
                break
            f.write(chunk)
    return dest


def pull_game_clips(game_pk: int, out_dir: str | None = None, *, limit=None,
                    calls=None, cookie=None, bearer=None) -> list[dict]:
    """Fetch every tracked pitch's clip for a game into `out_dir` (default data/clips).

    `calls` optionally filters by umpire-call substring(s), e.g. ["strike","ball"]
    for takes only, or ["play","foul"] for contact. Default None = pull everything
    (we annotate contact clips too — they carry the command-vs-damage signal).
    """
    out_dir = out_dir or os.path.join(_ROOT, "data", "clips")
    pitches = fetch_game_pitches(game_pk)
    if calls:
        wants = [c.lower() for c in calls]
        pitches = [p for p in pitches if any(w in (p.call or "").lower() for w in wants)]
    if limit:
        pitches = pitches[:limit]
    counters: dict[str, int] = {}
    rows: list[dict] = []
    for pr in pitches:
        if not pr.play_id:
            continue
        last = (pr.trackman.pitcher_name or "pitcher").split()[-1].lower()
        last = re.sub(r"[^a-z]", "", last) or "pitcher"
        ab = _abbr(pr.trackman.pitch_type)
        key = f"{last}_{ab}"
        counters[key] = counters.get(key, 0) + 1
        stem = f"{key}_{counters[key]}"
        payload = fetch_play_analytics(game_pk, pr.play_id, cookie=cookie, bearer=bearer)
        urls = extract_video_urls(payload)
        if not urls:
            rows.append({"stem": stem, "play_id": pr.play_id, "status": "no-video-url"})
            continue
        dest = os.path.join(out_dir, f"{stem}.mp4")
        download(urls[0], dest, cookie=cookie, bearer=bearer)
        rows.append({"stem": stem, "play_id": pr.play_id, "status": "ok",
                     "pitcher": pr.trackman.pitcher_name, "type": pr.trackman.pitch_type,
                     "hand": pr.trackman.pitcher_throws, "call": pr.call, "file": dest})
    return rows


def main():
    p = argparse.ArgumentParser(description="Auto-pull per-pitch clips for a game.")
    p.add_argument("game_pk", type=int)
    p.add_argument("--out", default=None, help="clip output dir (default data/clips)")
    p.add_argument("--limit", type=int, default=None, help="only the first N pitches")
    p.add_argument("--calls", default=None,
                   help="comma-sep umpire-call substrings to keep (e.g. 'strike,ball' "
                        "for takes, 'play,foul' for contact). default: pull everything")
    p.add_argument("--cookie", default=None)
    p.add_argument("--bearer", default=None)
    p.add_argument("--probe", metavar="GUID", default=None,
                   help="dump the raw analytics(video) JSON for ONE playId and exit "
                        "(use to pin the real video schema once auth works)")
    a = p.parse_args()

    try:
        if a.probe:
            payload = fetch_play_analytics(a.game_pk, a.probe, cookie=a.cookie, bearer=a.bearer)
            print(json.dumps(payload, indent=2)[:4000])
            print("\nvideo urls found:", extract_video_urls(payload))
            return
        calls = [c.strip() for c in a.calls.split(",")] if a.calls else None
        rows = pull_game_clips(a.game_pk, a.out, limit=a.limit, calls=calls,
                               cookie=a.cookie, bearer=a.bearer)
        ok = sum(1 for r in rows if r["status"] == "ok")
        for r in rows:
            print(f"  {r['status']:14s} {r['stem']}")
        print(f"\n{ok}/{len(rows)} clips -> {a.out or 'data/clips'}  (ready for annotate.html)")
    except AuthError as e:
        print(f"AUTH: {e}")
        sys.exit(2)


if __name__ == "__main__":
    main()
