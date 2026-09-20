# Stats API Video Auth + Payload (private notes) — Jul 3 2026

How to pull per-pitch video (and pitch tracking) from MLB's Stats API, verified
live Jul 3 2026 with Zac's Okta login.

> **The live token is NOT in git.** It's a 1-hour secret. It lives in the
> gitignored `command-cv/data/.mlb_token` (or set `MLB_STATS_BEARER`). This doc
> holds the *mechanism* only.

## Two-step auth model (this is the whole trick)

| Step | Endpoint | Gate | Header |
|---|---|---|---|
| 1. Get the JSON (pitch data + video URLs) | `statsapi.mlb.com/api/v1/game/{gamePK}/{guid}/analytics?hydrate=analytics(video)` | **Okta SSO** | `Authorization: Bearer <access_token>` |
| 2. Download the `.mp4` | `sporty-clips.mlb.com/<hash>.mp4` | **Cloudflare** (bot UA block) | browser `User-Agent` + `Referer` — **NO bearer** |

The `.mp4` URLs are NOT Okta-gated. Cloudflare just 403s a non-browser UA
("Attention Required" HTML). `video.py::_cdn_headers()` sends a real Chrome UA +
`Referer: https://statsapi.mlb.com/` → HTTP 200. Verified: CENTERFIELD clip =
358 frames, ~60fps, 1280×720.

## Getting the Bearer (expires in 1 hour)

1. In Chrome, hit the Okta login for statsapi (the research-tool login).
2. After login the browser lands on a redirect like:
   ```
   https://statsapi.mlb.com/#id_token=<jwt>&access_token=<jwt>&token_type=Bearer&expires_in=3600&scope=profile+openid&...
   ```
3. Copy the **`access_token`** value (the one after `&access_token=`, NOT
   `id_token`). That's the Bearer.
4. Save to `command-cv/data/.mlb_token` (gitignored) or `export MLB_STATS_BEARER=...`.
5. It dies in 3600s (1 hr) → re-login to refresh. (A cookie-session or a
   long-lived service token from Adam would avoid the hourly refresh — ask.)

Zac's token privileges (`groups` claim, sub=`zbridger@astros.com`, team-id-117):
`research-tool-user`, `stats-api-team`,
`stats-api-privilege-can-get-all-analytics-endpoint-hydrations`,
`...-basic-analytics-endpoint`, `...-hydrations-on-guids`, `...-basic-guids`,
`...-weather-data`, `...-basic-bat-tracking`. = full analytics access.

## The video schema (pinned)

```
payload["video"]["playbackGroups"][i]
    ["mediaSourceType"]                        # angle name
    ["playbackRenditions"][j]["playbackUrl"]   # .mp4
    ["playbackRenditions"][j]["rank"]          # lower = higher priority
```

10 angles on the MLB example (game 824821): BROADCAST (rank 0),
ALTERNATE_BROADCAST (1), HIGH_HOME (3), **CENTERFIELD (4)**, PITCHCAST (5),
1B/3B BATTER/PITCHER_SIDEVIEW (6-9), MID_HOME (999). All 4000K MP4.

**ANGLE, CORRECTED 2026-09-19 -- READ THIS BEFORE COPYING THE LINE BELOW IT.**
The MiLB ball detector is trained on **BROADCAST** and `rolling_pull.py --angle`
defaults to BROADCAST, refusing any game whose banked clips came down in another
geometry. The sentence that used to sit here said CENTERFIELD, it was written for
the July MLB pull, and copying it into the MiLB harvest is what brought Daikin Park
and Sutter Health down in an unseen geometry (Sutter returned 0 of 14 clips).
So: **MiLB / detector work = BROADCAST.** The legacy MLB path
`src/ingest/video.py` still defaults to `--angle CENTERFIELD` (pitcher→catcher
axis, chosen for the plate-plane homography + glove tracking) and is left that way
on purpose -- every MLB clip in the project was pulled with it. Name the angle
explicitly rather than relying on either default.

## BONUS — the analytics payload IS Trackman-equivalent (big)

The same authed call also returns per-pitch tracking, so **Model A's "actual"
plate location can come straight from this endpoint — no separate Trackman DB
access needed**:

- `metaData.stat.play.pitchData.strikeZoneTop / strikeZoneBottom / strikeZoneWidth(17.0)` — the zone box for the homography.
- `ballSegments.pitchSegment.trajectoryData.location = {x, z}` — **the plate crossing point** (our Model-A actual). Example: x=0.844, z=2.241.
- `releaseData` (release pos/speed/spin/extension), full `trajectoryData` (VAA/HAA/breaks).
- `hitSegment` (launch speed/angle, landing bearing/distance) — for contact pitches.
- `trackedEvents[]` with `playEventType` + `x`/`y` — includes `BALL_WAS_FIELDED` (17), `BALL_WAS_CAUGHT` (4) fielder positions. Same event vocabulary as `groundcontroltracking.Tracking.LK_Play_Event_Types`.
- `positionMap` — playerId per fielding position.

**Implication for the plan:** one authed call per playId gives us (a) the video,
(b) the strike-zone box for calibration, and (c) the actual plate_x/plate_z
ground truth — the entire Model-A + Model-B training pair in one request. We may
not need GC2/Trackman DB access at all for the calibration set.

## Usage

```
# probe one play (angles + ordered URLs)
python src/ingest/video.py <game_pk> --probe <guid> --bearer "$(cat data/.mlb_token)"

# pull a whole game's CENTERFIELD clips into data/clips/
python src/ingest/video.py <game_pk> --bearer "$(cat data/.mlb_token)"
# --calls "play,foul"  -> contact pitches only ;  --angle HIGH_HOME  -> other angle
```
