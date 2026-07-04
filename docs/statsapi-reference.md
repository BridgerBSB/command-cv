# MLB Stats API — Command CV reference (distilled Jul 3 2026)

Distilled from the official OpenAPI spec (`Stats API Documentation` v2.0.0) Zac
pasted. Base: `https://statsapi.mlb.com`. Only the endpoints/schemas that matter
for Command CV are captured here; the full spec has ~150 endpoints.

## Auth model (from `securitySchemes`)
Two schemes: **`basicAuth`** and **`bearerAuth` (JWT)**. Endpoints whose spec
lists `security: [{basicAuth},{bearerAuth}]` require auth — for us that's the
**Okta `access_token` Bearer** (see `2026-07-03-statsapi-video-auth.md`).
Everything else (schedule, people, sports, feed/live, gameLog) is **public**.

**Zac's token privileges** (`groups` claim) → what he can call:
`can-get-all-analytics-endpoint-hydrations`, `can-get-basic-analytics-endpoint`,
`can-get-hydrations-on-guids`, `can-get-basic-guids`, `can-get-weather-data`,
**`can-get-basic-bat-tracking`**. NOT present: `CAN_GET_SKELETAL_DATA`,
`CAN_GET_POSE_TRACKING_DATA`, hitter/pitcher biomechanics → those endpoints will
403 for him. Bat tracking (basic) + analytics + guids + weather = yes.

## Endpoints we use / should use

| Endpoint | Auth | Command-CV use |
|---|---|---|
| `GET /api/v1.1/game/{gamePk}/feed/live` | public | Per-game everything; we enumerate a pitcher's playIds here (A+ has NO pX/pZ). |
| `GET /api/v1/game/{gamePk}/{guid}/analytics?hydrate=analytics(video)` | **bearer** | **Our spine.** `AnalyticsRestObject` — video + zone + plate location + tracked events in one call. |
| `GET /api/v1/game/{gamePk}/guids` | **bearer** | **Cleaner play enumeration** than feed/live: returns every play's GUID with filters `isPitch/isHit/isPickoff` + `AnalyticsPlayMetadataRestObject` (already carries `pitchSegment`, `hitSegment`, `strikeZoneInfo`, metrics!). |
| `GET /api/v1/analytics/guids` / `/analytics/game` | **bearer** | **Bulk discovery by date/season/sportId** — could pull a pitcher's whole season of plays without per-game feed calls. Paginated (`limit`/`offset`). |
| `GET /api/v1/batTracking/game/{gamePk}/{playId}` | **bearer** (basic-bat-tracking ✓) | **`BatTrackingRestObject`: `impact.contactPoint{x,y,z}` = where bat met ball.** For CONTACT pitches this is the actual contact location AS DATA — may remove the manual "box the ball" step. Also attackAngle, batSpeed, sweetSpot, bat head/handle positions. |
| `GET /api/v1/game/{gamePk}/{guid}/contextMetrics` + `contextMetricsAverages` | **bearer** | Per-play `CalculatedMetricRestObject[]` (the Statcast metric list). |
| `GET /api/v1/weather/game/{gamePk}/{playId}` | **bearer** (weather ✓) | Per-play wind/air for ball-flight, if ever needed. |
| `GET /api/v1/sports/{sportId}/players?season=` | public | Resolve MLBAM id by scanning a level's roster (how we found Ogando 808010, Langford 815308). |
| `GET /api/v1/people/{personId}/stats?stats=gameLog&group=pitching&season=&sportId=` | public | A pitcher's game_pks for a season/level. |
| `GET /api/v1/people/search?names=` | public | Name→id, but **MLB-40man only** — misses MiLB (returned empty for Ogando/Langford). Use `sports/{id}/players` for MiLB. |
| `GET /api/v1/schedule?sportId=&startDate=&endDate=` | public | Alt way to enumerate games by date range. |
| `GET /api/v1/stats/search` | **bearer** | "Beast" pitch-level stat search — huge filter set (pitchTypes, plateLocationZ, exitVelocity, playIds, etc). Overkill for now. |

## `AnalyticsRestObject` — the key payload shape (what we parse)
```
strikeZoneInfo: { plateX, plateZ, strikeZoneTop, strikeZoneBottom, isStrike,
                  widthInches, depthInches, edgePositionZone{x,y,z}, ... }
metaData (StatSplits): play.pitchData{strikeZoneTop/Bottom/coordinates},
                       play.details.type/pitchHand, playId, count, ...
ballSegments:
  pitchSegment.trajectoryData.location{x,z}   <- PLATE CROSSING = Model-A actual
  pitchSegment.releaseData{releaseSpeed,spinRate,extension,spinVector,seamOrientation}
  pitchSegment.trajectoryData{VAA,HAA,horizontal/verticalBreak,zoneSpeed,pfxData}
  hitSegment.launchData / landingData          <- for contact pitches
  positions[]{position{x,y,z}, velocity, time} <- full ball path samples
trackedEvents[]: { playEvent (BALL_WAS_PITCHED/HIT/CAUGHT/FIELDED/RELEASED...),
                   playEventType(int), positionId, x, y, timeCode }
                 <- BALL_WAS_CAUGHT (type 4) x/y = glove catch position (DATA)
positionMap: { positionId -> person }          <- who was where
video.playbackGroups[].playbackRenditions[].playbackUrl  (+ mediaSourceType, rank)
bat: BatTrackingRestObject (if hydrated)
calibrationData: { radar/optical: {calibrationId, calibrationTime} }
```
NOTE: `StrikeZoneInfo` also exposes `plateX`/`plateZ` directly — cross-check
against `ballSegments.pitchSegment.trajectoryData.location` for the actual.

## What this changes for the plan
1. **Contact "actual" may be free.** `batTracking.impact.contactPoint{x,y,z}`
   gives the bat-ball contact location as data → on contact pitches we may not
   need Zac to hand-box the ball (where bat tracking exists). Verify coverage at
   A+ (basic-bat-tracking privilege is present).
2. **Glove-catch "actual" may be partly free.** `trackedEvents` `BALL_WAS_CAUGHT`
   x/y is the catch position — where HawkEye tracked it. Sparse at A+, but for
   MLB/tracked parks it removes manual catch annotation. Still need CV for the
   SETUP (intended target) glove — the API does NOT give the pre-pitch mitt.
3. **`/guids` + `/analytics/game` beat walking feed/live** for enumeration and
   already carry pitchSegment/strikeZoneInfo — fewer calls, richer rows.

## What the API does NOT give (still needs CV)
- **Catcher's SETUP glove (intended target)** — no endpoint returns the pre-pitch
  mitt position. This is exactly Command CV's edge; the SETUP glove is always a
  CV/annotation job.
- Per-frame catcher skeleton/pose (privilege-gated; Zac lacks it) — not needed.
