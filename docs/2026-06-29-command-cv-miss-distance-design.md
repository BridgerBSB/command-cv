# Command CV — Computer-Vision Miss-Distance / Command Tracker

**Created:** 2026-06-29 · **Branch:** `feature/pd-goals` · **Status:** DESIGN → building PoC
**Owner:** Zac Bridger (PD analyst)

Recreating the tool from Lucas Baquero's LinkedIn post: quantify *command* by
measuring how far a pitch missed the catcher's intended target — turning
"intent into measurable data" instead of the eye test.

---

## 1. The core insight (what makes this tractable)

We do **NOT** track the 95 mph ball through the video. That is nearly
impossible from standard broadcast FPS (motion blur, 30 fps).

- **Intended location** = where the catcher's glove is set → **computer vision**
  on a single selected frame (the "glove flash").
- **Actual location** = where the pitch crossed the plate → **Trackman / Hawk-Eye**
  `plate_x` / `plate_z` (already tagged, far more accurate than any CV ball-track).
- **Miss** = vector from intended → actual, expressed in real **inches** + a
  **direction** (arm-side / glove-side / up / down).

This is exactly the reference method ("compared to a Trackman's tagged location").

## 2. Inputs / outputs

**Inputs per pitch:**
- One video frame (the glove-flash frame) OR a clip + a frame selector.
- A Trackman/Statcast pitch row: `plate_x`, `plate_z`, `sz_top`, `sz_bot`,
  `pitcher_throws` (hand), `pitch_id`, optional pitch type / velo.

**Output per pitch:** a **Strike-Zone View** report (PNG/PDF) — strike-zone box,
crosshair = intended (glove), circle = actual (Trackman), dashed miss line, miss
distance (in) + direction label. Plus a row appended to a per-pitcher
**command profile** CSV for session-over-session tracking.

## 3. Pipeline (stages)

```
[0] INGEST      video + trackman row   (provided clip  | MLB/MiLB Stats API feeder)
[1] FRAME PICK  choose glove-flash frame (manual idx/ts | auto pose/motion v2)
[2] DETECT      home plate + strike zone + glove in that frame (trained model | assisted)
[3] CALIBRATE   homography: image px -> Trackman plate-plane (ft), anchored on plate (17")
[4] COMPUTE     intended=(glove in zone coords); actual=(plate_x,plate_z); miss vec+mag+dir
[5] REPORT      render Strike-Zone View; append to player command profile
[6] AGGREGATE   per-pitcher command profile over sessions (avg miss, directional bias)
```

Build order is **value-first**: stages [3]-[6] (geometry + calibration + report +
profile) are the engine and are built/tested **first** with synthetic frames.
Stages [0]-[2] (ingest + detect + the trained model) are feeders bolted on after.

## 4. Module layout (`command-cv/`)

| File | Role |
|---|---|
| `src/geometry.py` | Strike-zone constants, Trackman coord frame, miss vector + direction. cv2-free. |
| `src/calibration.py` | Homography from image reference points → plate plane (ft). cv2-optional. |
| `src/report.py` | Strike-Zone View renderer (matplotlib, Astros colors). cv2-free. |
| `src/detect.py` | Detection interface: trained-model backend + classical/assisted fallback. **lazy** cv2. |
| `src/profile.py` | Per-pitcher command-profile CSV append + session rollups. |
| `src/pipeline.py` | Orchestrates frame + trackman row → report + profile row. |
| `src/ingest/statsapi.py` | MLB/MiLB Stats API feeder (schedule → game → per-pitch playId + pitchData). |
| `scripts/synth_frame.py` | Draw a synthetic glove-flash frame (plate + zone + glove) for testing. |
| `scripts/render_synthetic_report.py` | End-to-end smoke → report PNG to eyeball. |
| `tests/` | Unit tests: geometry math, calibration round-trip, direction labels. |

## 5. Coordinate frame + the things we must NOT guess

- Trackman `plate_x`,`plate_z` in **feet**, origin = center of plate front, z up.
- Rulebook plate half-width = 8.5 in = **0.7083 ft** → zone box x-edges `±0.7083`.
- `sz_top`,`sz_bot` are **batter-specific** (from the Trackman row); synthetic
  defaults 3.5 / 1.5 ft.
- **plate_x SIGN convention** (which side is catcher-left) and **arm-side vs
  glove-side** label both depend on a convention that MUST be validated against
  one known pitch on the work laptop (DB access). Until then: magnitude is exact;
  the L/R + arm/glove *label* sits behind a single documented constant flagged
  `# VERIFY`. (Blocking rule: never guess — make it explicit + verify.)

## 6. Known limitation — glove parallax (documented, not hidden)

The glove sits ~1.5-3 ft behind the plate-front plane and is not exactly on the
calibration surface; from the elevated CF camera this introduces a small
scale/vertical bias. The reference tool accepts this. v1 replicates their method
and **labels the report "approximate."** Optional v2 refinement: depth-correct
using the glove's known size (~12 in) as a second scale cue.

## 7. "Train our model" — the detector

The CV detector for **glove / home plate / strike-zone** is a trained object/keypoint
model (Ultralytics YOLO is the planned backend). Training data = frames scraped from
MiLB/MLB via the Stats API `playId` clips, hand-labeled (glove box, plate keypoints).
v1 ships an **assisted/classical fallback** (plate via line/edge detection, glove via
click-or-template) so the engine is usable before the model is trained; the model is
a drop-in upgrade behind the same `detect.py` interface.

## 8. Video acquisition (the open feeder question)

MLB Stats API (`statsapi.mlb.com`, no key) gives schedule, live feed, per-pitch
`playId` GUID + `pitchData.coordinates` (pX/pZ). Mapping `playId` → an actual
**downloadable** clip is the uncertain part (entitlement/Film Room). Design treats
ingest as pluggable: **(a)** user-provided clips work today; **(b)** Stats-API
auto-fetch is a feeder we validate separately once the clip-URL path / any key is
confirmed. Core value does not block on it.

## 9. Out of scope (YAGNI for v1)

- Real-time / live in-game processing.
- Ball-flight CV tracking (Trackman owns "actual").
- Multi-camera fusion. A Posit Connect deployment (Connect can't decode video).

## 10. Verification gates

- Geometry/calibration unit tests (round-trip a known point).
- **Render-and-look** (BLOCKING rule #18): every report rendered to PNG and
  visually inspected — worst cases (miss far outside zone, glove at zone edge).
- plate_x sign + arm/glove label validated against ONE known DB pitch before any
  coach-facing use.
