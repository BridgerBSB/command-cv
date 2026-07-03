# Command CV — Calibration + Training Plan (Jul 3 2026)

Goal: make the CV-only miss number reproduce what Trackman/Hawk-Eye would
say, so we can measure command on pitches with **no Trackman/Hawk-Eye data**.

## The two models (Zac's "Model A vs Model B")

| | **Model A — Trackman-anchored** | **Model B — CV-only** |
|---|---|---|
| Intended target | catcher glove at SETUP (CV) | catcher glove at SETUP (CV) |
| Actual location | **Trackman `plate_x`/`plate_z`** (data) | glove at CATCH, or ball at CONTACT (CV) |
| Ruler | strike-zone homography (17″ plate + `strikeZoneTop/Bottom`) | glove size (~12″) — no zone anchor |
| Role | **ground truth + training label** | the thing we calibrate/train to match A |
| Works on contact pitches? | yes (every pitch has a plate crossing) | yes, via "Mark Contact" (box the ball) |

**The whole idea:** on pitches that have BOTH, Model A is the answer and Model
B is measured against it → we learn the pixel→inches calibration + residual.
Then on no-Trackman pitches, Model B (corrected) stands in for A.

## Phases

### Phase 1 — Assemble the training set (BLOCKED on Zac's creds)
1. **Query all tracked pitches + `plate_x`/`plate_z`** the same way the visuals
   do — **pitcher's view to match the video** (negate `plate_x`, per
   `coordinates.md` Arm Farm convention). Include `strikeZoneTop/Bottom`,
   pitch type, hand, `playId` GUID. The Stats API path already returns all of
   this (`src/ingest/statsapi.py`, LIVE-verified).
2. **Pull the clips** for those playIds via `src/ingest/video.py` (Brodie's
   `analytics(video)` endpoint). Needs auth — see "Lead-me-through" below.
   → clips land in `data/clips/<pitcher>_<type>_<n>.mp4`, annotator-ready.

### Phase 2 — Label (pairs video pixels ↔ Trackman ground truth)
- Annotate glove SETUP + CATCH (or CONTACT) in `tools/annotate.html`.
- Each tagged pitch now has: (video pixel coords of setup + catch) paired with
  (Trackman `plate_x`/`plate_z` actual). This pairing IS the training data.

### Phase 3 — Calibrate the ruler (strike-zone homography)
- Per camera/venue, solve `src/calibration.py` homography from plate corners +
  the zone box → pixel→plate-plane inches. Now the CV catch point comes out in
  **real inches anchored to the zone**, not glove-size-scaled.
- **Validate:** on the paired set, compare CV-derived actual vs Trackman
  `plate_x`/`plate_z`. Report the residual (bias + spread). Fit a correction so
  Model B ≈ Model A where both exist.

### Phase 4 — Train the glove detector (remove manual tagging at scale)
- Labeled setup/catch frames → YOLO (`scripts/train_yolo.py`), OR test
  BaseballCV's pre-trained glove model first (may skip training-from-scratch).
- Once the detector is good, the SETUP glove (and CATCH glove) are found
  automatically → Model B runs without a human clicking every clip.

### Phase 5 — Apply to no-Trackman pitches
- CV finds glove setup + catch → calibrated homography → miss in inches,
  corrected by the Phase-3 residual. This is the command number where there's
  no Trackman/Hawk-Eye. Where Trackman exists, show both (agreement = trust).

## Plane honesty (don't silently mix)
Trackman = plate-front plane; glove-catch = receive plane (behind); contact =
in front. The ball breaks between them. Label each distinctly. For command, the
target lives in the RECEIVE plane, so setup→catch is the purest self-consistent
measure; a glove-size depth correction is the v2 bridge.

## Status / next
- Phase 1 code exists; blocked only on carrying Zac's auth into `video.py`.
- `--probe <guid>` will dump the real analytics(video) JSON so we pin the exact
  `.mp4` schema (`extract_video_urls` is schema-defensive until then).
