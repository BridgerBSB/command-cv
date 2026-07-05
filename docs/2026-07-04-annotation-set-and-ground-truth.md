# Command CV — Bootstrap Annotation Set + Ground-Truth Contract (Jul 4 2026)

Establishes the state of the training data + the Model-A ground-truth contract
after Zac hand-annotated the first pitcher bootstrap set. Pairs with the memory
note `command-cv-status.md` and the calibration plan
`docs/2026-07-03-calibration-training-plan.md`.

---

## What we have now

### Clips pulled (authed MLB Stats API, `scripts/pull_pitcher_clips.py`)

| Pitcher | MLBAM | Hand | Level | Clips on disk | Manifest rows |
|---|---|---|---|---|---|
| Joan Ogando | 808010 | R | Asheville A+ (sportId 13) | 50 | 50 |
| Colby Langford | 815308 | L | Asheville A+ | 30 | 32 (2 undownloaded — token expired) |

A+ = single **BROADCAST** angle (behind-pitcher, mitt visible). No CENTERFIELD
feed at A+. Clips ~60 fps, 1280×720. Gitignored under
`data/clips/<name>_2026/`.

### Annotations (Zac, browser `tools/annotate.html`)

- **49 / 50 Ogando clips annotated** Jul 4 2026. The 1 skipped clip was a
  stray side-camera angle (not the broadcast angle) — correctly skipped.
- Ingested via `scripts/ingest_annotation.py` → `data/annotations/*.json`
  (49) + `data/yolo/{images,labels}/train/` (98 new mitt crops: setup+catch).
- **fps FIX (this session):** `annotate.html` hardcoded FPS=30; the API clips
  are ~60 fps. Every tagged frame index mapped to the wrong extracted frame
  (browser frame N @ N/30s vs cv2 frame N @ N/60s). Added an fps override field
  (default 60), wired through the frame/seek math. **Verified**: box overlays on
  extracted setup+catch frames land on the catcher's glove across si_1 / sl_11 /
  fc_1 → frame alignment confirmed, all 49 trustworthy.
- Langford (30 clips) not yet annotated. Set **hand = L** in the annotator for
  those (drives arm-side/glove-side label).

### `endpoint_kind` convention (Zac Q, answered)

- **CATCH** = mitt receives it (take / whiff / ball-in-dirt the catcher gloves) →
  box the mitt at receive.
- **CONTACT** = batter hit it (foul / BIP) → box the **ball where the bat meets
  it**.
- The miss **number is identical either way** (setup-box → endpoint-box
  displacement). `endpoint_kind` only tags which, for the later
  **command-vs-damage** filter ("was that nuke off a missed target or a good
  pitch he just hit?"). Marking a hit as CATCH is harmless to the miss value;
  going forward tag hits as CONTACT so the filter works.

---

## Model-A ground-truth contract (plate_x / plate_z) — ESTABLISHED

Every clip's manifest row (`data/clips/<name>_2026/<name>_2026_manifest.csv`)
carries the Trackman/HawkEye ground truth pulled from the authed
`analytics(video)` payload — **no GC2/Trackman DB access needed**:

| Column | Meaning | Coverage |
|---|---|---|
| `plate_x` | horizontal plate crossing, ft (**Model-A actual**) | 50/50 Ogando, 32/32 Langford |
| `plate_z` | vertical plate crossing, ft | full |
| `sz_top` / `sz_bot` | that batter's strike-zone bounds, ft | full |
| `call` | outcome (Ball / In play / …) | full |
| `pitch_type`, `hand` | arsenal + throwing hand | full |
| `play_id`, `game_pk`, `date`, `cf_url` | provenance + video URL | full |

**plate_z** values are clean pitch heights (2.0–4.5 ft). **sz_top/sz_bot** are
the reference that lets us build the **pixel→inches zone homography** from the
video (plate width 17" horizontal + sz bounds vertical), converting the
CV/glove pixel miss into true inches.

### The ONE open item — plate_x sign (`# VERIFY` in `src/geometry.py`)

The miss **magnitude is exact**. Only the **arm-side vs glove-side WORD** depends
on confirming plate_x's sign convention against one known DB pitch on the work
laptop (flagged `_ARM_SIDE_X_SIGN`). Not blocking the pipeline — a labeling
polarity check.

---

## Miss numbers so far — RELATIVE only (Method B), calibration pending

`scripts/methodb_batch.py` → `output/command_profile_b.csv`. Ogando rollup
(n=55 incl. old GC2 clips): **avg_miss 20.9″**, horiz_bias −0.9″, vert_bias
+0.8″.

**BLOCKING caveat (`docs/BACKLOG.md`):** Method B measures inches off the
**glove size (~12″), NOT the strike-zone reference**. So absolute inches are
signal-only, not trustworthy — the 60–72″ outliers (si_16, si_23) are where
setup/catch gloves are far apart in pixels and the glove-scaling breaks.
**Now unblocked:** sz_top/sz_bot in every manifest → build the zone homography →
convert to true inches calibrated against `plate_x`/`plate_z` (Model A) on the
pitches that have both.

---

## Where we are on the two models

- **Model A (Trackman actual, `plate_x`/`plate_z`)** — ✅ captured for every clip.
  Production spine + training label. Only the sign-word verify is open.
- **Model B (CV glove from video)** — target = observed setup glove (Zac's box);
  actual = catch/contact glove-or-ball box. Currently glove-size-relative;
  calibrate to Model A via the zone homography → then it reproduces the Trackman
  number from video, extended to pitches where Trackman is dark.

---

## Next steps (in order)

1. **Train YOLOv8 mitt detector** on the 98 labeled crops
   (`scripts/train_yolo.py`, pretrained `yolov8n.pt`, CPU) → auto-locate the
   glove instead of clicking. Reference: BaseballCV pre-trained `glove_tracking`
   is a fallback if from-scratch transfer is weak.
2. **Zone-homography calibration** — pixel→inches from plate width (17") +
   sz_top/sz_bot; validate CV miss vs `plate_x`/`plate_z` on pitches with both.
3. **Finish Langford pull** (30→50; re-grab Okta token, hand=L) + annotate.
4. **Verify plate_x sign** on one known DB pitch (work laptop).
5. Auto-apply YOLO to all ~1,108 season pitches (Ogando 437 + Langford 671) →
   Zac audits.

---

## Industry reference (LinkedIn, Harpalsinh Gohil — "Football Match Intelligence")

Validates our stack choice: **YOLOv8 detection** + **Lucas-Kanade optical flow**
(camera pan/tilt compensation) + **K-Means HSV color clustering** (team
separation, zero manual labeling). For us:
- YOLOv8 for glove detection — same path we're on. ✅
- Optical-flow camera compensation — relevant later for multi-frame glove
  tracking on the panning broadcast feed; not needed for the single setup/catch
  frames.
- HSV color clustering — the classical color detector we already tried was noisy
  on the busy orange MiLB background (locks onto the mask cage), which is exactly
  why we're training YOLO instead.
