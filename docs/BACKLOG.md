# Command CV — Backlog / Open Notes

Running list of things to revisit. Newest first.

## Measurement reference — Method B has NO strike-zone anchor (Jul 3 2026)

**The note Zac asked to record.** Method B (`src/miss_methodb.py`) measures
the miss in inches using the **catcher's mitt size (~12″) as the ruler**
(`scale = glove_diameter_in / diam_px`). There is **no strike-zone / plate
reference** in that path, so:

- The absolute inch value is only as good as the assumed 12″ mitt + the
  camera angle. On the Ogando down-the-line side-cam the horizontal axis is
  weak and the numbers ran large (avg 20.7″, sl_2 = 41″).
- It also folds in the **catcher's own receiving movement** (reach / frame /
  shift setup→catch), which is not pitcher "miss."

**Fix / direction (see the calibration-training plan):** anchor the ruler to
the strike zone via `src/calibration.py` (home-plate 17″ + `strikeZoneTop/
Bottom` DLT homography), and use **Trackman `plate_x`/`plate_z` as ground
truth** on pitches that have it to calibrate the CV-only number. Report
Method-B numbers as **relative** signal until the calibration lands.

Treat today's 12 Ogando numbers as directional (which pitch missed more,
up/down tendency), NOT as trustworthy absolute inches.

## Training volume

12 annotated clips (24 frames) is a proof-of-concept label set only. The
glove detector will need far more labeled frames (Zac: "we may need to train
way more"). Two feeders: (1) hand-annotate more clips, (2) test BaseballCV's
pre-trained `glove_tracking.pt` / RF-DETR to skip training-from-scratch.
