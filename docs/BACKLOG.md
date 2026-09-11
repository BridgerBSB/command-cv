# Command CV — Backlog / Open Notes

Running list of things to revisit. Newest first.

## 12 of 55 labels were ingested from the WRONG video (Sep 11 2026) - OPEN

Found while auditing step 1. `scripts/ingest_annotation.py` resolves a clip by
bare filename (`--clip` -> `--clips-dir` -> next to the JSON -> `data/clips/<name>`
-> cwd). Twelve Ogando stems exist in BOTH `data/clips/` (Jun 29 GC2 zip, side
cam) and `data/clips/ogando_2026/` (Jul 3/8 API pull): cu_1, cu_3, fc_1, fc_2,
fc_3, ff_1..ff_5, sl_1, sl_2. For all 12 the saved training image matches the ZIP
clip (diff 0.0) and not the API clip (diff 34-62). 43 others match the API clip
exactly.

Two different cases inside the 12:
- **cu_3, ff_1..ff_5 (saved Jul 2):** labeled on the zip clips. Their API twins
  were downloaded Jul 8. These labels do not describe the API pitches at all.
- **cu_1, fc_1..fc_3, sl_1, sl_2 (saved Jul 5):** the label coordinates may be on
  the API clip, but the saved training image came from the zip clip, so the YOLO
  box sits on the wrong picture. UNVERIFIED which way; check by eye.

What it can touch (NOT yet measured):
1. `scripts/calibrate_report.py` joins `data/annotations` to
   `ogando_2026_manifest.csv` by stem, so the Jul 5 affine LOO (median 4.2 in)
   may include zip-clip glove pixels paired with API plate_x/plate_z.
2. The Jul 5 mitt detector (98 crops) trained on `data/yolo/images`, which holds
   the mismatched crops for up to 24 samples.
3. `tighten.py` (the 3.9 in LOO) auto-detects on API clips and does not read the
   annotations, but it runs that detector.

**HALF FIXED Sep 11 2026.** `scripts/ingest_annotation.py` no longer picks a clip
silently: an explicit `--clip` / `--clips-dir` wins, and otherwise every place the
name could live is searched and a name matching MORE THAN ONE video is REFUSED,
naming the candidates. Each annotation now records `clip_path` / `clip_bytes` /
`clip_frames`, so provenance is checkable instead of inferred. Frame extraction
decodes in order rather than seeking (a seek lands 0-2 frames early; verified
different on a real clip). Guards in `tests/test_ingest_clip_resolution.py`, proven
against the pre-fix resolver: it picks `data/clips/ogando_ff_1.mp4` (the zip clip)
while the new one refuses.

STILL OPEN, none of it measured:
1. Re-ingest the 6 Jul-5 labels (cu_1, fc_1..fc_3, sl_1, sl_2) with
   `--clips-dir data/clips/ogando_2026`, after checking by eye that each box lands
   on the glove in the API clip.
2. Decide what to do with the 6 Jul-2 zip-clip labels (cu_3, ff_1..ff_5): they
   describe pitches the API set does not contain. Drop them from the API pool or
   keep them tagged to the zip clips.
3. Re-run `calibrate_report.py` after 1 and 2 and compare the LOO median against
   the 4.2 in on record.
4. Rebuild the YOLO crops and decide whether the mitt detector needs a retrain
   (up to 24 of 98 training samples paired a box with the wrong image).

Related: cv2 `set(CAP_PROP_POS_FRAMES)` lands 0-2 frames EARLY on these clips
(tested against in-order decode on 4 clips). `ingest_annotation.py`,
`src/video.py` and `src/auto_glove.py` all seek. Harmless for a held setup glove;
up to 33 ms off at the catch.

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
