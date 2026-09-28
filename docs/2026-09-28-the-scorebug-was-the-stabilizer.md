# The scorebug was the stabilizer (2026-09-27/28)

PROGRESS-LOG rows 88-100. The short version: the per-clip camera correction was matching the
broadcast OVERLAY, not the field, so every re-framed clip was treated as if the camera had not
moved. Fixing that took the video-vs-data plate error from 1.5-8 in to 0.3-0.5 in at every
park, and a camera fit on one game now works on another game at the same park.

## What was wrong

`stabilize_park.py` v1 estimated a homography from each clip to one reference frame with ORB
features over the WHOLE frame. The scorebug is pixel-identical in every clip, so on any clip
whose framing differed from the reference, ORB's strongest "inliers" were the scorebug and
RANSAC returned roughly the identity. The run still printed hundreds of inliers and a zoom of
~1.00, so nothing looked wrong.

Tell: Isotopes wide clips visibly zoomed out ~25%, yet reported zoom 0.92-1.00. With the
scorebug masked, ORB found 7-64 inliers on the same pairs (it had nothing real); SIFT with the
mask found ~45 and agreed across three independent clips at zoom 1.26-1.28.

## The fix (v2, `*_stab2*` outputs; v1 outputs are void)

1. Mask every overlay before matching: `overlay_boxes` per park in
   `data/landmarks/<game>.json` (scorebug, and the pitch clock, whose digits change).
2. SIFT + Lowe ratio test instead of ORB.
3. Require inliers spread across the frame, not clustered on one board.
4. `--chain`: when a clip no longer matches the reference (Frawley's 7th-8th innings under
   stadium lights), match it to the last clip that did and compose. Costs ~0.1-0.2 in.
5. Cache set-up frames so a rerun skips the ~1 h mitt-detector pass.

Then each park's reference-view camera is fit to ground LANDMARKS (foul lines, the 9 ft mound
circle, the plate bar) plus the fit-half ball tracks, and graded on held-out clips.

## Results (video vs data at the plate, held-out, median in)

| park | before | after |
|---|---|---|
| Asheville 822515 | 0.58 | 0.30 |
| Frawley 821819 | 1.48 | 0.37 |
| Frawley 821848, camera never fit on this game | 2.01 | 0.40 |
| Durham 816698 | 8.35 | 0.40 |
| Isotopes 815419 | 3.40 | 0.33 |
| Isotopes 815443, never fit on this game | - | 1.36 (n 27; limited by its ball tracks) |

## What I got wrong on the way

Row 88 explained Durham's leftover ~5 in as a centre-field camera "timing limit". It was this
bug. The explanation was physically plausible, and it was wrong.

## Checks to keep

- Eyeball two or three clips' reported zoom against how they LOOK. Hundreds of inliers at
  zoom 1.00 on a clip that is visibly re-framed means the overlay did the matching.
- Before explaining a residual with physics, rule out the pipeline in front of it.
- A ball-only camera fit can be right along the pitch path and wrong on the ground (Frawley,
  row 92). Project the field lines and look.
