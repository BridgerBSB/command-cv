# The detector gate was reading the wrong frames

Sep 13 2026. Every detector measurement on record was taken on frames that contain no
ball. The detector was never undertrained. Corrected, the same weights file scores
**97.7% found at 3.8-4.5 px top-1**, where the record says 29-48% found at 167-308 px.

## The defect

Two sources carry a column called `frame_idx` and they count from different origins.

| source | `frame_idx` means | range on game 824821 |
|---|---|---|
| OpenCommand `gloveball_tracks` | frames from **RELEASE** | -28 .. +30 |
| our `ball_labels_<game>.csv` | frames from the **start of our clip file** | 179 .. 213 |

`detect_ball_eval.py` and `detect_diagnose.py` both did

```python
want = {int(r.frame_idx): (r.baseball_center_x, r.baseball_center_y) for r in seg.itertuples()}
frames, i, hi = {}, 0, max(want)
while i <= hi: ...        # read OUR clip, keep the frames in `want`
```

with `r` coming from OpenCommand. So `max(want)` was about +30, and the loop read
**frames 0 to 30 of our clip** - the pitcher still standing on the mound, roughly three
seconds before release - then scored those detections against the ball's **mid-flight**
position. The pitch crosses the plate around frame 205.

That is the whole of the "44 px", "167 px" and "308 px" errors. There was no ball in any
frame either script ever looked at.

## How it was found

Not by any aggregate. The labels and the truth were merged on their shared key:

```
labels   5540 rows over 277 plays
oc track 8029 rows over 288 plays
overlap  0 rows            <- 277 plays in common, zero (play_id, frame_idx) matches
```

**Zero overlap on a key shared by 277 plays is the finding.** No top-1, best-of-K, rank
or confidence statistic could have surfaced it, because every one of them was computed
over the wrong rows. See the correction added to
`.claude/rules/metric-must-separate-failure-modes.md`.

## The mapping, derived two independent ways

`absolute_clip_frame = oc_frame_idx + shift`, per play.

| derivation | how | median | sd |
|---|---|---|---|
| geometric | `f0 - T_plate * fps`, **no pixels used** | 181.9 | 2.29 |
| empirical | slide an integer shift, best match of our projected path to OC's detected path, **pixels only** | 183.0 | 2.36 |

They disagree by a median of **+0.57 frames**, with 263 of 277 plays inside 2 frames.
That residual bias is expected and explained: `ball_clock_align.py` scans `f0` in whole
frames while the true plate crossing falls between frames, so the geometric shift carries
a sub-frame offset of up to 0.5.

Supporting evidence that OC frame 0 is release: their `release_s_hint` spans only
-0.0084 to +0.0084 s, i.e. it is the sub-frame remainder of a release at frame 0.

## The other headline: our labels are good to 0.65 inches

At the winning shift, our geometry-derived label and OpenCommand's **independent**
detection describe the same ball:

| | |
|---|---|
| median | **2.15 px = 0.65 in** |
| p90 | 3.89 px |
| max | 6.83 px |
| plays under 5 px | 273 / 277 |
| plays under 10 px | **277 / 277** |

This is one number validating the pose solve, the Statcast trajectory, the clock
alignment and the label generation together, against a source that shares none of our
code. It is a stronger statement than anything previously on record, and it was
unavailable until the two frame spaces were reconciled.

It also confirms by construction what `output/ball_path_check_824821.png` shows by eye:
the label circle sits on the ball in all 15 panels.

## The A/B: the frames were the entire variable

The record's numbers came from earlier weights, so "the fix did it" needed isolating from
"the epochs did it". Both mappings were run against the **same `best.pt`**, same clips,
same truth:

| arm | mode | frames | found% | top1 px | bestK px | bestK in | conf med |
|---|---|---|---:|---:|---:|---:|---:|
| OLD `shift=0` | tile | 155 | 61.9% | 54.64 | 45.71 | 13.713 | 0.198 |
| OLD `shift=0` | full | 155 | 45.8% | 43.98 | 37.41 | 11.222 | 0.195 |
| **NEW mapped** | tile | 171 | **97.7%** | **4.49** | **2.64** | **0.792** | **0.468** |
| **NEW mapped** | full | 171 | **97.7%** | **3.77** | **2.77** | **0.830** | **0.460** |

The OLD arm reproduces the "epoch 13" table in
`2026-09-12-detector-gate-undertrained.md` **digit for digit** (61.9 / 54.64 / 45.71 /
0.198 and 45.8 / 43.98 / 37.41 / 0.195). `best.pt` has not been written since Sep 12
16:43, so that run and this one used the same weights file, which is why the match is
exact rather than merely close. The epochs moved none of it.

Note the confidence column. Confidence is computed with **no reference to truth at all**,
so 0.195 against 0.46 on identical weights can only be caused by the images being
different. That is the cleanest single piece of evidence in the table.

Per-frame detail from `detect_diagnose.py` on the corrected frames, same weights:

    00f695d8 f195  truth (631,299)   #1 conf .416   5.2 px off   (#2  1.5 px)
    018f0e61 f191  truth (584,281)   #1 conf .460   6.3 px off
    01dde385 f199  truth (649,301)   #1 conf .469   6.9 px off   (#2  4.1 px)

Every box 19-21 px, ball-sized. Two to five detections per frame rather than a crowd of
false positives, because there is now a ball in the frame to find.

## What this retracts

From `2026-09-12-detector-gate-undertrained.md`, all void:

- "undertrained, not broken" - the diagnosis, and the evidence for it
- "the ball IS detected in 2 of 3 frames, ranked #3" - read off a pre-pitch frame
- "confidence 0.02-0.20 everywhere, the signature of a model that has not converged"
- "tiling amplifies false positives" - tile vs full was noise on ball-free frames
- "the real gate is not runnable"
- "what it would take: more epochs"

What survives is the instrument, not the diagnosis: reporting best-of-K beside top-1 is
still correct practice. It simply was not this bug, and it could not have been.

## The real gate, run at last

The gate needs at least 8 ball points per clip and was getting about 7 at 29% found, so
it was declared not runnable. At 97.7% over 27-31 tracked frames it gets about 27, and
`scripts/gate_our_detections.py` now puts our detections through the same joint camera
solve. Same clips, same candidate runs, same frames, same trajectory, same solver, same
evaluation - **only the pixel coordinates differ**, and every function is imported from
`ablate_no_box.py` rather than restated.

Detections are taken **top-1 by confidence**, never best-of-K: at solve time there is no
truth to choose with, and picking by distance would use the answer to select the input.

Run on **59 clips** - the same 59 the Sep 12 blind-zone result used, so the two are
directly comparable. Box-assisted game camera Cx -14.040 Cz 32.552.

| arm | joint camera | off box-assisted | box corners \|x\|,\|z\| in | glove-plane gap in |
|---|---|---|---|---|
| OC detections | -13.697 / 33.138 | 0.34 / 0.59 ft | 0.665 / 0.600 | 1.075 |
| **OUR detections** | -13.984 / 32.697 | **0.06 / 0.15 ft** | **0.982 / 1.005** | **1.056** |

**The OC arm reproduces the Sep 12 blind-zone run exactly** - camera -13.697 / 33.138,
off 0.34 / 0.59 ft, against a box-assisted -14.040 / 32.552. Those are the published
figures to the digit. That is the harness validating itself against a known result before
its other arm is read, and it is the reason the comparison can be trusted at all.

Our arm supplied 18-30 points per clip, within 0-3 of OC's on all 59.

**Box corners is the evaluation that cannot be gamed** - neither no-box arm ever sees the
box, so those 4 corners are a held-out object at a known world position. Ours lands at
about 1.0 in against OC's 0.6-0.7 in. The glove-plane gap, the plane the miss is actually
read at, is 1.056 in against OC's 1.075 in.

An earlier 12-clip smoke run gave the same picture (ours 0.28 / 0.02 ft off, corners
1.34 / 1.47 in, gap 1.156 in), so the result is not an artefact of the clip count.

**This is the measurement A+ was blocked on.** A camera solved from our own detections,
with no drawn box anywhere in the fit, lands where the box-assisted one does.

## What it does NOT claim

- **Our camera landing closer to the box-assisted solve than OC's own arm (0.06 / 0.15
  vs 0.34 / 0.59 ft) is NOT a claim of being better than OpenCommand.** The evaluation
  that cannot be gamed - held-out box corners - is *worse* for us, 0.98 / 1.01 in against
  their 0.67 / 0.60 in. The fair reading is that our detections support a camera about as
  good as theirs, not a better one. Camera position on one game is also a single draw.
- **One game, one park, one broadcast camera.** 824821 only.
- **0.79-0.83 in is not comparable to OpenCommand's 0.4 in figure.** Ours is a screen
  distance between two detections; theirs is a ball-at-plate error after a camera solve.
  Treating them as one quantity would repeat in miniature the exact mistake this document
  records. What can be said is that our detections sit about 2.7 px from OC's and our
  labels 2.15 px from OC's, so the detector is near the noise floor of its own target.
- **`ball@plate` is still circular** and is reported only so the trap stays visible.
  Scoring a pose on the ball run it was fitted to proves nothing.
- **None of this is measured at A+.** This is MLB video, where a Statcast trajectory
  exists to generate labels from. The val split held out a whole unseen park and camera,
  which is the honest test available here, but transfer to the A+ feed is unproven and is
  the point of the exercise.

## Blast radius

`frame_idx` was grepped across every `.py` in the project. Exactly two files read our
video at an OpenCommand index, and both are detector-eval only:

| file | status |
|---|---|
| `scripts/detect_ball_eval.py` | **was broken**, fixed |
| `scripts/detect_diagnose.py` | **was broken**, fixed |
| `scripts/ablate_no_box.py` | unaffected - `frame_idx` only ever enters as `t0 + frame_idx/fps`, and a constant offset is absorbed by the fitted `t0` |
| `scripts/parity_opencommand_pose.py` | unaffected, same reason |
| `scripts/build_ball_dataset.py` | unaffected - reads our own labels, already absolute |
| `zone_blind.py`, `zone_zoom.py`, `box_presence_sweep.py`, `render_zone_demo.py` | unaffected, same reason |

**So nothing shipped is retracted.** The blind zone result (0.84 in median, camera off
0.34/0.59 ft), the five-park zone-graphic sweep and the pinhole parity solve all stand
untouched. That they were never affected is itself consistent with the diagnosis: they
are the paths where a constant frame offset cancels.

## The fix

`scripts/oc_frame_map.py` holds the mapping as **one** definition, imported by both
consumers, because the defect was two scripts independently doing the same wrong thing.

- `build_shifts(game, clips)` returns the per-play shift. A play with no label gets **no
  entry** and the caller skips it. There is deliberately no fallback to 0, since a silent
  default is the failure being fixed.
- `verify_shifts(...)` re-derives the shift empirically and **raises** if the two
  derivations disagree by more than 3 frames, or if our labels sit more than 12 px from
  OC's ball. Both consumers call it before reading a single frame.

### The guard was proven able to fail, and via the intended assert

A guard that has only ever passed is a claim, and one that goes red for the wrong reason
has not been proven either. Both traps were hit here before the check was trusted:

| injection | expected | result |
|---|---|---|
| real shifts | PASS | PASS - 1.00 fr, 2.15 px |
| `+5` frames | disagreement | RED, disagreement (4.00 fr > 3.0) |
| labels jittered +-40 px | residual | RED, residual (28.46 px > 12.0) |
| `-183` frames (the original bug) | vacuity | RED, vacuity |

The first attempt injected `+40` and `-183` frames, and both went red **via the
non-vacuity assert** - `verify_shifts` only searches +-8 frames, so a larger injection
can never join and the numeric checks were never exercised. The second attempt moved the
labels by a constant +40 px to trip the residual assert, and tripped the **disagreement**
assert instead (8 frames): a constant offset along a moving path is partly absorbable by
sliding the frame, so that perturbation does not separate the two modes. Only per-row
random jitter, which no single shift can absorb, isolates the residual assert. Both
misfires are recorded rather than relabelled as passes.

## Prevention

Added to `.claude/rules/metric-must-separate-failure-modes.md`: before trusting any
number computed across two sources, **join them and print how many rows matched**. Zero,
or far below both inputs, is the finding. A shared column *name* is not a shared
*definition*.

## Files

- `scripts/oc_frame_map.py` (new) - the mapping and its verification
- `scripts/detect_ball_eval.py`, `scripts/detect_diagnose.py` - map before reading
- `docs/2026-09-12-detector-gate-undertrained.md` - superseded banner added
- `output/frame_shift_824821.csv` - per-play shifts, both derivations, residuals
