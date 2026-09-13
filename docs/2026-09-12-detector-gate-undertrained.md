# The detector gate: failed, and the reason is undertraining, not a broken dataset

> **SUPERSEDED Sep 13 2026 - THE MEASUREMENTS BELOW ARE VOID.** Every number in this
> document was collected on the WRONG FRAMES. OpenCommand's `frame_idx` is relative to
> RELEASE; `detect_ball_eval.py` and `detect_diagnose.py` used it as an index into our
> own clip, so they read frames 0-30 - the pitcher still standing on the mound, three
> seconds before the pitch - and graded those detections against a mid-flight ball
> position. There was no ball in any frame either script looked at.
>
> Re-run on the corrected frames, the same weights give **97.7% found and 3.8-4.5 px
> top-1** where this page records 29-48% and 167-308 px. The conclusions drawn here -
> "undertrained", "the real gate is not runnable", "tiling amplifies false positives",
> "confidence 0.02-0.20 means not converged" - do not survive, and the epoch-7
> rank-#3 evidence was read off the wrong picture too.
>
> What DOES survive is the metric lesson: top-1 alone could not separate a ranking
> failure from a blindness failure, and that is why `.claude/rules/metric-must-separate-failure-modes.md`
> exists. It simply was not the defect here.
>
> See **`docs/2026-09-13-the-gate-was-reading-the-wrong-frames.md`**.

Sep 12 2026. Training reached **epoch 8 of 30** across five watchdog kills on this
box, best `mAP50 0.101` at epoch 7. The cheap pixel gate
(`scripts/detect_ball_eval.py`) was then run on those weights, before spending a
camera solve on them.

## The gate failed, clearly

| mode | frames | found% | err px med | err px p90 | err in med | conf med |
|---|---:|---:|---:|---:|---:|---:|
| tile (320 px windows) | 155 | 29.0% | 308.69 | 661.58 | 92.6 | 0.132 |
| full (imgsz 1280) | 155 | 48.4% | 167.34 | 660.38 | 50.2 | 0.158 |

Against a ball ~16 px wide in a 1280 px frame, and against OpenCommand's ~0.4 in
ball-at-plate error. **The real gate - our detections into `ablate_no_box.py` - is
therefore not runnable**, and that is the correct outcome rather than a disappointing
one: at 29% found over ~25 frames that is ~7 detections per clip, below the >=8 the
joint solve needs, and points 167 px wrong would produce a confidently wrong camera.
A wrong number here is worse than no number.

## Tiling did WORSE than one full-frame pass, and that is a method lesson

29.0% vs 48.4%, despite tiling matching training conditions exactly (320 px windows
at native scale, which is how the crops were built). The cause is the merge rule:
the best detection is taken across ~15-20 windows, so every extra window is another
chance for a false positive to outscore the true ball. **With a weak detector,
tiling amplifies false positives.** Revisit tiling only once confidence is
well-separated.

## The diagnosis: undertrained. Evidence, not vibes

`scripts/detect_diagnose.py` draws truth against our top-5 detections on the full
frame and prints each detection's distance to truth.

    00f695d8 f11  truth (631,299)   #1 conf .202 108 px off ... #3 conf .106  40.0 px off
    018f0e61 f10  truth (584,281)   #1 conf .099 658 px off (in the CROWD)
    01dde385 f13  truth (649,301)   #1 conf .026  41.9 px off ... #3 conf .021  15.5 px off

Three things follow:

1. **Every box is 20-24 px** - ball-sized, matching `BOX_SCALE 1.8` on a ~11 px ball.
   The model learned the right SCALE. A structurally broken run would fire on
   arbitrary large texture.
2. **The ball IS detected in 2 of 3 frames, ranked #3** (40.0 px and 15.5 px). It can
   see the ball; it scores it below a distractor. That is a ranking problem, which
   more epochs fix, not a perception failure, which they would not.
3. **Confidence is 0.02-0.20 everywhere** - uniformly uncertain, the signature of a
   model that has not converged. 8 of 30 epochs, LR still high, mAP oscillating
   0.039-0.101 across those 8.

The false positives are bright fragmented texture **in the crowd**, i.e. ball-like,
not random grass. The model has learned "small bright round thing" and has not yet
learned to reject other small bright things.

**THE AGGREGATE METRIC HID THIS.** `detect_ball_eval.py` scores only the top-1
detection, so a correctly-found ball sitting at rank #3 counts as a 40 px miss - or,
worse, as "found" at the wrong place. A median error of 167 px and "the ball is
found within 16 px but ranked third" are the same row in that table. Any future
detector metric here should report best-of-top-K distance alongside top-1, or it will
keep confusing a ranking problem with a blindness problem.

## UPDATE at epoch 13 (same day): epochs worked, and the ranking hypothesis is now obsolete

Re-ran the gate on epoch-13 weights, with the fixed best-of-top-K metric:

| mode | found% | top1 px | bestK px | bestK in | med rank | conf med |
|---|---:|---:|---:|---:|---:|---:|
| tile | **61.9%** | 54.64 | 45.71 | 13.7 | **1.0** | 0.198 |
| full | 45.8% | 43.98 | 37.41 | 11.2 | 2.0 | 0.195 |

Against the epoch-7 numbers above: **top-1 error 167 px -> 44 px, and tile's found%
29.0% -> 61.9%**. Nine epochs did a great deal. The reason that improvement was
invisible in training is that `mAP50` sat flat in a 0.08-0.12 band the whole time -
mAP at fixed IoU thresholds is a poor proxy for the thing we actually care about,
which is pixel distance to a ~16 px ball.

**The ranking diagnosis is now OBSOLETE, not confirmed.** `bestK` (37-46 px) is only
marginally below `top1` (44-55 px), and tile's `med rank` is **1.0** - the closest
detection is already ranked first. The ball is no longer being outranked; what remains
is genuine localisation imprecision. The diagnosis was right at epoch 7 and has been
overtaken, which is a better outcome than being wrong but is not the same thing.

**Tiling now BEATS full-frame** (61.9% vs 45.8%), reversing epoch 7 exactly as the
stated mechanism predicted: tiling only lost because weak confidence let a false
positive from some other window outrank the true ball, and once confidence separated,
the extra windows started helping. That is the hypothesis surviving a test it could
have failed.

**Still not good enough for the real gate.** 44 px is ~13 inches against OpenCommand's
0.4 in. Feeding those points into the joint camera solve would still produce a
confidently wrong camera. Closer, not there.

## What this does and does not block

**Does not block anything shipped.** The blind camera result (`08c2b3a9`,
`2026-09-12-zone-graphic-is-in-our-clips.md`) used OpenCommand's detections and
stands untouched: 0.84 in median, camera off 0.34/0.59 ft.

**Does not block MLB.** `ball_clock_align.py` already finds the ball with no detector
at all (17x over baseline) by sliding the known trajectory's clock. The detector was
never needed at MLB.

**Blocks A+ only**, which is the whole point of building it: there is no Statcast
trajectory to lean on and no drawn box to calibrate against, so a real detector is
the only route.

## What it would take

More epochs - the diagnosis says they would help. What stopped us is not the method:
**five watchdog kills**, 8 of 30 epochs completed, ~340 s/epoch, on a 16 GB machine
where other applications hold ~13 GB. That is arithmetic, not tuning. Options are a
machine that can hold the job, or a much smaller run
(`--fraction`, fewer epochs) accepting a weaker detector.

## Bug history

- The 4th kill was misdiagnosed as an AMP/startup failure because the log ended at
  the AMP check. `--no-amp` was built for it (`b007c8e1`); the next attempt got past
  startup and died mid-epoch-9 at batch 78/1554 - the ordinary failure. The fix
  targeted a cause that probably did not exist, and on the resume path ultralytics
  overrode the flag anyway (`resume=True` restores checkpoint args over explicit
  ones).
- `detect_ball_eval.py` first died with `RuntimeError: Invalid device string: '0'`:
  ultralytics accepts `device="0"`, torch's `.to()` wants `"cuda:0"` (`51c0fa01`).
- The restart loop (killed with its child) assumed the watchdog takes the python
  process and leaves the PowerShell parent. It takes the whole task.
