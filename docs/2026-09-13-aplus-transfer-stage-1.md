# A+ transfer, stage 1: the detector sees baseballs, but we cannot yet say which one is the pitch

Sep 13 2026, immediately after the frame-mapping fix
(`2026-09-13-the-gate-was-reading-the-wrong-frames.md`) unblocked the detector. A+ is the
whole point of building it, and this is the first honest look.

**Verdict: inconclusive, and the reason is a SEARCH problem, not established detector
failure.** Three ways of isolating the pitched ball were tried and all three were invalid.
Writing them down so the next attempt does not repeat them.

## What does transfer

The detector fires on **real baseballs** in the Asheville A+ feed, which it never trained
on (it saw Camden, Detroit, Colorado). On `ogando_cu_2` it tracked an unmistakable
baseball for 18 consecutive frames at confidence 0.25-0.60, correctly boxed and
correctly sized. Rendered and looked at: `output/aplus_probe.png`.

Encouraging, and not nothing: the learned features survive a different broadcast operation.
A+ clips are also **1280x720 at ~59.6 fps, identical to the MLB clips**, so the ball's
pixel scale carries over and no re-scaling is needed.

## What does not

**In 3 Ogando clips, the pitched ball was never tracked.** Two of the three produced no
sustained moving run at all, while still firing on 203-315 frames each - i.e. a large
number of stationary false positives.

## The three criteria that failed, and why

### 1. "A detection that MOVES is the ball" - defeated by the camera

The 18-frame run above is a **stationary white object near the left wall**, carried across
the frame by a camera pan. Its trace gives it away: x runs 316 -> 7 (right to left),
step ~15 px/frame **decelerating to 1-5 px/frame**, coming to rest at the frame edge as
the pan stops. A pitch does not decelerate to a halt at the edge of frame.

Worse, the direction is wrong. This is a side view, so the pitch travels **left to right**;
the artefact ran right to left.

> Apparent motion at A+ is dominated by the CAMERA. "Does it move?" cannot separate a
> ball from a wall light - the same two-causes-one-number trap as the frames bug, in a
> new costume.

### 2. "Find the window where the camera is quiet" - defeated by the players

Step 1 measured the A+ camera as static *during flight* (max 1.62 px over 55 clips), so
the flight should sit inside a quiet window. A downsampled whole-frame mean-absdiff found
exactly one quiet window on `ogando_cu_2`: **frames 444-464, after the play is over**.

The reason is already in this project's own record: step 1 rejected phase correlation as a
camera-motion instrument because **players dominate**. A whole-frame difference has the
identical flaw - the pitcher's delivery and the batter's swing swamp it, so the pitch
frames never read as quiet. I rebuilt a known-bad instrument in a new form.

(That run also exposed a plain instrument bug first: the *median* of a uint8 frame
difference is integer-valued and collapses to 0, making a 35th-percentile threshold
exactly `0.00` and matching nothing. Mean, not median. Fixed before the result above.)

### 3. "High confidence means the ball" - defeated by sheer false-positive volume

203-464 frames per clip carry a detection. Confidence alone ranks a wall light above a
ball as happily as the reverse.

## The likely reason, stated as a hypothesis and NOT as a finding

**The A+ BROADCAST is a side view.** Observed directly on `ogando_cu_2`
(`output/aplus_fullframe.png`): pitcher left, catcher centre, umpire and batter right,
crowd behind. This contradicts the note on record describing A+ BROADCAST as
"behind-pitcher"; it needs confirming across more clips before the note is rewritten.

If that holds, it matters a great deal:

| | MLB training data | A+ |
|---|---|---|
| ball approaches | nearly head-on | transversely across frame |
| appearance | compact blob, ~11-20 px | elongated, motion-blurred streak |
| pixel speed | small early, accelerating in | large and roughly constant |

A detector trained only on compact head-on blobs has good reason to miss a fast transverse
streak. **This is a hypothesis.** It predicts that the pitch is genuinely not being
detected at A+, rather than detected-but-unidentifiable - and those call for opposite next
steps, so it must be tested before either is chosen.

## The chicken-and-egg that actually blocks this

At MLB, labels are free because the camera pose is known (from the drawn zone box), so the
trajectory can be projected and the clock slid until it lands on something bright and
moving. At A+:

- the manifests **do** carry a full 9-param trajectory (`traj_x0..z2`, `zone_time`,
  `rel_x/y/z`), so the earlier claim that A+ has "no Statcast trajectory to lean on" is
  wrong for token-pulled clips
- but there is **no drawn zone box**, so there is no pose
- fitting a pose needs ball pixels; knowing which pixels are the ball needs a pose

## Routes forward, none attempted

1. **Global fit over many clips.** Search for the (camera pose, per-clip `t0`) that makes
   projected trajectories land on *some* detection in each frame, RANSAC-style. Uses the
   detections without having to identify them first, and the trajectories are already in
   the manifests. Most principled, most work.
2. **Register out the camera.** Align consecutive frames on background features, then look
   for the small object moving against a now-static background. Directly attacks failure
   #1. Note step 1 already rejected phase correlation for this; feature tracking on
   background corners was the replacement it recommended.
3. **Hand-label a handful of A+ flights**, fit a pose from those, then bootstrap exactly as
   at MLB. Least clever, most certain, and the labels would also test the streak
   hypothesis directly.
4. **Test the streak hypothesis first**, since it is cheap and decides between the others:
   hand-mark the ball on ~10 A+ flight frames and simply ask whether the detector fires
   there at all.

Route 4 before 1-3.

## What NOT to do

- **Don't** retry "detections that move" - the camera moves more than the ball.
- **Don't** retry a whole-frame difference as a camera-motion instrument - players
  dominate it, and this project rejected that once already for phase correlation.
- **Don't** read the 18-frame 0.60-confidence run as success. It is a wall light.
- **Don't** conclude the detector fails at A+ from this page. Three search procedures
  failed; that is a different claim, and the streak hypothesis has not been tested.
- **Don't** rewrite the "behind-pitcher" note in `command-cv-status.md` on one clip.

## Files

- `output/aplus_probe.png` - the 18-frame run, crops (the wall light)
- `output/aplus_fullframe.png` - full frames showing the side view and the pan
- scratch probes were not promoted to `scripts/`, since all three criteria were invalid
