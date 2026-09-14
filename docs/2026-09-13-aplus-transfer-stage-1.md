# A+ transfer: the detector DOES track the pitch; a stationary false positive outranks it

Sep 13 2026, immediately after the frame-mapping fix
(`2026-09-13-the-gate-was-reading-the-wrong-frames.md`) unblocked the detector. A+ is the
whole point of building it, and this is the first honest look.

> **RESOLVED, later the same session - read this first.** The verdict below was
> "inconclusive", and it was wrong in the good direction. Once the *actual flight window*
> was located (by eye, via a contact strip) the detector was found tracking the pitched
> ball cleanly: **`ogando_cu_2` frames 188-198, confidence rising to 0.564, an
> unmistakable baseball with visible seams**, descending into the catcher's mitt.
> See "The answer" below. The framing that survived is the important half - this was a
> SEARCH problem, never established detector failure. The transverse-streak hypothesis is
> **falsified**: the A+ ball is a compact round blob, exactly what the model trained on.

**Original verdict (superseded): inconclusive, and the reason is a SEARCH problem, not
established detector failure.** Three ways of isolating the pitched ball were tried and
all three were invalid. Writing them down so the next attempt does not repeat them.

## What does transfer

The detector fires on **real baseballs** in the Asheville A+ feed, which it never trained
on (it saw Camden, Detroit, Colorado). On `ogando_cu_2` it tracked an unmistakable
baseball for 18 consecutive frames at confidence 0.25-0.60, correctly boxed and
correctly sized. Rendered and looked at: `output/aplus_probe.png`.

Encouraging, and not nothing: the learned features survive a different broadcast operation.
A+ clips are also **1280x720 at ~59.6 fps, identical to the MLB clips**, so the ball's
pixel scale carries over and no re-scaling is needed.

## What does not

**Top-1 does not hold the ball.** Each clip carries a stationary high-confidence
attractor that claims top-1 on most frames - on `ogando_cu_2` a crowd object at
(826.5, 236.5), conf 0.25-0.35, holding top-1 on ~35 of 56 flight-window frames with a
per-frame step of 0.1-1.9 px; on `ogando_ch_1` something on the catcher's gear at
(800-804, 462-464), flat to 0.0-1.0 px across 55 consecutive frames.

That is what fragments the real track, and it is why the first three searches failed.

~~In 3 Ogando clips, the pitched ball was never tracked.~~ **WRONG - superseded by "The
answer" below.** It was never tracked *by the searches I ran*, because none of them
looked in the flight window.

## The three criteria that failed, and why

### 1. "A detection that MOVES is the ball" - defeated by a SECOND BALL

> **CORRECTED Sep 14 2026.** This section originally said the 18-frame run was a
> stationary wall light carried by a camera pan. **That was wrong**, and it was asserted
> in a commit message too. Sparse-feature flow measured over the whole clip puts camera
> motion at a median of **0.15 px/frame** (p90 1.58, max 6.72) - the camera is essentially
> static. Subtracting it changes the track's step from 15.1 to 15.1 px/frame, i.e. not at
> all. The object is genuinely moving.

The 18-frame run is a **real baseball**, just not the pitched one: x runs 316 -> 7 (right
to left), decelerating, ~300 px net, ending near the mound. That is the catcher's
**return throw**. The first probe's crops showed an unmistakable baseball, which should
have been the clue - I attributed real motion to the camera without measuring the camera.

So the actual difficulty is that **a clip contains more than one moving baseball**, and
the return throw is longer and straighter than the pitch, so any length-and-smoothness
ranking prefers it.

> "Does it move?" cannot separate the pitch from the return throw. What does is
> DIRECTION: the pitch runs pitcher -> catcher (left to right here, x 600 -> 641), the
> return throw runs catcher -> pitcher (right to left, x 316 -> 7). Timing and the
> accelerate-vs-decelerate signature agree with it.

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

> **FALSIFIED the same session.** The A+ ball renders as a compact, sharply-defined round
> blob with visible seams (`output/aplus_zoom_cu2.png`), not a streak, and the detector
> reaches 0.564 on it. The side-view observation stands; the appearance consequence drawn
> from it does not. Stated as a hypothesis, tested, discarded - which cost one render
> instead of a dataset rebuild.

## The answer

Locating the flight window **by eye** - a contact strip at stride 10
(`output/aplus_strip_cu2.png`) - showed the delivery at roughly f170-205 on
`ogando_cu_2`. Every earlier search had been looking elsewhere: the "moving run" at
f260-277 (the wall light) and the "quiet window" at f444-464 (after the play).

Tracking by **nearest neighbour from a seed rather than by confidence** - so the crowd
attractor cannot hijack it - over the detector's top-10:

| frame | x | y | step px | conf |
|---|---|---|---|---|
| 188 | 599.9 | 348.3 | - | 0.132 |
| 193 | 615.4 | 367.4 | 7.2 | 0.243 |
| 195 | 622.3 | 377.9 | 6.4 | 0.543 |
| 196 | 622.9 | 383.7 | 5.9 | **0.564** |
| 198 | 628.1 | 397.1 | 9.0 | 0.420 |
| 204 | 641.2 | 446.7 | 18.2 | 0.102 |

Right and down, accelerating 6 -> 18 px/frame, arriving at the mitt. `output/aplus_zoom_cu2.png`
shows a baseball inside the circle, seams visible at f191. **That is the pitch.**

Confidence peaks mid-flight and collapses at both ends (0.07-0.13), which is exactly when
the crowd attractor takes top-1 back.

`ogando_ch_1` remains unresolved: top-1 is attractor-locked across f140-240, which spans
its delivery. But **that trace records top-1 only**, so it cannot separate "not detected"
from "detected at rank 3" - the precise trap that
`.claude/rules/metric-must-separate-failure-modes.md` exists for. It has not been
re-tested with top-K, and no claim is made about it.

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

**Route 4 was run, and it answered the question: the ball is visible AND the detector fires
on it.** So this is an identification problem, and the shape of the fix is now known:

> **Build tracks from the top-K detections across frames, not from top-1.** Score
> candidate tracks by length, smoothness and consistent acceleration, and take the best
> one. The ball is already in the top-10 on the flight frames; it simply is not rank 1,
> because a stationary crowd object outranks it. No retraining is implied.

Route 1 (global fit over many clips) stays the principled endpoint and gets easier once
tracks are reliable. Route 3 (hand-label a few A+ flights) is now only needed if top-K
tracking proves insufficient.

### The top-K tracker was attempted and does NOT work yet - validation red twice

Built with `ogando_cu_2` as a validation case, because its answer is known (f188-204). It
never recovered it, so nothing it says about any other clip has been read.

**v1** seeded chains from every detection, shared a global `used` set, and ranked by step
slope. It put chains at f168-181 on top - straightness 0.28-0.57, confidence 0.11-0.14,
i.e. jitter. Two defects: ranking by slope *rewards* a chain that hops between unrelated
detections, since hopping makes steps grow fastest; and the shared `used` set let an
earlier junk chain **consume the ball's detections** so the real track could never form.

**v2** dropped the shared set, added a straightness filter (>= 0.85, justified by the
physics of a ~0.25 s flight rather than by the known answer), and ranked by
length x straightness with slope demoted to a filter. On `cu_2`: 221 tracks -> 25 after
the step floor -> **1** after straightness. That survivor is **f209-215**, straightness
0.971 - the **return throw**, not the pitch.

**Diagnosis, not yet fixed.** The ball's own path is straight enough to pass: net
displacement ~107 px over a path length of ~114 px, straightness ~0.94. So the chain is
almost certainly **over-extending past the catch** into the return throw that begins
around f210, and the merged path bends below the threshold. The fix is to terminate a
chain when direction reverses or the object stops, and to score **sub-chains** rather than
only maximal ones.

**Deliberately not tuned further.** Adjusting a scorer until it selects the answer already
known for one clip is fitting to that clip, which is the failure
`.claude/rules/grep-guard-criteria.md` exists to prevent. `ogando_ch_1` reported 0
surviving tracks in both versions; with validation red that number means nothing and no
claim is made from it.

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
