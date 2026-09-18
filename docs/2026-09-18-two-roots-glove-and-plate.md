# Two Roots Off One Tree: Glove-and-Catch vs plate_x/z (Sep 18 2026)

Zac, this session: *"we need to know where the glove started and where the ball
was caught, which fits our original idea, with the z and x plate being the
fallback or the other model here. Make it clear these relate but are parallel
roots off the same idea tree."*

That is right, and it is already the locked framework in
`2026-07-09-two-models-and-planes.md`. This note does two things that doc could
not: it says what the **ball detector changed** about each root, and it names the
**first runnable experiment** on the glove root, because the data for it is
already on disk.

Nothing here supersedes the Jul 9 doc. Read that one first for the plane
distinction; this is the Sep 18 amendment.

---

## The tree

Both roots answer the same question -- *how far was the pitch from where the
catcher asked for it* -- and they differ only in what supplies the **actual**.

```
                        MISS DISTANCE = target - actual
                                 |
         target = the catcher's SET-UP GLOVE, from video (shared by both roots)
                                 |
        +------------------------+-------------------------+
        |                                                  |
  ROOT A  actual = plate_x / plate_z                 ROOT B  actual = where the
          plane: FRONT OF THE PLATE                          ball was CAUGHT
          source: DATA (analytics payload, or GC2)           plane: RECEIVE, ~1.5-3 ft
          status: built + validated, 3.9 in LOO              behind the plate
          needs: the data to exist for that pitch            source: VIDEO only
                                                             status: the ball half
                                                             is now measurable
                                                             needs: nothing gated
```

They are **different numbers by design**, not two attempts at one number. The
ball is still travelling and dropping between the plate front and the mitt, so
Root B's actual sits behind and below Root A's. Neither is wrong. Label which
plane a value lives in, every time.

**The bridge is the real prize, and it is unchanged from Jul 9:** once both roots
run on the same pitches, train on the *discrepancy* -- learn the plane offset per
pitch from shape and speed, so Root B reproduces the plate-front number from
video alone. That is what turns "video command" into "Trackman-grade video
command", and it is why the roots must be kept separate rather than averaged.

---

## What the ball detector changed (Sep 18)

The detector trained today reads **2.15 px median on a held-out park** and finds
the ball in 99% of frames on full 1280x720 video, with no pose, no cadence
assumption and no drawn zone graphic. Three consequences, one per root.

**Root B's actual is no longer blocked on the glove detector.** The Jul 9 doc
said Root B "needs the glove detector to be rock-solid". That was true when the
only way to say where the ball ended up was to find the mitt. It is not true now:
the ball's own pixel path is measured directly, and its last frame before the
mitt closes is a **ball** position, not a mitt position. The glove detector is
still needed for the **target** half. That is a much smaller ask -- the set-up
glove is stationary for many frames before release, which is the easy case.

**Root A is no longer gated on the analytics endpoint.** `plate_x` / `plate_z`
are in GC2 for every tracked pitch we have, at a scale the per-pitch Okta call
will never reach. The authed payload remains the right source when we are pulling
clips anyway (it comes back in the same request), but the DB is the source for
anything at scale. Root A's real constraint was never the endpoint; it is that
below AAA the pitch may not be tracked at all, which is exactly the hole Root B
exists to fill.

**Root B is the one that works where we most need it.** Measured this week: the
public feed carries tracked `pitchData` on 327/327 AAA pitches and **0 of 276**
at A+, **0 of 298** at AA. Root A simply cannot run at those levels from the
public feed. Root B needs pixels, which we have at every level that serves video.

---

## The annotations already answer the first question

`data/annotations/*.json` holds **55 hand annotations, every one carrying BOTH
endpoints**:

```json
{"setup": {"frame": 151, "cx": 770, "cy": 446, "w": 48, "h": 40},
 "catch": {"frame": 228, "cx": 763, "cy": 452, "w": 46, "h": 27},
 "endpoint_kind": "catch"}
```

41 are `catch`, 14 are `contact` (the ball was hit, so there is no receive point
-- those are not Root B rows and must not be pooled with the 41).

**Both boxes are the GLOVE, not the ball.** That is the first thing to measure,
and it is a defect waiting to happen if it is assumed either way: a catch box of
46 x 27 px on footage where the ball is ~8-11 px means the mitt centroid and the
ball centre can differ by a real number of inches, systematically, in the
direction the mitt travels to receive. A systematic offset would ride straight
into every Root B number, would look like a plane offset, and would be absorbed
by any affine fit -- which is to say it would be invisible exactly where it
matters (`metric-must-separate-failure-modes.md`).

### Experiment 1, runnable today, no DB and no token

For each of the 41 `catch` clips: run the detector, take the ball at the
annotated catch frame, and measure `ball - glove_centroid` in pixels,
**decomposed along and across the ball's direction of travel** (the decomposition
that caught the half-frame bias on 09-17; ALONG isolates timing, ACROSS isolates
geometry, and both separate a systematic BIAS from random scatter, which a
contact sheet cannot).

### RESULT, same day: the experiment as designed cannot run, and why is the finding

**0 of 41 measured.** 15 clips gave no chain; 26 gave a chain ending 1-20 frames
BEFORE the annotated catch frame. At ~60 fps a 10-frame gap is ~0.17 s, in which
a 90 mph pitch covers ~22 ft, so this is not the detector fumbling a last frame.

A 10-clip probe settled it. At the annotated catch frame the detector fires **not
at all** in 5 of 10, and in the other 5 its nearest candidate sits **65 to 345 px
from the mitt**. The ball is inside the glove by then. Nothing to detect.

**Consequence for Root B, and it is a design change, not a setback.** The
receive-plane "actual" cannot be *the detected ball at the catch frame*. It has
to be the ball's pixel path, fitted over the frames where the ball is clean and
**evaluated at the receive time**. That needs no new machinery:
`plate_scale_check.py` already interpolates an observed pixel path to a named
time, and this is the same operation one plane further downstream. It also makes
the original mitt-vs-ball question answerable -- compare the *extrapolated* ball
position against the mitt box.

The guard is why this is a finding rather than a wrong number: with nothing
measured the script refuses to print an offset and says the failure is about the
window or the weights. A version that fell back to the nearest available
detection would have reported a confident 65-345 px "mitt offset" assembled
entirely from things that are not the ball.

The three outcomes below remain the right reading *once the extrapolation
version runs*:

| result | reading |
|---|---|
| offset near zero, low scatter | the mitt centroid is a fine proxy; the 55 annotations are usable as Root B truth as they stand |
| systematic ALONG offset | the annotated frame is not the receive instant, or the mitt is already closing -- fix the frame, not the model |
| systematic ACROSS offset | the mitt sits off the ball's line at receive; Root B must use the BALL, and the annotations become target-only |

All three are useful. None of them can be guessed from the box sizes.

**Caveat that has to travel with the result, CORRECTED 2026-09-18.** An earlier
version of this file said all 55 were "one pitcher at one park (Ogando,
Asheville)". That was wrong -- asserted from the `ogando_` filename prefix rather
than looked up. Joining the stems to `ogando_2026_manifest.csv` gives **seven
ballparks**:

| park | annotations | of which `catch` |
|---|---:|---:|
| HomeTrust (Asheville) | 27 | 18 |
| First National Bank Field (Greensboro) | 7 | 5 |
| Truist (Winston-Salem) | 7 | 7 |
| Frawley (Wilmington) | 6 | 5 |
| Fluor Field (Greenville) | 4 | 4 |
| Bowling Green | 2 | 1 |
| AdventHealth (Rome) | 2 | 1 |

This makes the experiment **stronger**, not weaker: a mitt-vs-ball offset that is
real geometry should replicate across seven cameras, and one that is an artifact
of a single broadcast should not. Report the offset per park as well as pooled --
if it only appears at Asheville it is a camera fact, not a receiving fact.

The caveat that genuinely remains is **one pitcher**. All 55 are Ogando, so
anything specific to how one catcher receives him is not separable here.

---

## What each root needs next

**Root B (glove + catch).**
1. Experiment 1 above -- is the mitt centroid the ball.
2. A set-up-glove detector. Different problem from the ball: large, stationary,
   many frames of it, but low contrast against a crouching catcher.
3. The receive point in FEET, not pixels. `plate_scale_check.py` already shows
   our poses put the plate within +0.21 in at Isotopes and +0.22 in at Frawley,
   and OpenCommand pins Cy at a constant across 30 parks and still lands 0.04 in
   -- so metric recovery does NOT need a surveyed landmark, which was retracted
   as a blocker today. A landmark stays worth marking as an independent check.

**Root A (plate_x/z).**
1. Pull `plate_x` / `plate_z` from GC2 rather than per-pitch from the endpoint,
   for anything above a single game.
2. Keep the per-game affine that currently absorbs the plane offset, and stop
   calling that offset an error. It is physics.

**The bridge.** Blocked on nothing except both roots running on the same pitches
at one park. Isotopes is the obvious candidate: AAA, tracked 327/327, 327 clips
on disk, the detector's best park.

---

## What NOT to do

- **Don't** average, blend, or reconcile the two actuals. They are different
  planes; a disagreement between them is the SIGNAL the bridge is trained on.
- **Don't** call Root A a fallback in the code or in a column name. It is the
  data-backed root and it is more accurate where it exists. "Fallback" is only
  true from the point of view of the standalone goal.
- **Don't** treat the 14 `contact` annotations as Root B rows. No receive point
  exists on a batted ball.
- **Don't** quote any number from the 55 annotations without saying "one pitcher,
  one park".
- **Don't** let a mitt-vs-ball offset get absorbed by a fit before it has been
  measured. Measure it first, then decide what to fit.
