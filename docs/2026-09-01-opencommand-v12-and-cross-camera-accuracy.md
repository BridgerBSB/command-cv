# OpenCommand v1.2, and what it proves about inches across cameras

Deep read of `github.com/tomdoyo/open-command` at v1.2.0, plus an analysis of
their published pose data that they do not run themselves.

The driving question: **when we say a pitch missed by 14 inches, how do we know
that number means the same thing from a different camera, in a different park,
at a different zoom?** OpenCommand is currently the only place that question can
be answered at scale, because they ship 1.3 million posed clips across 30 parks
and two seasons.

Supersedes the method sections of
`bsb-brain/projects/command-cv-external-research.md` (captured Aug 17 against
v1.0.0). Their pipeline has changed materially since.

License note: CC BY-NC-SA 4.0, non-commercial. We read their code and
reimplement the geometry. We do not vendor their files or their data.

---

## 1. What changed since our Aug 17 capture

We read v1.0.0. They are now on v1.2.0, and two of the three changes land
directly on things we flagged.

| Version | Date | Change |
|---|---|---|
| 1.0.0 | Aug 11 | Initial release |
| 1.1.0 | Aug 21 | 2024 season added. Target selection becomes highest glove **discounted by lateness** |
| 1.2.0 | Aug 27 | Naive per-pitcher offset replaced by a **two-level hierarchical empirical-Bayes model** |

### 1a. A correction to something I told you on Aug 21

I said their inferred target "forces that pitcher's mean residual to zero by
construction" and "mechanically buys the ~1 inch headline improvement." That was
a fair reading of v1.0.0, where inferred target = naive + `mean(actual - naive)`
per pitcher and pitch type. It is no longer true, and they found it before we did.

v1.2.0 keeps that old method in the table under the name **`fixed offset`**, and
publishes the diagnostics that expose exactly the artifact I described:

- **Flatness.** Fixed offset reads 8.08 in at n=10 against 10.14 in on the full
  season. The metric moves 2 inches purely on sample size. Inferred reads 9.74 at
  n=10 against 9.91 full, which is flat.
- **Bias.** Fixed offset carries a bias of **-2.02 in at n=10**. Inferred carries
  -0.25 in.
- **Held out, 50/50 random split.** Naive train 14.53 / test 14.56. Inferred train
  12.89 / **test 13.20**, a gap of +0.27 in. The improvement survives a holdout.

So the gain is real, and they replaced the estimator that would have faked it.
The remaining honest caveat is theirs, stated plainly in the README: the model
"assumes every pitcher perfectly calibrates his pitches, but most pitchers are
probably an inch or two off."

**The lesson to carry:** the flatness table and the bias-vs-n row are exactly the
tests our own blocking rule 19 asks for. A metric that reads *better* on small
samples is a metric being fit to its own noise. When we ship a per-pitcher
command number, that table goes in the validation before anything else.

### 1b. The v1.2 target model, in plain terms

Two things are now fit per pitcher, shrunk toward a league prior by empirical
Bayes with DerSimonian-Laird variance:

1. **Glove dependence** (`xx`, `xz`, `zx`, `zz`): how much the target actually
   moves per inch the glove moves. Their own docstring names the failure this
   fixes: some pitchers never look at the glove (they name Misiorowski), and some
   move more than an inch per inch of glove movement (they name Skenes). A slope
   of zero says the glove is not the target for that pitcher.
2. **Offset**: whether a pitcher's pitches finish at the glove or start there and
   break off it, shrunk so that n=1 does not produce a 0 inch miss.

The cross terms (`xz`, `zx`) exist because a pitcher's offset can differ by
target cluster. Their example: Hendricks' sinker lands lower than the glove on
the glove side but not the arm side. They note they tested clustering and this
was simpler and as good.

This is a genuinely better idea than anything we have, and it is independent of
the geometry. It is also the piece most likely to be **wrong for us**, because it
is fit on MLB pitchers. An A-ball pitcher who misses by 15 inches is not
"calibrated with an offset," he is missing.

---

## 2. How they get inches, and where the camera enters

World frame: feet, origin at the plate's back tip, x = catcher's right, y =
toward the pitcher, z = up. Everything is a pinhole camera with 7 parameters
`(Cx, Cy, Cz, pan, tilt, roll, f)`.

The camera is handled in **two stages, at two different grains**, and this split
is the whole answer to the cross-camera question.

**Phase 1, per GAME (the mount).** The CF camera is a fixed mount that pans,
tilts and zooms per pitch, but does not move between pitches. So they solve
*where it is* once per game. Every clip with a plausible ball run gets a 7-param
solve against 12+ observations: 8 or more ball pixels, plus the 4 strike-zone box
corners at triple weight. The ball's true 3D position at any instant is free from
the Statcast 9-param trajectory. A clock offset `t0` is fitted and thrown away.
A clip only **votes** if its ball reprojects within **3 px RMSE**, and the game's
camera position is the **median of the votes**, minimum 2.

**Phase 2, per CLIP (the PTZ).** Freeze the position. Refit only pan, tilt, roll
and focal length from the **4 box corners alone**, on that clip's own peak-glove
frame. The ball is deliberately excluded here, and their comment says why:
"camera often moves mid-ball flight."

Two constants are **pinned, not solved**:

- `CY_FT = 400`. Camera depth is degenerate against focal length: move back and
  zoom in and you get the same pixels. They validated the pin on a 140-clip
  harness, where a free `Cy` rails against its bounds while the targets move
  **0.06 in**.
- `SETUP_DEPTH_FT = -1.75`. The plane the catcher sets up on, 1.75 ft behind the
  plate tip. Glove pixels unproject onto that plane.

Two details worth stealing outright:

- **They fit the box against a REFERENCE zone (`sz_top_ref`/`sz_bot_ref`), never
  the pitch's own `sz_top`/`sz_bot`.** Statcast re-measures the zone on takes and
  leaves the batter's default on swings, so fitting the camera against the pitch's
  own `sz` would fit the pose against the pitch's own outcome. That is a
  tautological-display trap sitting inside a geometry solve, and they saw it.
- **The zone plane moves by era.** 17/12 ft through 2025, 8.5/12 ft from 2026
  under ABS, because the broadcast moved the drawn box while `plate_x/z` did not.
  A hardcoded plane would silently bias every 2026 number.

---

## 3. The cross-camera evidence, including the part they never published

### 3a. What they publish

A single pooled median across all 30 parks:

| Season | pitches | end to end | reproj_px median | ball at plate, X | Z |
|---|---:|---:|---:|---:|---:|
| 2024 | 722,012 | 92.01% | 2.29 | 0.34 in | 0.40 in |
| 2025 | 724,005 | 90.01% | 2.28 | 0.35 in | 0.39 in |

**Two independent seasons, 1.27 million pitches, and the numbers do not move.**
Different broadcast crews, different production, camera hardware replaced in some
parks between years. That is meaningful evidence of camera-independence, and it
is not labeled as such anywhere in their repo.

The tail is the caveat. `reproj_px` p90 is 13.4 and 14.0 against a median of 2.3,
roughly a 6x spread, and 23.5% of clips reproject worse than 5 px.

### 3b. What they do not publish, and what it says

`park` is stamped on every row of `camera_poses.csv.gz` and `targets.csv.gz`. It
is **never used in a single validation**. Their accuracy is reported at one grain
only, and a camera problem confined to one stadium cannot be seen at that grain.

So I ran it. 635,925 checked clips, 30 parks, 2,391 games, 2025.
Reproducible via `scripts/analyze_opencommand_poses.py`.

**Per-park spread (inches):**

| | min | median | max | spread |
|---|---:|---:|---:|---:|
| bias X | -0.233 | -0.056 | +0.105 | 0.338 |
| bias Z | -0.158 | +0.117 | +0.324 | 0.482 |
| abs err X | 0.270 | 0.349 | 0.532 | 0.262 |
| abs err Z | 0.316 | 0.390 | 0.534 | 0.218 |
| reproj p90 | 9.79 | 13.69 | 19.99 | 10.20 |

**The headline: the method holds across 30 different cameras to within about half
an inch.** No park is broken. The worst systematic bias in the league is KC at
+0.324 in vertical. The worst precision is ATH at 0.532 in horizontal. For a
metric whose league spread is 8 to 16 inches, a half-inch camera effect is not
going to reorder anybody.

Three findings underneath that:

**1. Only about 30% of the residual is the park.** Decomposing the bias into
between-park and within-park (game to game) variance:

| axis | between-park sd | within-park sd | share that is PARK |
|---|---:|---:|---:|
| X | 0.102 in | 0.164 in | 27.7% |
| Z | 0.136 in | 0.206 in | 30.3% |

The majority of what is left varies **game to game inside the same stadium**, not
stadium to stadium. That is the strongest possible endorsement of solving the
mount **per game** rather than per park, and it means a per-park correction table
would recover less than a third of an already small residual. Not worth building.

**2. The remaining bias is a shared systematic, not noise.**
`corr(park X bias, park Z bias) = -0.606`. Parks that read low in X read high in
Z. Independent per-camera error would show no correlation. Something in the model
is common to all 30 cameras, most likely the fixed aim point at `(0, 0, 3)` or
the interaction between the pinned `Cy` and camera height. It is worth about a
tenth of an inch and nobody has chased it.

**3. Accuracy does not degrade off-axis or at zoom.** This is the one that
matters most for us:

| |pan| quintile | least | q1 | q2 | q3 | most |
|---|---:|---:|---:|---:|---:|
| abs err X (in) | 0.341 | 0.344 | 0.338 | 0.364 | 0.349 |
| abs err Z (in) | 0.387 | 0.390 | 0.393 | 0.411 | 0.394 |

Flat. **A projective model does not care where in the frame you are.** This is
exactly the failure mode our 6-DOF affine has and cannot avoid: an affine is a
planar approximation, so it fits well near frame center and drifts at the edges,
and a catcher setting up on the outside corner is at the edge. The evidence that
the pinhole fixes it is right here, across 636,000 clips.

Zoom is nearly flat too, with only the widest quintile slightly worse
(0.445 in Z against 0.37 to 0.40 elsewhere).

**4. Ball sample size has a floor and a ceiling.** Error by the number of ball
points feeding the position vote:

| n_fit | 1-8 | 9-12 | 13-16 | 17-24 | 25+ |
|---|---:|---:|---:|---:|---:|
| abs err X (in) | 0.420 | 0.374 | 0.324 | 0.317 | 0.381 |
| abs err Z (in) | 0.482 | 0.455 | 0.426 | 0.407 | 0.365 |

It buys real accuracy up to about 16 to 24 points and then stops. Useful for us:
we do not need to detect the ball on every frame, we need about 16 good ones.

---

## 4. What this means for our build

**Our A+ situation is genuinely harder, and now we can say exactly how.**

Their phase 2 refits pan/tilt/roll/zoom **per clip from the drawn strike-zone
box**. Our Asheville feed is behind-pitcher broadcast and almost certainly draws
no box. Their `f0` seed also comes from the box width. So phase 2 as written does
not port, and neither does their focal-length initialization.

What survives the port, and what has to be replaced:

| Their piece | Ports to A+? |
|---|---|
| Pinhole 7-param model, `project` / `unproject` | Yes, unchanged |
| Pinned `Cy = 400`, pinned glove depth -1.75 ft | Yes, though our camera is closer and the constant needs its own degeneracy check |
| Phase 1 mount solve from ball runs | Yes, if we can get the 9-param trajectory |
| Phase 2 per-clip PTZ from box corners | **No.** No drawn box |
| `clean_teleports` | Yes, verbatim in spirit, about 30 lines |
| Reference-zone discipline (`sz_ref` not pitch `sz`) | Yes, and it is a trap we would otherwise hit |
| Hierarchical target inference | Technically yes, but fit on MLB pitchers. Suspect at A+ |

**The per-clip PTZ gap is the real problem, and the per-game/per-park split above
tells us how bad it is.** Their own data says most of the residual varies game to
game, which is what phase 1 already handles. But phase 2 exists because the
camera pans, tilts and zooms *per pitch*, and without a box we have nothing to
re-fit it against on a given clip. Three options, in the order I would try them:

1. **Ball-only per-clip refit.** If our detector sees enough of the flight, refit
   pan/tilt/roll/f from the ball run on that clip alone with C frozen. Their
   reason for excluding the ball in phase 2 is that the camera moves mid-flight,
   which is a broadcast problem. If our A+ camera is more static, the objection
   may not apply to us, and we should measure that rather than assume it.
2. **HawkEye anchors.** We have real 3D positions per frame for the catcher, the
   umpire and every fielder. Those are calibration correspondences on demand, in
   every park, with no drawn box required. OpenCommand can never have this. If
   the ball-only refit is too thin, this is the fallback and it is our genuine
   structural edge.
3. **Fixed home-plate geometry.** The plate itself is a known 17 in object at a
   known location. Their tested `glove_tracking` model detects `homeplate`, and
   our own Jul 5 test measured it at 0.26 to 0.37 confidence on the A+
   behind-pitcher angle, occluded by the catcher and umpire. Weak, but it is a
   third anchor if the first two struggle.

**The number to beat is now well defined.** Our affine is at LOO median 3.9 in.
Their pinhole is at 0.35 in. If we build the pose solve on MLB video first, where
the box exists and the benchmark is published, and we cannot get near 0.4 in,
then the problem is us and not the data. That is why MLB video comes before A+.

---

## 5. Open items this raises

- **Phase 0 is unchanged and still blocking.** We do not have the 9-param
  trajectory in our manifests (`plate_x, plate_z, sz_top, sz_bot` only), and
  `src/ingest/statsapi.py` skips any pitch missing `pitchData.coordinates`.
  Everything above depends on getting `x0..az`.
- **Does our A+ broadcast draw a K-zone box?** One look at a pulled clip answers
  it and decides whether phase 2 ports at all.
- **How static is the A+ camera mid-flight?** Decides option 1 above. Measurable
  from clips we already have on disk.
- **We still have zero external validity.** Their credibility rests on BB% at
  +0.564 and Location+ at -0.607, not on the pipeline. We have neither.
- **Their stabilization numbers give us a sample floor for free:** Cronbach alpha
  hits 0.5 at about 55 pitches and 0.7 at about 129. Below roughly 50 pitches a
  per-pitcher command number is mostly noise. That is a real gate we will need,
  and it is the kind of floor that gets asked about the moment a coach sees a
  leaderboard.

## 6. Sources

- Repo: <https://github.com/tomdoyo/open-command> (read at v1.2.0, commit `984d26e1`)
- Data: <https://huggingface.co/datasets/tomdoyo/open-command>
- Project account: <https://x.com/open_command> (79 posts since Jul 16 2026;
  timeline not machine-readable, see note below)
- Author account: <https://x.com/tomdoyo>
- Announcement: <https://x.com/tomdoyo/status/2087272169852088752>

The X timeline could not be pulled programmatically. The fxtwitter API serves the
profile but not the timeline, and `syndication.twimg.com` refuses the connection.
Individual post URLs still work through `api.fxtwitter.com` if we have the status
id, so specific posts are recoverable one at a time. Content indexed by search
suggests the account posts leaderboards, target maps and per-pitcher trends
(example surfaced: Yamamoto's splitter at 5.19 in, against a league splitter
average of 11.00 in).
