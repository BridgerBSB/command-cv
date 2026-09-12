# Step 2a: our pinhole solve reproduces OpenCommand's published poses

Sep 11 2026. Before any of our own detections go into a camera solve, the solve
itself has to be right. The only independent answer key at scale is OpenCommand's
published `camera_poses.csv.gz`, so we reimplemented their geometry from reading
their code and fed it THEIR inputs for one game.

## Result, game 824821 (WSH @ BAL, Camden Yards, 2026-06-28)

| | ours | OpenCommand |
|---|---:|---:|
| clips posed | 280 | 280 (all matched by play_id) |
| camera position Cx, Cz (ft) | -13.9456, 32.5784 | -13.9456, 32.5784 |
| position votes / candidate ball runs | 211 / 284 | not published |
| ball at plate, median abs error X | 0.404 in | 0.404 in |
| ball at plate, median abs error Z | 0.395 in | 0.395 in |

Per clip, pan / tilt / roll match to 1e-11 rad and focal length to 3e-5 px.

**One clip of 280 differs, and only in its accuracy check, not its pose.**
`1574ec26-2b76-39fa-8bb1-05151c3f094a`: both solvers fit its ball run at about
12.5 px RMSE, four times the 3 px vote gate, so neither side lets it vote on the
camera position. The two optimizers (their batched Levenberg-Marquardt, our scipy
`least_squares` with the same loss, bounds and scaling) settle that bad run's
clock offset `t0` in different places, which moves its reprojection check (70.6 vs
13.4 px). Its pose is identical. Nothing downstream reads that check.

## What was reimplemented

- `src/pinhole.py`: camera basis (aim at (0, 0, 3), pan then tilt then roll by
  Rodrigues), `project`, `unproject`, Statcast 9-param trajectory, zone plane by
  era (17/12 ft through 2025, 8.5/12 ft from 2026 under ABS). Cy pinned at 400 ft.
- `scripts/parity_opencommand_pose.py`: phase 1 position votes (ball runs + box
  corners at weight 3, soft_l1, restart from the median for runs over 3 px, median
  of votes with RMSE under 3 px), phase 2 per-clip PTZ from the 4 corners alone,
  and their accuracy check.
- `tests/test_pinhole.py`: project/unproject round trip exact to 1e-9 in, basis
  orthonormal, aim point at frame centre, catcher's right on screen LEFT from CF
  (a sign slip there would mirror every miss), zone plane by era.

## Inputs and where they came from

All public, `data/opencommand/` (gitignored). OpenCommand is CC BY-NC-SA 4.0: we
read their code and data, we do not ship either.

- `2026/pbp_info.csv.gz`: Statcast trajectory and reference zone per pitch
- `2026/raw/strikezone_tracking.csv.gz`: their detected zone box per clip
- `2026/raw/gloveball_tracks/824821.csv.gz`: their ball and glove pixels per frame
- `2026/camera_poses.csv.gz`: the answer key
- Code read at `github.com/tomdoyo/open-command` commit `984d26e1`

Also pulled with Zac's token, for step 2b: every pitch of this game from MLB's
BROADCAST angle (289 clips + authed analytics payloads, `data/clips/mlb_824821/`).
One clip (`0b0cf4be...`) cannot seek to its last frames; a re-download returned the
same file, so it is the source, not our download.

## Two things learned on the way that change the plan

1. **The zone box is a drawn broadcast graphic, and our copy of the video does not
   have it.** **[CORRECTED Sep 12 2026 - THIS IS WRONG. The graphic IS in our
   BROADCAST clips, including in `c4958213` itself, from frame 0. A per-frame sweep
   of 6 clips found it on 100% of live plate-camera frames; the frames without it
   are replays and other camera shots inside the same clip. See
   `2026-09-12-zone-graphic-is-in-our-clips.md`. The claim below about CENTERFIELD /
   HIGH_HOME / PITCHCAST was NOT re-checked - only BROADCAST clips are on disk.]**
   OpenCommand's README: broadcasts draw the zone "(17in width,
   sz_top/sz_bot)", read "at a snapshot pre-pitch", and "sometimes broadcasts don't
   draw a strikezone box". Frames 0 to 240 of our BROADCAST clip for play
   `c4958213` show no graphic, yet their box lands exactly on the zone in front of
   the catcher in our frame. So our feed is the same camera framing without the
   overlay. For the MLB benchmark we can use their box pixels. For our own pipeline
   we cannot count on detecting a box, even at MLB.
2. **Their scale checks out against their own model.** A 17 in plate at 400 ft
   through f of about 16,000 px is about 57 px; their boxes in this game measure 52
   to 58 px wide.
3. **Same footage, different clock (Sep 12 2026).** Their ball pixels sit 117 px from
   our projected path if you assume their `frame_idx` counts from the plate crossing,
   which reads like a different feed. Sweep the offset instead and the residual drops to
   **0.9 to 1.5 px at about -0.4 s**, i.e. their frame zero is near release. So their
   video is the same camera and framing as ours, with the zone graphic burned in, and
   their detections and boxes can be used against our clips once the clock is aligned.
   No angle MLB serves us carries the graphic: checked BROADCAST, CENTERFIELD, HIGH_HOME
   and PITCHCAST end to end. `ALTERNATE_BROADCAST` is the one sporty-clips feed not yet
   pulled and is the last candidate.

## Step 2b: the ball can be found with no detector at all

`scripts/ball_clock_align.py`. The camera pose and the trajectory are both known, so
the ball's pixel path is known up to ONE unknown: which frame of the clip the pitch
crosses the plate. Slide that offset and score how strongly a small bright moving
object sits on the predicted path.

**40 clips, game 824821:**

| | |
|---|---:|
| score, median | 165.0 |
| baseline (median over all offsets), median | 9.6 |
| score / baseline | 17.3x |
| plate-crossing frame | 201 to 212 |
| beats MIRRORED path | 40 / 40 |
| beats TIME-REVERSED path | 39 / 39 scored (1 not scorable) |

Rendered check (`output/ball_path_check_824821.png`): the predicted circle sits on
the ball in all 15 panels, 3 pitches x 5 frames, from 16 frames out to the crossing.

### Two wrong versions first, and why they were wrong

1. **Blob detector + fit.** Found 568 candidate runs on the first clip and its best
   fit to the true trajectory was 12.7 px. It detects players and crowd, not the ball.
2. **Raw frame-difference along the path.** REJECTED by its own control: a DIFFERENT
   pitch's trajectory scored as well as the right one (median 8.55 vs 12.16; on one
   clip the wrong trajectory won 18.75 to 4.12). The score rewarded any path crossing
   busy pixels.

The fix was a score that can only respond to a small bright object: white top-hat
(keeps blobs smaller than the kernel), local contrast against a ring around the point,
and the MEDIAN over the run so most of the flight must agree.

### The control matters more than the score

"Another pitch from the same game" is a WEAK control: pitches trace nearly the same
screen path, so it does not separate. The controls that work are physically
impossible: the same path MIRRORED left-right, and the same path REVERSED in time.
Both keep speed, length and pixel scale, so only the path differs.

One nuance worth keeping: a control can be **unscorable**, because a mirrored or
reversed path can leave the frame and never collect enough sample points. The first
summary counted that NaN as a loss and printed "39/40" on a run where nothing actually
lost (clip `09fad439`, CU). The script now prints the denominator and says how many
were not scorable.

### Why this matters beyond MLB

At the winning offset the projected ball position on every flight frame is a LABEL
with no hand annotation: 800 labels from 40 clips, `output/ball_labels_824821.csv`.
That is the training set for a real ball detector, which is what A+ needs, where there
is no trajectory to lean on and no drawn box to calibrate against.

## A+ sample sizes, whatever anchor wins (Sep 12 2026)

Any per-game camera solve needs enough clips from that one camera. What we hold per
game, counting only OUR pitcher's pitches:

| pitcher | pitches | games | clips per game |
|---|---:|---:|---|
| Ogando | 149 | 13 | 60, 33, 23, 7, 6, 6, 5, 3, 2, 1, 1, 1, 1 |
| Langford | 32 | 11 | 4, 4, 4, 3, 3, 3, 3, 3, 2, 1 (one game 0 on disk) |

Three Ogando games clear 20 clips; Langford clears none. That looks bleak and it is the
wrong number to design around. **The camera does not care who is pitching**, so a per-game
solve uses EVERY pitch in the game. From the public feed, those same A+ games carry:

| game | date | venue | our pitches | pitches in the game |
|---|---|---|---:|---:|
| 822515 | 2026-04-25 | HomeTrust Park | 60 | 362 |
| 822246 | 2026-04-12 | First National Bank Field | 33 | 322 |
| 821922 | 2026-04-04 | AdventHealth Stadium | 23 | 280 |
| 821780 | 2026-04-19 | Truist Stadium | 7 | 340 |
| 821847 | 2026-05-02 | Frawley Stadium | 6 | 295 |
| 822512 | 2026-05-07 | HomeTrust Park | 6 | 283 |

Every one of them has a `playId`, so a game-scoped pull yields 280-362 clips per camera,
which is five times the 59 the MLB joint solve is being tested on. `pull_pitcher_clips.py`
is pitcher-scoped, so this needs a game-scoped puller (the MLB one used this session lives
in the scratchpad) plus one token paste per game. A 5-clip game is not the constraint.

## Step 2c: the box CAN be dropped, if the camera is solved across many clips at once

`scripts/ablate_no_box.py`, 59 clips of game 824821, OpenCommand's ball detections
throughout so the DETECTOR is never the variable. Every variant is scored on the zone-box
corners (a known-position object the no-box fits never see) and on how far it disagrees
with the box pose about a fixed pixel at the catcher's setup depth, which is where the
miss is read.

| variant | ball@plate abs X, Z (in) | BOX CORNERS abs X, Z (in) | glove-plane gap (in) |
|---|---|---|---:|
| BOX (published shape) | 0.387, 0.355 | 0.051, 0.046 | 0 (reference) |
| NOBOX, per clip | 0.167, 0.183 | **18.4, 29.7** | **42.6** |
| NOBOX_C (C from box votes, aim from ball) | 0.214, 0.321 | 0.485, 0.500 | 0.69 |
| **NOBOX_JOINT (no box anywhere)** | 0.120, 0.230 | **0.665, 0.600** | **1.08** |

Camera position: the joint ball-only solve lands at Cx -13.70, Cz 33.14 against the
box-assisted Cx -14.04, Cz 32.55. **Off by 0.34 ft and 0.59 ft with no box at all.**

**Two things this settles.**

1. **Per-clip ball-only is dead**, and would have looked like the winner on the naive
   metric. It posts the BEST ball-at-plate number in the table (0.167 in) while missing
   the zone corners by up to 30 inches and disagreeing with the real camera by 42 in at
   the glove plane. Its camera scatters 17 ft in X and 37 ft in Z and rails at a solver
   bound on 21 of 59 clips. A 7-parameter fit to ~21 points buys the ball path and
   nothing else.
2. **One camera position fitted jointly across many clips replaces the box**, at a cost
   of about an inch at the plane where command is measured. Against miss distances of 8
   to 16 inches that is a real but acceptable tax, and it is far below the 3.9 in our
   old affine calibration carries.

Caveats, none of them small: this used OpenCommand's ball detections, and ours will be
worse; the glove-plane gap is measured against the box pose, which is a good reference
but not ground truth; `Cy` is pinned at 400 ft, which is an MLB centre-field number and
must be re-checked for the closer, behind-the-pitcher A+ camera; and A+ has to be tested
on its own footage, not inferred from this.

## Next
The box question is closed, so the path to A+ is now concrete:

1. **Train a ball detector on the auto-labels** (step 2b, 800 labels per 40 clips, free).
   Then re-run the joint solve with OUR detections replacing OpenCommand's, still at MLB,
   still scored on held-out box corners. That isolates the detector as the only change and
   says what our own vision costs us on top of the 1.08 in the joint solve already costs.
2. **Game-scoped pull at A+.** The joint solve needs many clips from one camera, and at A+
   only a game-scoped pull provides them (280-362 pitches per game vs our pitcher's 5-60).
   `pull_pitcher_clips.py` is pitcher-scoped; the game-scoped puller used for MLB this
   session is still in the scratchpad and needs promoting. One token paste per game.
3. **Re-check the pinned camera distance for A+.** `Cy = 400 ft` is an MLB centre-field
   number. The Asheville camera is behind the pitcher and closer, so the pin that is
   harmless at MLB may not be, and it needs its own degeneracy check before any A+ number
   is believed.
4. **Only then the miss itself:** glove detection at setup gives the target, the joint-solve
   camera turns those pixels into inches at the setup plane, and the actual stays
   `plate_x`/`plate_z` from tracking. That is the original Model A, with a ruler that now
   survives its own controls.
