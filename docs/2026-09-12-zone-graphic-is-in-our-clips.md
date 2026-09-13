# The zone graphic IS in our MLB clips (corrects Sep 11)

Sep 12 2026. `2026-09-11-step2-pinhole-parity.md` says, under "Two things learned
on the way that change the plan":

> **The zone box is a drawn broadcast graphic, and our copy of the video does not
> have it.** ... Frames 0 to 240 of our BROADCAST clip for play `c4958213` show no
> graphic ... So our feed is the same camera framing without the overlay.

and

> No angle MLB serves us carries the graphic: checked BROADCAST, CENTERFIELD,
> HIGH_HOME and PITCHCAST end to end.

**The first claim is wrong, and it is wrong on its own test case.** The graphic is
in our BROADCAST clips, including `c4958213`, from frame 0.

## How it was found

Not by looking for it. Zac asked to see "the box zone" and the honest answer on
record was "we can't get one, every feed MLB serves is clean." Rendering the zone
we PROJECT (`scripts/render_zone_demo.py`) put a raw frame beside an annotated one,
and the raw panel - nothing drawn on it by us - had a white rectangle in it.

`ball_path_check_824821.png` had been showing that rectangle since Sep 11. It was
read as part of our own overlay. `ball_clock_align.py` draws exactly three things:
a green circle, a black caption bar, yellow text. Nothing white, nothing
rectangular.

## The measurement

`scripts/box_presence_sweep.py`. The projected quad is known, so score every frame:

    mean(border ring of the quad) - mean(its inner and outer neighbours)

A drawn bright line makes the border brighter than BOTH neighbours. Six clips,
every frame, `hi` = frames scoring >= 8 grey levels:

| play | frames | hi | hi% | peak |
|---|---:|---:|---:|---:|
| **c4958213** (the doc's own play) | 764 | 251 | 32.9% | 39.69 |
| 00f695d8 | 408 | 408 | 100.0% | 46.58 |
| 018f0e61 | 410 | 410 | 100.0% | 51.33 |
| 01dde385 | 912 | 324 | 35.5% | 45.26 |
| 0305f843 | 530 | 263 | 49.6% | 50.42 |
| 04438004 | 392 | 388 | 99.0% | 37.73 |

**The sub-100% clips are not flickering.** They are longer clips containing replays
and other camera shots, and the score correctly collapses on those frames:
`c4958213` f382 = -0.1 (a fielder in the outfield), `01dde385` f456 = -4.1 (an
outfield wall ad), `0305f843` f265 = 4.4 (the mound). That behaviour is the
metric's own negative control - it does not fire when the plate camera is not live,
which it would if it were merely picking up the catcher's white jersey.

Filmstrip: `output/box_presence_strip.png`. Raw-vs-projected: `output/zone_zoom.png`.

## Our projection lands on it

`scripts/zone_zoom.py`, layers separated (the first attempt drew OpenCommand's box
over ours and hid the agreement). Projected zone corners vs OpenCommand's observed
box pixels, per edge:

| play | L | T | R | B |
|---|---:|---:|---:|---:|
| 00f695d8 | 0.3 | 0.2 | 0.3 | 0.2 |
| 018f0e61 | 0.2 | 0.1 | 0.2 | 0.1 |

Sub-pixel on all eight edges. **Read this for what it is:** these poses come from
`parity_opencommand_824821.csv`, the BOX-ASSISTED solve, which was fitted with those
four corners at weight 3. Agreeing with an input you were given is consistency, not
independent confirmation.

## The blind version - RUN, and it holds (same day)

`scripts/zone_blind.py`. The pose here comes from `ablate_no_box.fit_joint`: ONE
shared camera position fitted across 59 clips from ball pixels ALONE (297 params,
2468 residuals), each clip keeping its own pan/tilt/roll/f/t0. **No box anywhere in
the fit**, so the drawn box is a genuinely held-out object.

| | Cx | Cz |
|---|---:|---:|
| box-assisted game solve | -14.040 | 32.552 |
| **blind joint solve** | **-13.697** | **33.138** |

Off by 0.34 ft and 0.59 ft - reproducing the Sep 11 ablation exactly.

| play | blind vs drawn box, max edge | quad skew |
|---|---:|---:|
| 00f695d8 | 0.628 in | 0.65 / 0.88 px |
| 018f0e61 | 1.954 in | 3.03 / 2.20 px |
| 01dde385 | 0.842 in | 1.34 / 0.89 px |

Median max-edge 0.842 in, worst 1.954 in, at ~0.30 in/px.

**The important part is that it got WORSE.** The box-assisted render agreed to
0.1-0.3 px; this one disagrees by 0.5-6.5 px. That is the expected direction and it
is the evidence the test is real - a blind fit returning the same sub-pixel number
would have meant the box had leaked into it. It also sits where the Sep 11 ablation
put `NOBOX_JOINT` (0.665 / 0.600 in on held-out corners), which is the independent
check on this check. In `output/zone_blind.png` the row-2 clip is visibly offset,
which is the 1.954 in case showing itself rather than being averaged away.

**SKEW, and what OpenCommand's format throws away.** Our zone is a true projected
quadrilateral; their stored box is four scalars (`x_left`, `y_top`, `x_right`,
`y_bottom`) and so is axis-aligned by construction. The measured non-rectangularity
is 0.65 to 3.03 px - real, and unrepresentable in their format. Anything comparing
against their stored box inherits that as a floor.

## What this changes, and what it does not

**Unaffected:**
- **Step 2c, "the box can be dropped."** That ablation used OpenCommand's box pixels
  and ball detections throughout; whether OUR clips also show a box is irrelevant to
  whether the SOLVE needs one. 1.08 in at the glove plane stands.
- **The A+ motivation.** Asheville genuinely draws no box - that was measured
  separately on 55 hand-labelled clips across 7 parks
  (`2026-09-01-opencommand-v12-and-cross-camera-accuracy.md` section 4a) and is not
  touched by this.

**Changed:**
- At MLB we can read the zone box off OUR OWN clips. The MLB benchmark no longer has
  to borrow `strikezone_tracking.csv.gz` from OpenCommand to have a box to score
  against, which removes a CC BY-NC-SA dependency from the benchmark path.
- It is a free, per-pitch, known-world-position calibration target on every MLB clip -
  useful for validating the detector and the camera solve at scale.

## Scope - now FIVE parks, five broadcasters (extended same day)

The first pass covered one game, and that was the stated weakness. `box_presence_sweep.py`
now reads `data/opencommand/poses_<game>.csv` as well as the parity CSV, so the other
four games are testable. Five clips swept per game:

| game | park | peak score | hi% range |
|---|---|---:|---|
| 824821 | BAL Camden | 37-51 | 33-100% |
| 824259 | DET Comerica | 47-66 | 27-100% |
| 824341 | COL Coors | 51-55 | 22-80% |
| 823850 | MIA loanDepot | 59-63 | 17-100% |
| 824018 | LAA Angel Stadium | 52-60 | 21-92% |

**Every clip at every park carries it**, with peaks at the four new parks equal to or
higher than Camden's. So this is not a MASN quirk - it is systematic across MLB
broadcasts.

The varying hi% is the same replay explanation, and it is self-consistent: hi% tracks
INVERSELY with clip length, which is what you expect if the graphic is on during the
live plate camera and long clips carry more replay. The two longest clips in the set
(`034a4e69` 1602 frames, `02a8477d` 1678 frames) score 21.7% and 20.9%; clips of
396-456 frames score 100%.

**Verified by eye, not only by score** (`output/zone_zoom_COL.png`,
`output/zone_zoom_MIA.png`): the untouched RAW panel shows the white rectangle at
Coors Field and at loanDepot Park - a day game and a night game, different
broadcasters - and our projected quad lands on it to 0.1-0.3 px on all eight edges.
The MIA render also exercises the no-labels path (`using clip midpoint`, frames 213
and 184), which is correct because the zone is world-fixed and the camera is static in
flight: the frame choice changes the picture, not the quad.

### Still NOT re-checked

Only **BROADCAST** was tested, because only BROADCAST clips exist on this machine. The
Sep 11 claim about CENTERFIELD, HIGH_HOME and PITCHCAST is neither confirmed nor
refuted.

## The lesson worth keeping

The Sep 11 conclusion came from eyeballing ONE play, and the line directly above it
quotes OpenCommand's README saying "sometimes broadcasts don't draw a strikezone
box" - which is exactly the warning that a single clip cannot settle it. A sweep
over every frame of six clips took minutes and is not cherry-pickable.

Same family as the `crop_check` alarm earlier the same day: a 16-20 px object is
below what the eye resolves in a full frame, and both the false alarm (a box I said
was on a tire advert, which was a real ball) and this false negative came from
looking at too few pixels too casually. Measure it, and include a control that can
fail.
