# Command CV — Two Models, Two Planes, and the Standalone-CV Goal (Jul 9 2026)

Zac direction. Locks the conceptual framework so we don't conflate two genuinely
different measurements. Read this before touching calibration or the "actual".

---

## The two planes (this is the whole distinction)

A pitch is a moving object; WHERE it is depends on WHEN you measure it:

| Measurement | Plane | Source | What it is |
|---|---|---|---|
| **plate_x / plate_z** | **front of the plate** (the rulebook crossing plane) | Trackman/Hawk-Eye (DATA, from the authed analytics payload) | Where the ball crosses the front edge of home plate. The canonical "location." |
| **glove-catch** | **receive plane** (~1.5-3 ft BEHIND the plate) | our CV (glove detected in video) | Where the catcher actually catches it. |

**They are different numbers by design.** The ball is still travelling (and dropping)
between the plate front and the catcher's glove, so glove-catch ≠ plate_x/z. Neither
is "wrong" — they measure different points. Do not silently mix them; label which
plane a value lives in.

- Target (intended) lives in the **receive plane** — it's the catcher's set-up glove.
- plate_x/z lives in the **plate-front plane**.
- So even "target vs plate_x/z" spans two planes; the per-game affine absorbs the
  ~constant offset. This is acceptable for command (a few inches), documented here
  so nobody "fixes" the discrepancy thinking it's a bug — it is physics.

---

## The two MODELS

### Model A — plate_x/z backed (SHIP FIRST, immediate results)
Actual = Trackman `plate_x`/`plate_z`. Target = detected set-up glove. Miss =
target − actual. This is what's built + validated today (season board, 3.9in LOO).
**Fastest path to coach-facing numbers. Depends on the data being present.**

### Model B — standalone CV (THE REAL GOAL)
Actual = **glove-catch from video** (no Trackman). Target = set-up glove. Both from
CV → a command number that needs **no data backup at all**, works anywhere there's
video (non-Hawk-Eye parks, amateur, spring, side fields).

- Model B's numbers WILL differ from Model A (different planes) and it loses some
  data (pitches with no clean catch — blocked balls, foul-tips, etc.).
- **But we need it.** A fully functional model that stands on its own is the point;
  Trackman is a crutch, not the destination.
- **The future lever (Zac, "this is huge"):** once we have both, TRAIN on the
  discrepancy between Model B (glove-catch) and Model A (plate_x/z) — learn the
  plane-offset / break correction per pitch, so Model B reproduces the plate-front
  number from video alone. That's the bridge from "video command" to "Trackman-grade
  video command."

**Order of operations:** ship Model A now (data-backed, immediate). Build Model B in
parallel (needs the glove detector to be rock-solid — see below). Then learn A↔B.

---

## plate_x sign — LOCKED (Jul 9 2026)

Physical anchor: clip `si_12` is a **Hit-By-Pitch, plate_x = −2.15, struck a RHB**.
RHB stands on the 3B side → **negative plate_x = 3B side, positive = 1B side.**
For a RHP, arm-side run goes toward 3B → **RHP arm-side = negative plate_x** (LHP
mirrors). Encoded in `src/geometry.py::_ARM_SIDE_X_SIGN` (`R:-1, L:+1`). The miss
MAGNITUDE never depended on this; now the arm-side/glove-side WORD is correct too,
so command can read as "misses arm-side with the slider" — the coaching-useful form.

---

## Glove / target model — trained on ONE catcher pool (BLOCKING for scale)

**Current state:** the detector + set-up-target were annotated ONLY on Asheville
(A+) catchers — Ogando's/Langford's battery. It detects THAT pool well.

**The gap:** different catchers set up differently (glove height, how still they
hold, how they receive). A glove is a glove so the DETECTOR mostly transfers, but
set-up habits + receive styles vary, and we have ZERO validation on other
affiliates' catchers (AA/AAA/other orgs).

**Requirement:** to scale Command CV beyond Asheville, annotate MULTIPLE catchers
across levels — not just more Asheville clips. Zac: "I will annotate as many gloves
as needed." Priority order for annotation:
1. The 69 Asheville hard-cases (`output/retrain_hard_cases.txt`) — fixes the elbow-grab.
2. Then catchers at other levels as we extend to those pitchers.

plate_x/z is the prime importance for immediate results; the glove model is the
prerequisite for Model B (standalone) and for scaling past one battery.

---

## What's next (and what is explicitly LATER)
- NOW: retrain glove on hard-cases (Zac annotating) → tighter Model A board.
- NOW: Model A is shippable; sign is locked.
- LATER (separate `/spec` + `/plan`, per Zac): **how this lives in Arm Farm** —
  storing per-pitch command data, the tab, click-to-video, aggregation. Do NOT
  design the storage/UI until plate_x/z + glove tracking are nailed. Flagged, parked.
- FUTURE: Model B standalone + the A↔B discrepancy learner.
