# Command CV — Miss-Distance / Command Tracker

Quantify pitcher **command** by measuring how far a pitch missed the catcher's
intended target — recreating the LinkedIn tool by Lucas Baquero. Turns "intent
into measurable data" instead of the eye test.

> **Key idea:** we do NOT track the ball in video. The miss is the gap between
> where the catcher set the target and where the pitch actually ended up.

## Two methods (pick by camera angle)

- **Method A — Trackman (center-field camera).** intended = glove (CV on the
  setup frame); actual = Trackman `plate_x`/`plate_z`; calibrate the frame to the
  plate plane via a homography. Best when the strike-zone plane faces the lens.
  Modules: `geometry.py` + `calibration.py` + `pipeline.py`; feed = `ingest/statsapi.py`.
- **Method B — glove-size scale (side / down-the-line camera, e.g. GC2 clips).**
  intended = mitt at the **setup** frame; actual = mitt at the **catch** frame;
  miss = their displacement scaled to inches by the mitt's real ~12" size (the
  "size of the glove" idea). No Trackman, no plate homography, immune to the
  oblique angle. Modules: `glove_track.py` + `miss_methodb.py`; run `methodb_pitch.py`.
  Verified on a real GC2 Ogando clip: **7.2 in arm-side/up** off a low target.

See `docs/2026-06-29-command-cv-miss-distance-design.md` for the full design.

## Status (2026-06-29) — working PoC

| Stage | State |
|---|---|
| Geometry (miss vector, inches, arm/glove + up/down labels) | ✅ built + unit-tested |
| Calibration (pixel → Trackman plane homography, pure NumPy DLT) | ✅ built + round-trip tested |
| Strike-Zone View report (matplotlib, Astros colors) | ✅ built + render-and-look verified |
| Per-pitcher command profile CSV + session rollups | ✅ built |
| Pipeline orchestrator | ✅ built |
| **Stats API ingest (actual loc + video `playId`)** | ✅ **live-verified** (261 pitches, game 814841) |
| Detection interface (glove/plate/zone) | ✅ interface + manual backend; YOLO backend stubbed |
| **Provided-video path** (decode clip → flash frame → annotate → report) | ✅ built + tested end-to-end (synthetic clip) |
| Click-to-annotate / labeling tool | ✅ `scripts/annotate_clip.py` |
| **Trained glove/plate detector** | ⬜ TODO — needs labeled real frames |
| Video clip fetch from `playId` | ⬜ **needs MLB key** — `/guids` `/analytics` return 401; `/content` has no free MiLB clip |

## Quick start

```bash
pip install -r requirements.txt
python -m pytest -q                              # 7 tests
python scripts/render_synthetic_report.py        # synthetic Strike-Zone View
python scripts/demo_statsapi_report.py 814841 0   # REAL pitch from Stats API
python -m src.ingest.statsapi 814841              # dump tracked pitches + playIds

# Provided-video path (the "feed a clip" workflow):
python scripts/make_test_clip.py                  # synth clip + coords for smoke test
python scripts/clip_to_report.py --clip output/test_clip.mp4 \
    --coords output/test_clip_coords.json --frame 20 \
    --plate-x 0.45 --plate-z 1.85 --sz-top 3.6 --sz-bot 1.55 --hand R \
    --pitcher "Test" --pitch-id SMOKE1
# For a REAL clip: annotate the flash frame by clicking, then report:
python scripts/annotate_clip.py --clip mypitch.mp4 --frame 120 --out coords.json
python scripts/clip_to_report.py --clip mypitch.mp4 --coords coords.json --frame 120 \
    --game 814841 --play-id <GUID>     # pulls the matching Trackman row
```
Outputs land in `output/` (gitignored).

## How a real pitch flows

1. **Ingest** — `src/ingest/statsapi.py` pulls every tracked pitch of a game:
   `plate_x/plate_z`, strike-zone top/bottom, pitcher hand/name, pitch type, and
   the `playId` GUID for the clip. (No DB, no key.)
2. **Detect** — locate home plate + strike zone + glove in the glove-flash frame.
   Today: `ManualDetections` (provided coords). Planned: `YoloDetector`.
3. **Calibrate** — `solve_homography` maps frame pixels → Trackman plate plane (ft),
   anchored on the 17" plate + strike-zone corners.
4. **Compute** — `compute_miss` → miss in inches + arm/glove + up/down.
5. **Report** — `render_strike_zone_view` → the Strike-Zone View PNG.
6. **Profile** — `src/profile.py` appends the row for session-over-session trends.

## Annotate → train → apply (the click-tool workflow)

For real clips, label the catcher's mitt, then train the detector to auto-find it.

```bash
# 1. (optional) pick the setup + catch frames
python scripts/contact_sheet.py --clip "ogando_ff_1.mp4" --start 78 --end 124 --step 3 --crop

# 2. annotate: scrub (a/d), click mitt center+edge, s=setup  c=catch  t=Trackman  w=write
python scripts/annotate_tool.py --clip "ogando_ff_1.mp4" --hand R
#    -> data/annotations/ogando_ff_1.json + YOLO samples in data/yolo/

# 3. real numbers for everything annotated so far
python scripts/methodb_batch.py        # -> output/command_profile_b.csv + rollup

# 4. train the mitt detector once enough clips are labeled
pip install ultralytics
python scripts/train_yolo.py --epochs 100
#    -> data/yolo/runs/mitt/weights/best.pt  (wire into src/detect.py YoloDetector)
```

`data/` is local (gitignored): clips aren't in the repo, annotations + training
images regenerate from them.

## Known caveats (read before any coach-facing use)

- **plate_x sign + arm/glove label** sit behind one documented constant flagged
  `# VERIFY` — validate against ONE known DB pitch on the work laptop first.
- **Glove parallax** — the glove is ~1.5-3 ft behind the plate plane; v1 labels
  reports "glove location approximate" (same limitation as the reference tool).
- **Hawk-Eye sparsity** — untracked MiLB parks lack pX/pZ; those pitches are
  skipped (no "actual" to compare).
