# Command CV — Miss-Distance / Command Tracker

Quantify pitcher **command** by measuring how far a pitch missed the catcher's
intended target — recreating the LinkedIn tool by Lucas Baquero. Turns "intent
into measurable data" instead of the eye test.

> **Key idea:** we do NOT track the ball in video. The **catcher's glove**
> (intended target) comes from **computer vision** on one frame; the **actual**
> pitch location comes from **Trackman / Hawk-Eye** (`plate_x` / `plate_z`). The
> miss is the vector between them, in real inches + a direction.

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
| **Trained glove/plate detector** | ⬜ TODO — needs labeled MiLB frames |
| Video clip fetch from `playId` (Film Room entitlement) | ⬜ open question |

## Quick start

```bash
pip install -r requirements.txt
python -m pytest -q                              # 7 tests
python scripts/render_synthetic_report.py        # synthetic Strike-Zone View
python scripts/demo_statsapi_report.py 814841 0   # REAL pitch from Stats API
python -m src.ingest.statsapi 814841              # dump tracked pitches + playIds
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

## Known caveats (read before any coach-facing use)

- **plate_x sign + arm/glove label** sit behind one documented constant flagged
  `# VERIFY` — validate against ONE known DB pitch on the work laptop first.
- **Glove parallax** — the glove is ~1.5-3 ft behind the plate plane; v1 labels
  reports "glove location approximate" (same limitation as the reference tool).
- **Hawk-Eye sparsity** — untracked MiLB parks lack pX/pZ; those pitches are
  skipped (no "actual" to compare).
