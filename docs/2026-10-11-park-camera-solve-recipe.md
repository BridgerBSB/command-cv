# Solving a new park's camera - the recipe (reconstructed 2026-10-11)

Why this exists: a solved park camera is what turns a park into (a) a miss distance in inches
and (b) camera-projected training labels (PROGRESS-LOG 102, 113). Six parks have been solved,
each by hand, and the sequence lived only in queue scripts and logs. This is that sequence,
rebuilt from `output/_queue*.sh`, `output/_adv_821922.log`, the script docstrings and the
progress log. Every command below was run at least once on a real park; the two steps that
were never scripted were marked **UNSCRIPTED**; step 5 now has a script (make_holdout_split.py), step 7 (landmarks) still does not.

**LABELS ONLY? Skip the landmarks.** If the park is wanted for training labels and not for a
miss in inches, steps 7-9 are unnecessary: a camera fitted to the ball alone (step 6, with
`--stab`) projects labels within ~1 px of the landmark camera's at all five solved parks
(PROGRESS-LOG 115). `scripts/label_new_park.py --game g --dir D` runs steps 2, 4, 5, 6 and 10 and
stops at the eye check. You still mark the overlay boxes (step 3) by hand first.

Worked examples: AdventHealth 821922 (behind the pitcher, rows 107), Durham 816698 (CF view,
rows 87-91), Asheville 822515 (side view, row 100).

Notation: `g` = game_pk, `D` = clip dir (`data/clips/pull_<g>` for a downloaded game),
`V4 = data/yolo_ball_runs/milb4_v4/weights/best.pt`. Run from `command-cv/`. GPU steps use the
ROCm venv on the AMD laptop (`~/.venvs/cv-rocm`); never two GPU jobs at once.

---

## 0. Pick the game

- Park must be ADMIT in `docs/park_admission.csv` (v4's tracks there were eye-checked real).
- The pitch-level manifest comes from the banked browser pull: `output/cvpull/cvpull_<g>.csv`
  (123 games banked, URLs permanent). Plan of record: `output/pull_plan_2026.csv`.
- Prefer a game with the most pitches carrying `plate_x/plate_z` + the 9-param trajectory -
  `compare_models_at_plate.py` and `command_miss.py` grade only those.

## 1. Clips

```
python scripts/pull_from_browser_csv.py --csv output/cvpull/cvpull_<g>.csv --name pull_<g>
python tools/download_manifest_clips.py data/clips/pull_<g>
```
Resumable; writes `<D>/<stem>.mp4` + `<D>/pull_<g>_manifest.csv`. Disk: ~120 clips is a few GB.
(822258 / 821924 / 821925 have a manifest and no mp4s - step 1's second command was never run.)

## 2. v4 ball tracks + eye check

```
python scripts/detect_ball_tracks.py --dir D --game g --weights V4 --device 0 --tag _v4 --window-pick
```
- `--window-pick` at side-view parks (the return throw otherwise wins longest-chain, row 104);
  `--two-pass` at Durham/Asheville-style views. `--lo/--hi` bound the flight frames
  (AdventHealth: `--lo 150 --hi 260`); the script prints the window that would have sufficed.
- Output `output/detect_tracks_<g>_v4.csv` + raw `dets_<g>_v4.csv` (project_labels needs the raw
  dets later). The script says it: these are detections, not verified. Render a random sheet and
  LOOK before going on.

## 3. Reference view + overlay mask (hand step, ~5 min)

Create `data/landmarks/<g>.json`:
```json
{"game": g, "overlay_boxes": [[x0,y0,x1,y1], ...], "overlay_note": "...",
 "ref_stem": "<a clip showing the most field>", "ref_frame": 150}
```
- Mask EVERY broadcast overlay (scorebug, corner bars). An unmasked scorebug is what broke
  stabilization at every park until row 91 (27% zoom error, all-green output).
- Pick the WIDEST reference view: the landmark fit needs foul-line chalk / mound / plate bar in
  frame (Isotopes' first reference showed only the mound and under-determined the camera, row 90).

## 4. Stabilize every clip into the reference view

```
python scripts/stabilize_park.py --game g --dir D --tracks output/detect_tracks_<g>_v4.csv --chain --tag c
```
(`--ref-clip <mp4> --ref-frame N` when the reference lives at another game of the same park, e.g.
Isotopes 815443 -> 815419_0284 f135.) Runtime: the mitt pass dominates, ~1.5 h for 327 clips,
cached to `setups_<g>.csv` for reruns. Writes `stab2_H_<g>c.csv`, `detect_tracks_<g>_v4stab2c.csv`,
`targets_<g>_stab2c.csv`, `stab2_clips_<g>c.csv`.

Read the printout: clips with no usable homography, clips mapped through a chain, and the zoom
histogram. A zoom histogram stacked exactly at 1.000 at a park known to re-frame = the overlay
mask is wrong.

## 5. Held-out split + fit-half seeds

```
python scripts/make_holdout_split.py --game g --tracks output/detect_tracks_<g>_v4.csv
```
Writes two files and REFUSES to overwrite either:
- `output/holdout_<g>.csv` - one column `stem`. Never used by any fit. v6_run, grade_holdout and
  project_labels all key off this file, so it must exist BEFORE step 6 and never change after.
- `output/seeds_<g>_v4fit.csv` - the RAW v4 track rows of the other half.

The rule (sorted tracked stems, odd positions held out, even positions seed) was recovered from
the five solved parks; `--check` reproduces Asheville, AdventHealth and Frawley exactly. Isotopes
815419 and Durham 816698 used a hand-picked pool (165 of 242, 25 of 46 tracked clips) whose
filter was never recorded, so their files stay as they are. Written 2026-10-11 (was step 5's
"UNSCRIPTED" gap).

## 6. Ball-only starting pose

```
python scripts/fit_park_pose.py --game g --dir D --seeds output/seeds_<g>_v4fit.csv --out output/park_pose_<g>_v4fit.npz --render
```
Multi-start. LOOK at the render: home plate and the rubber are drawn from the fitted camera and
never entered the fit, so landing on the real plate is the only honest check here. A ball-only
pose is good for LABELS, not inches - (Cy, f) trade off and Cy can rail (Durham, row 87).

## 7. Ground landmarks - **UNSCRIPTED**

`data/landmarks/<g>_feats.npz`: pixel points on the reference frame for the foul lines
(|x| = y on the ground), the 9 ft mound circle centred (0, 59), and the plate bar, extracted by
HSV colour segmentation and eye-checked. Record what was REJECTED and why in the json
(`"rejected"`), because the wrong-landmark failures were the expensive ones:
- Durham's "rubber" slab was not the rubber (row 88).
- AdventHealth's bright slab is ~9 ft behind the pivot foot (a mat / rosin bag), and from a low
  behind-the-pitcher camera only the NEAR mound edge is real - the far edge is the crest
  (all-edge fit 18 px -> near-edge 8 px, row 107).
- Greensboro: both landmark fits put the plate 47-95 px high; it stayed ball-only (row 108).

Gap to close: a `extract_landmarks.py --game g` that writes the npz + a render with every point
drawn, so the eye check is one look instead of a session.

## 8. Joint camera: landmarks + fit-half ball

```
python scripts/fit_pose_landmarks.py --game g --dir D --seeds output/seeds_<g>_v4fit.csv \
    --stab output/detect_tracks_<g>_v4stab2c.csv --ball-weight 1 --joint-from-init \
    --init output/park_pose_<g>_v4fit.npz --out output/park_pose_<g>_s2lm1.npz
```
`--ball-weight 0` first is worth it: a landmarks-only camera makes every ball track an
independent check (Durham landmarks-only 5.12 in vs joint 0.45 in after the overlay fix).

## 9. Grade on the held-out half

```
python scripts/command_miss.py --game g --dir D --pose output/park_pose_<g>_s2lm1.npz \
    --tracks output/detect_tracks_<g>_v4stab2c.csv --targets output/targets_<g>_stab2c.csv --tag _s2c
python scripts/grade_holdout.py --game g --tags _s2c
```
B-A gap = |video plate crossing - tracking plate crossing|, inches, held-out clips only.
Solved parks for reference: AdventHealth 0.15, Isotopes ~0.3, Asheville 0.30, Frawley ~0.4,
Durham 0.45. Check the SIGNED bias too - a median of |gap| hides a constant offset.

## 10. Camera-projected labels (what v6 trains on)

```
python scripts/project_labels.py --game g --dir D --pose output/park_pose_<g>_s2lm1.npz --stab-tag c --all-frames
python scripts/render_proj_labels.py --game g --clips D
```
Fit-half stems only (held-out stems are never labelled). LOOK at the sheet and add a row to
`docs/proj_label_admission.csv`: CONFIRMED rows (a v4 candidate agrees) and UNCONFIRMED rows get
separate verdicts, because unconfirmed means "detector blind spot" at some parks (Isotopes,
Frawley: ADMIT) and "ball hidden behind batter/catcher" at others (Durham, Asheville: REJECT).
Then add `(g, D)` to `TRAIN_GAMES` and a grading row to `PARKS` in the next training script.

---

## Time per park (what is measured, what is not)

| Step | Measured |
|---|---|
| 1 download | not measured on this laptop |
| 2 detect v4 | ~4 min for 82 clips at Isotopes on the AMD GPU (v6 run, 2026-10-11) |
| 4 stabilize | ~1.5 h for 327 clips (Surface, mitt pass); cached on rerun |
| 5, 7 | hand work; no time recorded |
| 6, 8 | not measured |
| 10 | seconds per game (row 113) |
