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
   have it.** OpenCommand's README: broadcasts draw the zone "(17in width,
   sz_top/sz_bot)", read "at a snapshot pre-pitch", and "sometimes broadcasts don't
   draw a strikezone box". Frames 0 to 240 of our BROADCAST clip for play
   `c4958213` show no graphic, yet their box lands exactly on the zone in front of
   the catcher in our frame. So our feed is the same camera framing without the
   overlay. For the MLB benchmark we can use their box pixels. For our own pipeline
   we cannot count on detecting a box, even at MLB.
2. **Their scale checks out against their own model.** A 17 in plate at 400 ft
   through f of about 16,000 px is about 57 px; their boxes in this game measure 52
   to 58 px wide.

## Next: step 2b

Swap THEIR ball detections for OURS on the same 289 clips, keep everything else,
and see whether ball at plate stays near 0.40 in. That isolates our detector as the
only change. Then swap the box for a box-free anchor, which is the A+ problem.
