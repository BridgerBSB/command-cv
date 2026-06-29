"""Provided-video path: clip + glove-flash frame + annotations + Trackman -> report.

The end-to-end CLI for the "feed a clip" workflow (Option 1). Detection here is
the human-supplied coords JSON (the ManualDetections backend); swap in the trained
YoloDetector later with no other change.

Examples:
  # synthetic smoke (uses make_test_clip.py output):
  python scripts/make_test_clip.py
  python scripts/clip_to_report.py --clip output/test_clip.mp4 \
      --coords output/test_clip_coords.json --frame 20 \
      --plate-x 0.45 --plate-z 1.85 --sz-top 3.6 --sz-bot 1.55 --hand R \
      --pitcher "Test Pitcher" --pitch-id SMOKE1

  # real clip + pull the Trackman row from the Stats API by game + playId:
  python scripts/clip_to_report.py --clip mypitch.mp4 --coords coords.json \
      --time 4.2 --game 814841 --play-id aa35fbce-7cdf-38a9-b5ae-6650d41f1bec
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.calibration import (home_plate_world_points, solve_homography,
                             strikezone_world_points)
from src.detect import FrameDetections
from src.pipeline import TrackmanPitch, process_pitch
from src.report import render_frame_overlay
from src.video import frame_at_index, frame_at_time, probe, save_frame


def _trackman_from_args(a) -> TrackmanPitch:
    if a.game and a.play_id:
        from src.ingest.statsapi import fetch_game_pitches
        for r in fetch_game_pitches(a.game):
            if r.play_id == a.play_id:
                return r.trackman
        raise SystemExit(f"play_id {a.play_id} not found in game {a.game}")
    if a.plate_x is None or a.plate_z is None:
        raise SystemExit("supply either --game/--play-id or --plate-x/--plate-z")
    return TrackmanPitch(
        pitch_id=a.pitch_id or "MANUAL", plate_x=a.plate_x, plate_z=a.plate_z,
        pitcher_throws=a.hand, sz_top=a.sz_top, sz_bot=a.sz_bot,
        pitcher_name=a.pitcher or "")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--clip", required=True)
    p.add_argument("--coords", required=True, help="JSON: plate_pts, zone_pts, glove_px")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--frame", type=int)
    g.add_argument("--time", type=float)
    p.add_argument("--game", type=int)
    p.add_argument("--play-id")
    p.add_argument("--plate-x", type=float)
    p.add_argument("--plate-z", type=float)
    p.add_argument("--sz-top", type=float)
    p.add_argument("--sz-bot", type=float)
    p.add_argument("--hand", default="R")
    p.add_argument("--pitcher", default="")
    p.add_argument("--pitch-id", default="")
    p.add_argument("--outdir", default=None)
    a = p.parse_args()

    info = probe(a.clip)
    print(f"clip: {info.width}x{info.height} {info.fps:.1f}fps {info.frame_count}f "
          f"{info.duration_s:.1f}s")

    frame = frame_at_index(a.clip, a.frame) if a.frame is not None \
        else frame_at_time(a.clip, a.time)

    with open(a.coords) as f:
        c = json.load(f)
    det = FrameDetections(
        plate_pts=[tuple(x) for x in c["plate_pts"]],
        zone_pts=[tuple(x) for x in c["zone_pts"]],
        glove_px=tuple(c["glove_px"]))

    tm = _trackman_from_args(a)

    outdir = a.outdir or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output")
    os.makedirs(outdir, exist_ok=True)
    profile_csv = os.path.join(outdir, "command_profile.csv")

    # Run the engine (calibrate -> miss -> Strike-Zone View + profile row).
    szv = os.path.join(outdir, "clip_strike_zone_view.png")
    res = process_pitch(det, tm, report_path=szv, profile_csv=profile_csv)
    print(f"{tm.pitcher_name or 'pitch'}: {res.miss.direction}")

    # Annotated frame overlay (reticle on glove + zone + actual projected to px).
    H = solve_homography(
        list(det.plate_pts) + list(det.zone_pts),
        home_plate_world_points() + strikezone_world_points(res.zone.z_bot, res.zone.z_top))
    actual_px = H.world_to_pixel(tm.plate_x, tm.plate_z)
    save_frame(frame, os.path.join(outdir, "clip_flash_frame.png"))
    ovl = os.path.join(outdir, "clip_overlay.png")
    render_frame_overlay(frame, det.glove_px, ovl, zone_px=det.zone_pts,
                         actual_px=actual_px,
                         label=f"{tm.pitcher_name or 'Target'} — {res.miss.miss_in:.1f}in "
                               f"{res.miss.horiz_label}/{res.miss.vert_label}")
    print("wrote", szv)
    print("wrote", ovl)
    print("profile ->", profile_csv)


if __name__ == "__main__":
    main()
