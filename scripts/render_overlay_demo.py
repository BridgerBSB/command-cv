"""Demo: reticle-on-glove overlay on a frame (recreates reference cv image.jpg).

Builds the synthetic frame, then overlays the glove reticle + strike-zone quad,
plus the actual pitch projected back to pixels via the calibration homography.
Run: python scripts/render_overlay_demo.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.synth_frame import build
from src.calibration import (home_plate_world_points, solve_homography,
                             strikezone_world_points)
from src.report import render_frame_overlay


def main() -> None:
    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "output")
    os.makedirs(out_dir, exist_ok=True)
    frame_path = os.path.join(out_dir, "synthetic_frame.png")
    sf = build(draw_path=frame_path)
    det = sf.detections

    # Calibrate pixel<->world to project the (real) actual pitch back onto pixels.
    img_pts = list(det.plate_pts) + list(det.zone_pts)
    world_pts = home_plate_world_points() + strikezone_world_points(
        sf.zone.z_bot, sf.zone.z_top)
    H = solve_homography(img_pts, world_pts)
    actual_world = (0.45, 1.85)            # a real Trackman pX/pZ would go here
    actual_px = H.world_to_pixel(*actual_world)

    out = os.path.join(out_dir, "frame_overlay.png")
    render_frame_overlay(
        frame_path, det.glove_px, out,
        zone_px=det.zone_pts, actual_px=actual_px,
        label="Intended target (glove)")
    print("wrote", out)


if __name__ == "__main__":
    main()
