"""Generate a synthetic 'glove-flash' frame + ground-truth pixel coords.

Lets us test calibration end-to-end (pixel glove -> world ft) without real video.
A synthetic perspective homography maps world feet -> image pixels; we project
the plate corners, strike-zone corners, and a planted glove point, draw a simple
behind-the-plate scene, and return the pixel coords + the planted glove world.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.calibration import (home_plate_world_points, solve_homography,
                             strikezone_world_points)
from src.detect import FrameDetections
from src.geometry import StrikeZone

IMG_W, IMG_H = 960, 720


@dataclass
class SyntheticFrame:
    detections: FrameDetections
    glove_world: tuple[float, float]   # planted ground-truth (x_ft, z_ft)
    zone: StrikeZone
    image_path: str | None


def _world_to_pixel_homography(zone: StrikeZone):
    """A synthetic perspective: map zone world corners -> a trapezoid in pixels
    (top edge narrower than bottom = looking slightly down from CF cam)."""
    world = strikezone_world_points(zone.z_bot, zone.z_top)  # TL,TR,BR,BL
    # Desired pixel trapezoid (perspective): top narrower, shifted up.
    cx = IMG_W / 2
    pix = [
        (cx - 70, 300),   # TL
        (cx + 70, 300),   # TR
        (cx + 95, 470),   # BR
        (cx - 95, 470),   # BL
    ]
    return solve_homography(world, pix)   # maps world(ft) -> pixel here


def build(glove_world=(-0.5, 3.25), draw_path: str | None = None) -> SyntheticFrame:
    zone = StrikeZone.from_trackman(sz_top=3.6, sz_bot=1.55)
    H_w2p = _world_to_pixel_homography(zone)

    def w2p(x, z):
        return H_w2p.pixel_to_world(x, z)  # H here is world->pixel; name reused

    plate_world = home_plate_world_points()         # 2 pts at z=0
    zone_world = strikezone_world_points(zone.z_bot, zone.z_top)
    plate_px = [w2p(x, z) for (x, z) in plate_world]
    zone_px = [w2p(x, z) for (x, z) in zone_world]
    glove_px = w2p(*glove_world)

    det = FrameDetections(plate_pts=plate_px, glove_px=glove_px, zone_pts=zone_px,
                          confidence=1.0)

    image_path = None
    if draw_path:
        image_path = _draw(draw_path, plate_px, zone_px, glove_px)

    return SyntheticFrame(detections=det, glove_world=glove_world, zone=zone,
                          image_path=image_path)


def _draw(path, plate_px, zone_px, glove_px) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon

    fig = plt.figure(figsize=(IMG_W / 100, IMG_H / 100), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, IMG_W); ax.set_ylim(IMG_H, 0)  # image coords: y down
    ax.set_facecolor("#3f5d34")  # field green
    ax.axis("off")
    # strike zone (perspective quad)
    ax.add_patch(Polygon(zone_px, closed=True, fill=False, edgecolor="yellow", lw=2))
    # home plate (line between the two front corners + a small pentagon hint)
    (lx, ly), (rx, ry) = plate_px
    ax.plot([lx, rx], [ly, ry], color="white", lw=3)
    # glove (catcher mitt) as a dark circle
    ax.scatter([glove_px[0]], [glove_px[1]], s=900, c="#2a2a2a", edgecolors="tan", lw=3)
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path


if __name__ == "__main__":
    out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "output", "synthetic_frame.png")
    sf = build(draw_path=out)
    print("planted glove world:", sf.glove_world)
    print("glove px:", sf.detections.glove_px)
    print("wrote", out)
