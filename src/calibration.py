"""Calibration: map image pixels -> Trackman plate-plane feet via a homography.

We anchor the mapping on reference points whose real-world (x, z) in the
strike-zone plane are known: the home-plate corners (17 in wide) and/or the
strike-zone box corners. Given >=4 correspondences we solve a 3x3 homography
with the Direct Linear Transform (DLT) in pure NumPy — no OpenCV required, so
calibration is unit-testable without a CV stack.

Reference-point real-world coords use the same frame as geometry.py:
  x = horizontal feet (catcher view, 0 = plate center), z = vertical feet.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .geometry import PLATE_HALF_FT


@dataclass
class Homography:
    """A pixel -> world (ft) projective map and its inverse."""
    H: np.ndarray          # 3x3, maps [px, py, 1] -> world (x, z)
    H_inv: np.ndarray      # 3x3, world (x, z) -> pixel

    def pixel_to_world(self, px: float, py: float) -> tuple[float, float]:
        v = self.H @ np.array([px, py, 1.0])
        return float(v[0] / v[2]), float(v[1] / v[2])

    def world_to_pixel(self, x: float, z: float) -> tuple[float, float]:
        v = self.H_inv @ np.array([x, z, 1.0])
        return float(v[0] / v[2]), float(v[1] / v[2])


def solve_homography(
    image_pts: np.ndarray | list,
    world_pts: np.ndarray | list,
) -> Homography:
    """Solve the 3x3 homography mapping image px -> world ft via DLT.

    image_pts / world_pts: matching (N>=4, 2) arrays. world is (x_ft, z_ft).
    """
    src = np.asarray(image_pts, dtype=float)
    dst = np.asarray(world_pts, dtype=float)
    if src.shape[0] < 4 or src.shape != dst.shape:
        raise ValueError("need >=4 matching image/world point pairs")

    A = []
    for (x, y), (u, v) in zip(src, dst):
        A.append([-x, -y, -1, 0, 0, 0, u * x, u * y, u])
        A.append([0, 0, 0, -x, -y, -1, v * x, v * y, v])
    A = np.asarray(A, dtype=float)

    # Solution is the right-singular vector of A with smallest singular value.
    _, _, Vt = np.linalg.svd(A)
    H = Vt[-1].reshape(3, 3)
    if abs(H[2, 2]) > 1e-12:
        H = H / H[2, 2]
    return Homography(H=H, H_inv=np.linalg.inv(H))


def home_plate_world_points() -> list[tuple[float, float]]:
    """Canonical world (x_ft, z_ft) of the home-plate front edge, at z=0.

    Two points (left/right front corners of the plate) define horizontal scale.
    Order matches `home_plate_pixel_points` expectations.
    """
    return [(-PLATE_HALF_FT, 0.0), (+PLATE_HALF_FT, 0.0)]


def strikezone_world_points(z_bot: float, z_top: float) -> list[tuple[float, float]]:
    """Four world corners of the strike-zone box (catcher view), ft.

    Order: top-left, top-right, bottom-right, bottom-left.
    """
    return [
        (-PLATE_HALF_FT, z_top),
        (+PLATE_HALF_FT, z_top),
        (+PLATE_HALF_FT, z_bot),
        (-PLATE_HALF_FT, z_bot),
    ]
