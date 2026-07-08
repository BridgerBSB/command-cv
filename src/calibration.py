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


@dataclass
class Affine:
    """A pixel -> world (ft) affine map: world = A @ [px, py, 1].

    Affine (6 DOF: scale, rotation, shear, translation) is more robust than a
    full projective homography (8 DOF) when only a handful of correspondences
    are available, at the cost of not modeling perspective foreshortening. For
    a narrow region of the frame (the receive area) over a fixed camera it is a
    good approximation.
    """
    A: np.ndarray          # 2x3, maps [px, py, 1] -> (x_ft, z_ft)

    def pixel_to_world(self, px: float, py: float) -> tuple[float, float]:
        v = self.A @ np.array([px, py, 1.0])
        return float(v[0]), float(v[1])


def solve_affine(image_pts: np.ndarray | list, world_pts: np.ndarray | list) -> Affine:
    """Least-squares affine mapping image px -> world ft.

    image_pts / world_pts: matching (N>=3, 2) arrays. world is (x_ft, z_ft).
    Solves world = A @ [px, py, 1] in the least-squares sense (over-determined
    for N>3), so more correspondences reduce noise instead of over-fitting.
    """
    src = np.asarray(image_pts, dtype=float)
    dst = np.asarray(world_pts, dtype=float)
    if src.shape[0] < 3 or src.shape != dst.shape:
        raise ValueError("need >=3 matching image/world point pairs")
    # [px, py, 1] design matrix; solve for the 3x2 coefficient block, transpose to 2x3.
    M = np.hstack([src, np.ones((src.shape[0], 1))])   # (N, 3)
    coef, *_ = np.linalg.lstsq(M, dst, rcond=None)      # (3, 2)
    return Affine(A=coef.T)                              # (2, 3)


def solve_affine_ransac(image_pts, world_pts, thresh_in: float = 7.0,
                        n_iter: int = 300, seed: int = 7):
    """Robust affine via RANSAC — rejects outlier correspondences.

    Auto-detected catch gloves are ~half clean + some catastrophic misses. A
    plain least-squares fit is dragged by the bad ones; RANSAC samples minimal
    3-point fits, scores each by how many points land within `thresh_in` INCHES,
    keeps the largest consensus, and refits on those inliers only.

    Returns (Affine, inlier_mask). inlier_mask is a bool array over the inputs.
    """
    src = np.asarray(image_pts, dtype=float)
    dst = np.asarray(world_pts, dtype=float)
    n = src.shape[0]
    if n < 3:
        raise ValueError("need >=3 points")
    rng = np.random.default_rng(seed)
    thresh_ft = thresh_in / 12.0
    best_mask, best_count = None, -1
    for _ in range(n_iter):
        idx = rng.choice(n, size=3, replace=False)
        try:
            aff = solve_affine(src[idx], dst[idx])
        except Exception:
            continue
        M = np.hstack([src, np.ones((n, 1))])
        pred = (aff.A @ M.T).T                       # (n, 2) world ft
        err = np.hypot(*(pred - dst).T)
        mask = err <= thresh_ft
        if mask.sum() > best_count:
            best_count, best_mask = int(mask.sum()), mask
    if best_mask is None or best_count < 3:
        return solve_affine(src, dst), np.ones(n, dtype=bool)
    return solve_affine(src[best_mask], dst[best_mask]), best_mask


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
