"""src/pinhole.py geometry: projection and back-projection must be exact inverses on a
depth plane, the camera basis must be orthonormal, and the aim point must land at
frame center when pan/tilt/roll are zero. DB-free, no video."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src import pinhole as ph  # noqa: E402


def test_round_trip_exact_on_depth_plane():
    rng = np.random.default_rng(0)
    worst = 0.0
    for _ in range(200):
        C = np.array([rng.uniform(-30, 30), ph.CY_FT, rng.uniform(10, 50)])
        pan, tilt, roll = rng.uniform(-0.05, 0.05, 3)
        f, yp = rng.uniform(8000, 20000), rng.uniform(-2, 2)
        X = np.c_[rng.uniform(-1.5, 1.5, 20), np.full(20, yp), rng.uniform(1, 4, 20)]
        xz = ph.unproject(C, pan, tilt, roll, f, ph.project(C, pan, tilt, roll, f, X), yp)
        worst = max(worst, float(np.abs(xz - X[:, [0, 2]]).max()))
    assert worst * 12 < 1e-9, worst


def test_basis_orthonormal():
    B = np.array(ph.camera(np.array([-13.9, ph.CY_FT, 32.6]), 0.01, -0.004, 0.001))
    assert np.allclose(B @ B.T, np.eye(3), atol=1e-12)


def test_aim_point_is_frame_center():
    px = ph.project(np.array([-13.9, ph.CY_FT, 32.6]), 0, 0, 0, 16000, ph.AIM)
    assert np.allclose(px, [ph.W / 2, ph.H / 2], atol=1e-9)


def test_catchers_right_is_screen_left_from_cf():
    """From a centre-field camera, +x (catcher's right) must appear LEFT of centre.
    OpenCommand pairs world x=-17/24 with the box's x_right pixel, so a sign slip
    here would mirror every miss."""
    C = np.array([0.0, ph.CY_FT, 30.0])
    left = ph.project(C, 0, 0, 0, 16000, np.array([1.0, 0.0, 3.0]))
    assert left[0] < ph.W / 2


def test_zone_plane_moves_for_abs():
    assert ph.zone_y(2025) == 17 / 12 and ph.zone_y(2026) == 8.5 / 12
