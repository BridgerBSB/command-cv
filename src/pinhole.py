"""Pinhole camera for broadcast baseball video: world feet <-> screen pixels.

Reimplemented from reading OpenCommand's poselib.py (github.com/tomdoyo/open-command,
commit 984d26e1, CC BY-NC-SA 4.0). Their code is not vendored; the geometry is
written out again here so it can be checked term by term against their published
camera poses (scripts/parity_opencommand_pose.py).

World frame (Statcast): feet, origin at the back tip of home plate, x = catcher's
right, y = toward the pitcher, z = up.

Camera: 7 numbers. Position C = (Cx, Cy, Cz) and orientation (pan, tilt, roll)
relative to a base frame aimed at (0, 0, 3), plus focal length f in pixels.
Cy is PINNED at 400 ft by the caller: camera depth is degenerate against focal
length (move back and zoom in, same pixels).
"""
import numpy as np

W, H = 1280, 720
CY_FT = 400.0
AIM = np.array([0.0, 0.0, 3.0])


def zone_y(season):
    """Depth (ft) of the plane the broadcast draws the zone box on: the plate front
    (17/12 ft) through 2025, mid-plate (8.5/12 ft) from 2026 under ABS."""
    return 17 / 12 if int(season) <= 2025 else 8.5 / 12


def _rot(axis, angle):
    """Rodrigues rotation by `angle` radians about unit `axis`."""
    a = axis / np.linalg.norm(axis)
    k = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(angle) * k + (1 - np.cos(angle)) * (k @ k)


def camera(C, pan, tilt, roll):
    """(right, up, fwd) unit vectors of the camera at C."""
    fwd = AIM - np.asarray(C, float)
    fwd = fwd / np.linalg.norm(fwd)
    right = np.array([fwd[1], -fwd[0], 0.0])          # fwd x z_hat
    right = right / np.linalg.norm(right)
    up = np.cross(right, fwd)
    R = _rot(fwd, roll) @ _rot(right, tilt) @ _rot(up, pan)   # pan, then tilt, then roll
    return R @ right, R @ up, R @ fwd


def project(C, pan, tilt, roll, f, X):
    """World point(s) (N, 3) or (3,) -> pixel(s)."""
    right, up, fwd = camera(C, pan, tilt, roll)
    rel = np.atleast_2d(X) - np.asarray(C, float)
    depth = rel @ fwd
    px = np.stack([W / 2 + f * (rel @ right) / depth,
                   H / 2 - f * (rel @ up) / depth], axis=-1)
    return px[0] if np.ndim(X) == 1 else px


def unproject(C, pan, tilt, roll, f, pix, y_plane):
    """Pixel(s) -> world (x, z) where the pixel's ray crosses the plane y = y_plane."""
    right, up, fwd = camera(C, pan, tilt, roll)
    p = np.atleast_2d(pix).astype(float)
    ray = ((p[:, :1] - W / 2) / f) * right + ((H / 2 - p[:, 1:]) / f) * up + fwd
    C = np.asarray(C, float)
    s = (y_plane - C[1]) / ray[:, 1]
    pt = C + s[:, None] * ray
    out = pt[:, [0, 2]]
    return tuple(out[0]) if np.ndim(pix) == 1 else out


def statcast_traj(p):
    """Statcast 9-parameter trajectory (x0..az, a = full acceleration) -> t -> (x, y, z)."""
    def traj(t):
        t = np.asarray(t, float)
        return np.stack([p["x0"] + p["vx0"] * t + 0.5 * p["ax"] * t ** 2,
                         p["y0"] + p["vy0"] * t + 0.5 * p["ay"] * t ** 2,
                         p["z0"] + p["vz0"] * t + 0.5 * p["az"] * t ** 2], axis=-1)
    return traj
