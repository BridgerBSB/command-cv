"""trajectory_truth() flattens the authed payload's 9-param flight polynomial.

Shape copied from a real Ogando 821922 payload (Sep 10 2026). Two properties:
  1. the polynomial evaluated at zone_time reproduces `location` (the plate
     crossing we already ship as plate_x/plate_z) -- proves the coefficient
     ORDER (c0 + c1 t + c2 t^2) is read correctly;
  2. a payload with no trajectory yields None in every slot, never 0.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from pull_pitcher_clips import _TRAJ_COLS, pitch_truth, trajectory_truth  # noqa: E402

_PAYLOAD = {
    "metaData": {"stat": {"play": {"pitchData": {"strikeZoneTop": 3.4, "strikeZoneBottom": 1.6},
                                   "details": {"type": {"description": "Sinker"},
                                               "pitchHand": {"code": "R"}}}}},
    "ballSegments": {"pitchSegment": {
        "trajectoryData": {
            "trajectoryPolynomialX": [-2.770278292574579, 8.744507015253154, -6.981942628172895],
            "trajectoryPolynomialY": [54.13385722600862, -138.53752947333456, 15.077210263605988],
            "trajectoryPolynomialZ": [5.40401713215662, 0.932266192992694, -7.878013228372806],
            "validTimeInterval": [0.0, 0.42],
            "zoneTime": 0.39775637533962316,
            "location": {"x": -0.39670895770009296, "y": 1.4166666667, "z": 4.528450424446956},
        },
        "releaseData": {"releasePosition": {"x": -2.7293747785, "y": 53.4837246704, "z": 5.4082206601}},
    }},
}


def _ev(c0, c1, c2, t):
    return c0 + c1 * t + c2 * t * t


def test_polynomial_reproduces_plate_crossing():
    tr = pitch_truth(_PAYLOAD)
    zt = tr["zone_time"]
    x = _ev(tr["traj_x0"], tr["traj_x1"], tr["traj_x2"], zt)
    z = _ev(tr["traj_z0"], tr["traj_z1"], tr["traj_z2"], zt)
    assert abs(x - tr["plate_x"]) < 1e-9, (x, tr["plate_x"])
    assert abs(z - tr["plate_z"]) < 1e-9, (z, tr["plate_z"])
    y = _ev(tr["traj_y0"], tr["traj_y1"], tr["traj_y2"], zt)
    assert abs(y - 17 / 12) < 0.01, y          # plate front is 17in from the tip
    assert (tr["t_start"], tr["t_end"]) == (0.0, 0.42)
    assert abs(tr["rel_y"] - 53.48) < 0.01


def test_missing_trajectory_is_none_not_zero():
    tr = trajectory_truth({}, {})
    assert set(tr) == set(_TRAJ_COLS)
    assert all(v is None for v in tr.values())


def test_partial_polynomial_does_not_raise():
    tr = trajectory_truth({"trajectoryPolynomialX": [1.0, 2.0]}, {})
    assert (tr["traj_x0"], tr["traj_x1"], tr["traj_x2"]) == (1.0, 2.0, None)
