"""End-to-end calibration test: planted glove world -> pixels -> recovered world.

Uses the synthetic frame's ground-truth pixel coords. The recovered intended
location must match the planted glove world to sub-inch precision, proving the
homography calibration + pipeline wiring are correct without any real video.
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.synth_frame import build
from src.pipeline import TrackmanPitch, process_pitch


def test_glove_world_round_trips_through_calibration():
    planted = (-0.5, 3.25)
    sf = build(glove_world=planted)

    tm = TrackmanPitch(pitch_id="TEST1", plate_x=0.40, plate_z=1.90,
                       pitcher_throws="R", sz_top=3.6, sz_bot=1.55)
    res = process_pitch(sf.detections, tm)

    # Recovered intended (glove) location should match the planted world coords.
    assert math.isclose(res.miss.intended_x, planted[0], abs_tol=0.02)  # < 0.25 in
    assert math.isclose(res.miss.intended_z, planted[1], abs_tol=0.02)

    # And the miss vector points from glove to the Trackman pitch location.
    assert math.isclose(res.miss.actual_x, 0.40)
    assert math.isclose(res.miss.actual_z, 1.90)
    assert res.miss.miss_in > 0


def test_two_plate_points_alone_is_rejected():
    sf = build()
    sf.detections.zone_pts = None  # only 2 plate points remain
    tm = TrackmanPitch(pitch_id="X", plate_x=0.0, plate_z=2.5)
    try:
        process_pitch(sf.detections, tm)
        assert False, "expected ValueError for <4 calibration points"
    except ValueError:
        pass
