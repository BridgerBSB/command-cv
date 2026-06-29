import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.geometry import IN_PER_FT, PLATE_HALF_FT, StrikeZone, compute_miss


def test_perfect_command_is_zero_miss():
    m = compute_miss(0.0, 2.5, 0.0, 2.5, pitcher_throws="R")
    assert m.miss_in == 0.0
    assert m.horiz_label == "on-line"
    assert m.vert_label == "level"


def test_miss_magnitude_in_inches():
    # 1 ft right, 1 ft up => sqrt(2)*12 inches
    m = compute_miss(0.0, 2.0, 1.0, 3.0, pitcher_throws="R")
    assert math.isclose(m.miss_in, math.hypot(12.0, 12.0), rel_tol=1e-9)
    assert math.isclose(m.horiz_in, 12.0)
    assert math.isclose(m.vert_in, 12.0)


def test_vertical_labels():
    up = compute_miss(0.0, 2.0, 0.0, 3.0)
    assert up.vert_label == "up"
    down = compute_miss(0.0, 3.0, 0.0, 2.0)
    assert down.vert_label == "down"


def test_arm_glove_side_mirrors_by_hand():
    # Same physical horizontal miss labels opposite for RHP vs LHP.
    rhp = compute_miss(0.0, 2.5, 0.5, 2.5, pitcher_throws="R")
    lhp = compute_miss(0.0, 2.5, 0.5, 2.5, pitcher_throws="L")
    assert {rhp.horiz_label, lhp.horiz_label} == {"arm-side", "glove-side"}


def test_zone_dimensions():
    z = StrikeZone.from_trackman(sz_top=3.5, sz_bot=1.5)
    assert math.isclose(z.width_ft, 2 * PLATE_HALF_FT)
    assert math.isclose(z.height_ft, 2.0)
