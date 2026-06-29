import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.miss_methodb import GlovePx, compute_miss_b


def test_zero_displacement_is_on_target():
    g = GlovePx(800, 500, 40)
    m = compute_miss_b(g, g)
    assert m.miss_in == 0.0
    assert m.horiz_label == "on-line" and m.vert_label == "level"


def test_glove_size_scale_inches():
    # mitt 40px = 12in -> 0.3 in/px. Catch 40px right, 20px up (image y down).
    setup = GlovePx(800, 500, 40)
    catch = GlovePx(840, 480, 40)
    m = compute_miss_b(setup, catch, glove_diameter_in=12.0)
    assert math.isclose(m.scale_in_per_px, 12.0 / 40.0)
    assert math.isclose(m.horiz_in, 40 * 0.3)
    assert math.isclose(m.vert_in, 20 * 0.3)          # +20px up -> positive
    assert m.vert_label == "up"
    assert math.isclose(m.miss_in, math.hypot(12.0, 6.0))


def test_catch_below_setup_is_down():
    setup = GlovePx(800, 480, 40)
    catch = GlovePx(800, 520, 40)        # 40px lower in image -> pitch low
    m = compute_miss_b(setup, catch)
    assert m.vert_label == "down"


def test_arm_glove_side_mirrors_by_hand():
    setup = GlovePx(800, 500, 40)
    catch = GlovePx(830, 500, 40)
    r = compute_miss_b(setup, catch, pitcher_throws="R")
    l = compute_miss_b(setup, catch, pitcher_throws="L")
    assert {r.horiz_label, l.horiz_label} == {"arm-side", "glove-side"}
