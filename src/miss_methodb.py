"""Method B — self-contained command miss from a single side-angle clip.

intended = catcher's mitt at the SETUP frame (target held before the pitch).
actual   = catcher's mitt at the CATCH frame (where he received the ball).
miss     = the image displacement between them, scaled to inches using the
           mitt's own known real size as the ruler (pixels-per-inch at the
           glove's depth) — so a side / down-the-line camera doesn't corrupt it.

This is the right method when there is no center-field camera and no Trackman
row. Vertical miss comes from image-y (inverted: image y grows downward),
horizontal from image-x. The arm-side / glove-side WORD depends on pitcher hand
and which image direction is "toward the plate's right" — flagged VERIFY, same
as geometry.py. Magnitude is convention-independent.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

DEFAULT_GLOVE_DIAMETER_IN = 12.0   # catcher's mitt ~12-12.5"

# Does increasing image-x correspond to arm-side for this hand? The camera here
# looks down a baseline; sign must be confirmed against one known pitch. VERIFY.
_ARM_SIDE_DX_SIGN = {"R": +1.0, "L": -1.0}


@dataclass(frozen=True)
class GlovePx:
    """A mitt detection in pixels."""
    cx: float
    cy: float
    diameter_px: float


@dataclass(frozen=True)
class MissB:
    miss_in: float
    horiz_in: float      # signed, +/- in inches (image-x * scale)
    vert_in: float       # signed, + = pitch ended up HIGHER than target
    horiz_label: str
    vert_label: str
    direction: str
    scale_in_per_px: float


def compute_miss_b(
    setup: GlovePx,
    catch: GlovePx,
    pitcher_throws: str = "R",
    glove_diameter_in: float = DEFAULT_GLOVE_DIAMETER_IN,
    deadband_in: float = 1.0,
) -> MissB:
    """Miss in inches from setup->catch mitt displacement, glove-size scaled."""
    # Average the two diameter estimates for the pixels->inches ruler.
    diam_px = (setup.diameter_px + catch.diameter_px) / 2.0
    if diam_px <= 0:
        raise ValueError("glove diameter in pixels must be > 0")
    scale = glove_diameter_in / diam_px        # inches per pixel at glove depth

    dx_px = catch.cx - setup.cx
    dy_px = catch.cy - setup.cy
    horiz_in = dx_px * scale
    vert_in = -dy_px * scale                   # invert: image y grows downward
    miss_in = math.hypot(horiz_in, vert_in)

    hand = (pitcher_throws or "R").upper()[:1]
    arm_sign = _ARM_SIDE_DX_SIGN.get(hand, +1.0)
    if abs(horiz_in) <= deadband_in:
        horiz_label = "on-line"
    else:
        horiz_label = "arm-side" if (horiz_in * arm_sign) > 0 else "glove-side"
    if abs(vert_in) <= deadband_in:
        vert_label = "level"
    else:
        vert_label = "up" if vert_in > 0 else "down"

    parts = [p for p in (horiz_label, vert_label) if p not in ("on-line", "level")]
    direction = f"{miss_in:.1f} in " + (" / ".join(parts) if parts else "on target")
    return MissB(miss_in=miss_in, horiz_in=horiz_in, vert_in=vert_in,
                 horiz_label=horiz_label, vert_label=vert_label,
                 direction=direction, scale_in_per_px=scale)
