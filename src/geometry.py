"""Strike-zone geometry + miss-vector math, in the Trackman coordinate frame.

All horizontal/vertical positions are in FEET in Trackman's plate frame:
  - x : horizontal, 0 = center of plate, measured from the CATCHER's view.
  - z : vertical, 0 = ground, up positive.

This module is intentionally cv2-free and pure-numeric so the report engine and
unit tests run without any computer-vision dependency.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

# ---- physical constants -----------------------------------------------------

PLATE_WIDTH_IN = 17.0                 # rulebook home plate width
PLATE_HALF_FT = (PLATE_WIDTH_IN / 2.0) / 12.0   # 0.70833... ft
IN_PER_FT = 12.0

# Default strike-zone vertical bounds (ft) when a Trackman row omits them.
# Real values are batter-specific (sz_top / sz_bot columns) and should be passed in.
DEFAULT_SZ_TOP_FT = 3.5
DEFAULT_SZ_BOT_FT = 1.5

# ---- arm-side / glove-side label convention --------------------------------
# Sign of "arm-side" in the plate_x frame, per pitcher hand.
# CONFIRMED Jul 9 2026 via a physical anchor: Ogando clip si_12 is a Hit-By-Pitch
# with plate_x = -2.15 that struck a RIGHT-handed batter. A RHB stands on the
# 3B side, so NEGATIVE plate_x = 3B side. A RHP's arm-side run goes toward the
# 3B side / the RHB, so RHP arm-side = negative plate_x -> sign -1.0 (LHP mirrors).
# horiz = actual_x - intended_x (below) is in this same plate_x frame, so an
# arm-side miss for a RHP has horiz < 0, and (horiz * -1.0) > 0 -> "arm-side".
_ARM_SIDE_X_SIGN = {
    "R": -1.0,   # RHP: arm-side = 3B side = negative plate_x (si_12 HBP-on-RHB)
    "L": +1.0,   # LHP: mirror = 1B side = positive plate_x
}


@dataclass(frozen=True)
class StrikeZone:
    """Strike-zone box in Trackman feet (catcher view)."""
    x_left: float = -PLATE_HALF_FT
    x_right: float = +PLATE_HALF_FT
    z_bot: float = DEFAULT_SZ_BOT_FT
    z_top: float = DEFAULT_SZ_TOP_FT

    @classmethod
    def from_trackman(cls, sz_top: float | None, sz_bot: float | None) -> "StrikeZone":
        return cls(
            z_top=float(sz_top) if sz_top is not None else DEFAULT_SZ_TOP_FT,
            z_bot=float(sz_bot) if sz_bot is not None else DEFAULT_SZ_BOT_FT,
        )

    @property
    def width_ft(self) -> float:
        return self.x_right - self.x_left

    @property
    def height_ft(self) -> float:
        return self.z_top - self.z_bot


@dataclass(frozen=True)
class MissResult:
    """The command-miss between intended (glove) and actual (Trackman)."""
    intended_x: float          # ft
    intended_z: float          # ft
    actual_x: float            # ft
    actual_z: float            # ft
    miss_in: float             # total miss magnitude, inches
    horiz_in: float            # signed horizontal miss (actual - intended), inches
    vert_in: float             # signed vertical miss (actual - intended), inches
    horiz_label: str           # "arm-side" | "glove-side" | "on-line"
    vert_label: str            # "up" | "down" | "level"
    direction: str             # human summary, e.g. "2.8 in arm-side / up"


def compute_miss(
    intended_x: float,
    intended_z: float,
    actual_x: float,
    actual_z: float,
    pitcher_throws: str = "R",
    deadband_in: float = 0.75,
) -> MissResult:
    """Miss vector from intended (glove) to actual (pitch), in inches + labels.

    `pitcher_throws` ("R"/"L") only affects the arm-side/glove-side *word*; the
    numbers are convention-independent. `deadband_in` suppresses a direction word
    when the component is within ~1 ball-radius of dead-on.
    """
    horiz_ft = actual_x - intended_x
    vert_ft = actual_z - intended_z
    horiz_in = horiz_ft * IN_PER_FT
    vert_in = vert_ft * IN_PER_FT
    miss_in = math.hypot(horiz_in, vert_in)

    hand = (pitcher_throws or "R").upper()[:1]
    arm_sign = _ARM_SIDE_X_SIGN.get(hand, +1.0)

    if abs(horiz_in) <= deadband_in:
        horiz_label = "on-line"
    else:
        # If the miss is in the same x-direction as arm-side, it's arm-side.
        horiz_label = "arm-side" if (horiz_in * arm_sign) > 0 else "glove-side"

    if abs(vert_in) <= deadband_in:
        vert_label = "level"
    else:
        vert_label = "up" if vert_in > 0 else "down"

    parts = []
    if horiz_label not in ("on-line",):
        parts.append(horiz_label)
    if vert_label not in ("level",):
        parts.append(vert_label)
    where = " / ".join(parts) if parts else "on target"
    direction = f"{miss_in:.1f} in {where}"

    return MissResult(
        intended_x=intended_x, intended_z=intended_z,
        actual_x=actual_x, actual_z=actual_z,
        miss_in=miss_in, horiz_in=horiz_in, vert_in=vert_in,
        horiz_label=horiz_label, vert_label=vert_label, direction=direction,
    )
