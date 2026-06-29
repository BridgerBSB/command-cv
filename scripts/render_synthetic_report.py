"""Smoke test: synthetic miss -> Strike-Zone View, no DB / no video needed.

Run:  python scripts/render_synthetic_report.py
Produces output/synthetic_strike_zone_view.png to eyeball (render-and-look gate).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.geometry import StrikeZone, compute_miss
from src.report import render_strike_zone_view


def main() -> None:
    # Synthetic batter-specific zone (Trackman-style)
    zone = StrikeZone.from_trackman(sz_top=3.6, sz_bot=1.55)

    # Intended (glove) high & glove-side-left; actual low & to the right — a big miss,
    # mirroring the reference report's shape.
    intended_x, intended_z = -0.92, 3.78
    actual_x, actual_z = 0.74, 1.58

    miss = compute_miss(intended_x, intended_z, actual_x, actual_z,
                        pitcher_throws="R")
    print("miss:", miss.direction,
          f"(h={miss.horiz_in:+.1f} in, v={miss.vert_in:+.1f} in)")

    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "output")
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, "synthetic_strike_zone_view.png")
    render_strike_zone_view(
        miss, zone, out,
        pitch_label="Pitch #1767053249",
        subtitle="RHP · Synthetic test",
        approximate=True,
    )
    print("wrote", out)


if __name__ == "__main__":
    main()
