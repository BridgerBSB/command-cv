"""Full-chain demo on a REAL pitch from the MLB/MiLB Stats API.

Pulls a game's pitches, picks one, and renders a Strike-Zone View using the real
Trackman pX/pZ as 'actual'. The intended/glove target is a PLACEHOLDER here
(zone center) because the trained glove detector / a real clip is the missing
input — swap that one line for `detect.YoloDetector` output when the model lands.

Run:  python scripts/demo_statsapi_report.py [game_pk] [pitch_index]
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.geometry import StrikeZone, compute_miss
from src.ingest.statsapi import fetch_game_pitches
from src.report import render_strike_zone_view


def main() -> None:
    game_pk = int(sys.argv[1]) if len(sys.argv) > 1 else 814841
    idx = int(sys.argv[2]) if len(sys.argv) > 2 else 0

    recs = fetch_game_pitches(game_pk)
    if not recs:
        print("no tracked pitches"); return
    rec = recs[idx]
    tm = rec.trackman
    zone = StrikeZone.from_trackman(tm.sz_top, tm.sz_bot)

    # PLACEHOLDER intended target: middle-middle. Replace with detected glove.
    intended_x, intended_z = 0.0, (zone.z_top + zone.z_bot) / 2.0

    miss = compute_miss(intended_x, intended_z, tm.plate_x, tm.plate_z,
                        pitcher_throws=tm.pitcher_throws)
    print(f"{tm.pitcher_name} {tm.pitch_type}: actual pX={tm.plate_x:+.2f} "
          f"pZ={tm.plate_z:.2f} | {miss.direction}")
    print("  NOTE: intended target is a placeholder (middle-middle), not a real glove.")

    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "output")
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, "demo_real_pitch_report.png")
    render_strike_zone_view(
        miss, zone, out,
        pitch_label=f"Pitch #{tm.pitch_id[:8]}",
        subtitle=f"{tm.pitcher_name} · {tm.pitch_type} · {tm.pitcher_throws}HP",
        approximate=True,
    )
    print("wrote", out)


if __name__ == "__main__":
    main()
