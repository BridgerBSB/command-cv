"""Season command board: aggregate command miss (inches) by pitch type across ALL
of a pitcher's games. Each game is calibrated independently (its own camera) with
the RANSAC pixel->inches fit, then per-pitch misses are pooled.

Reads the cached auto-detections (output/auto_detections.jsonl) + the manifest.
Run AFTER scripts/tighten.py has detected the pitcher's clips.

  python scripts/command_board.py --pitcher ogando [--min-game 8]
"""
import argparse
import csv
import json
import os
import sys
import collections

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.calibration import solve_affine_ransac

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pitcher", default="ogando")
    p.add_argument("--min-game", type=int, default=8,
                   help="min usable auto-detections in a game to calibrate it")
    a = p.parse_args()

    cache_p = os.path.join(ROOT, "output", "auto_detections.jsonl")
    cache = {json.loads(l)["stem"]: json.loads(l) for l in open(cache_p)}
    man = {r["stem"]: r for r in csv.DictReader(
        open(os.path.join(ROOT, "data", "clips", f"{a.pitcher}_2026",
                          f"{a.pitcher}_2026_manifest.csv")))}

    by_game = collections.defaultdict(list)
    for stem, r in man.items():
        c = cache.get(stem)
        if not c or not c.get("ok"):
            continue
        try:
            px, pz = float(r["plate_x"]), float(r["plate_z"])
        except (TypeError, ValueError):
            continue
        by_game[r["game_pk"]].append((stem, r["pitch_type"], c["setup"], c["catch"], px, pz))

    misses = []          # (pitch_type, miss_in)
    n_games_used = 0
    for g, rows in by_game.items():
        if len(rows) < a.min_game:
            continue
        catch = np.array([x[3] for x in rows], float)
        wld = np.array([[x[4], x[5]] for x in rows], float)
        aff, mask = solve_affine_ransac(catch, wld)
        if mask.sum() < 5:
            continue
        n_games_used += 1
        for stem, pt, setup, cxy, px, pz in rows:
            tx, tz = aff.pixel_to_world(*setup)
            misses.append((pt, float(np.hypot((tx - px) * 12, (tz - pz) * 12))))

    if not misses:
        print("no calibratable games yet"); return

    print(f"\n{a.pitcher.title()} 2026 SEASON command board "
          f"({len(misses)} pitches, {n_games_used} games)\n")
    by_pt = collections.defaultdict(list)
    for pt, m in misses:
        by_pt[pt].append(m)
    print(f"{'pitch':16s} {'n':>4} {'avg in':>8} {'median in':>10}")
    for pt, ms in sorted(by_pt.items(), key=lambda kv: -len(kv[1])):
        print(f"{pt:16s} {len(ms):>4} {np.mean(ms):>7.1f} {np.median(ms):>10.1f}")
    allm = np.array([m for _, m in misses])
    print(f"\n{'OVERALL':16s} {len(allm):>4} {allm.mean():>7.1f} {np.median(allm):>10.1f}")


if __name__ == "__main__":
    main()
