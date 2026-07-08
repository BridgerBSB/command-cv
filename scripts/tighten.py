"""Hands-off TIGHTEN: auto-detect setup+catch gloves on a game's clips (our fast
mitt model, CPU-fine) and calibrate pixel->inches with a ROBUST (RANSAC) fit that
rejects the bad auto-catches. No GPU, no clicking.

Setup auto-detection is validated (2.3px). Catch auto (glove-motion) is ~half
clean with some catastrophic misses -> RANSAC over the per-game catch<->plate_x/z
correspondences keeps only the consensus inliers, so more pitches TIGHTEN the fit
instead of the outliers wrecking it.

Resumable: each clip's auto-detection is cached to output/auto_detections.jsonl,
so a killed run picks up where it left off on re-run (the CPU can be slow).

Run:
  python scripts/tighten.py --pitcher ogando                 # all pulled games
  python scripts/tighten.py --pitcher ogando --game 822515   # one game
"""
import argparse
import csv
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.auto_glove import detect_setup_catch
from src.calibration import solve_affine_ransac

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "output", "auto_detections.jsonl")


def _manifest(pitcher):
    p = os.path.join(ROOT, "data", "clips", f"{pitcher}_2026", f"{pitcher}_2026_manifest.csv")
    return {r["stem"]: r for r in csv.DictReader(open(p))}


def _cache_load():
    d = {}
    if os.path.exists(CACHE):
        for ln in open(CACHE):
            try:
                r = json.loads(ln); d[r["stem"]] = r
            except Exception:
                pass
    return d


def auto_detect_all(pitcher, model, only_game=None):
    """Auto-detect setup+catch for every clip; cache per-clip (resumable)."""
    man = _manifest(pitcher)
    cache = _cache_load()
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    clips = sorted(glob.glob(os.path.join(ROOT, "data", "clips", f"{pitcher}_2026", f"{pitcher}_*.mp4")))
    with open(CACHE, "a") as out:
        for clip in clips:
            stem = os.path.splitext(os.path.basename(clip))[0]
            r = man.get(stem)
            if not r:
                continue
            if only_game and str(r["game_pk"]) != str(only_game):
                continue
            if stem in cache:
                continue
            d = detect_setup_catch(clip, model)
            rec = {"stem": stem, "game_pk": r["game_pk"], "ok": d.get("ok", False)}
            if d.get("ok"):
                rec.update(setup=d["setup"][:2], catch=d["catch"][:2])
            out.write(json.dumps(rec) + "\n"); out.flush()
            cache[stem] = rec
            print(f"  {stem}: {'ok' if rec['ok'] else 'skip'}")
    return cache


def tighten(pitcher, only_game=None):
    man = _manifest(pitcher)
    cache = _cache_load()
    # assemble per-game (setup_px, catch_px, plate_x, plate_z)
    games = {}
    for stem, rec in cache.items():
        if not rec.get("ok"):
            continue
        r = man.get(stem)
        if not r or (only_game and str(r["game_pk"]) != str(only_game)):
            continue
        try:
            px, pz = float(r["plate_x"]), float(r["plate_z"])
        except (TypeError, ValueError):
            continue
        games.setdefault(r["game_pk"], []).append(
            (stem, rec["setup"], rec["catch"], px, pz))

    print(f"\n{'game':>10} {'n':>3} {'inliers':>8} {'LOO median in':>14} {'LOO RMSE in':>12}")
    pooled = []
    for g, rows in sorted(games.items(), key=lambda kv: -len(kv[1])):
        if len(rows) < 5:
            continue
        catch = np.array([r[2] for r in rows], float)
        wld = np.array([[r[3], r[4]] for r in rows], float)
        aff, mask = solve_affine_ransac(catch, wld)
        nin = int(mask.sum())
        # LOO on inliers only
        ci, wi = catch[mask], wld[mask]
        errs = []
        for i in range(len(ci)):
            keep = [j for j in range(len(ci)) if j != i]
            if len(keep) < 3:
                continue
            a2, _ = solve_affine_ransac(ci[keep], wi[keep])
            d = (np.array(a2.pixel_to_world(*ci[i])) - wi[i]) * 12.0
            errs.append(float(np.hypot(*d)))
        if errs:
            e = np.array(errs); pooled.extend(errs)
            print(f"{g:>10} {len(rows):>3} {nin:>8} {np.median(e):>12.1f}in {np.sqrt(np.mean(e**2)):>10.1f}in")

    if pooled:
        e = np.array(pooled)
        print(f"\nTIGHTENED LOO (RANSAC inliers): median {np.median(e):.1f}in  "
              f"RMSE {np.sqrt(np.mean(e**2)):.1f}in  n={len(e)}")
        print("Baseline (5-7 hand pts/game): median 4.2in. Lower = tightened.")
    else:
        print("\nNo game had >=5 usable auto-detections yet.")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pitcher", default="ogando")
    p.add_argument("--game", default=None, help="game_pk to scope (default: all pulled games)")
    p.add_argument("--detect-only", action="store_true")
    a = p.parse_args()

    from ultralytics import YOLO
    model = YOLO(os.path.join(ROOT, "data", "yolo", "runs", "mitt", "weights", "best.pt"))
    print("auto-detecting setup+catch (resumable)...")
    auto_detect_all(a.pitcher, model, only_game=a.game)
    if not a.detect_only:
        tighten(a.pitcher, only_game=a.game)


if __name__ == "__main__":
    main()
