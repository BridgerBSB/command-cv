"""Per-game affine calibration + leave-one-out validation of pixel->inches.

The command miss needs the setup-glove pixel expressed in the same feet frame as
the Trackman actual (plate_x, plate_z). We get that per-game transform empirically:
each game is a fixed camera, and every pitch carries a known (plate_x, plate_z).
Pairing each pitch's CATCH-glove pixel with its Trackman location gives
correspondences; we fit an affine pixel->feet map per game.

VALIDATION (the honest part): leave-one-out. For each pitch, fit on the OTHER
pitches, predict this pitch's (plate_x, plate_z) from its catch-glove pixel, and
measure the error in inches. Low LOO error => the pixel->inches transform is
trustworthy. This is reported BEFORE any command miss is trusted.

Caveat modeled here: the catch plane sits slightly behind the plate plane (the
ball breaks between them), so catch-glove != exact plate crossing. That offset is
mostly absorbed by the fit; its per-pitch variation shows up as LOO residual —
i.e. it is INCLUDED in the reported error, not hidden.

Run:  python scripts/calibrate_report.py [--min-clips 5]
"""
import argparse
import csv
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.calibration import solve_affine

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ANN_DIR = os.path.join(ROOT, "data", "annotations")


def _load(pitcher="ogando"):
    """Return list of dicts: stem, game_pk, setup(px,py), catch(px,py), plate_x, plate_z."""
    man = {}
    mpath = os.path.join(ROOT, "data", "clips", f"{pitcher}_2026",
                         f"{pitcher}_2026_manifest.csv")
    for r in csv.DictReader(open(mpath)):
        man[r["stem"]] = r
    out = []
    for f in sorted(glob.glob(os.path.join(ANN_DIR, f"{pitcher}_*.json"))):
        a = json.load(open(f))
        stem = a["stem"]
        r = man.get(stem)
        if not r or not a.get("setup") or not a.get("catch"):
            continue
        try:
            px, pz = float(r["plate_x"]), float(r["plate_z"])
        except (TypeError, ValueError):
            continue
        out.append({
            "stem": stem, "game_pk": r["game_pk"],
            "setup": (a["setup"]["cx"], a["setup"]["cy"]),
            "catch": (a["catch"]["cx"], a["catch"]["cy"]),
            "plate_x": px, "plate_z": pz,
        })
    return out


def _by_game(rows):
    g = {}
    for r in rows:
        g.setdefault(r["game_pk"], []).append(r)
    return g


def validate_game(rows):
    """Leave-one-out: predict each pitch's (plate_x, plate_z) from its catch pixel.

    Returns per-pitch errors in inches (horizontal, vertical, radial).
    """
    img = np.array([r["catch"] for r in rows], dtype=float)
    wld = np.array([[r["plate_x"], r["plate_z"]] for r in rows], dtype=float)
    n = len(rows)
    errs = []
    for i in range(n):
        keep = [j for j in range(n) if j != i]
        if len(keep) < 3:
            return None
        aff = solve_affine(img[keep], wld[keep])
        pred = np.array(aff.pixel_to_world(*img[i]))
        d = (pred - wld[i]) * 12.0  # feet -> inches
        errs.append((abs(d[0]), abs(d[1]), float(np.hypot(*d))))
    return np.array(errs)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pitcher", default="ogando")
    p.add_argument("--min-clips", type=int, default=5)
    a = p.parse_args()

    rows = _load(a.pitcher)
    games = _by_game(rows)
    fit_games = {g: rs for g, rs in games.items() if len(rs) >= a.min_clips}

    print(f"{a.pitcher}: {len(rows)} usable clips across {len(games)} games; "
          f"{len(fit_games)} games have >= {a.min_clips} clips (calibratable)\n")

    all_rad = []
    print(f"{'game':>10} {'n':>3}  {'LOO horiz':>10} {'LOO vert':>9} {'LOO radial (RMSE in)':>22}")
    for g, rs in sorted(fit_games.items(), key=lambda kv: -len(kv[1])):
        e = validate_game(rs)
        if e is None:
            continue
        h = float(np.sqrt(np.mean(e[:, 0] ** 2)))
        v = float(np.sqrt(np.mean(e[:, 1] ** 2)))
        rad = float(np.sqrt(np.mean(e[:, 2] ** 2)))
        all_rad.extend(e[:, 2].tolist())
        print(f"{g:>10} {len(rs):>3}  {h:>8.1f}in {v:>7.1f}in {rad:>18.1f}in")

    if all_rad:
        ar = np.array(all_rad)
        print(f"\nPOOLED LOO radial error: mean {ar.mean():.1f}in  "
              f"median {np.median(ar):.1f}in  RMSE {np.sqrt(np.mean(ar**2)):.1f}in  "
              f"p90 {np.percentile(ar,90):.1f}in  (n={len(ar)})")
        print("\nInterpretation: this is how far off, in inches, the calibration is when "
              "predicting a pitch it was NOT fit on. It is the trust floor for any "
              "calibrated command miss.")
    else:
        print("\nNo game had enough clips to validate — need more pitches per game "
              "(auto-detect gloves on a full game's pitches).")


if __name__ == "__main__":
    main()
