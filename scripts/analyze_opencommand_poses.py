"""Per-park / per-game geometry accuracy of OpenCommand's published camera poses.

OpenCommand (tomdoyo/open-command, CC BY-NC-SA) stamps `park` on every pose row
and publishes a single POOLED accuracy median across all 30 MLB parks. It never
breaks that number down by park, so a camera problem confined to one stadium is
invisible at the grain they report.

This script runs that breakdown, because "are our inches right from a different
camera and a different view" is the question we have to answer for ourselves at
A+, and their data is the only place it can currently be measured at scale.

What it answers:
  1. How far apart are the 30 parks on systematic bias and on precision?
  2. Is the residual a property of the PARK (a per-park correction is available)
     or of the GAME (the per-game mount solve is already the right unit)?
  3. Does error grow off-axis or at zoom extremes? That is the affine-vs-pinhole
     argument, and it is what "a different view" means in practice.
  4. How many ball observations does a pose actually need?

Input (~74 MB, not vendored; their license is non-commercial, we read not ship):
  https://huggingface.co/datasets/tomdoyo/open-command/resolve/main/2025/camera_poses.csv.gz

Run:
  python scripts/analyze_opencommand_poses.py <path-to-camera_poses.csv.gz>

Findings as of 2026-09-01 are written up in
docs/2026-09-01-opencommand-v12-and-cross-camera-accuracy.md.
"""
import os
import sys

import pandas as pd

MIN_CLIPS_PER_GAME = 50   # a game needs a real sample before its median means anything


def load(src):
    assert os.path.isfile(src), f"not a file: {src}"
    df = pd.read_csv(src)
    v = df.dropna(subset=["balltp_err_x_in", "balltp_err_z_in"])
    print(f"loaded {len(df):,} pose rows, {len(v):,} ({100 * len(v) / len(df):.1f}%) "
          f"carry a ball-at-plate check | {v.park.nunique()} parks | {v.game_pk.nunique():,} games")
    return v


def pooled(v):
    """The grain OpenCommand publishes."""
    print("\nPOOLED (their published grain)")
    print(f"  bias      X {v.balltp_err_x_in.median():+.3f} in   Z {v.balltp_err_z_in.median():+.3f} in")
    print(f"  |err| med X {v.balltp_err_x_in.abs().median():.3f} in   Z {v.balltp_err_z_in.abs().median():.3f} in")
    print(f"  reproj_px median {v.reproj_px.median():.2f}  p90 {v.reproj_px.quantile(.9):.2f}")


def per_park(v):
    """Signed bias (systematic, the dangerous one) and |err| (precision), per park."""
    g = v.groupby("park")
    out = pd.DataFrame({
        "n": g.size(),
        "bias_x": g.balltp_err_x_in.median(),
        "bias_z": g.balltp_err_z_in.median(),
        "abs_x": g.balltp_err_x_in.apply(lambda s: s.abs().median()),
        "abs_z": g.balltp_err_z_in.apply(lambda s: s.abs().median()),
        "reproj_med": g.reproj_px.median(),
        "reproj_p90": g.reproj_px.quantile(.90),
    }).sort_values("bias_z")
    pd.set_option("display.width", 200)
    print(f"\nPER PARK (n={len(out)}), sorted by Z bias")
    print(out.round(3).to_string())

    print("\nSPREAD ACROSS PARKS (never published)")
    for c in ("bias_x", "bias_z", "abs_x", "abs_z", "reproj_med", "reproj_p90"):
        s = out[c]
        print(f"  {c:11s} min {s.min():+7.3f}  median {s.median():+7.3f}  "
              f"max {s.max():+7.3f}   spread {s.max() - s.min():6.3f}")
    print(f"\n  corr(park X bias, park Z bias) = "
          f"{out.bias_x.corr(out.bias_z):+.3f}   (a shared systematic, not independent noise)")
    return out


def park_vs_game(v):
    """Variance decomposition: is the bias the camera, or the day?"""
    key = ["park", "game_pk"]
    sizes = v.groupby(key).size()
    pg = v.groupby(key)[["balltp_err_x_in", "balltp_err_z_in"]].median()[sizes >= MIN_CLIPS_PER_GAME]
    print(f"\nPARK vs GAME  (games with >= {MIN_CLIPS_PER_GAME} checked clips: {len(pg):,})")
    for ax in ("balltp_err_x_in", "balltp_err_z_in"):
        between = pg[ax].groupby("park").mean().std()
        within = pg[ax].groupby("park").std().mean()
        icc = between ** 2 / (between ** 2 + within ** 2)
        print(f"  {ax}: between-park sd {between:.3f} in, within-park (game to game) sd "
              f"{within:.3f} in -> {100 * icc:.1f}% of the variance is PARK")


def view_dependence(v):
    """Does accuracy degrade off-axis or at zoom extremes? An affine would."""
    v = v.assign(absx=v.balltp_err_x_in.abs(), absz=v.balltp_err_z_in.abs())
    for col, label in (("pan", "|pan| (how far off-axis the camera is aimed)"),
                       ("f_px", "focal length (zoom)")):
        base = v[col].abs() if col == "pan" else v[col]
        v["_bin"] = pd.qcut(base, 5, labels=[f"q{i}" for i in range(5)])
        print(f"\nERROR BY {label} QUINTILE")
        print(v.groupby("_bin", observed=True)[["absx", "absz", "reproj_px"]].median().round(3).to_string())

    print("\nERROR BY n_fit (ball points feeding the position vote)")
    v["_nb"] = pd.cut(v.n_fit, [-1, 0, 8, 12, 16, 24, 9999],
                      labels=["0", "1-8", "9-12", "13-16", "17-24", "25+"])
    print(v.groupby("_nb", observed=True)[["absx", "absz"]]
          .agg(["median", "size"]).round(3).to_string())


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    v = load(sys.argv[1])
    pooled(v)
    per_park(v)
    park_vs_game(v)
    view_dependence(v)
