"""Batch: every saved annotation -> Method B miss -> command profile + summary.

Reads data/annotations/*.json (from annotate_tool.py), computes the glove-size-
scaled setup->catch miss per pitch, writes output/command_profile_b.csv, and
prints a per-pitch table + a per-pitcher rollup (avg miss + directional bias).

Run: python scripts/methodb_batch.py [--glove-in 12]
"""
import argparse
import csv
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.miss_methodb import GlovePx, compute_miss_b

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIELDS = ["stem", "pitcher", "pitch_type", "hand", "setup_frame", "catch_frame",
          "miss_in", "horiz_in", "vert_in", "horiz_label", "vert_label",
          "scale_in_per_px"]


def _gp(d):
    return GlovePx(cx=d["cx"], cy=d["cy"], diameter_px=(d["w"] + d["h"]) / 2.0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--glove-in", type=float, default=12.0)
    a = p.parse_args()

    ann_paths = sorted(glob.glob(os.path.join(ROOT, "data", "annotations", "*.json")))
    if not ann_paths:
        raise SystemExit("no annotations yet — run scripts/annotate_tool.py first")

    rows = []
    for ap in ann_paths:
        with open(ap) as f:
            ann = json.load(f)
        miss = compute_miss_b(_gp(ann["setup"]), _gp(ann["catch"]),
                              pitcher_throws=ann.get("hand", "R"),
                              glove_diameter_in=a.glove_in)
        rows.append({
            "stem": ann["stem"], "pitcher": ann["pitcher"],
            "pitch_type": ann["pitch_type"], "hand": ann.get("hand", "R"),
            "setup_frame": ann["setup"]["frame"], "catch_frame": ann["catch"]["frame"],
            "miss_in": miss.miss_in, "horiz_in": miss.horiz_in, "vert_in": miss.vert_in,
            "horiz_label": miss.horiz_label, "vert_label": miss.vert_label,
            "scale_in_per_px": miss.scale_in_per_px,
        })

    out_csv = os.path.join(ROOT, "output", "command_profile_b.csv")
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)

    print(f"{'pitch':18s} {'type':5s} {'miss':>6s}  direction")
    print("-" * 50)
    for r in rows:
        where = "/".join(x for x in (r["horiz_label"], r["vert_label"])
                         if x not in ("on-line", "level")) or "on target"
        print(f"{r['stem']:18s} {r['pitch_type']:5s} {r['miss_in']:5.1f}in  {where}")

    # per-pitcher rollup (full precision; round only here at display)
    by = {}
    for r in rows:
        by.setdefault(r["pitcher"], []).append(r)
    print("\nrollup:")
    for pit, rs in by.items():
        n = len(rs)
        avg = sum(x["miss_in"] for x in rs) / n
        hb = sum(x["horiz_in"] for x in rs) / n
        vb = sum(x["vert_in"] for x in rs) / n
        print(f"  {pit}: n={n}  avg_miss={avg:.1f}in  "
              f"horiz_bias={hb:+.1f}in  vert_bias={vb:+.1f}in")
    print(f"\nwrote {out_csv}")


if __name__ == "__main__":
    main()
