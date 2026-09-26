"""Command miss per pitch, measured BOTH ways off ONE target.

TARGET (shared by both roots) = the catcher's glove at SET-UP, before release.
  hand  -> data/annotations/<stem>.json `setup` box (Zac's labels), where one exists
  auto  -> src/auto_glove.detect_setup_catch (mitt detector, the long stable hold;
           validated Jul 2026 against 49 hand labels at a median 2.3 px)
  The glove pixel is unprojected onto the plane y = GLOVE_DEPTH_FT (-1.75 ft, i.e. 1.75 ft
  behind the plate tip: OpenCommand's median catch depth, pinned the same way they pin it).

ACTUAL, two roots (docs/2026-09-18-two-roots-glove-and-plate.md):
  A  (the fallback, data)  = manifest plate_x / plate_z, from the tracking system
  B  (video only)          = the detector's ball path evaluated at zone_time and unprojected
                             onto the plate-front plane, the exact computation that
                             compare_models_at_plate.py grades (last 10 frames)
  miss = actual - target, inches, x in the plate_x convention (+ = catcher's right).

WHAT MAKES THE TWO ROOTS CHECK EACH OTHER. They share the target, so
  miss_B - miss_A = actual_B - actual_A
i.e. their disagreement is exactly the video's plate-crossing error, which is independently
known (Asheville v4: 0.45 in median spread). If the two miss distributions differ by more
than that, something other than the ball is wrong.

PLANES, stated so nobody mixes them later: the target lives on the glove plane (-1.75 ft),
both actuals on the plate-front plane (+17/12 ft). OpenCommand compares across the same two
planes; the ball keeps moving between them, which is physics, not error.

CAVEAT: a ball-only camera fit determines the projection well and the camera DISTANCE
weakly (PROGRESS-LOG row 30). Unprojecting the glove 3.2 ft behind the plane the pose was
checked on inherits that. The A-root miss at this park is therefore partly a camera check.

    python scripts/command_miss.py --game 822515 --dir data/clips/ogando_2026 \\
        --pose output/park_pose_822515_pass1.npz --tracks output/detect_tracks_822515_v4.csv
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from src import pinhole as ph  # noqa: E402
import compare_models_at_plate as cmp  # noqa: E402

GLOVE_DEPTH_FT = -1.75
LAST_K = 10
# OpenCommand's plausibility screen for a target, inches (x about plate centre, z height).
PLAUS_X_IN = 20.0
# Hand labels drawn on a DIFFERENT VIDEO with the same stem (the Jun 29 GC2 side-cam zip),
# docs/BACKLOG.md "12 of 55 labels were ingested from the WRONG video". Rendered 2026-09-26
# on the broadcast clip: ff_2 / ff_4 setup boxes sit on empty dirt in front of the plate,
# si_9 / si_13 sit on the glove. These six describe no pitch in this manifest.
WRONG_VIDEO_LABELS = {"ogando_cu_3", "ogando_ff_1", "ogando_ff_2", "ogando_ff_3",
                      "ogando_ff_4", "ogando_ff_5"}


def plausible(x_in, z_in, pt):
    # The manifest spells pitch types out ("Four-seam FB", "Sinker"); OpenCommand's screen is
    # keyed on codes. Matching the codes only would silently give every pitch the
    # non-fastball bounds.
    p = str(pt).upper()
    ff = p == "FF" or "FOUR-SEAM" in p
    fb = ff or p in ("SI", "FC", "FT") or any(k in p for k in ("SINKER", "CUTTER", "TWO-SEAM"))
    lo = 10.0 if fb else 6.0
    hi = 50.0 if ff else 44.0
    return abs(x_in) <= PLAUS_X_IN and lo <= z_in <= hi


def med(s):
    s = pd.Series(s).dropna()
    return (np.nan, 0) if s.empty else (float(s.median()), len(s))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", type=int, required=True)
    ap.add_argument("--dir", required=True)
    ap.add_argument("--pose", required=True)
    ap.add_argument("--tracks", required=True)
    ap.add_argument("--mitt-weights", default="data/yolo/runs/mitt/weights/best.pt")
    ap.add_argument("--no-auto", action="store_true", help="hand targets only (no GPU)")
    a = ap.parse_args()
    d = a.dir if os.path.isabs(a.dir) else os.path.join(ROOT, a.dir)
    assert os.path.isdir(d), d

    z = np.load(os.path.join(ROOT, a.pose))
    C, (pan, tilt, roll), f = z["C"], z["ang"], float(z["f"])
    man = pd.read_csv(os.path.join(d, os.path.basename(d) + "_manifest.csv"))
    man = man[man.game_pk == a.game].set_index("stem")
    assert len(man), "game %d not in manifest" % a.game
    trk = pd.read_csv(os.path.join(ROOT, a.tracks))

    hand = {}
    for p in glob.glob(os.path.join(ROOT, "data", "annotations", "*.json")):
        j = json.load(open(p, encoding="utf-8"))
        if j.get("stem") in WRONG_VIDEO_LABELS:
            continue
        if j.get("stem") in man.index and j.get("setup"):
            hand[j["stem"]] = (float(j["setup"]["cx"]), float(j["setup"]["cy"]))

    auto = {}
    if not a.no_auto:
        from ultralytics import YOLO
        from src.auto_glove import detect_setup_catch
        mitt = YOLO(os.path.join(ROOT, a.mitt_weights))
        for i, stem in enumerate(man.index, 1):
            clip = os.path.join(d, stem + ".mp4")
            if not os.path.exists(clip):
                continue
            r = detect_setup_catch(clip, mitt)
            if r.get("ok"):
                auto[stem] = r["setup"][:2]
            if i % 10 == 0:
                print("  auto target %d/%d" % (i, len(man)), flush=True)

    rows = []
    for stem, r in man.iterrows():
        if pd.isna(r.get("plate_x")):
            continue
        ax_A, az_A = float(r.plate_x), float(r.plate_z)
        g = trk[trk.stem == stem].sort_values("frame").tail(LAST_K)
        ax_B = az_B = np.nan
        if len(g) >= cmp.MIN_PTS:
            ex, ez = cmp.clip_error(g, r, C, pan, tilt, roll, f)
            ax_B, az_B = ax_A + ex / 12.0, az_A + ez / 12.0
        for src, tg in (("hand", hand.get(stem)), ("auto", auto.get(stem))):
            if tg is None:
                continue
            tx, tz = ph.unproject(C, pan, tilt, roll, f, tg, GLOVE_DEPTH_FT)
            row = dict(stem=stem, target_src=src, pitch_type=r.get("pitch_type"),
                       call=r.get("call"), target_px=tg[0], target_py=tg[1],
                       target_x_in=tx * 12, target_z_in=tz * 12,
                       plaus=plausible(tx * 12, tz * 12, r.get("pitch_type")),
                       A_dx=(ax_A - tx) * 12, A_dz=(az_A - tz) * 12,
                       B_dx=(ax_B - tx) * 12, B_dz=(az_B - tz) * 12)
            row["A_miss"] = float(np.hypot(row["A_dx"], row["A_dz"]))
            row["B_miss"] = float(np.hypot(row["B_dx"], row["B_dz"]))
            row["B_minus_A_in"] = float(np.hypot(row["B_dx"] - row["A_dx"], row["B_dz"] - row["A_dz"]))
            rows.append(row)

    out = pd.DataFrame(rows)
    p = os.path.join(ROOT, "output", "command_miss_%d.csv" % a.game)
    out.to_csv(p, index=False)

    print("\ngame %d  pose %s  pitches with truth %d  | hand targets %d  auto targets %d"
          % (a.game, os.path.basename(a.pose), int(man.plate_x.notna().sum()), len(hand), len(auto)))
    for src in ("hand", "auto"):
        s = out[out.target_src == src]
        if s.empty:
            continue
        ok = s[s.plaus]
        print("\nTARGET = %s glove  (%d pitches, %d pass the plausibility screen)" % (src, len(s), len(ok)))
        print("  %-26s %8s %8s   %s" % ("", "median", "n", "(all pitches / plausible only)"))
        for lab, col in (("miss A  (plate_x/z data)", "A_miss"), ("miss B  (video ball)", "B_miss"),
                         ("B vs A disagreement", "B_minus_A_in")):
            m1, n1 = med(s[col]); m2, n2 = med(ok[col])
            print("  %-26s %8.2f %8d   / %.2f (n %d)" % (lab, m1, n1, m2, n2))
    both = out.pivot_table(index="stem", columns="target_src", values=["target_x_in", "target_z_in"])
    if ("target_x_in", "hand") in both and ("target_x_in", "auto") in both:
        dd = np.hypot(both[("target_x_in", "hand")] - both[("target_x_in", "auto")],
                      both[("target_z_in", "hand")] - both[("target_z_in", "auto")]).dropna()
        if len(dd):
            print("\nhand vs auto TARGET on the same pitch: median %.2f in, p90 %.2f in, n %d"
                  % (dd.median(), dd.quantile(.9), len(dd)))
    print("\nwrote", p)


if __name__ == "__main__":
    main()
