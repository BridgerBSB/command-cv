"""v7: v6's recipe plus projected labels from MORE parks. Does breadth fix the unseen-park loss?

WHY (PROGRESS-LOG 114-115). v6 helped every park its labels came from and was WORSE at the one
park it never saw (AdventHealth 821922, 17 of 22 clips, p .006). Two readings: projected labels
only ever help their own park (then the road is labels at every park), or four parks were too
narrow and more parks generalise (then v7 beats v6 at AdventHealth). Row 115 made a new park's
labels cheap (ball-only camera, label_new_park.py), so this is now testable.

ONE CHANGE FROM v6: the training list adds every NEW park admitted in
docs/proj_label_admission.csv. Same init (v4), same epochs (5), same replay fraction (0.25),
same val folder, same grading flags. AdventHealth stays out of training so it is still unseen.

GRADING. v4 / v6 / v7 side by side, inches vs tracking, on:
  - the held-out halves of v6's five games (validated landmark cameras), as in v6
  - AdventHealth 821922, never seen by any of the three
  - each new park's held-out half, under its BALL-ONLY camera. Fine for comparing models
    (same camera for all three) but the inch SCALE there is not validated - (Cy, f) trade off
    in a ball-only fit (fit_park_pose docstring) - so read those rows as relative only.

Resumable: every step skips when its output exists (v6's _v4g/_v6g detections are reused).
    .\\scripts\\run_detached.ps1 -Name v7 -Command "<venv python> -u scripts\\v7_run.py"
"""
import argparse
import glob
import os
import random
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import v6_run as v6  # noqa: E402  (shares sh / admitted_labels / map_tracks / mapping_self_check)

OUT = v6.OUT

# Candidate new parks: (game, clip dir, detect flags). Only those with a ledger row are used.
NEW = [
    ("815046", "data/clips/pull_815046", ["--window-pick"]),
    ("817611", "data/clips/pull_817611", ["--window-pick"]),
    ("826785", "data/clips/pull_826785", ["--window-pick"]),
    ("817256", "data/clips/pull_817256", ["--window-pick"]),
    ("827562", "data/clips/pull_827562", ["--window-pick"]),
    ("816989", "data/clips/pull_816989", ["--window-pick"]),
    ("822161", "data/clips/pull_822161", ["--window-pick"]),
    # third batch (2026-10-11 overnight) - used once admitted, e.g. by a v8:
    #   v7_run.py --name milb8_proj --tag _v8g --also milb7_proj:_v7g
    ("814941", "data/clips/pull_814941", ["--window-pick"]),
    ("815216", "data/clips/pull_815216", ["--window-pick"]),
    ("815256", "data/clips/pull_815256", ["--window-pick"]),
    ("817085", "data/clips/pull_817085", ["--window-pick"]),
    ("817284", "data/clips/pull_817284", ["--window-pick"]),
    ("817590", "data/clips/pull_817590", ["--window-pick"]),
    ("821786", "data/clips/pull_821786", ["--window-pick"]),
    ("827337", "data/clips/pull_827337", ["--window-pick"]),
]


def admitted_new():
    led = pd.read_csv(os.path.join(ROOT, "docs", "proj_label_admission.csv"), dtype=str)
    ok = []
    for g, d, fl in NEW:
        row = led[led.game_pk == g]
        if len(row) == 1 and "ADMIT" in (row.iloc[0].confirmed_verdict, row.iloc[0].unconfirmed_verdict):
            ok.append((g, d, fl))
        else:
            print("   %s: not admitted in proj_label_admission.csv -> left out" % g, flush=True)
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--replay", type=float, default=0.25)
    ap.add_argument("--name", default="milb7_proj", help="run name under data/yolo_ball_runs")
    ap.add_argument("--tag", default="_v7g", help="detection tag for this model")
    ap.add_argument("--also", default="", help="earlier runs to grade beside it, 'run:tag,...' "
                    "(e.g. milb7_proj:_v7g for a v8)")
    a = ap.parse_args()
    run = os.path.join(ROOT, "data", "yolo_ball_runs", a.name)
    w_new = os.path.join(run, "weights", "best.pt")
    models = [(v6.V4, "_v4g"), (v6.W6, "_v6g")]
    for spec in filter(None, a.also.split(",")):
        nm, tg = spec.split(":")
        w = os.path.join(ROOT, "data", "yolo_ball_runs", nm, "weights", "best.pt")
        assert os.path.exists(w), w
        models.append((w, tg))
    models.append((w_new, a.tag))
    grades = os.path.join(OUT, "%s_grades.txt" % a.tag.strip("_").rstrip("g"))

    new = admitted_new()
    assert new, "no new park is admitted yet - nothing for v7 to add over v6"
    train_games = v6.TRAIN_GAMES + [(g, d) for g, d, _f in new]
    parks = v6.PARKS + [(g, d, "park_pose_%s_ballonly.npz" % g, "c", True, fl) for g, d, fl in new]
    print("v7 adds %d park(s): %s" % (len(new), ", ".join(g for g, _d, _f in new)), flush=True)

    print("STEP 0  mapping self-check (new parks)", flush=True)
    worst = [x for x in (v6.mapping_self_check(g, "c") for g, _d, _f in new) if x is not None]
    assert worst and max(worst) < 0.01, "mapper disagrees with stabilize_park at a new park: %s" % worst

    print("STEP 1-2  labels + crops", flush=True)
    for g, d in train_games:
        crops = os.path.join(ROOT, "data", "yolo_proj", g)
        if glob.glob(os.path.join(crops, "images", "train", "*.jpg")):
            continue
        lab = v6.admitted_labels(g)
        lp = os.path.join(OUT, "proj_labels_%s_admitted.csv" % g)
        lab.to_csv(lp, index=False)
        print("   %s: %d admitted labels (confirmed %d)" % (g, len(lab), int(lab.confirmed.sum())),
              flush=True)
        v6.sh(["scripts/build_ball_dataset.py", "--labels", lp, "--clips", d, "--out", crops,
               "--all-train", "--jpg"])

    y = os.path.join(ROOT, "data", "yolo_" + a.name.split("_")[0], "data.yaml")
    os.makedirs(os.path.dirname(y), exist_ok=True)
    v4img = sorted(glob.glob(os.path.join(ROOT, "data", "yolo_milb4", "images", "train", "*.png")))
    random.Random(0).shuffle(v4img)
    keep = v4img[:int(len(v4img) * a.replay)]
    proj = []
    for g, _d in train_games:
        proj += sorted(glob.glob(os.path.join(ROOT, "data", "yolo_proj", g, "images", "train", "*.jpg")))
    lst = os.path.join(os.path.dirname(y), "train.txt")
    with open(lst, "w") as f:
        f.write("\n".join(keep + proj) + "\n")
    with open(y, "w") as f:
        f.write("train: %s\n" % lst)
        f.write("val: %s\n" % os.path.join(ROOT, "data", "yolo_milb4", "images", "val"))
        f.write("names:\n  0: ball\n")
    print("STEP 3  train list: %d v4 replay (of %d) + %d projected = %d"
          % (len(keep), len(v4img), len(proj), len(keep) + len(proj)), flush=True)

    done = 0
    rc = os.path.join(run, "results.csv")
    if os.path.exists(rc):
        done = max(0, sum(1 for _ in open(rc)) - 1)
    if done < a.epochs:
        print("STEP 4  train (%d of %d epochs done)" % (done, a.epochs), flush=True)
        v6.sh(["scripts/train_ball_detector.py", "--data", y, "--model", v6.V4, "--epochs",
               str(a.epochs), "--batch", "16", "--imgsz", "320", "--device", "0", "--name", a.name]
              + (["--resume"] if done else []))
    assert os.path.exists(w_new), w_new

    print("STEP 5-6  detect + map", flush=True)
    for g, d, _pose, ht, held, flags in parks:
        stems = ["--stems-file", os.path.join(OUT, "holdout_%s.csv" % g)] if held else []
        for w, tag in models:
            if not os.path.exists(os.path.join(OUT, "detect_tracks_%s%s.csv" % (g, tag))):
                v6.sh(["scripts/detect_ball_tracks.py", "--dir", d, "--game", g, "--weights", w,
                       "--device", "0", "--tag", tag] + flags + stems)
            if not os.path.exists(os.path.join(OUT, "detect_tracks_%s%ss.csv" % (g, tag))):
                v6.map_tracks(g, tag, tag + "s", ht)

    new_ids = {g for g, _d, _f in new}
    tags = ",".join(tg + "s" for _w, tg in models)
    with open(grades, "w", encoding="utf-8") as f:
        for g, d, pose, _ht, held, _fl in parks:
            what = ("NEW PARK, held-out half, BALL-ONLY camera (relative only)" if g in new_ids
                    else "HELD-OUT half" if held else "NEVER SEEN by v4 / v6 / v7")
            f.write("\n### %s   %s   %s\n" % (g, what, pose))
            f.flush()
            v6.sh(["scripts/compare_models_at_plate.py", "--game", g, "--dir", d,
                   "--tags", tags, "--pose", os.path.join("output", pose)]
                  + (["--holdout"] if held else []), out=f)
    print("RUN COMPLETE %s -> %s" % (a.name, grades), flush=True)


if __name__ == "__main__":
    main()
