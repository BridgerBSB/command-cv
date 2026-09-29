"""v5: fine-tune v4 on the streamed games that passed the eye check, then grade in INCHES.

Resumable like v4_finish.py: each step skips when its output exists.

  1. data   data/yolo_milb5/data.yaml -- train = the v4 train folder + every streamed game
            whose row in docs/park_admission.csv reads labeller=v4stream, verdict=ADMIT.
            val = the v4 val folder, UNCHANGED, so v4 and v5 are scored on identical data.
  2. train  from v4's best.pt (fine-tune, not from scratch), --resume safe.
  3. detect each grading park with v5, SAME flags as that park's v4 run.
  4. grade  compare_models_at_plate.py, v4 vs v5, video ball vs tracking data at the plate,
            in inches -> output/v5_grades.txt. Never mAP (it misled three times).

    python scripts/v5_run.py --epochs 10
"""
import argparse
import os
import subprocess
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
V4 = os.path.join(ROOT, "data", "yolo_ball_runs", "milb4_v4", "weights", "best.pt")
NAME = "milb5_v5"
RUN = os.path.join(ROOT, "data", "yolo_ball_runs", NAME)
W = os.path.join(RUN, "weights", "best.pt")
# (game, clip dir, v4 tag, v5 tag, detect flags matching the v4 run, grading extras)
PARKS = [
    ("822515", "data/clips/ogando_2026", "_v4", "_v5", ["--two-pass"],
     ["--pose", "output/park_pose_822515_pass1.npz"]),
    ("816698", "data/clips/aaa_816698", "_v4", "_v5", ["--two-pass"], []),
    ("821922", "data/clips/ogando_2026", "_ogando_v4wp", "_ogando_v5wp",
     ["--lo", "150", "--hi", "260", "--window-pick"],
     ["--pose", "output/park_pose_821922_s2lm2.npz"]),
]


def sh(args, out=None):
    print(">>", " ".join(args), flush=True)
    r = subprocess.run([sys.executable, "-u"] + args, cwd=ROOT,
                       stdout=out, stderr=subprocess.STDOUT if out else None, text=True)
    if r.returncode:
        print("   FAILED rc=%d" % r.returncode, flush=True)
        sys.exit(r.returncode)


def admitted():
    led = pd.read_csv(os.path.join(ROOT, "docs", "park_admission.csv"), dtype=str)
    ok = led[(led.labeller == "v4stream") & (led.verdict == "ADMIT")].game_pk.tolist()
    return [g for g in ok if os.path.isdir(os.path.join(ROOT, "data", "yolo_stream", g, "images"))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=10)
    a = ap.parse_args()
    games = admitted()
    assert games, "no streamed game is ADMITTED in docs/park_admission.csv yet"
    y = os.path.join(ROOT, "data", "yolo_milb5", "data.yaml")
    os.makedirs(os.path.dirname(y), exist_ok=True)
    train = [os.path.join(ROOT, "data", "yolo_milb4", "images", "train")] + [
        os.path.join(ROOT, "data", "yolo_stream", g, "images", "train") for g in games]
    with open(y, "w") as f:
        f.write("train:\n" + "".join("  - %s\n" % p for p in train))
        f.write("val: %s\n" % os.path.join(ROOT, "data", "yolo_milb4", "images", "val"))
        f.write("names:\n  0: ball\n")
    print("v5 data: v4 train + %d streamed games: %s" % (len(games), ", ".join(games)), flush=True)

    done = 0
    rc = os.path.join(RUN, "results.csv")
    if os.path.exists(rc):
        done = max(0, sum(1 for _ in open(rc)) - 1)
    if done < a.epochs:
        sh(["scripts/train_ball_detector.py", "--data", y, "--model", V4, "--epochs",
            str(a.epochs), "--batch", "16", "--imgsz", "320", "--device", "0", "--name", NAME]
           + (["--resume"] if done else []))
    assert os.path.exists(W), W
    for g, d, _t4, t5, flags, _x in PARKS:
        if os.path.exists(os.path.join(ROOT, "output", "detect_tracks_%s%s.csv" % (g, t5))):
            continue
        sh(["scripts/detect_ball_tracks.py", "--dir", d, "--game", g, "--weights", W,
            "--device", "0", "--tag", t5] + flags)
    with open(os.path.join(ROOT, "output", "v5_grades.txt"), "w", encoding="utf-8") as f:
        for g, d, t4, t5, _fl, extra in PARKS:
            f.write("\n### %s   v4=%s  v5=%s\n" % (g, t4, t5))
            f.flush()
            sh(["scripts/compare_models_at_plate.py", "--game", g, "--dir", d,
                "--tags", "%s,%s" % (t4, t5)] + extra, out=f)
    print("V5 COMPLETE -> output/v5_grades.txt", flush=True)


if __name__ == "__main__":
    main()
