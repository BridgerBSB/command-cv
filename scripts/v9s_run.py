"""v9s: v8r fine-tuned on SEGRA's projected labels. Does it find the ball over SEGRA's crowd?

WHY (PROGRESS-LOG 129). SEGRA Stadium (Fayetteville home, 66 games, half of every A-level pitch)
is a centre-field camera whose last stretch of flight crosses the seats directly behind the
plate. With the chain bridge (--bridge 10) its camera is unbiased but the held-out gate reads
0.86 in, over 0.47, because the ball is localised ~2x worse than at the passing parks. The
projected-label sheet shows the cause: on the frames the detector misses, the ball is plainly
visible over the crowd (docs/proj_label_admission.csv, 827289) - a background the training
set does not cover.

ONE CHANGE FROM v8r: start from v8r's weights and add 827289's admitted FIT-HALF labels, with a
replay sample of v8r's own training list so the other parks are not forgotten. The held-out
half of 827289 is never labelled (project_labels.py) and is what this grades on.

GRADE. v8r vs v9s on 827289's held-out stems, same window and bridge, mapped into the same
reference view, inches at the plate through the same bridged ball-only camera
(compare_models_at_plate.py --holdout). Then other SEGRA games, never seen, are the real test.

    .\\scripts\\run_detached.ps1 -Name v9s -Command "<venv python> -u scripts\\v9s_run.py"
"""
import argparse
import glob
import os
import random
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import v6_run as v6  # noqa: E402

OUT = v6.OUT
G, D = "827289", "data/clips/hou_827289"
V8R = os.path.join(ROOT, "data", "yolo_ball_runs", "milb8r_proj", "weights", "best.pt")
FLAGS = ["--window-pick", "--lo", "160", "--hi", "230", "--bridge", "10"]
POSE = "park_pose_827289_ballonly.npz"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--replay", type=float, default=0.15, help="share of v8r's train list replayed")
    ap.add_argument("--name", default="milb9s_segra")
    ap.add_argument("--tag", default="_v9sg")
    a = ap.parse_args()
    run = os.path.join(ROOT, "data", "yolo_ball_runs", a.name)
    w_new = os.path.join(run, "weights", "best.pt")

    print("STEP 1  SEGRA crops", flush=True)
    crops = os.path.join(ROOT, "data", "yolo_proj", G)
    if not glob.glob(os.path.join(crops, "images", "train", "*.jpg")):
        lab = v6.admitted_labels(G)
        lp = os.path.join(OUT, "proj_labels_%s_admitted.csv" % G)
        lab.to_csv(lp, index=False)
        print("   %s: %d admitted labels (confirmed %d)" % (G, len(lab), int(lab.confirmed.sum())), flush=True)
        v6.sh(["scripts/build_ball_dataset.py", "--labels", lp, "--clips", D, "--out", crops,
               "--all-train", "--jpg"])
    seg = sorted(glob.glob(os.path.join(crops, "images", "train", "*.jpg")))
    assert seg, "no SEGRA crops built"

    print("STEP 2  train list", flush=True)
    base = [l.strip() for l in open(os.path.join(ROOT, "data", "yolo_milb8r", "train.txt")) if l.strip()]
    random.Random(0).shuffle(base)
    keep = base[:int(len(base) * a.replay)]
    y = os.path.join(ROOT, "data", "yolo_" + a.name.split("_")[0], "data.yaml")
    os.makedirs(os.path.dirname(y), exist_ok=True)
    lst = os.path.join(os.path.dirname(y), "train.txt")
    with open(lst, "w") as f:
        f.write("\n".join(keep + seg) + "\n")
    with open(y, "w") as f:
        f.write("train: %s\n" % lst)
        f.write("val: %s\n" % os.path.join(ROOT, "data", "yolo_milb4", "images", "val"))
        f.write("names:\n  0: ball\n")
    print("   %d v8r replay (of %d) + %d SEGRA = %d" % (len(keep), len(base), len(seg), len(keep) + len(seg)),
          flush=True)

    done = 0
    rc = os.path.join(run, "results.csv")
    if os.path.exists(rc):
        done = max(0, sum(1 for _ in open(rc)) - 1)
    if done < a.epochs:
        print("STEP 3  train from v8r (%d of %d epochs done)" % (done, a.epochs), flush=True)
        v6.sh(["scripts/train_ball_detector.py", "--data", y, "--model", V8R, "--epochs", str(a.epochs),
               "--batch", "16", "--imgsz", "320", "--device", "0", "--name", a.name]
              + (["--resume"] if done else []))
    assert os.path.exists(w_new), w_new

    print("STEP 4  detect the held-out half, both models", flush=True)
    stems = ["--stems-file", os.path.join(OUT, "holdout_%s.csv" % G)]
    for w, tag in ((V8R, "_v8rSg"), (w_new, a.tag)):
        if not os.path.exists(os.path.join(OUT, "detect_tracks_%s%s.csv" % (G, tag))):
            v6.sh(["scripts/detect_ball_tracks.py", "--dir", D, "--game", G, "--weights", w,
                   "--device", "0", "--tag", tag] + FLAGS + stems)
        if not os.path.exists(os.path.join(OUT, "detect_tracks_%s%ss.csv" % (G, tag))):
            v6.map_tracks(G, tag, tag + "s", "c")

    grades = os.path.join(OUT, "v9s_grades.txt")
    with open(grades, "w", encoding="utf-8") as f:
        f.write("### %s  HELD-OUT half, bridged ball-only camera %s\n" % (G, POSE))
        f.flush()
        v6.sh(["scripts/compare_models_at_plate.py", "--game", G, "--dir", D,
               "--tags", "_v8rSgs,%ss" % a.tag, "--pose", os.path.join("output", POSE), "--holdout"], out=f)
    print("RUN COMPLETE %s -> %s" % (a.name, grades), flush=True)


if __name__ == "__main__":
    main()
