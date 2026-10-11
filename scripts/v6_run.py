"""v6: fine-tune v4 on CAMERA-PROJECTED labels from every solved park, then grade in INCHES on
clips it never trained on.

WHY (PROGRESS-LOG 111). v5 fine-tuned v4 on v4's OWN streamed detections and graded worse:
self-labels add parks but no information v4 did not already have. The one thing that moved a
park was INDEPENDENT labels - the known trajectory projected through a validated camera (row
102, Isotopes 815443: usable video pitches 11 -> 31 of 37). v6 asks whether that generalises:
labels from five games at four parks, graded on their held-out halves and on a park it never
sees. If it does, solving more park cameras is the road to the season; if not, it is not.

Resumable: every step skips when its output exists.
  1. labels  output/proj_labels_<g>.csv filtered by docs/proj_label_admission.csv: confirmed
             rows always, unconfirmed rows only where that park's eye check ADMITs them.
  2. crops   build_ball_dataset.py per game -> data/yolo_proj/<g>/ (all train, jpg).
  3. data    data/yolo_milb6/data.yaml: a --replay slice of v4's train crops + every projected
             crop. val = v4's val folder, UNCHANGED, so v4 and v6 share it.
  4. train   fine-tune from v4's best.pt, --epochs (row 102 used 5), device 0.
  5. detect  v4 AND v6 at every grading park with the SAME flags: the HELD-OUT stems of a game
             v6 trained on (--stems-file), every clip of a park it never saw.
  6. map     both models' tracks into the park's reference view through the SAVED per-clip
             homographies (the good ones, stabilize_park's rule). Self-check first: re-mapping
             v4's raw tracks must reproduce v4's saved stab2c file.
  7. grade   compare_models_at_plate.py on the reference-view tracks with the park's validated
             camera -> output/v6_grades.txt. Inches vs tracking, never mAP.

Run detached (long-jobs-on-this-laptop.md), from the ROCm venv on the AMD laptop:
    .\\scripts\\run_detached.ps1 -Name v6 -Command "<venv python> -u scripts\\v6_run.py"
"""
import argparse
import glob
import os
import random
import subprocess
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from project_labels import good_homographies  # noqa: E402

V4 = os.path.join(ROOT, "data", "yolo_ball_runs", "milb4_v4", "weights", "best.pt")
NAME = "milb6_proj"
RUN = os.path.join(ROOT, "data", "yolo_ball_runs", NAME)
W6 = os.path.join(RUN, "weights", "best.pt")
OUT = os.path.join(ROOT, "output")

# Games with camera-projected labels: (game, clip dir).
TRAIN_GAMES = [
    ("815419", "data/clips/aaa_815419"),
    ("815443", "data/clips/pull_815443"),
    ("816698", "data/clips/aaa_816698"),
    ("821819", "data/clips/aplus_821819"),
    ("822515", "data/clips/ogando_2026"),
]
# Grading parks: (game, clip dir, validated pose, stab2 H tag, held out?, detect flags).
# Flags match each park's earlier v4/v5 runs where those existed (v5_run.py PARKS).
PARKS = [
    ("815419", "data/clips/aaa_815419", "park_pose_815419_s2lm1.npz", "c", True, ["--window-pick"]),
    ("815443", "data/clips/pull_815443", "park_pose_815419_s2lm1.npz", "c", True, ["--window-pick"]),
    ("816698", "data/clips/aaa_816698", "park_pose_816698_s2lm1.npz", "c", True, ["--two-pass"]),
    ("821819", "data/clips/aplus_821819", "park_pose_821819_s2lm1.npz", "c", True, ["--window-pick"]),
    ("822515", "data/clips/ogando_2026", "park_pose_822515_s2lm1.npz", "c", True, ["--two-pass"]),
    # AdventHealth: a validated camera (0.15 in, row 107) and NO projected labels -> v6 never saw it
    ("821922", "data/clips/ogando_2026", "park_pose_821922_s2lm2.npz", "c", False,
     ["--lo", "150", "--hi", "260", "--window-pick"]),
]


def sh(args, out=None):
    print(">>", " ".join(args), flush=True)
    r = subprocess.run([sys.executable, "-u"] + args, cwd=ROOT,
                       stdout=out, stderr=subprocess.STDOUT if out else None, text=True)
    if r.returncode:
        print("   FAILED rc=%d" % r.returncode, flush=True)
        sys.exit(r.returncode)


def admitted_labels(g):
    """Projected labels this park's eye check admits (docs/proj_label_admission.csv)."""
    led = pd.read_csv(os.path.join(ROOT, "docs", "proj_label_admission.csv"), dtype=str)
    row = led[led.game_pk == g]
    assert len(row) == 1, "game %s has no single row in proj_label_admission.csv" % g
    row = row.iloc[0]
    lab = pd.read_csv(os.path.join(OUT, "proj_labels_%s.csv" % g))
    keep = pd.Series(False, index=lab.index)
    if row.confirmed_verdict == "ADMIT":
        keep |= lab.confirmed.astype(bool)
    if row.unconfirmed_verdict == "ADMIT":
        keep |= ~lab.confirmed.astype(bool)
    return lab[keep]


def map_tracks(g, src_tag, dst_tag, h_tag):
    """Raw-pixel tracks -> the park's reference view, through the saved good homographies,
    nearest H frame per point (stabilize_park.py's rule, imported via good_homographies)."""
    dst = os.path.join(OUT, "detect_tracks_%s%s.csv" % (g, dst_tag))
    trk = pd.read_csv(os.path.join(OUT, "detect_tracks_%s%s.csv" % (g, src_tag)))
    H = good_homographies(pd.read_csv(os.path.join(OUT, "stab2_H_%s%s.csv" % (g, h_tag))))
    rows, dropped = [], 0
    for stem, t in trk.groupby("stem"):
        h = H[H.stem == stem]
        if h.empty:
            dropped += 1
            continue
        hf = h.frame.to_numpy()
        for r in t.itertuples():
            k = int(np.argmin(np.abs(hf - r.frame)))
            M = h.iloc[k][["h%d" % i for i in range(9)]].to_numpy(float).reshape(3, 3)
            q = M @ np.array([r.x, r.y, 1.0])
            rows.append(dict(stem=stem, game_pk=r.game_pk, frame=int(r.frame), x=q[0] / q[2],
                             y=q[1] / q[2], conf=r.conf, h_frame=int(hf[k])))
    pd.DataFrame(rows).to_csv(dst, index=False, float_format="%.17g")
    print("   mapped %s%s -> %s: %d clips, %d with no good H dropped"
          % (g, src_tag, dst_tag, trk.stem.nunique() - dropped, dropped), flush=True)
    return dst


def mapping_self_check(g, h_tag):
    """Re-map v4's raw tracks and compare with the stab2 file stabilize_park wrote for them.
    A mismatch means the mapper is not the stabilizer's rule and every grade below is void."""
    raw = os.path.join(OUT, "detect_tracks_%s_v4.csv" % g)
    ref = os.path.join(OUT, "detect_tracks_%s_v4stab2%s.csv" % (g, h_tag))
    if not (os.path.exists(raw) and os.path.exists(ref)):
        print("   self-check %s: skipped (no raw v4 + stab2 pair)" % g, flush=True)
        return None
    mine = pd.read_csv(map_tracks(g, "_v4", "_v4selfcheck", h_tag))
    theirs = pd.read_csv(ref)
    m = mine.merge(theirs, on=["stem", "frame"], suffixes=("_m", "_t"))
    if not len(m):
        print("   self-check %s: NO overlapping rows (the stab2 file came from other tracks)" % g,
              flush=True)
        return None
    d = np.hypot(m.x_m - m.x_t, m.y_m - m.y_t)
    print("   self-check %s: %d shared points, max diff %.2g px, %d over 0.01 px"
          % (g, len(m), float(d.max()), int((d > 0.01).sum())), flush=True)
    return float(d.max())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--replay", type=float, default=0.25, help="fraction of v4 train crops kept")
    a = ap.parse_args()

    # 6 (first, it is seconds): the mapper must reproduce the stabilizer before anything uses it
    print("STEP 0  mapping self-check", flush=True)
    worst = [x for x in (mapping_self_check(g, ht) for g, _d, _p, ht, _h, _f in PARKS)
             if x is not None]
    assert worst, "no park could be self-checked; refusing to grade through an unchecked mapper"
    assert max(worst) < 0.01, "mapper disagrees with stabilize_park by %.3f px" % max(worst)

    # 1-2: labels + crops
    print("STEP 1-2  labels + crops", flush=True)
    for g, d in TRAIN_GAMES:
        crops = os.path.join(ROOT, "data", "yolo_proj", g)
        if glob.glob(os.path.join(crops, "images", "train", "*.jpg")):
            continue
        lab = admitted_labels(g)
        lp = os.path.join(OUT, "proj_labels_%s_admitted.csv" % g)
        lab.to_csv(lp, index=False)
        print("   %s: %d admitted labels (confirmed %d)" % (g, len(lab), int(lab.confirmed.sum())),
              flush=True)
        sh(["scripts/build_ball_dataset.py", "--labels", lp, "--clips", d, "--out", crops,
            "--all-train", "--jpg"])

    # 3: data
    y = os.path.join(ROOT, "data", "yolo_milb6", "data.yaml")
    os.makedirs(os.path.dirname(y), exist_ok=True)
    v4img = sorted(glob.glob(os.path.join(ROOT, "data", "yolo_milb4", "images", "train", "*.png")))
    random.Random(0).shuffle(v4img)
    keep = v4img[:int(len(v4img) * a.replay)]
    new = []
    for g, _d in TRAIN_GAMES:
        new += sorted(glob.glob(os.path.join(ROOT, "data", "yolo_proj", g, "images", "train", "*.jpg")))
    assert new, "no projected crops were built"
    lst = os.path.join(os.path.dirname(y), "train.txt")
    with open(lst, "w") as f:
        f.write("\n".join(keep + new) + "\n")
    with open(y, "w") as f:
        f.write("train: %s\n" % lst)
        f.write("val: %s\n" % os.path.join(ROOT, "data", "yolo_milb4", "images", "val"))
        f.write("names:\n  0: ball\n")
    print("STEP 3  train list: %d v4 replay (of %d) + %d projected = %d"
          % (len(keep), len(v4img), len(new), len(keep) + len(new)), flush=True)

    # 4: train
    done = 0
    rc = os.path.join(RUN, "results.csv")
    if os.path.exists(rc):
        done = max(0, sum(1 for _ in open(rc)) - 1)
    if done < a.epochs:
        print("STEP 4  train (%d of %d epochs done)" % (done, a.epochs), flush=True)
        sh(["scripts/train_ball_detector.py", "--data", y, "--model", V4, "--epochs",
            str(a.epochs), "--batch", "16", "--imgsz", "320", "--device", "0", "--name", NAME]
           + (["--resume"] if done else []))
    assert os.path.exists(W6), W6

    # 5-6: detect both models, map both into the reference view
    print("STEP 5-6  detect + map", flush=True)
    for g, d, _pose, ht, held, flags in PARKS:
        stems = ["--stems-file", os.path.join(OUT, "holdout_%s.csv" % g)] if held else []
        for w, tag in ((V4, "_v4g"), (W6, "_v6g")):
            if not os.path.exists(os.path.join(OUT, "detect_tracks_%s%s.csv" % (g, tag))):
                sh(["scripts/detect_ball_tracks.py", "--dir", d, "--game", g, "--weights", w,
                    "--device", "0", "--tag", tag] + flags + stems)
            if not os.path.exists(os.path.join(OUT, "detect_tracks_%s%ss.csv" % (g, tag))):
                map_tracks(g, tag, tag + "s", ht)

    # 7: grade
    with open(os.path.join(OUT, "v6_grades.txt"), "w", encoding="utf-8") as f:
        for g, d, pose, _ht, held, _fl in PARKS:
            f.write("\n### %s   %s   %s\n" % (g, "HELD-OUT half (v6 trained on the other half)"
                                            if held else "NEVER SEEN by v6", pose))
            f.flush()
            sh(["scripts/compare_models_at_plate.py", "--game", g, "--dir", d,
                "--tags", "_v4gs,_v6gs", "--pose", os.path.join("output", pose)]
               + (["--holdout"] if held else []), out=f)
    print("V6 COMPLETE -> output/v6_grades.txt", flush=True)


if __name__ == "__main__":
    main()
