"""Camera-projected training labels at a NEW park, with a ball-only camera (no landmarks).

WHY (PROGRESS-LOG 114-115). v6 showed camera-projected labels lift the detector where its own
self-labels (v5) did not. The expensive step in solving a park was the hand landmark fit
(docs/2026-10-11-park-camera-solve-recipe.md step 7). That fit is needed for INCHES, not for
LABELS: at the five solved parks (2026-10-11), labels projected through a camera fitted to the
ball alone landed a median 0.47-1.10 px from the landmark camera's labels (p90 1.17-2.17, worst
5.5 px); 1 of 7,931 shared labels was off by more than half a ball (Durham). fit_park_pose.py's
docstring said a ball-only pose is good for labels; that is the measurement. Caveat: the
ball-only camera is fitted to v4's own tracks, so its labels sit closer to v4's detections
(0.53-0.68 px) than the landmark camera's do (0.68-1.36 px).

WHAT IT RUNS, each step skipped when its output exists (resumable, long-jobs rule):
  1. detect   v4, --window-pick                    -> detect_tracks_<g>_v4.csv, dets_<g>_v4.csv
  2. stab     stabilize_park --chain --tag c --no-setups -> detect_tracks_<g>_v4stab2c.csv, stab2_H_<g>c.csv
              (needs data/landmarks/<g>.json with overlay_boxes - mark them by eye first).
              --no-setups skips the glove pass labels never use: Dell Diamond 815046 in 62 s
              instead of ~30 min, with all 2,666 track points and all 1,358 labels identical
              to the with-set-ups run on the same reference (2026-10-11). No glove targets are
              written, so a labels-only park has no miss distance until it is re-stabilized.
  3. split    make_holdout_split                   -> holdout_<g>.csv, seeds_<g>_v4fit.csv
  4. camera   fit_park_pose --stab --render        -> park_pose_<g>_ballonly.npz (+ png)
  5. labels   project_labels --all-frames          -> proj_labels_<g>.csv
  6. sheet    render_proj_labels                   -> proj_labels_<g>_sheet.png
  7. camera   render_ref_pose                      -> park_pose_<g>_ref.png

IT STOPS THERE. Admission is an eye check, recorded in docs/proj_label_admission.csv:
  park_pose_<g>_ref.png  plate / rubber / 9 ft mound circle / foul lines drawn from a camera that
                         never saw them, ON THE REFERENCE VIEW. (Not park_pose_<g>.png: that one
                         draws reference-view points on clip 0's raw frame and reads as offset
                         at any park whose framing moves - found at Dell Diamond 815046.)
  proj_labels_<g>_sheet  confirmed AND unconfirmed tiles on the ball. Unconfirmed labels are the
                         ones that can teach a hidden ball (behind the pitcher, batter, catcher).
Nothing here trains on anything.

    python scripts/label_new_park.py --game 815046 --dir data/clips/pull_815046
"""
import argparse
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "output")
V4 = os.path.join(ROOT, "data", "yolo_ball_runs", "milb4_v4", "weights", "best.pt")


def sh(args):
    print(">>", " ".join(args), flush=True)
    r = subprocess.run([sys.executable, "-u"] + args, cwd=ROOT)
    if r.returncode:
        print("   FAILED rc=%d" % r.returncode, flush=True)
        sys.exit(r.returncode)


def have(name):
    return os.path.exists(os.path.join(OUT, name))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", type=int, required=True)
    ap.add_argument("--dir", required=True)
    ap.add_argument("--detect-flags", default="--window-pick",
                    help="space-separated extra detect_ball_tracks flags (e.g. '--two-pass')")
    a = ap.parse_args()
    g = a.game
    lm = os.path.join(ROOT, "data", "landmarks", "%d.json" % g)
    assert os.path.exists(lm), "mark overlay boxes first: %s (stabilize_park refuses without them)" % lm

    print("STEP 1  detect (v4)", flush=True)
    if not have("detect_tracks_%d_v4.csv" % g):
        sh(["scripts/detect_ball_tracks.py", "--dir", a.dir, "--game", str(g), "--weights", V4,
            "--device", "0", "--tag", "_v4"] + a.detect_flags.split())

    print("STEP 2  stabilize", flush=True)
    if not have("detect_tracks_%d_v4stab2c.csv" % g):
        sh(["scripts/stabilize_park.py", "--game", str(g), "--dir", a.dir,
            "--tracks", "output/detect_tracks_%d_v4.csv" % g, "--chain", "--tag", "c", "--no-setups"])

    print("STEP 3  held-out split", flush=True)
    if not have("holdout_%d.csv" % g):
        sh(["scripts/make_holdout_split.py", "--game", str(g),
            "--tracks", "output/detect_tracks_%d_v4.csv" % g])

    print("STEP 4  ball-only camera", flush=True)
    pose = "park_pose_%d_ballonly.npz" % g
    if not have(pose):
        sh(["scripts/fit_park_pose.py", "--game", str(g), "--dir", a.dir,
            "--seeds", "output/seeds_%d_v4fit.csv" % g,
            "--stab", "output/detect_tracks_%d_v4stab2c.csv" % g,
            "--out", "output/" + pose, "--render"])

    print("STEP 5  projected labels", flush=True)
    if not have("proj_labels_%d.csv" % g):
        sh(["scripts/project_labels.py", "--game", str(g), "--dir", a.dir, "--pose", "output/" + pose,
            "--stab-tag", "c", "--all-frames"])

    print("STEP 6  eye-check sheet", flush=True)
    if not have("proj_labels_%d_sheet.png" % g):
        sh(["scripts/render_proj_labels.py", "--game", str(g), "--clips", a.dir])

    print("STEP 7  reference-view camera check", flush=True)
    if not have("park_pose_%d_ref.png" % g):
        sh(["scripts/render_ref_pose.py", "--game", str(g), "--dir", a.dir, "--pose", "output/" + pose])

    print("LABELS READY, NOT ADMITTED -> look at output/park_pose_%d_ref.png and "
          "output/proj_labels_%d_sheet.png, then add a row to docs/proj_label_admission.csv" % (g, g),
          flush=True)


if __name__ == "__main__":
    main()
