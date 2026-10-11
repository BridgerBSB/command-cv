"""Miss distance at a NEW park: ball-only camera + glove set-ups, graded in inches on held-out clips.

WHY A BALL-ONLY CAMERA IS ENOUGH (PROGRESS-LOG 121, 2026-10-11). At the five parks with a
hand-landmark camera, a camera fitted to the ball alone (fit_park_pose --stab, as in
label_new_park.py) moved the glove TARGET a median 0.18-0.76 in and each park's median miss by
at most 0.80 in (Durham 12.71 -> 11.91), and its held-out plate-crossing gap was 0.20-0.37 in
vs the landmark camera's 0.20-0.47. Misses are 11-15 in, so the landmark fit (recipe step 7,
hand work) is not needed for a miss distance. It is still the stronger CHECK at a home park.

WHAT IT RUNS, each step skipped when its output exists (resumable, long-jobs rule):
  0. clips    refuses to start unless every manifest clip is on disk (a half-downloaded
              game would silently become a smaller game)
  1. detect   v4 ball tracks                          -> detect_tracks_<g>_v4.csv, dets_<g>_v4.csv
  2. stab     stabilize_park --chain --tag c, WITH the glove set-up pass
                                                      -> targets_<g>_stab2c.csv, detect_tracks_<g>_v4stab2c.csv
              (needs data/landmarks/<g>.json with overlay_boxes - mark them by eye first)
  3. split    make_holdout_split                      -> holdout_<g>.csv, seeds_<g>_v4fit.csv
  4. camera   fit_park_pose --stab                    -> park_pose_<g>_ballonly.npz
  5. look     render_ref_pose                         -> park_pose_<g>_ref.png  (EYE CHECK)
  6. miss     command_miss --tag _bo                  -> command_miss_<g>_bo.csv
  7. grade    grade_holdout --tags _bo                (prints the held-out inches)

IT STOPS THERE. The camera is admitted for misses only after the ref render is looked at
(plate under the plate, rubber under the pivot foot, circle on the dirt edge) and the held-out
gap is in the solved parks' range (0.15-0.47 in median). Boards come from command_board_org.py.

    python scripts/miss_new_park.py --game 814874 --dir data/clips/pull_814874
"""
import argparse
import os
import subprocess
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "output")
V4 = "data/yolo_ball_runs/milb4_v4/weights/best.pt"


def sh(args):
    print("  $ " + " ".join(args), flush=True)
    r = subprocess.run([sys.executable, "-u"] + args, cwd=ROOT)
    if r.returncode:
        sys.exit("FAILED (%d): %s" % (r.returncode, " ".join(args)))


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
    d = a.dir if os.path.isabs(a.dir) else os.path.join(ROOT, a.dir)
    lm = os.path.join(ROOT, "data", "landmarks", "%d.json" % g)
    assert os.path.exists(lm), "mark overlay boxes first: %s (stabilize_park refuses without them)" % lm

    man = pd.read_csv(os.path.join(d, os.path.basename(d) + "_manifest.csv"))
    man = man[man.game_pk == g]
    on_disk = sum(os.path.exists(os.path.join(d, s + ".mp4")) for s in man.stem)
    print("STEP 0  clips on disk %d of %d in the manifest" % (on_disk, len(man)), flush=True)
    assert on_disk == len(man), "download incomplete -- run tools/download_manifest_clips.py %s" % a.dir

    print("STEP 1  detect (v4)", flush=True)
    if not have("detect_tracks_%d_v4.csv" % g):
        sh(["scripts/detect_ball_tracks.py", "--dir", a.dir, "--game", str(g), "--weights", V4,
            "--device", "0", "--tag", "_v4"] + a.detect_flags.split())

    print("STEP 2  stabilize + glove set-ups", flush=True)
    if not have("targets_%d_stab2c.csv" % g):
        sh(["scripts/stabilize_park.py", "--game", str(g), "--dir", a.dir,
            "--tracks", "output/detect_tracks_%d_v4.csv" % g, "--chain", "--tag", "c"])

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
            "--out", "output/" + pose])

    print("STEP 5  reference-view camera check", flush=True)
    if not have("park_pose_%d_ref.png" % g):
        sh(["scripts/render_ref_pose.py", "--game", str(g), "--dir", a.dir, "--pose", "output/" + pose])

    print("STEP 6  command miss", flush=True)
    if not have("command_miss_%d_bo.csv" % g):
        sh(["scripts/command_miss.py", "--game", str(g), "--dir", a.dir, "--pose", "output/" + pose,
            "--tracks", "output/detect_tracks_%d_v4stab2c.csv" % g,
            "--targets", "output/targets_%d_stab2c.csv" % g, "--tag", "_bo"])

    print("STEP 7  held-out grade (inches)", flush=True)
    sh(["scripts/grade_holdout.py", "--game", str(g), "--tags", "_bo"])

    print("MISS READY, CAMERA NOT ADMITTED -> look at output/park_pose_%d_ref.png; held-out gap "
          "above should sit in 0.15-0.47 in" % g, flush=True)


if __name__ == "__main__":
    main()
