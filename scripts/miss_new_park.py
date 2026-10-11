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

--ours-only runs the whole chain on OUR pitchers' clips alone (pitching team's parent org 117 in
pitch_context.py): a subset folder data/clips/hou_<g>/ of hard links (no extra disk) and a
filtered manifest. The board never uses the other team's pitches, so this roughly halves a game;
the camera still gets ~150 tracked clips, against 30-60 seeds at the solved Ogando parks. The
scope is written to output/miss_scope_<g>.txt and a rerun in the other scope is refused, because
every step skips on an existing output and would silently mix the two.

IT STOPS THERE. The camera is admitted for misses only after the ref render is looked at
(plate under the plate, rubber under the pivot foot, circle on the dirt edge) and the held-out
gap is in the solved parks' range (0.15-0.47 in median). Boards come from command_board_org.py.

    python scripts/miss_new_park.py --game 814874 --dir data/clips/pull_814874
"""
import argparse
import os
import shutil
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


def ours_only_dir(g, d):
    """data/clips/hou_<g>/ = hard links to the clips our pitchers threw, plus a filtered manifest."""
    ctx_p = os.path.join(OUT, "pitch_context_%d.csv" % g)
    if not os.path.exists(ctx_p) or "pitch_team_org" not in pd.read_csv(ctx_p, nrows=1):
        sh(["scripts/pitch_context.py", "--game", str(g), "--dir", d])
    ctx = pd.read_csv(ctx_p)
    ours = set(ctx.loc[ctx.pitch_team_org == 117, "stem"])
    man = pd.read_csv(os.path.join(d, os.path.basename(d) + "_manifest.csv"))
    sub = man[(man.game_pk == g) & man.stem.isin(ours)]
    assert len(sub), "no Houston pitches in game %d" % g
    hd = os.path.join(ROOT, "data", "clips", "hou_%d" % g)
    os.makedirs(hd, exist_ok=True)
    copied = 0
    for s in sub.stem:
        src, dst = os.path.join(d, s + ".mp4"), os.path.join(hd, s + ".mp4")
        if os.path.exists(dst):
            continue
        try:
            os.link(src, dst)
        except OSError:
            shutil.copy2(src, dst)
            copied += 1
    sub.to_csv(os.path.join(hd, "hou_%d_manifest.csv" % g), index=False)
    print("  ours-only: %d of %d clips (Houston pitching) -> %s%s"
          % (len(sub), int((man.game_pk == g).sum()), hd,
             "  (%d COPIED, hard link failed)" % copied if copied else ""), flush=True)
    return hd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", type=int, required=True)
    ap.add_argument("--dir", required=True)
    ap.add_argument("--ours-only", action="store_true",
                    help="only the clips Houston pitchers threw (subset folder data/clips/hou_<g>/)")
    ap.add_argument("--weights", default=V4,
                    help="ball detector; v4 by default. SEGRA (1B-side mirror view) needs v8r "
                         "(data/yolo_ball_runs/milb8r_proj/weights/best.pt): on 12 rendered clips v8r "
                         "followed the pitch on ~9, v4 on ~4 (2026-10-11). File names keep the legacy "
                         "_v4 (stabilize_park and make_holdout_split hard-code it); the weights actually "
                         "used are recorded in output/miss_scope_<g>.txt")
    ap.add_argument("--detect-flags", default="--window-pick",
                    help="space-separated extra detect_ball_tracks flags (e.g. '--two-pass')")
    a = ap.parse_args()
    g = a.game
    d = a.dir if os.path.isabs(a.dir) else os.path.join(ROOT, a.dir)
    lm = os.path.join(ROOT, "data", "landmarks", "%d.json" % g)
    assert os.path.exists(lm), "mark overlay boxes first: %s (stabilize_park refuses without them)" % lm

    man = pd.read_csv(os.path.join(d, os.path.basename(d) + "_manifest.csv"))
    man = man[man.game_pk == g]
    scope = "ours-only" if a.ours_only else "all"
    if a.weights != V4:   # legacy markers say only the scope; v4 runs keep matching them
        scope += " weights=" + a.weights
    sp = os.path.join(OUT, "miss_scope_%d.txt" % g)
    if os.path.exists(sp):
        prev = open(sp).read().strip()
        assert prev == scope, ("game %d was run %s; this run is %s and every step would skip on the "
                               "old outputs. Move them aside first." % (g, prev, scope))
    on_disk = sum(os.path.exists(os.path.join(d, s + ".mp4")) for s in man.stem)
    print("STEP 0  clips on disk %d of %d in the manifest" % (on_disk, len(man)), flush=True)
    assert on_disk == len(man), "download incomplete -- run tools/download_manifest_clips.py %s" % a.dir
    open(sp, "w").write(scope)
    if a.ours_only:
        d = ours_only_dir(g, d)
        a.dir = os.path.relpath(d, ROOT)

    print("STEP 1  detect (%s)" % a.weights, flush=True)
    if not have("detect_tracks_%d_v4.csv" % g):
        sh(["scripts/detect_ball_tracks.py", "--dir", a.dir, "--game", str(g), "--weights", a.weights,
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
