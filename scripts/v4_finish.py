"""Finish the v4 run from wherever it stopped. Safe to run any number of times.

Each step checks whether its output already exists and skips if so, so a relaunch after a
crash, a memory reap or a reboot picks up exactly where the last attempt died:

  1. train     -- resumes from last.pt if the run has fewer than EPOCHS rows in results.csv
  2. detect x3 -- one per grading park, skipped when detect_tracks_<game>_v4.csv exists
  3. grade x3  -- compare_models_at_plate, always re-run (seconds), written to
                  output/v4_grades.txt so the answer is in one file

Launched by scripts/overnight_watchdog.ps1 when the original cv-v4 chain is no longer running
and has not finished. Never runs two GPU jobs at once: the watchdog only calls it when no
v4 python process is alive.
"""
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUN = os.path.join(ROOT, "data", "yolo_ball_runs", "milb4_v4")
W = os.path.join(RUN, "weights", "best.pt")
EPOCHS = 15
PARKS = [  # (game, clip dir, extra grading args)
    ("822515", "data/clips/ogando_2026", ["--pose", "output/park_pose_822515_pass1.npz"]),
    ("816698", "data/clips/aaa_816698", []),
    ("815419", "data/clips/aaa_815419", []),
]


def sh(args, out=None):
    print(">>", " ".join(args), flush=True)
    r = subprocess.run([sys.executable, "-u"] + args, cwd=ROOT,
                       stdout=out, stderr=subprocess.STDOUT if out else None, text=True)
    if r.returncode:
        print("   FAILED rc=%d" % r.returncode, flush=True)
        sys.exit(r.returncode)


def epochs_done():
    p = os.path.join(RUN, "results.csv")
    return max(0, sum(1 for _ in open(p)) - 1) if os.path.exists(p) else 0


def main():
    done = epochs_done()
    print("v4 epochs recorded: %d/%d" % (done, EPOCHS), flush=True)
    if done < EPOCHS:
        sh(["scripts/train_ball_detector.py", "--data", "data/yolo_milb4/data.yaml",
            "--epochs", str(EPOCHS), "--batch", "16", "--imgsz", "320", "--device", "0",
            "--name", "milb4_v4", "--resume"])
    assert os.path.exists(W), W
    for g, d, _ in PARKS:
        if os.path.exists(os.path.join(ROOT, "output", "detect_tracks_%s_v4.csv" % g)):
            print("detect %s: already done, skipping" % g, flush=True)
            continue
        sh(["scripts/detect_ball_tracks.py", "--dir", d, "--game", g, "--weights", W,
            "--device", "0", "--two-pass", "--tag", "_v4"])
    with open(os.path.join(ROOT, "output", "v4_grades.txt"), "w", encoding="utf-8") as f:
        for g, d, extra in PARKS:
            f.write("\n### %s\n" % g)
            f.flush()
            sh(["scripts/compare_models_at_plate.py", "--game", g, "--dir", d,
                "--tags", "_v1,_v3,_v4"] + extra, out=f)
    print("V4 COMPLETE -> output/v4_grades.txt", flush=True)


if __name__ == "__main__":
    main()
