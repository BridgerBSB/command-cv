"""Download a game, detect, label, DELETE the video, move on. The loop that scales.

THE ARITHMETIC THAT FORCES THIS SHAPE. A game is ~300 clips at ~6.2 MB = ~1.8 GB. The 8
venues already banked are 4,813 clips = ~30 GB against 25.5 GB free, and the full 63-venue
plan is ~36,500 clips = ~226 GB. Keeping the video is not an option at any of those sizes.

But the video is not the artifact. Measured on what is already on disk: 2,243 clips occupy
13.9 GB while the labelled crops they produce occupy 0.32 GB. So a game costs ~1.8 GB
TRANSIENT and leaves ~40 MB PERMANENT, and the whole 63-venue pull leaves about 5 GB. Disk
stops being the constraint the moment the mp4s are treated as scratch.

NO TOKEN IS NEEDED HERE. The Okta bearer gates the analytics JSON, which is already banked in
the cvpull CSVs. sporty-clips URLs need no auth (probed with a Chrome UA, a plain UA and no
referer -- all 200) and do not expire: a URL harvested in July still served in September. So
this can grind for hours, unattended, long after the token is dead.

BUILT FOR A MACHINE THAT KILLS LONG JOBS. See .claude/rules/long-jobs-on-this-laptop.md --
three background jobs were reaped for memory on 2026-09-18. So every stage writes its output
before the next begins, a game that already has labels is skipped, and the mp4s are deleted
as soon as the labels exist. A kill costs the game in flight, never the run, and re-running
resumes.

IT STOPS RATHER THAN FILLING THE DISK. --min-free-gb is checked before each game and the loop
exits cleanly when it would be crossed, because a disk that hits zero takes the desktop with
it and there is nothing to debug afterwards.

WHAT IT DOES NOT DO: verify. detect_ball_tracks emits DETECTIONS and this turns them into
labels, but nobody has LOOKED. Each game's render is kept (labels_from_detector writes one)
and a park whose sheet has not been read is not a park whose labels are trusted --
.claude/rules/render-and-look.md.

    python scripts/rolling_pull.py --weights data/yolo_ball_runs/milb4park/weights/best.pt
    python scripts/rolling_pull.py --weights ... --games 817407,825026 --min-free-gb 8
"""
from __future__ import annotations

import argparse
import csv as _csv
import glob
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src.ingest.video import download  # noqa: E402

MANIFEST_COLS = (["stem", "game_pk", "date", "venue", "play_id", "pitch_type", "hand",
                  "plate_x", "plate_z", "sz_top", "sz_bot", "call", "angle",
                  "clip_downloaded", "cf_url"]
                 + ["traj_x0", "traj_x1", "traj_x2", "traj_y0", "traj_y1", "traj_y2",
                    "traj_z0", "traj_z1", "traj_z2", "t_start", "t_end", "zone_time",
                    "rel_x", "rel_y", "rel_z"])


def free_gb():
    return shutil.disk_usage(ROOT).free / 2 ** 30


def run(cmd, label):
    p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    if p.returncode != 0:
        print("    %s FAILED rc=%d" % (label, p.returncode))
        print("      " + (p.stderr or "")[-500:].replace("\n", "\n      "))
        return None
    return p.stdout or ""


def fetch_game(csv_path, clip_dir, workers, max_clips):
    """Download one game's clips and write the manifest detect_ball_tracks expects."""
    rows = list(_csv.DictReader(open(csv_path, newline="", encoding="utf-8")))
    rows = [r for r in rows if r.get("cf_url")]
    if max_clips:
        rows = rows[:max_clips]
    os.makedirs(clip_dir, exist_ok=True)
    name = os.path.basename(clip_dir)
    for i, r in enumerate(rows):
        r["stem"] = "%s_%04d" % (name, i)

    def get(r):
        p = os.path.join(clip_dir, r["stem"] + ".mp4")
        if os.path.exists(p) and os.path.getsize(p) > 10000:
            return True
        try:
            download(r["cf_url"], p)
            return True
        except Exception:                                    # noqa: BLE001
            return False

    ok = 0
    with ThreadPoolExecutor(workers) as ex:
        futs = {ex.submit(get, r): r for r in rows}
        for fu in as_completed(futs):
            good = fu.result()
            futs[fu]["clip_downloaded"] = int(good)
            ok += good
    man = os.path.join(clip_dir, name + "_manifest.csv")
    with open(man, "w", newline="", encoding="utf-8") as f:
        w = _csv.DictWriter(f, fieldnames=MANIFEST_COLS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    return ok, len(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--cvpull-dir", default=os.path.join(ROOT, "output", "cvpull"))
    ap.add_argument("--games", default="", help="comma-separated gamePks; default = all banked")
    ap.add_argument("--min-free-gb", type=float, default=6.0)
    ap.add_argument("--max-clips", type=int, default=0, help="0 = every clip in the CSV")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--lo", type=int, default=150)
    ap.add_argument("--hi", type=int, default=250)
    ap.add_argument("--device", default="0")
    ap.add_argument("--keep-clips", action="store_true",
                    help="do NOT delete the mp4s. Only for debugging one game.")
    a = ap.parse_args()

    want = set(int(x) for x in a.games.split(",") if x.strip())
    csvs = sorted(glob.glob(os.path.join(a.cvpull_dir, "cvpull_*.csv")))
    jobs = []
    for c in csvs:
        pk = int(os.path.basename(c).split("_")[1].split(".")[0])
        if want and pk not in want:
            continue
        jobs.append((pk, c))
    print("%d banked game(s); %.1f GB free, floor %.1f GB\n"
          % (len(jobs), free_gb(), a.min_free_gb))

    summary = []
    for pk, csv_path in jobs:
        lab = os.path.join(ROOT, "output", "det_labels_%d.csv" % pk)
        if os.path.exists(lab):
            print("== %d  already labelled, skipping" % pk)
            continue
        if free_gb() < a.min_free_gb + 2.5:
            print("\nSTOPPING: %.1f GB free is under the floor plus one game. Nothing was "
                  "half-written." % free_gb())
            break
        rows = list(_csv.DictReader(open(csv_path, newline="", encoding="utf-8")))
        venue = rows[0].get("venue", "?") if rows else "?"
        print("== %d  %s  (%d clips)" % (pk, venue[:34], len(rows)))
        t0 = time.time()

        clip_dir = os.path.join(ROOT, "data", "clips", "pull_%d" % pk)
        ok, n = fetch_game(csv_path, clip_dir, a.workers, a.max_clips)
        print("    downloaded %d/%d  (%.1f GB free)" % (ok, n, free_gb()))
        if not ok:
            summary.append((pk, venue, 0, 0, "download failed"))
            continue

        out = run([sys.executable, os.path.join(ROOT, "scripts", "detect_ball_tracks.py"),
                   "--dir", clip_dir, "--game", str(pk), "--weights", a.weights,
                   "--lo", str(a.lo), "--hi", str(a.hi), "--device", a.device],
                  "detect")
        if out is None:
            summary.append((pk, venue, ok, 0, "detect failed"))
            continue
        line = [x for x in out.splitlines() if x.startswith("game %d:" % pk)]
        print("    " + (line[0] if line else "detect done"))

        out = run([sys.executable, os.path.join(ROOT, "scripts", "labels_from_detector.py"),
                   "--game", str(pk), "--dir", clip_dir, "--measure-diameter",
                   "--sample", "6", "--no-report-clip-size"], "label")
        if out is None:
            summary.append((pk, venue, ok, 0, "label failed"))
            continue
        npts = 0
        for x in out.splitlines():
            if "labelled points" in x:
                print("    " + x.strip())
                try:
                    npts = int(x.strip().split("clips,")[1].split("labelled")[0])
                except Exception:                            # noqa: BLE001
                    pass

        if not a.keep_clips:
            # The labels exist, so the video is scratch. Deleting HERE, before the next game
            # starts, is what makes the disk arithmetic work at 63 venues.
            mp4s = glob.glob(os.path.join(clip_dir, "*.mp4"))
            freed = sum(os.path.getsize(p) for p in mp4s) / 2 ** 30
            for p in mp4s:
                os.remove(p)
            print("    deleted %d mp4s (%.2f GB), manifest kept  -> %.1f GB free"
                  % (len(mp4s), freed, free_gb()))
        summary.append((pk, venue, ok, npts, "%.1f min" % ((time.time() - t0) / 60)))

    print("\n" + "=" * 74)
    print("%-9s %-30s %6s %8s  %s" % ("game", "venue", "clips", "points", "note"))
    for pk, v, ok, npts, note in summary:
        print("%-9d %-30s %6d %8d  %s" % (pk, v[:30], ok, npts, note))
    print("=" * 74)
    print("%.1f GB free. Labels are DETECTIONS nobody has looked at -- read each game's "
          "det_labels_<pk>.png before trusting them." % free_gb())
    return 0


if __name__ == "__main__":
    sys.exit(main())
