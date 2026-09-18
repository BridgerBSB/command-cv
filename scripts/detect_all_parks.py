"""Run the ball detector over every clip directory on disk, one park at a time.

WHY A DRIVER AND NOT A FOR-LOOP IN A SHELL. Three reasons, all learned:
  - The GPU is 4.3 GB. Two detector processes fit and then do not, and the second one dies
    somewhere in the middle of a park with a partial CSV that looks finished. Serial by
    construction.
  - A park that yields nothing must be REPORTED, not skipped silently. L.P. Frans reported
    "0 tracks" for a week as if it were a property of the park; it was a stale differencing
    implementation. A run that ends with a table naming every park and its yield makes that
    kind of zero impossible to miss.
  - Parks differ in where the plate frame falls, so --lo/--hi are per-park. Keeping them in
    one table beside the directory is the only place they stay consistent between runs.

WHAT IT DOES NOT DO. It does not write labels. detect_ball_tracks emits DETECTIONS; turning
those into labels goes through audit_seed_tracks (which renders them), because a detection
that nobody looked at is not a label -- see .claude/rules/render-and-look.md.

    python scripts/detect_all_parks.py --weights data/yolo_ball_runs/milb4park/weights/best.pt
    python scripts/detect_all_parks.py --only aaa_816698,a_827106 --weights ...
    python scripts/detect_all_parks.py --skip-done --weights ...
"""
from __future__ import annotations

import argparse
import glob
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# (clip dir, game_pk, lo, hi). The windows come from the chain-end distribution each run
# prints; a park with no measurement yet keeps the wide default so an unseen park cannot
# read as "the detector fails there" (probe-capability-before-accepting-a-blocker.md).
PARKS = [
    ("aaa_815419", 815419, 178, 232),   # Isotopes, chain ends p05 203 / med 213 / p95 218
    ("aplus_821819", 821819, 178, 232),  # Frawley
    ("aaa_816698", 816698, 150, 250),   # Durham
    ("aa_817407", 817407, 150, 250),    # Riders -- pose fails here, detector does not need one
    ("a_827106", 827106, 150, 250),     # L.P. Frans -- broken cadence, which the detector ignores
    ("aplus_819261", 819261, 150, 250),  # Hops -- never yielded a plate frame by any method
    ("mlb_823850", 823850, 150, 250),   # loanDepot
    ("mlb_824018", 824018, 150, 250),   # Angel Stadium
    ("mlb_824259", 824259, 150, 250),   # Comerica
    ("mlb_824341", 824341, 150, 250),   # Coors
    # FOUR BALLPARKS THAT WERE ALREADY ON DISK. ogando_2026 is a PITCHER dir holding 149
    # clips from 13 games at 7 different A+ parks, so every game inside it is its own park
    # with its own camera. detect_ball_tracks now FILTERS the manifest by --game instead of
    # stamping it, which is what makes these reachable. Both of the first two are the
    # behind-the-pitcher archetype that differencing could never track and the detector
    # cracked at Hops today.
    ("ogando_2026", 822246, 150, 250),  # Greensboro, First National Bank Field -- 33 clips
    ("ogando_2026", 821922, 150, 250),  # Rome, AdventHealth Stadium -- 23 clips
]
# DELIBERATELY ABSENT, and each for a reason:
#   ogando_2026 / 822515 (60 clips, Asheville HomeTrust) is the HELD-OUT VAL PARK. Tracking
#     it is fine for evaluation; it must never reach a training set, so it is not queued
#     here where its labels would be one build command away from the train split.
#   The remaining games inside the two pitcher dirs are too small for a consensus flight
#     window to mean anything: Winston-Salem 7+3, Greenville 4+3, Bowling Green 2. Listed so
#     the next person knows they were counted and skipped, not missed.
#   mlb_824821 (289) and ogando_hardcases (69) predate the manifest convention -- guid-named
#     mp4s with a sidecar .json and no <dir>_manifest.csv. 824821's labels already exist in
#     data/yolo_ball5 from the pose path, so nothing is lost by leaving it here.


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--only", default="", help="comma-separated clip dir names")
    ap.add_argument("--skip-done", action="store_true",
                    help="skip a park whose output/detect_tracks_<pk>.csv already exists")
    ap.add_argument("--device", default="0")
    ap.add_argument("--conf", type=float, default=0.15)
    a = ap.parse_args()

    only = set(x.strip() for x in a.only.split(",") if x.strip())
    rows = []
    for name, pk, lo, hi in PARKS:
        d = os.path.join(ROOT, "data", "clips", name)
        out = os.path.join(ROOT, "output", "detect_tracks_%d.csv" % pk)
        if only and name not in only:
            continue
        n_clips = len(glob.glob(os.path.join(d, "*.mp4")))
        if not n_clips:
            rows.append((name, pk, 0, "NO CLIPS ON DISK", 0.0))
            continue
        if a.skip_done and os.path.exists(out):
            rows.append((name, pk, n_clips, "skipped (csv exists)", 0.0))
            continue
        cmd = [sys.executable, os.path.join(ROOT, "scripts", "detect_ball_tracks.py"),
               "--dir", d, "--game", str(pk), "--weights", a.weights,
               "--lo", str(lo), "--hi", str(hi), "--conf", str(a.conf),
               "--device", a.device]
        print("\n=== %s  (%d clips, f%d-%d) ===" % (name, n_clips, lo, hi), flush=True)
        t0 = time.time()
        p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
        mins = (time.time() - t0) / 60.0
        tail = [ln for ln in (p.stdout or "").splitlines() if ln.strip()]
        for ln in tail[-6:]:
            print("   " + ln)
        note = "rc=%d" % p.returncode
        for ln in tail:
            if ln.startswith("game %d:" % pk):
                note = ln.split(":", 1)[1].strip()
        if p.returncode != 0:
            print("   STDERR: " + (p.stderr or "")[-400:])
        rows.append((name, pk, n_clips, note, mins))

    print("\n" + "=" * 78)
    print("%-16s %-9s %6s  %-34s %6s" % ("park", "game_pk", "clips", "result", "min"))
    for name, pk, n, note, mins in rows:
        print("%-16s %-9d %6d  %-34s %6.1f" % (name, pk, n, note[:34], mins))
    print("=" * 78)
    print("A park reading 0 is a FINDING, not a skip. Render it before concluding anything.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
