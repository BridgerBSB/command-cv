"""Grow the ball-detector training set one game at a time, without keeping the video.

WHY. A game's clips are 1-2 GB and this laptop has single-digit GB free, so "download every
game, then label" cannot run here. But training only needs ~20 frames around each pitch,
cropped to 320 px -- a few MB per game. So each game is streamed: download -> detect ->
label -> crop -> delete the mp4s, and the next game starts on the space the last one freed.

PER GAME, every step skips if its output already exists, so a kill or reboot resumes:
  1. subset manifest   data/clips/stream_<g>/stream_<g>_manifest.csv (evenly spaced,
                       --max-clips, from a manifest that already holds cf_url -- no Okta)
  2. download          sporty-clips mp4s, browser UA, no credential (src/ingest/video.py)
  3. detect            detect_ball_tracks.py, v4 weights, --window-pick
  4. label             labels_from_detector.py --measure-diameter (no pose needed); writes a
                       RANDOM-sample sheet output/det_labels_<g>_stream.png for the eye check
  5. crops             build_ball_dataset.py --all-train -> data/yolo_stream/<g>/
  6. delete mp4s       unless --keep-video (the sheet and crops survive)

NOTHING IS ADMITTED HERE. A park enters training only after its sample sheet is looked at
and a verdict is written to docs/park_admission.csv, the same gate v3/v4 parks went through
(several were REJECTED for tracks on umpire masks and catcher gear). The dataset assembly
step reads that ledger.

    python scripts/stream_games.py --games 814874 816447 827289 817525 --max-clips 120
"""
import argparse
import glob
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src.ingest.video import download  # noqa: E402

WEIGHTS = "data/yolo_ball_runs/milb4_v4/weights/best.pt"
STATUS = os.path.join(ROOT, "output", "stream_status.json")


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def source_manifest(g):
    for p in glob.glob(os.path.join(ROOT, "data", "clips", "*", "*_manifest.csv")):
        if os.path.basename(os.path.dirname(p)).startswith("stream_"):
            continue
        x = pd.read_csv(p)
        if "cf_url" in x and "game_pk" in x and (x.game_pk.astype(str) == str(g)).any():
            x = x[(x.game_pk.astype(str) == str(g)) & x.cf_url.notna()]
            if len(x):
                return x
    raise SystemExit("no manifest with clip URLs for game %s" % g)


def sh(args, logf):
    log(">> " + " ".join(args))
    with open(logf, "a", encoding="utf-8") as f:
        r = subprocess.run([sys.executable, "-u"] + args, cwd=ROOT, stdout=f,
                           stderr=subprocess.STDOUT, text=True)
    return r.returncode


def status_set(g, **kw):
    s = json.load(open(STATUS)) if os.path.exists(STATUS) else {}
    s.setdefault(str(g), {}).update(kw)
    json.dump(s, open(STATUS, "w"), indent=1)


def run_game(g, a):
    name = "stream_%s" % g
    d = os.path.join(ROOT, "data", "clips", name)
    os.makedirs(d, exist_ok=True)
    logf = os.path.join(ROOT, "output", "stream_%s.log" % g)
    t0 = time.time()

    mp = os.path.join(d, name + "_manifest.csv")
    if not os.path.exists(mp):
        x = source_manifest(g).reset_index(drop=True)
        if len(x) > a.max_clips:
            step = len(x) / a.max_clips
            x = x.iloc[[int(i * step) for i in range(a.max_clips)]]
        x.to_csv(mp, index=False)
    man = pd.read_csv(mp)
    venue = man["venue"].iloc[0] if "venue" in man else "?"
    log("game %s (%s): %d clips" % (g, venue, len(man)))

    tracks = os.path.join(ROOT, "output", "detect_tracks_%s_stream.csv" % g)
    labels = os.path.join(ROOT, "output", "det_labels_%s_stream.csv" % g)
    crops = os.path.join(ROOT, "data", "yolo_stream", str(g))
    done_crops = os.path.exists(os.path.join(crops, "data.yaml"))

    if not done_crops:
        todo = [(r.cf_url, os.path.join(d, r.stem + ".mp4")) for r in man.itertuples()
                if not os.path.exists(os.path.join(d, r.stem + ".mp4"))]
        if todo:
            def one(u_p):
                try:
                    download(*u_p)
                    return 1
                except Exception:
                    return 0
            with ThreadPoolExecutor(8) as ex:
                ok = sum(ex.map(one, todo))
            log("  downloaded %d/%d" % (ok, len(todo)))

        if not os.path.exists(tracks):
            if sh(["scripts/detect_ball_tracks.py", "--dir", d, "--game", str(g),
                   "--weights", WEIGHTS, "--window-pick", "--tag", "_stream"], logf):
                status_set(g, venue=venue, state="detect_failed")
                return
        if not os.path.exists(labels):
            if sh(["scripts/labels_from_detector.py", "--game", str(g), "--dir", d,
                   "--tracks", tracks, "--measure-diameter", "--tag", "_stream"], logf):
                status_set(g, venue=venue, state="label_failed")
                return
        if sh(["scripts/build_ball_dataset.py", "--labels", labels, "--clips", d,
               "--out", crops, "--all-train", "--jpg"], logf):
            status_set(g, venue=venue, state="crop_failed")
            return

    tr = pd.read_csv(tracks) if os.path.exists(tracks) else pd.DataFrame()
    lab = pd.read_csv(labels) if os.path.exists(labels) else pd.DataFrame()
    n_img = len(glob.glob(os.path.join(crops, "images", "*", "*.jpg")))
    if not a.keep_video:
        for p in glob.glob(os.path.join(d, "*.mp4")):
            os.remove(p)
    status_set(g, venue=venue, state="ready_for_eye_check", clips=int(len(man)),
               tracked=int(tr.stem.nunique()) if len(tr) else 0,
               labels=int(len(lab)), crops=int(n_img),
               sheet="output/det_labels_%s_stream.png" % g,
               minutes=round((time.time() - t0) / 60, 1))
    log("  done: %d/%d clips tracked, %d labels, %d crops, %.1f min" % (
        tr.stem.nunique() if len(tr) else 0, len(man), len(lab), n_img, (time.time() - t0) / 60))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", nargs="+", required=True)
    ap.add_argument("--max-clips", type=int, default=120)
    ap.add_argument("--keep-video", action="store_true")
    a = ap.parse_args()
    for g in a.games:
        try:
            run_game(g, a)
        except SystemExit as e:
            log("  SKIP %s: %s" % (g, e))
            status_set(g, state="skipped", note=str(e))


if __name__ == "__main__":
    main()
