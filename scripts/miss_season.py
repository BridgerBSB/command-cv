"""Every 2026 game's misses: fetch Houston's pitches, run miss_new_park, gate, delete the video.

Zac, 2026-10-11: "we need the misses for every game of 2026". The pilot proved the per-game
chain (miss_new_park.py: ball-only camera + glove set-ups, graded in inches on held-out clips)
at Sugar Land and Corpus. This is that chain run over output/season_pull_2026.csv.

PER GAME, IN ORDER
  1. classify   which park, is its camera recipe known (docs/miss_park_registry.csv)
  2. aside      step outputs from ANOTHER clip folder of the same game (the Ogando pilot ran
                822515 & co. on one pitcher's clips) move to output/_aside_<g>_preseason/,
                or every miss_new_park step would skip on them. Printed, never deleted.
  3. fetch      Houston's pitches only (pitching team's parent org 117 in the game feed),
                prefetched during the previous game. Stems are numbered over the whole game.
  4. miss       miss_new_park.py --ours-only with the park's detect window
  5. gate       held-out B_ok gap (grade_holdout.grade) <= GATE_BOK_IN -> PASS, else FAIL
  6. delete     the mp4s (the cvpull CSV is the durable part; a re-run re-downloads)
  7. ledger     one row per game in docs/miss_season_ledger.csv, written as it finishes
Every PASS_EVERY passes, and at the end, the org board is rebuilt from the pilot games plus
every PASS (command_board_org.py --name season2026).

PARKS. A park is `ready` when a game there has passed with a known detect window; those games
run unattended. A `setup` park runs ONE game (its first, default window) and stops there: the
ledger prints the window that game measured, and the park becomes `ready` only after its
reference render (park_pose_<g>_ref.png) has been looked at and the registry row edited. That
is the one hand step per park, and it is the step miss_new_park.py says it stops for. A park
with no registry row needs its scorebug boxes marked first; `blocked` parks are skipped.

THE WINDOW IS PER PARK, NOT PER GAME (PROGRESS-LOG 125). A second game at a park run with the
default window let post-pitch chains into the tracker's consensus and wrecked the camera at
BOTH home parks; the window game 1 measured fixed both.

THE GATE. A game whose own camera does not reproduce the tracking system's plate crossing on
held-out clips does not go on the board. GATE_BOK_IN = 0.47 in is the top of the solved parks'
range (miss_new_park.py); admitted cameras graded 0.26-0.45 and every rejected one 1.13 or
worse, so the line does not sit in a crowd. Reference renders of passing games are still
meant to be looked at: --refs-sheet tiles the ones whose `eye` cell is blank.

BUILT FOR A MACHINE THAT KILLS LONG JOBS (long-jobs-on-this-laptop.md). Run it detached
(scripts/run_detached.ps1). A game killed mid-run has no ledger row and resumes from its
clips and step outputs on the next run; a finished game is never redone unless asked.

    python scripts/miss_season.py --dry-run                 # the plan: what runs, what waits
    python scripts/miss_season.py --max-games 3             # a short real run
    python scripts/miss_season.py --games 814876,817528
    python scripts/miss_season.py --refs-sheet              # eye-check sheets for PASS games
"""
from __future__ import annotations

import argparse
import csv as _csv
import glob
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "output")
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from grade_holdout import grade  # noqa: E402
from miss_new_park import STEP_OUTPUTS, V4, stale_outputs  # noqa: E402
from pitch_context import fetch as fetch_context  # noqa: E402
from rolling_pull import MIN_DOWNLOADED, fetch_game, free_gb  # noqa: E402

SEASON = os.path.join(OUT, "season_pull_2026.csv")
CVPULL = os.path.join(OUT, "cvpull")
REGISTRY = os.path.join(ROOT, "docs", "miss_park_registry.csv")
ADMISSION = os.path.join(ROOT, "docs", "miss_camera_admission.csv")
LEDGER = os.path.join(ROOT, "docs", "miss_season_ledger.csv")
LOGS = os.path.join(OUT, "miss_season", "logs")
HOU = 117
GATE_BOK_IN = 0.47
WEIGHTS = {"v4": V4, "v8r": "data/yolo_ball_runs/milb8r_proj/weights/best.pt"}
GAME_GB = 1.3            # Houston's half of a game, ~170-200 clips at ~6 MB, plus headroom
BOARD_NAME = "season2026"

# The pilot board (PROGRESS-LOG 125) measured these through landmark cameras before the
# ball-only route existed. 815419 holds every pitch of its game. The rest hold ONE pitcher's
# clips (the Ogando pull), so the season run redoes those games in full and its PASS
# supersedes the pilot entry on the board.
PILOT_BOARD = {815419: "_s2c", 822515: "_s2c", 822512: "_s2c", 822507: "_s2c", 822498: "_s2c",
               822504: "_s2c", 821922: "_s2lm2", 821847: "_s2c"}
PILOT_COMPLETE = {815419}

LEDGER_COLS = ["game_pk", "date", "level", "team", "side", "venue_id", "venue", "role",
               "clips_ours", "clips_missing", "heldout_n", "b_ok", "bok_gap_in", "gap_med_in",
               "gap_p90_in", "plaus", "verdict", "window_used", "window_measured", "dl_min",
               "run_min", "ref_png", "eye", "finished_ct", "note"]


def now_ct():
    try:
        from zoneinfo import ZoneInfo
        return datetime.now(ZoneInfo("America/Chicago")).strftime("%Y-%m-%d %H:%M")
    except Exception:                                    # noqa: BLE001  (no tzdata on the box)
        return datetime.now().strftime("%Y-%m-%d %H:%M") + " local"


def read_ledger():
    if not os.path.exists(LEDGER):
        return pd.DataFrame(columns=LEDGER_COLS)
    return pd.read_csv(LEDGER, dtype={"eye": str, "note": str})


def append_ledger(row):
    new = not os.path.exists(LEDGER)
    with open(LEDGER, "a", newline="", encoding="utf-8") as f:
        w = _csv.DictWriter(f, fieldnames=LEDGER_COLS, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerow(row)


def write_atomic(df, path):
    """Temp file + os.replace. The running job re-reads the registry and the ledger before
    every game; a half-written ledger would drop games from its done-set and re-run them."""
    tmp = path + ".tmp"
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def mark_eye(games):
    led = pd.read_csv(LEDGER, dtype=str)
    for g in games:
        i = led.index[(led.game_pk == str(g)) & (led.verdict == "PASS")]
        assert len(i), "%d has no PASS row in the ledger" % g
        led.loc[i, "eye"] = "ok " + now_ct()[:10]
    write_atomic(led, LEDGER)
    print("eye check recorded for %s" % ", ".join(map(str, games)))


def set_ready(venue_id, note=""):
    """Setup -> ready, at the window the park's PASSING trial measured. Run only after its
    reference render has been looked at (and recorded with --eye-ok)."""
    led = pd.read_csv(LEDGER, dtype=str)
    t = led[(led.venue_id == str(venue_id)) & (led.role == "setup") & (led.verdict == "PASS")]
    assert len(t), "venue %d has no PASSING trial" % venue_id
    t = t.iloc[-1]
    assert str(t.eye).startswith("ok"), "trial %s's render has not been eye-checked (--eye-ok)" % t.game_pk
    w = str(t.window_measured).split()
    assert len(w) == 4 and w[0] == "--lo" and w[2] == "--hi", "no measured window: %r" % t.window_measured
    reg = pd.read_csv(REGISTRY, dtype=str)
    i = reg.index[reg.venue_id == str(venue_id)]
    assert len(i) == 1 and reg.loc[i, "status"].iloc[0] == "setup", "venue %d is not in setup" % venue_id
    reg.loc[i, ["status", "lo", "hi"]] = ["ready", w[1], w[3]]
    reg.loc[i, "source_games"] = reg.loc[i, "source_games"].iloc[0] + " " + t.game_pk
    reg.loc[i, "note"] = reg.loc[i, "note"].iloc[0] + (
        " TRIAL %s (%s): PASS held-out %.2f in (B_ok n %s), render eye-checked. Window = the one it "
        "measured.%s" % (t.game_pk, now_ct()[:10], float(t.bok_gap_in), t.b_ok,
                         (" " + note) if note else ""))
    write_atomic(reg, REGISTRY)
    print("%s ready at --lo %s --hi %s" % (reg.loc[i, "venue"].iloc[0], w[1], w[3]))


def read_registry():
    r = pd.read_csv(REGISTRY, dtype={"lo": "Int64", "hi": "Int64"})
    return {int(x.venue_id): x._asdict() for x in r.itertuples(index=False)}


def sweep_downloads():
    """Copy cvpull_<pk>.csv files the Chrome pull left in Downloads into output/cvpull/.

    EVERY profile's Downloads, not just this process's: on this laptop Claude Code runs as
    `Owner` while Zac's Chrome saves to C:\\Users\\zbrid\\Downloads (2026-10-12; a sweep of `~`
    alone would have waited forever on files sitting in the other folder)."""
    got = 0
    for p in glob.glob(os.path.join(os.path.dirname(os.path.expanduser("~")), "*", "Downloads",
                                    "cvpull_*.csv")):
        b = os.path.basename(p)
        if not b[7:-4].isdigit():                       # "cvpull_123 (1).csv" duplicates
            continue
        dst = os.path.join(CVPULL, b)
        if not os.path.exists(dst):
            shutil.copy2(p, dst)
            got += 1
    return got


def pilot_games():
    """Games miss_new_park ran BEFORE this script: a scope marker with no season marker. The
    season marker is written before a game starts, so a season game killed mid-run is not one
    of these and resumes."""
    out = set()
    for p in glob.glob(os.path.join(OUT, "miss_scope_*.txt")):
        g = os.path.basename(p)[len("miss_scope_"):-4]
        if g.isdigit() and not os.path.exists(os.path.join(OUT, "miss_season_marker_%s.txt" % g)):
            out.add(int(g))
    return out


def done_games(ledger, retry_failed):
    keep = ledger if not retry_failed else ledger[ledger.verdict == "PASS"]
    return set(keep.game_pk.astype(int)) | PILOT_COMPLETE | pilot_games()


def plan(season, reg, ledger, a):
    """Every season game with a status. Runnable rows carry the park and the role."""
    done = done_games(ledger, a.retry_failed)
    # A park's trial is used up by any trial row - except under --retry-failed, where only a
    # PASSING trial counts, so a failed trial reruns (at the window now in the registry).
    setup_rows = ledger[ledger.role == "setup"]
    if a.retry_failed:
        setup_rows = setup_rows[setup_rows.verdict == "PASS"]
    setup_taken = set(setup_rows.venue_id.astype(int))
    want = {int(x) for x in a.games.split(",") if x.strip()}
    vids = {int(x) for x in a.venue_ids.split(",") if x.strip()}
    rows = []
    for r in season.sort_values(["date", "game_pk"]).itertuples(index=False):
        g, vid = int(r.game_pk), int(r.venue_id)
        if want and g not in want:
            continue
        if vids and vid not in vids:
            continue
        park = reg.get(vid)
        csv_p = os.path.join(CVPULL, "cvpull_%d.csv" % g)
        if g in done:
            st = "done"
        elif not os.path.exists(csv_p):
            st = "awaiting pull"
        elif park is None:
            st = "park needs scorebug boxes"
        elif park["status"] == "blocked":
            st = "park blocked"
        elif park["status"] == "ready":
            st = "run"
        elif park["status"] == "setup" and a.no_setup:
            st = "park in setup (--no-setup)"
        elif park["status"] == "setup" and vid not in setup_taken:
            st = "run setup"
            setup_taken.add(vid)
        else:
            st = "park in setup, awaiting review"
        rows.append(dict(game_pk=g, date=r.date, level=r.level, team=r.team, side=r.side,
                         venue_id=vid, venue=r.venue, status=st, park=park, csv=csv_p))
    return rows


def houston_play_ids(g):
    ctx = fetch_context(g)
    return set(ctx.loc[ctx.pitch_team_org == HOU, "play_id"])


def fetch(job):
    """Download Houston's clips of one game. Returns (ok, n, minutes)."""
    t0 = time.time()
    ours = houston_play_ids(job["game_pk"])
    if not ours:
        return 0, 0, 0.0
    d = os.path.join(ROOT, "data", "clips", "pull_%d" % job["game_pk"])
    ok, n = fetch_game(job["csv"], d, workers=8, max_clips=0, keep=ours)
    return ok, n, (time.time() - t0) / 60


def trim_manifest(g):
    """Drop rows whose clip never arrived. Called only above MIN_DOWNLOADED; the count goes in
    the ledger, so a short game is recorded as short rather than passing for a whole one."""
    d = os.path.join(ROOT, "data", "clips", "pull_%d" % g)
    mp = os.path.join(d, "pull_%d_manifest.csv" % g)
    m = pd.read_csv(mp)
    have = m.stem.map(lambda s: os.path.exists(os.path.join(d, s + ".mp4")))
    if not have.all():
        m[have].to_csv(mp, index=False)
    return int((~have).sum())


def aside_stale(g, park):
    """Move another pull's step outputs + landmark file out of the way. Never deletes."""
    moved = []
    stale = stale_outputs(g)
    if stale:
        dst = os.path.join(OUT, "_aside_%d_preseason" % g)
        os.makedirs(dst, exist_ok=True)
        for n in stale:
            shutil.move(os.path.join(OUT, n), os.path.join(dst, n))
            moved.append(n)
    lm = os.path.join(ROOT, "data", "landmarks", "%d.json" % g)
    boxes = json.loads(park["overlay_boxes"])
    if os.path.exists(lm):
        j = json.load(open(lm, encoding="utf-8"))
        if j.get("ref_stem") or j.get("overlay_boxes") != boxes:
            dst = os.path.join(ROOT, "data", "landmarks", "_aside_preseason")
            os.makedirs(dst, exist_ok=True)
            shutil.move(lm, os.path.join(dst, "%d.json" % g))
            moved.append("data/landmarks/%d.json" % g)
    if not os.path.exists(lm):
        json.dump(dict(game=g, overlay_boxes=boxes,
                       overlay_note="%s: copied from docs/miss_park_registry.csv (boxes marked on "
                                    "game %s)" % (park["venue"], park["source_games"]),
                       purpose="miss distance, season run (miss_season.py); no ref_stem, so "
                               "stabilize_park uses the set-up nearest the median"),
                  open(lm, "w", encoding="utf-8"), indent=2)
    return moved


def retry_aside(g):
    """--retry-failed: move this script's own outputs of a FAIL/ERROR game aside, numbered."""
    k = 1
    while os.path.exists(os.path.join(OUT, "_aside_%d_retry%d" % (g, k))):
        k += 1
    dst = os.path.join(OUT, "_aside_%d_retry%d" % (g, k))
    names = [n.format(g=g) for n in STEP_OUTPUTS] + ["miss_scope_%d.txt" % g,
                                                      "miss_season_marker_%d.txt" % g]
    for n in names:
        if os.path.exists(os.path.join(OUT, n)):
            os.makedirs(dst, exist_ok=True)
            shutil.move(os.path.join(OUT, n), os.path.join(dst, n))


def delete_clips(g):
    """Both folders. hou_<g> holds HARD LINKS into pull_<g>, so a file's bytes are freed only
    when its last link goes (st_nlink == 1) - counting every unlink reported twice the space."""
    freed = 0
    for d in ("pull_%d" % g, "hou_%d" % g):
        for p in glob.glob(os.path.join(ROOT, "data", "clips", d, "*.mp4")):
            st = os.stat(p)
            if st.st_nlink <= 1:
                freed += st.st_size
            os.remove(p)
    return freed / 2 ** 30


def window_measured(log_path):
    """The '--lo L --hi H would have sufficed' line detect_ball_tracks printed for this game
    (the LAST one: the log appends across resumed runs)."""
    found = ""
    try:
        for line in open(log_path, encoding="utf-8", errors="replace"):
            if "would have sufficed" in line:
                found = line.split("->")[-1].replace("would have sufficed", "").strip()
    except OSError:
        pass
    return found


def gate(bok_gap):
    """PASS when the held-out B_ok gap is inside the solved parks' range. NaN (no held-out clip
    whose track reached the plate) is no evidence, so it FAILS rather than passing by default."""
    return "PASS" if bok_gap <= GATE_BOK_IN else "FAIL"


def run_game(job, dl, a):
    g, park = job["game_pk"], job["park"]
    role = "setup" if job["status"] == "run setup" else "season"
    base = {k: job[k] for k in ("game_pk", "date", "level", "team", "side", "venue_id", "venue")}
    base.update(role=role)
    ok, n, dl_min = dl
    if n == 0:
        return dict(base, verdict="SKIP", note="no Houston pitches with a clip URL in the pull CSV",
                    finished_ct=now_ct())
    if ok < MIN_DOWNLOADED * n:
        print("    INCOMPLETE DOWNLOAD %d/%d -- left on disk, retried next run" % (ok, n))
        return None
    missing = trim_manifest(g)

    flags = "--window-pick"
    # The park's window whenever the registry has one: a ready park's games, and a setup park's
    # RETRIED trial (its first trial ran at the default window and measured this one).
    if not pd.isna(park["lo"]):
        flags += " --lo %d --hi %d" % (park["lo"], park["hi"])
    weights = WEIGHTS.get(str(park["weights"]), str(park["weights"]))
    cmd = [sys.executable, "-u", os.path.join(ROOT, "scripts", "miss_new_park.py"),
           "--game", str(g), "--dir", "data/clips/pull_%d" % g, "--ours-only",
           "--weights", weights,
           # '=' form: argparse reads a bare "--window-pick" value as an option of its own
           # (2026-10-11, the first season game errored on exactly that).
           "--detect-flags=" + flags]
    os.makedirs(LOGS, exist_ok=True)
    log_p = os.path.join(LOGS, "%d.log" % g)
    # BEFORE the run, so a kill mid-game leaves a season game (resumes), not a pilot one.
    open(os.path.join(OUT, "miss_season_marker_%d.txt" % g), "w").write(role)
    t0 = time.time()
    with open(log_p, "a", encoding="utf-8") as log:
        log.write("\n[%s] %s\n" % (now_ct(), " ".join(cmd)))
        log.flush()
        rc = subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT).returncode
    run_min = (time.time() - t0) / 60
    row = dict(base, clips_ours=ok - missing, clips_missing=(n - ok) + missing,
               window_used=flags.replace("--window-pick", "").strip() or "default",
               window_measured=window_measured(log_p), dl_min=round(dl_min, 1),
               run_min=round(run_min, 1), finished_ct=now_ct())
    if rc:
        tail = open(log_p, encoding="utf-8", errors="replace").read().strip().splitlines()[-1:]
        return dict(row, verdict="ERROR", note=(tail[0] if tail else "rc %d" % rc)[:300])
    r = grade(g, "_bo")
    verdict = gate(r["bok_gap"])
    return dict(row, heldout_n=r["n"], b_ok=r["b_ok"], bok_gap_in=r["bok_gap"],
                gap_med_in=r["gap_med"], gap_p90_in=r["gap_p90"], plaus=r["plaus"],
                verdict=verdict, ref_png="output/park_pose_%d_ref.png" % g, eye="",
                note="" if verdict == "PASS" else
                "held-out B_ok gap %.2f in over the %.2f in gate" % (r["bok_gap"], GATE_BOK_IN))


def board_games(ledger):
    games = {}
    for g, t in PILOT_BOARD.items():
        games[g] = t
    if os.path.exists(ADMISSION):
        adm = pd.read_csv(ADMISSION)
        for g in adm.loc[adm.verdict == "ADMIT", "game_pk"].astype(int):
            games[g] = "_bo"
    for g in ledger.loc[ledger.verdict == "PASS", "game_pk"].astype(int):
        games[g] = "_bo"                       # a season PASS supersedes a one-pitcher pilot run
    return games


def rebuild_board(ledger):
    games = board_games(ledger)
    cmd = [sys.executable, "-u", os.path.join(ROOT, "scripts", "command_board_org.py"),
           "--name", BOARD_NAME, "--games"] + ["%d:%s" % kv for kv in sorted(games.items())]
    p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    tail = (p.stdout or "").strip().splitlines()[-3:] if p.returncode == 0 else \
        (p.stderr or "").strip().splitlines()[-3:]
    print("    board (%d games): %s" % (len(games), " | ".join(tail)))


def refs_sheet(ledger, per=12, cols=3):
    """Tile the reference renders of PASS games not yet eye-checked, 12 to a sheet."""
    import cv2
    import numpy as np
    todo = ledger[(ledger.verdict == "PASS") & (ledger.eye.fillna("") == "")]
    out_dir = os.path.join(OUT, "miss_season")
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for k in range(0, len(todo), per):
        tiles = []
        for r in todo.iloc[k:k + per].itertuples(index=False):
            im = cv2.imread(os.path.join(ROOT, r.ref_png))
            if im is None:
                im = np.full((360, 640, 3), 255, np.uint8)
            im = cv2.resize(im, (640, int(im.shape[0] * 640 / im.shape[1])))
            im = cv2.copyMakeBorder(im, 34, 0, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))
            cv2.putText(im, "%d  %s  %s  held-out %.2f in" % (r.game_pk, r.date, r.venue[:24],
                                                               r.bok_gap_in),
                        (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (98, 45, 0), 2, cv2.LINE_AA)
            tiles.append(im)
        h = max(t.shape[0] for t in tiles)
        tiles = [cv2.copyMakeBorder(t, 0, h - t.shape[0], 0, 0, cv2.BORDER_CONSTANT,
                                    value=(255, 255, 255)) for t in tiles]
        while len(tiles) % cols:
            tiles.append(np.full_like(tiles[0], 255))
        grid = np.vstack([np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)])
        p = os.path.join(out_dir, "refs_%02d.png" % (k // per + 1))
        cv2.imwrite(p, grid)
        paths.append(p)
    print("%d PASS games awaiting an eye check -> %d sheet(s)" % (len(todo), len(paths)))
    for p in paths:
        print("  " + p)


def order(jobs):
    """Runnable games, best first: games at READY parks before setup trials (a ready game goes
    straight on the board; a trial needs an eye check before its park's other games can run),
    then by date."""
    run = [j for j in jobs if j["status"] in ("run", "run setup")]
    return sorted(run, key=lambda j: (j["status"] != "run", j["date"], j["game_pk"]))


def report_plan(jobs, a, swept):
    counts = pd.Series([j["status"] for j in jobs]).value_counts()
    print("season plan (%d games%s; %d new pull CSV(s) copied from Downloads):"
          % (len(jobs), ", filtered" if a.games or a.venue_ids else "", swept))
    for st, k in counts.items():
        print("  %-34s %4d" % (st, k))
    runnable = order(jobs)
    by_park = pd.Series([j["venue"] for j in runnable]).value_counts()
    if len(by_park):
        print("runnable by park: " + ", ".join("%s %d" % kv for kv in by_park.items()))
    waiting = [j for j in jobs if j["status"].startswith("park ")]
    if waiting:
        print("waiting parks: " + ", ".join(
            "%s (%s) %d" % (v, s.replace("park ", ""), k) for (v, s), k in
            pd.Series([(j["venue"], j["status"]) for j in waiting]).value_counts().items()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="print the plan and stop")
    ap.add_argument("--games", default="", help="comma-separated gamePks (default: the season)")
    ap.add_argument("--venue-ids", default="", help="comma-separated venue_ids")
    ap.add_argument("--max-games", type=int, default=0, help="stop after this many games run")
    ap.add_argument("--no-setup", action="store_true", help="skip setup parks' trial games")
    ap.add_argument("--retry-failed", action="store_true",
                    help="re-run FAIL / ERROR games (their outputs move to _aside_<g>_retry<k>/)")
    ap.add_argument("--min-free-gb", type=float, default=6.0)
    ap.add_argument("--keep-clips", action="store_true", help="debugging one game only")
    ap.add_argument("--pass-every", type=int, default=10, help="rebuild the board every N passes")
    ap.add_argument("--idle-exit-min", type=float, default=0,
                    help="when nothing is runnable, keep checking Downloads for new pull CSVs "
                         "this many minutes before exiting (0 = exit at once)")
    ap.add_argument("--refs-sheet", action="store_true", help="eye-check sheets and stop")
    ap.add_argument("--board", action="store_true", help="rebuild the board and stop")
    ap.add_argument("--eye-ok", default="", help="comma-separated PASS games whose ref render "
                                                 "was looked at and is right; records it and stops")
    ap.add_argument("--ready", type=int, default=0,
                    help="venue_id: setup -> ready at its eye-checked passing trial's window; stops")
    a = ap.parse_args()

    if a.eye_ok:
        return mark_eye([int(x) for x in a.eye_ok.split(",") if x.strip()])
    if a.ready:
        return set_ready(a.ready)
    ledger = read_ledger()
    if a.refs_sheet:
        return refs_sheet(ledger)
    if a.board:
        return rebuild_board(ledger)

    season = pd.read_csv(SEASON)
    swept = sweep_downloads()
    jobs = plan(season, read_registry(), ledger, a)
    report_plan(jobs, a, swept)
    if a.dry_run:
        return 0

    print("\n%.1f GB free, floor %.1f GB\n" % (free_gb(), a.min_free_gb))
    # RE-PLAN AFTER EVERY GAME. Pull CSVs land in Downloads while this runs (the Chrome pull
    # is hours), and a home game at a ready park must not wait behind a day of setup trials,
    # so each step re-reads the ledger, the registry and the CSVs and takes the best game.
    # `tried` holds every game attempted this run: a download that keeps failing, or a FAIL
    # under --retry-failed, is not picked twice.
    tried, ran, passes, idle_since = set(), 0, 0, None
    nxt, nxt_g = None, None
    with ThreadPoolExecutor(1) as pool:
        while not (a.max_games and ran >= a.max_games):
            ledger = read_ledger()
            got = sweep_downloads()
            if got:
                print("    %d new pull CSV(s) copied from Downloads" % got)
            runnable = order([j for j in plan(season, read_registry(), ledger, a)
                              if j["game_pk"] not in tried])
            if not runnable:
                idle_since = idle_since or time.time()
                if (time.time() - idle_since) / 60 >= a.idle_exit_min:
                    break
                time.sleep(60)
                continue
            idle_since = None
            job = runnable[0]
            g = job["game_pk"]
            tried.add(g)
            if a.retry_failed and g in set(ledger.game_pk.astype(int)):
                retry_aside(g)
            moved = aside_stale(g, job["park"])
            if moved:
                print("    moved aside (another pull of %d): %s" % (g, ", ".join(moved)))
            if nxt_g is not None and nxt_g != g:
                # The plan changed under the prefetch (new CSVs, a park made ready). Let it
                # finish; its clips stay cached for when that game's turn comes. Its own
                # failure is its own, never this game's.
                try:
                    nxt.result()
                except Exception as e:                 # noqa: BLE001
                    print("    (stale prefetch of %d failed: %s)" % (nxt_g, e))
            try:
                dl = nxt.result() if nxt_g == g else fetch(job)
            except Exception as e:                     # noqa: BLE001  (feed / network)
                print("== %d  fetch failed: %s -- left for the next run" % (g, e))
                dl = None
            # Prefetch the next game while this one runs: the download is network-bound and
            # the run is GPU/CPU-bound, so the overlap is free. Under the disk floor it is
            # fetched in line instead, after this game's clips are gone.
            nxt, nxt_g = None, None
            if len(runnable) > 1 and free_gb() > a.min_free_gb + 2 * GAME_GB:
                nxt, nxt_g = pool.submit(fetch, runnable[1]), runnable[1]["game_pk"]
            if dl is None:
                continue
            ran += 1
            print("== %d  %s %s %-24s %s  (%d runnable)" % (g, job["date"], job["level"],
                                                           job["venue"][:24], job["status"],
                                                           len(runnable)), flush=True)
            row = run_game(job, dl, a)
            if row is None:
                continue
            gb = 0.0 if a.keep_clips else delete_clips(g)
            append_ledger(row)
            ledger = read_ledger()
            gap = row.get("bok_gap_in")
            print("    %s  held-out %s in (B_ok n %s)  run %.1f min, download %.1f min, "
                  "%.2f GB freed -> %.1f GB free"
                  % (row["verdict"], "-" if gap is None or gap != gap else "%.2f" % gap,
                     row.get("b_ok", "-"), row.get("run_min") or 0, row.get("dl_min") or 0, gb,
                     free_gb()), flush=True)
            if row["verdict"] in ("FAIL", "ERROR"):
                print("    " + row.get("note", ""))
            if row["role"] == "setup" and row["verdict"] in ("PASS", "FAIL"):
                print("    SETUP game for %s: window measured '%s'. Look at %s; if it is right, "
                      "set the park ready with that window in docs/miss_park_registry.csv"
                      % (job["venue"], row.get("window_measured", ""), row.get("ref_png", "")))
            if row["verdict"] == "PASS":
                passes += 1
                if passes % a.pass_every == 0:
                    rebuild_board(ledger)
            if free_gb() < a.min_free_gb:
                print("\nSTOPPING: %.1f GB free is under the floor. Nothing was half-written."
                      % free_gb())
                break
        if nxt is not None:
            try:
                nxt.result()                       # an unused prefetch; its clips stay cached
            except Exception:                      # noqa: BLE001
                pass
    if passes:
        rebuild_board(read_ledger())
    print("\nthis run: %d game(s) attempted, %d passed" % (ran, passes))
    print("ledger:\n%s" % read_ledger().verdict.value_counts().to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
