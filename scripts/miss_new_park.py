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
import json
import os
import shutil
import subprocess
import sys
import time

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "output")
V4 = "data/yolo_ball_runs/milb4_v4/weights/best.pt"

# Every file a step below SKIPS ON, plus the files those steps write beside it and the set-up
# cache stabilize_park reads back. A copy of any of these from a DIFFERENT clip folder of the
# same game is silently reused: the Ogando pilot ran 822515 & co. on one pitcher's clips, and
# a full-game run would have kept its tracks, its held-out half (make_holdout_split refuses to
# redo one) and its set-up reference. miss_season.py moves these aside; a hand run refuses.
STEP_OUTPUTS = ["detect_tracks_{g}_v4.csv", "dets_{g}_v4.csv", "setups_{g}.csv",
                "targets_{g}_stab2c.csv", "detect_tracks_{g}_v4stab2c.csv",
                "stab2_clips_{g}c.csv", "stab2_H_{g}c.csv", "holdout_{g}.csv",
                "seeds_{g}_v4fit.csv", "park_pose_{g}_ballonly.npz", "park_pose_{g}_ref.png",
                "command_miss_{g}_bo.csv"]


def stale_outputs(g):
    """STEP_OUTPUTS that exist although this script never ran on game g (no scope marker)."""
    if os.path.exists(os.path.join(OUT, "miss_scope_%d.txt" % g)):
        return []
    return [n.format(g=g) for n in STEP_OUTPUTS if os.path.exists(os.path.join(OUT, n.format(g=g)))]


def sh(args):
    print("  $ " + " ".join(args), flush=True)
    t0 = time.time()
    r = subprocess.run([sys.executable, "-u"] + args, cwd=ROOT)
    if r.returncode:
        sys.exit("FAILED (%d): %s" % (r.returncode, " ".join(args)))
    # Per-step wall clock: "30 min a game" is not actionable, "detect 9 + glove 12" is.
    print("  [%s took %.1f min]" % (os.path.basename(args[0]), (time.time() - t0) / 60), flush=True)


def have(name):
    return os.path.exists(os.path.join(OUT, name))


def ours_only_dir(g, d):
    """data/clips/hou_<g>/ = hard links to the clips our pitchers threw, plus a filtered manifest."""
    ctx_p = os.path.join(OUT, "pitch_context_%d.csv" % g)
    man = pd.read_csv(os.path.join(d, os.path.basename(d) + "_manifest.csv"))
    # Refresh when the context predates pitch_team_org OR this clip folder: a context written
    # for another folder of the same game (the Ogando pilot) has none of these stems, and the
    # subset below would come out empty.
    if (not os.path.exists(ctx_p) or "pitch_team_org" not in pd.read_csv(ctx_p, nrows=1)
            or not set(man.stem[man.game_pk == g]) <= set(pd.read_csv(ctx_p).stem)):
        sh(["scripts/pitch_context.py", "--game", str(g), "--dir", d])
    ctx = pd.read_csv(ctx_p)
    ours = set(ctx.loc[ctx.pitch_team_org == 117, "stem"])
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


# THE WINDOW LADDER (2026-10-13). A park window measured on ONE game does not hold for the next:
# broadcast clip timing shifts game to game, so at Sugar Land the pitch crossed around f189 in
# 814875 and at or past f225 in 814868 - where --hi 225 cut 76% of the flights off before the
# plate and the camera failed (0.91 / 1.56 / 1543 in on three of seven games). detect_ball_tracks
# PRINTS "WINDOW TRUNCATION" past 5% of chains ending within 4 frames of --hi, and nothing read it.
# Now: past that same threshold the whole window SHIFTS later and detection re-runs. Shifting,
# not widening, keeps the window as narrow as the park needed - its narrowness is what keeps
# post-pitch junk chains out (PROGRESS-LOG 125).
TRUNC_FRAC = 0.05            # detect_ball_tracks' own WINDOW TRUNCATION threshold
TRUNC_MARGIN = 4             # ... and its "within 4 frames of --hi"
LADDER_SHIFT = 40
LADDER_STEPS = 3
LADDER_RESOLVED = 0.5          # a shift that cuts the clipped share below this fraction found the end


def _flag(flags, name):
    return int(flags[flags.index(name) + 1]) if name in flags else None


def truncation(g, hi):
    """Share of chains ending within TRUNC_MARGIN frames of --hi: flights cut off, not finished.

    Counted on the tracks the camera will use, AFTER the detector's consensus filter; the
    detector's own warning counts every chain before it. Same test, different rows: 814868 reads
    88% of 165 here vs 76% of 173 there, 814871 27% vs 20%, Corpus 817528 0.7% vs no warning."""
    t = pd.read_csv(os.path.join(OUT, "detect_tracks_%d_v4.csv" % g), usecols=["stem", "frame"])
    ends = t.groupby("stem").frame.max()
    return (float((ends >= hi - TRUNC_MARGIN).mean()) if len(ends) else 0.0), len(ends)


def detect_with_ladder(g, a):
    flags = a.detect_flags.split()
    wp = os.path.join(OUT, "miss_window_%d.txt" % g)
    if os.path.exists(wp) and have("detect_tracks_%d_v4.csv" % g):
        flags = open(wp).read().split()          # resume: the window the existing tracks used
    lo, hi = _flag(flags, "--lo"), _flag(flags, "--hi")
    for step in range(LADDER_STEPS + 1):
        if not have("detect_tracks_%d_v4.csv" % g):
            sh(["scripts/detect_ball_tracks.py", "--dir", a.dir, "--game", str(g), "--weights",
                a.weights, "--device", "auto", "--tag", "_v4"] + flags)
        if lo is None or hi is None:
            break                                # the default window: a trial, no park window yet
        open(wp, "w").write(" ".join(flags))
        frac, n = truncation(g, hi)
        if frac <= TRUNC_FRAC:
            break
        # After a shift, a share that fell by more than half means the flights now finish inside
        # the window; what still ends at the new --hi is post-pitch motion, not a cut-off pitch.
        # 814869 (2026-10-13): 62% -> 5% at 186/265, then 5% / 10% / 12% of junk at each new edge
        # walked the window to 266/345, past the pitch, and the camera fit collapsed (5,343 in).
        if step and frac < prev_frac * LADDER_RESOLVED:
            print("WINDOW LADDER: %.0f%% end at --hi %d after the shift (was %.0f%%); the flights "
                  "finish inside it now - stopping here" % (100 * frac, hi, 100 * prev_frac), flush=True)
            break
        prev_frac = frac
        if step == LADDER_STEPS:
            print("WINDOW LADDER: still %.0f%% of %d chains cut off at --hi %d after %d shifts; "
                  "the gate decides" % (100 * frac, n, hi, LADDER_STEPS), flush=True)
            break
        dst = os.path.join(OUT, "_aside_%d_trunc_hi%d" % (g, hi))
        os.makedirs(dst, exist_ok=True)
        for f in ("detect_tracks_%d_v4.csv" % g, "dets_%d_v4.csv" % g):
            if os.path.exists(os.path.join(OUT, f)):
                shutil.move(os.path.join(OUT, f), os.path.join(dst, f))
        flags = [str(lo + LADDER_SHIFT) if f == str(lo) and i and flags[i - 1] == "--lo" else
                 str(hi + LADDER_SHIFT) if f == str(hi) and i and flags[i - 1] == "--hi" else f
                 for i, f in enumerate(flags)]
        print("WINDOW LADDER: %.0f%% of %d chains end at --hi %d (cut off); re-detecting at "
              "--lo %d --hi %d" % (100 * frac, n, hi, lo + LADDER_SHIFT, hi + LADDER_SHIFT), flush=True)
        lo, hi = lo + LADDER_SHIFT, hi + LADDER_SHIFT


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
    ref_stem = json.load(open(lm, encoding="utf-8")).get("ref_stem")
    assert not ref_stem or ref_stem in set(man.stem), (
        "%s names ref_stem %s, which is not in this clip folder (a landmark file from another "
        "pull of the game); stabilize_park would reference it" % (lm, ref_stem))
    scope = "ours-only" if a.ours_only else "all"
    if a.weights != V4:   # legacy markers say only the scope; v4 runs keep matching them
        scope += " weights=" + a.weights
    stale = stale_outputs(g)
    assert not stale, ("game %d has step outputs this script did not write (another clip folder "
                       "of the game?) and every step would skip on them: %s. Move them aside "
                       "first (miss_season.py does)." % (g, ", ".join(stale)))
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
    detect_with_ladder(g, a)

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
