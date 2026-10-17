"""Re-pick every PASS game's set-up TARGETS with OpenCommand's rule. Zac, 2026-10-17: "flip it".

WHY. His clip review (2026-10-16, 134 pitches, docs/target_review/): OpenCommand's rule right on
108 of 131 graded, the hold rule the season ran with 97. Its wins are a target the catcher SHOWS
high and early and then drops; the hold rule took the drop by ~23 in. The rule is
src.auto_glove.select_setup_opencommand, which scripts/setup_rule_oc_test.py proved reproduces
every graded pick.

SETTINGS, from that test: step 3 (every 3rd frame, what the hold rule read: 107 right vs 108,
2 of 134 picks moved) and ONE px-per-inch for every park, 3.96 = the median of the graded games'
slopes (108 right either way, 4 picks moved). The fixed scale needs no camera, so a new game can
be targeted before its camera exists.

ONLY THE TARGET CHANGES. Each game keeps its camera, ball tracks and reference view. The new set-up
pixel goes into that reference view through the game's OWN saved homography (stab2_H_<g>c.csv) at
the good key frame nearest the new set-up frame. The camera holds still before the pitch: over
58,890 clips the mapped image centre moved a median 0.28 px between the set-up frame and the first
ball frame (95th 4.3 px, 2026-10-17).
  check: per pitch, B_minus_A_in (ball vs tracking, target-free) must be IDENTICAL in the new and
         old miss files on every pitch both have. It is the proof camera + ball were not touched.
  note:  the PASS/FAIL gate grades the held-out pitches that HAVE a target, so a different set of
         targetable pitches can move a game's gate number. Reported per game, never re-gated here.

PER GAME (resumable: a game with command_miss_<g>_bo_hold.csv is done):
  1. download Houston's clips (miss_season.fetch), rebuild data/clips/hou_<g>/ (ours_only_dir)
  2. glove pass, OpenCommand rule              -> setups_<g>_oc.csv (raw clip px)
  3. map into the reference view               -> targets_<g>_stab2c_oc.csv
  4. command_miss --targets ... --tag _bo_oc   -> command_miss_<g>_bo_oc.csv, then the check
  5. swap: _bo -> _bo_hold (kept), _bo_oc -> _bo (what the board reads)
  6. delete the clips
Afterwards: python scripts/miss_season.py --board (board + season cards).

    python scripts/retarget_season.py --games 822162          # one game, timed
    python scripts/retarget_season.py --workers 4 --worker 0  # a shard of all PASS games
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from src.auto_glove import OC_FIXED_PPI, OC_STEP, detect_setup_opencommand  # noqa: E402
import miss_season as MS  # noqa: E402
from miss_new_park import ours_only_dir  # noqa: E402
from stabilize_park import MIN_INLIERS, apply, zoom  # noqa: E402

OUT = os.path.join(ROOT, "output")
GLOVE = os.path.join(ROOT, "models", "od", "YOLO", "glove_tracking", "model_weights", "glove_tracking.pt")
STEP, PPI = OC_STEP, OC_FIXED_PPI      # setup_rule_oc_test.py, 2026-10-17 (module docstring)


def p(name, g):
    return os.path.join(OUT, name % g)


def pass_games():
    led = MS.read_ledger()
    return sorted({int(g) for g in led.loc[led.verdict == "PASS", "game_pk"]})


def anchors(g):
    s = pd.read_csv(p("detect_tracks_%d_v4.csv", g), usecols=["stem", "frame"]).groupby("stem").frame.min()
    return s.to_dict(), float(s.median())


def good_H(g):
    """{stem: [(frame, H)]} of the game's good homographies (stabilize_park's own test)."""
    h = pd.read_csv(p("stab2_H_%dc.csv", g)).dropna(subset=["h0"])
    h = h[(h.inliers >= MIN_INLIERS) & (h.spread.astype(str) == "True")]
    out = {}
    for r in h.itertuples():
        H = np.array([getattr(r, "h%d" % k) for k in range(9)], float).reshape(3, 3)
        out.setdefault(r.stem, []).append((int(r.frame), H))
    return out


def retarget(g, model):
    t0 = time.time()
    hou = ours_only_dir(g, os.path.join(ROOT, "data", "clips", "pull_%d" % g))
    man = pd.read_csv(os.path.join(hou, "hou_%d_manifest.csv" % g))
    own, med = anchors(g)
    Hs = good_H(g)
    setups, tgt, why = [], [], {}
    for s in man.stem:
        clip = os.path.join(hou, s + ".mp4")
        if not os.path.exists(clip):
            why["clip missing"] = why.get("clip missing", 0) + 1
            continue
        r = detect_setup_opencommand(clip, model, int(own.get(s, med)), PPI, step=STEP)
        if not r["ok"]:
            why[r["reason"]] = why.get(r["reason"], 0) + 1
            continue
        x, y, f = r["setup"]
        setups.append(dict(stem=s, x=x, y=y, frame=f))
        if s not in Hs:
            why["no good homography"] = why.get("no good homography", 0) + 1
            continue
        hf, H = min(Hs[s], key=lambda t: abs(t[0] - f))
        px, py = apply(H, x, y)
        tgt.append(dict(stem=s, px=px, py=py, raw_px=x, raw_py=y, frame=f, h_frame=hf, zoom=zoom(H)))
    t_glove = time.time() - t0
    pd.DataFrame(setups).to_csv(p("setups_%d_oc.csv", g), index=False)
    pd.DataFrame(tgt).to_csv(p("targets_%d_stab2c_oc.csv", g), index=False)
    cmd = [sys.executable, "-u", os.path.join(ROOT, "scripts", "command_miss.py"), "--game", str(g),
           "--dir", os.path.relpath(hou, ROOT), "--pose", "output/park_pose_%d_ballonly.npz" % g,
           "--tracks", "output/detect_tracks_%d_v4stab2c.csv" % g,
           "--targets", "output/targets_%d_stab2c_oc.csv" % g, "--tag", "_bo_oc"]
    rc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    assert rc.returncode == 0, "command_miss failed for %d: %s" % (g, rc.stdout[-500:] + rc.stderr[-500:])

    old, new = pd.read_csv(p("command_miss_%d_bo.csv", g)), pd.read_csv(p("command_miss_%d_bo_oc.csv", g))
    both = old[old.target_src == "auto"].merge(new[new.target_src == "auto"], on="stem", suffixes=("_o", "_n"))
    dba = (both.B_minus_A_in_o - both.B_minus_A_in_n).abs().max() if len(both) else 0.0
    assert not (dba > 1e-9), "game %d: B_minus_A moved by %.3g in - camera or ball changed" % (g, dba)
    go, gn = MS.grade(g, "_bo"), MS.grade(g, "_bo_oc")
    moved = np.hypot(both.target_x_in_o - both.target_x_in_n, both.target_z_in_o - both.target_z_in_n)
    row = dict(game_pk=g, clips=len(man), targets_old=int((old.target_src == "auto").sum()),
               targets_new=len(new), shared=len(both), moved_med_in=float(moved.median()),
               moved_gt_5in=int((moved > 5).sum()), bok_gap_old=go["bok_gap"], bok_gap_new=gn["bok_gap"],
               plaus_old=go["plaus"], plaus_new=gn["plaus"], glove_min=round(t_glove / 60, 1),
               skipped="; ".join("%s %d" % kv for kv in sorted(why.items())))
    shutil.move(p("command_miss_%d_bo.csv", g), p("command_miss_%d_bo_hold.csv", g))
    shutil.copy2(p("command_miss_%d_bo_oc.csv", g), p("command_miss_%d_bo.csv", g))
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, nargs="*", default=None, help="default: every PASS game")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--worker", type=int, default=0)
    a = ap.parse_args()
    games = a.games or pass_games()
    games = [g for g in games if g % a.workers == a.worker and not os.path.exists(p("command_miss_%d_bo_hold.csv", g))]
    print("%d game(s) to retarget (worker %d of %d)" % (len(games), a.worker, a.workers), flush=True)
    from ultralytics import YOLO
    model = YOLO(GLOVE)
    log = os.path.join(OUT, "retarget_season_w%d.csv" % a.worker)
    jobs = [dict(game_pk=g, csv=os.path.join(MS.CVPULL, "cvpull_%d.csv" % g)) for g in games]
    with ThreadPoolExecutor(1) as ex:                  # the next game downloads while this one runs
        for job, dl in zip(jobs, ex.map(MS.fetch, jobs)):
            g = job["game_pk"]
            t0 = time.time()
            try:
                row = retarget(g, model)
            except Exception as e:                     # noqa: BLE001 - one bad game never stops the shard
                row = dict(game_pk=g, error=str(e)[:300])
            finally:
                freed = MS.delete_clips(g)
            row.update(dl_ok=dl[0], dl_n=dl[1], dl_min=round(dl[2], 1), run_min=round((time.time() - t0) / 60, 1),
                       freed_gb=round(freed, 2), finished_ct=MS.now_ct())
            pd.DataFrame([row]).to_csv(log, mode="a", header=not os.path.exists(log), index=False)
            print("  %d  %s" % (g, "  ".join("%s=%s" % (k, row[k]) for k in row if k != "game_pk")), flush=True)


if __name__ == "__main__":
    main()
