"""THE REAL GATE: put OUR ball detections through the joint camera solve.

Every detector number so far is a pixel distance. The question the project actually
asks is whether a camera solved from OUR detections lands where the box-assisted one
does - because that is what A+ needs, where there is no drawn zone box to lock onto and
no Statcast trajectory to slide a clock against.

This was blocked until Sep 13 2026 for a reason that turned out to be a bug: the gate
needs at least 8 ball points per clip, and the eval reported 29% found (about 7 per
clip). That 29% was measured on pre-pitch frames - see
docs/2026-09-13-the-gate-was-reading-the-wrong-frames.md. Corrected, detection runs at
97.7% over 27-31 tracked frames, so the solve has about 27 points and the gate runs.

WHAT IS HELD FIXED, so the detector is the only variable:
  same clips, same candidate runs, same frames, same trajectory, same solver, same
  evaluation. ONLY the pixel coordinates change - OpenCommand's detections in one arm,
  ours in the other. Everything is imported from ablate_no_box.py rather than restated,
  because two copies of a solve is how the two arms silently stop being comparable.

TOP-1 BY CONFIDENCE, DELIBERATELY. At solve time there is no truth to pick a detection
with, so taking best-of-K here would be cheating: it would use the answer to choose the
input. The arm uses the highest-confidence detection per frame and drops frames with
none. `full` mode is the default because its closest detection is already ranked first
(med rank 1.0 against tile's 2.0).

READ THE OUTPUT IN THIS ORDER:
  camera        |Cx|,|Cz| against the box-assisted game solve. This is the headline.
  box corners   the 4 drawn-zone corners unprojected at the zone plane, in inches. The
                no-box arms never see the box, so for them it is a held-out object at a
                known world position - the only evaluation here that cannot be gamed.
  glove plane   disagreement with the BOX pose about one pixel at the catcher's setup
                depth. That is where the miss is actually read.
  ball@plate    kept only so the trap stays visible. Scoring a pose on the ball run it
                was fitted to is circular - ablate_no_box.py's docstring has the history.

    python scripts/gate_our_detections.py --limit 60 --mode full
"""
import argparse
import os
import sys

import cv2
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from src import pinhole as ph  # noqa: E402
import parity_opencommand_pose as par  # noqa: E402
import ablate_no_box as abn  # noqa: E402
import detect_ball_eval as dbe  # noqa: E402
import oc_frame_map as ofm  # noqa: E402


def detect_run(model, clip_path, seg, shift, mode, conf):
    """Our detections on the SAME frames, returned in OC's release-relative index.

    The frame is found at `frame_idx + shift` in our clip; the returned frame_idx stays
    RELATIVE, because everything downstream turns it into a time as t0 + frame_idx/fps
    and t0's bounds are expressed in release-relative seconds. Only the pixels change.
    """
    want = {int(r.frame_idx) + shift: int(r.frame_idx) for r in seg.itertuples()}
    cap = cv2.VideoCapture(clip_path)
    frames, i, hi = {}, 0, max(want)
    while i <= hi:
        ok, f = cap.read()
        if not ok:
            break
        if i in want:
            frames[i] = f
        i += 1
    cap.release()
    fn = dbe.detect_full_topk if mode == "full" else dbe.detect_tiled_topk
    rows = []
    for abs_i, rel_i in sorted(want.items()):
        fr = frames.get(abs_i)
        if fr is None:
            continue
        ds = fn(model, fr, conf, k=1)          # top-1 by CONFIDENCE, never by distance
        if not ds:
            continue
        rows.append((rel_i, float(ds[0][0]), float(ds[0][1])))
    return pd.DataFrame(rows, columns=["frame_idx", "baseball_center_x",
                                       "baseball_center_y"])


def evaluate(prep, C_game, label):
    """Joint ball-only solve over `prep`, scored exactly as ablate_no_box scores it."""
    xj = abn.fit_joint(prep, C_game[0], C_game[2])
    C_joint = np.array([xj[0], ph.CY_FT, xj[1]])
    rows = []
    for k, (c, fit_seg, held, t0_box) in enumerate(prep):
        pose_box = np.concatenate([C_game, par.fit_ptz(c, C_game)])
        pose_j = np.concatenate([C_joint, xj[2 + 5 * k: 6 + 5 * k]])
        ex, ez = abn.err_at_plate(c, pose_j, xj[6 + 5 * k], held)
        bx, bz = abn.box_corner_err_in(c, pose_j)
        rows.append(dict(play_id=c["play_id"], n_fit=len(fit_seg), plate_x=ex, plate_z=ez,
                         box_x=bx, box_z=bz,
                         glove_gap=abn.glove_plane_gap_in(c, pose_j, pose_box)))
    d = pd.DataFrame(rows)
    d["arm"] = label
    return C_joint, d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", type=int, default=824821)
    ap.add_argument("--limit", type=int, default=60)
    ap.add_argument("--mode", default="full", choices=["full", "tile"])
    ap.add_argument("--conf", type=float, default=0.10)
    ap.add_argument("--device", default="0")
    ap.add_argument("--weights", default=os.path.join(
        ROOT, "data", "yolo_ball_runs", "ball_3park_b8", "weights", "best.pt"))
    a = ap.parse_args()
    assert os.path.exists(a.weights), f"no weights: {a.weights}"
    clips_dir = os.path.join(ROOT, "data", "clips", f"mlb_{a.game}")

    clips = par.load_clips(a.game)[:a.limit]
    shifts = ofm.build_shifts(a.game, clips)
    ofm.verify_shifts(a.game, clips, shifts)

    from ultralytics import YOLO
    model = YOLO(a.weights)
    model.to(f"cuda:{a.device}" if str(a.device).isdigit() else a.device)
    print(f"weights {os.path.basename(a.weights)} | mode {a.mode} | conf {a.conf}\n")

    prep_oc, prep_us, votes, skipped = [], [], [], []
    for c in clips:
        runs = par.candidate_runs(c)
        if not runs or len(runs[0]) < abn.MIN_FIT_PTS + abn.HOLDOUT_FR:
            continue
        seg = runs[0]
        pid = c["play_id"]
        clip_path = os.path.join(clips_dir, f"{pid}_BROADCAST.mp4")
        # the box-assisted fit is what defines the reference camera, and it uses OC's
        # pixels in BOTH arms - it is the answer we are grading against, not a variable
        x_box = abn._fit(c, seg.iloc[:-abn.HOLDOUT_FR], True)
        votes.append(x_box[:2])
        prep_oc.append((c, seg.iloc[:-abn.HOLDOUT_FR], seg.iloc[-abn.HOLDOUT_FR:], x_box[6]))

        if pid not in shifts or not os.path.exists(clip_path):
            skipped.append((pid, "no shift or no clip"))
            continue
        ours = detect_run(model, clip_path, seg, shifts[pid], a.mode, a.conf)
        if len(ours) < abn.MIN_FIT_PTS + abn.HOLDOUT_FR:
            skipped.append((pid, f"only {len(ours)} detections"))
            continue
        prep_us.append((c, ours.iloc[:-abn.HOLDOUT_FR], ours.iloc[-abn.HOLDOUT_FR:], x_box[6]))
        print(f"  {pid[:8]}  oc {len(seg):3d} pts   ours {len(ours):3d} pts", flush=True)

    assert prep_oc and prep_us, "one arm is empty - the comparison is vacuous"
    med = np.median(np.array(votes), axis=0)
    C_game = np.array([med[0], ph.CY_FT, med[1]])

    print(f"\nclips: {len(prep_oc)} with OC points, {len(prep_us)} with ours")
    if skipped:
        # never a bare short run: say who dropped and why
        print(f"dropped from OUR arm ({len(skipped)}):")
        for pid, why in skipped[:10]:
            print(f"   {pid[:8]}  {why}")
    print(f"\nbox-assisted game camera: Cx {C_game[0]:8.3f}  Cz {C_game[2]:8.3f}\n")

    out = []
    for prep, label in ((prep_oc, "OC detections"), (prep_us, "OUR detections")):
        Cj, d = evaluate(prep, C_game, label)
        out.append(d)
        print(f"{label:16} joint camera  Cx {Cj[0]:8.3f}  Cz {Cj[2]:8.3f}   "
              f"(off {abs(Cj[0] - C_game[0]):5.2f} ft, {abs(Cj[2] - C_game[2]):5.2f} ft)")

    print(f"\n{'arm':16} {'clips':>6} {'BOX CORNERS |x|,|z| in':>26} "
          f"{'glove-plane gap in':>20} {'ball@plate |x|,|z|':>22}")
    for d in out:
        print(f"{d.arm.iloc[0]:16} {len(d):6} "
              f"{d.box_x.abs().median():13.3f} {d.box_z.abs().median():11.3f} "
              f"{d.glove_gap.median():20.3f} "
              f"{d.plate_x.abs().median():11.3f} {d.plate_z.abs().median():9.3f}")

    p = os.path.join(ROOT, "output", f"gate_our_detections_{a.game}.csv")
    pd.concat(out, ignore_index=True).to_csv(p, index=False)
    print(f"\n{p}")
    print("\nBOX CORNERS is the evaluation that cannot be gamed: neither no-box arm ever")
    print("sees the box, so those 4 corners are a held-out object at a known position.")


if __name__ == "__main__":
    main()
