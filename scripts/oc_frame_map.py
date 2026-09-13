"""OpenCommand's frame_idx is RELATIVE TO RELEASE. Ours is the absolute clip index.

This is one definition, imported by every consumer, because the failure it fixes was
two scripts independently doing the same wrong thing.

  OpenCommand  gloveball_tracks.frame_idx spans about -28..+30 and frame 0 is release
               (release_s_hint is the sub-frame remainder and is ~0, +-0.008 s)
  ours         ball_labels_<game>.csv frame_idx is the decode index of OUR clip file,
               179..213 on game 824821

Reading OUR video at THEIR frame number therefore lands ~180 frames early - three
seconds before the pitch, the pitcher still on the mound. detect_ball_eval.py and
detect_diagnose.py both did exactly that, which is why the detector "found" the ball
at 44-308 px: it was being shown pre-pitch frames and graded against a mid-flight
ball position. See docs/2026-09-13-the-gate-was-reading-the-wrong-frames.md.

WHY THE MAPPING IS TRUSTWORTHY. It is derived two independent ways and they agree:

  geometric  shift = f0 - T_plate*fps, where f0 is the absolute plate frame our own
             ball_clock_align.py locked onto and T_plate comes from the Statcast
             trajectory. Uses no pixels.
  empirical  slide an integer shift and take the one that best matches our projected
             ball path to OC's detected ball path. Uses only pixels.

On game 824821, 277 plays: empirical median 183.0 (sd 2.36), geometric median 181.9
(sd 2.29), disagreeing by a median of +0.57 frames with 263/277 inside 2 frames. The
+0.5 bias is the expected artefact of f0 being scanned in whole-frame steps while the
true plate crossing falls between frames.

The residual at the winning shift is 2.15 px median (0.65 in), 277/277 under 10 px -
i.e. our geometry-derived labels and OpenCommand's independent detections describe the
same ball. That is the check that makes this mapping a measurement and not a guess.
"""
import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# f0 is an integer frame while the plate crossing is continuous, so the geometric
# shift carries a sub-frame bias. Anything beyond this is a real disagreement.
MAX_SHIFT_DISAGREE_FR = 3.0
# our label vs OC's detection on the same frame, game 824821: median 2.15 px, max 6.83
MAX_LABEL_RESIDUAL_PX = 12.0


def labels_path(game):
    return os.path.join(ROOT, "output", f"ball_labels_{game}.csv")


def load_labels(game):
    """Our absolute-frame ball labels, or None if this game was never aligned."""
    p = labels_path(game)
    return pd.read_csv(p) if os.path.exists(p) else None


def build_shifts(game, clips):
    """{play_id: int shift} such that  absolute_clip_frame = oc_frame_idx + shift.

    `clips` is parity_opencommand_pose.load_clips(game) - each entry carries T_plate
    and fps. Plays we never aligned get no entry, and the caller must skip them rather
    than fall back to 0, which is the bug this module exists to prevent.
    """
    lab = load_labels(game)
    assert lab is not None, (
        f"no {labels_path(game)} - run ball_clock_align.py --labels first. Without it "
        f"there is no way to place OC's release-relative frames in our clip.")
    f0 = lab.groupby("play_id").frame_idx.max()
    out = {}
    for c in clips:
        pid = c["play_id"]
        if pid not in f0.index:
            continue
        out[pid] = int(round(float(f0[pid]) - c["T_plate"] * c["fps"]))
    assert out, "no play had both a label and a clip - the shift map is vacuous"
    return out


def verify_shifts(game, clips, shifts, verbose=True):
    """Check the geometric shifts against the pixels, and RAISE if they disagree.

    A silent mapping is what caused the original defect, so this is not optional
    decoration: it re-derives the shift empirically from the ball paths and asserts
    the two agree, then asserts our labels actually land on OC's ball.
    """
    lab = load_labels(game)
    oc = {c["play_id"]: c["df"][c["df"].baseball_center_x.notna()] for c in clips}
    emp, resid = [], []
    for pid, s in shifts.items():
        o = oc.get(pid)
        g = lab[lab.play_id == pid]
        if o is None or len(o) < 8 or g.empty:
            continue
        lx = g.set_index("frame_idx")[["x", "y"]]
        best = None
        for t in range(s - 8, s + 9):
            j = lx.join(o.assign(a=o.frame_idx + t).set_index("a")
                        [["baseball_center_x", "baseball_center_y"]], how="inner").dropna()
            if len(j) < 6:
                continue
            d = float(np.median(np.hypot(j.x - j.baseball_center_x,
                                         j.y - j.baseball_center_y)))
            if best is None or d < best[1]:
                best = (t, d)
        if best is not None:
            emp.append(best[0] - s)
            resid.append(best[1])
    assert emp, "verify_shifts matched nothing - the check is vacuous, not passing"
    dis, res = float(np.median(np.abs(emp))), float(np.median(resid))
    if verbose:
        print(f"frame shift: {len(shifts)} plays | geometric vs empirical median "
              f"|disagreement| {dis:.2f} fr | label-vs-OC residual {res:.2f} px "
              f"({res * 0.30:.2f} in) over {len(resid)} plays")
    assert dis <= MAX_SHIFT_DISAGREE_FR, (
        f"frame shift disagrees by {dis:.2f} frames (limit {MAX_SHIFT_DISAGREE_FR}) - "
        f"the release-relative mapping does not hold for game {game}")
    assert res <= MAX_LABEL_RESIDUAL_PX, (
        f"our labels sit {res:.2f} px from OC's ball (limit {MAX_LABEL_RESIDUAL_PX}) - "
        f"the pose or the clock alignment is wrong for game {game}")
    return dis, res
