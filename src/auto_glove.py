"""Auto-detect the SETUP and CATCH glove in a pitch clip — no annotation.

The trained mitt detector (data/yolo/runs/mitt/weights/best.pt) is reliable, so
we read the glove's trajectory across the clip and read off two moments from its
MOTION, no ball tracking needed (ball detection is too noisy at 60fps A+):

  SETUP = the long stable hold before the pitch (catcher holds the target).
          -> the longest low-motion run of glove detections in the first ~65%.
          STATUS: SOLVED. Validated vs 49 hand labels = median 2.3px error.

  CATCH = the glove settling right after the pitch-flight detection gap.
          STATUS: EXPERIMENTAL / ~half-reliable (median ~31px, some catastrophic
          misses). Glove-motion alone can't separate a real big command-miss from
          a detection error, and a confidence filter (flight-gap + multi-frame
          settle + bounded travel) did NOT cleanly separate good from bad on the
          49-clip validation. DO NOT trust the catch output for calibration yet.

NEXT (the robust catch signal): the BALL AT REST. A caught pitch stops the ball in
the glove for ~10-15 frames -> a tight multi-frame stable ball cluster (distinct
from scattered per-frame false positives), which is both unambiguous catch timing
AND the actual-location pixel (a better calibration anchor than the glove center).
See scripts test in the Jul 5 session. Build `detect_catch_ball()` and prefer it;
keep this motion-based catch only as a fallback.

Setup detection here is production-ready; the catch half is a WIP checkpoint.
"""
from __future__ import annotations

import numpy as np


SETUP_CUTOFF_FRAC = 0.65     # a set-up run's centre frame must sit in the first 65% of the clip


def _glove_track(clip_path: str, model, step: int = 3, conf: float = 0.25,
                 imgsz: int = 640, half: bool = True, stop=None):
    """[(frame, x, y, conf)] — highest-conf glove per sampled frame.

    Frames are read STRAIGHT THROUGH (grab the skipped ones, decode every `step`-th), so
    `frame` is the sequential index counted from 0 -- the project convention
    (index-and-time-origins.md #2). The earlier `cap.set(POS_FRAMES)` seek cost 7.0 s of
    a 16.8 s clip (662 frames, Asheville) and lands 0-2 frames early, so a set-up `frame`
    from before 2026-10-11 can sit up to 2 frames off this one; the set-up PIXEL is a
    median over the hold and moved <= 0.85 px on 10 clips.

    half=True runs the detector in FP16 on the GPU: 0.018 vs 0.041 s/frame on the AMD
    laptop, box centres a median 0.19 px / max 0.50 px from FP32 over 60 frames
    (2026-10-11). Batching frames was tried and is SLOWER on this GPU (batch 16: 0.051
    s/frame), so frames go one at a time.

    ON CPU IT IS FORCED OFF. Ultralytics does NOT ignore `quantize=16` on CPU (the old note
    said it did - true of `half`, not of 8.4's rename): on the 2-core Hostinger VPS the
    OpenCommand glove model (yolo11x, 57M params) took 7.62 s/frame with it vs 1.36 s/frame
    in FP32 (2026-10-14).

    stop(track, frame) -> bool is asked after every sampled frame; True ends the read there.
    The track up to that frame is exactly what a full read would have produced (sequential
    decode is deterministic), which is what makes an early stop provably equivalent.
    """
    import cv2
    cap = cv2.VideoCapture(clip_path)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    track = []
    # A multi-class model (OpenCommand glove_tracking.pt: glove/homeplate/baseball/rubber/na)
    # must be held to its glove class, or the "highest-conf box" below can be the plate or the
    # ball. The single-class in-house mitt model is unaffected.
    names = getattr(model, "names", None) or {}
    glove_cls = [k for k, v in names.items() if v == "glove"] if len(names) > 1 else []
    kw = {"classes": glove_cls} if glove_cls else {}
    # ultralytics 8.4 renamed half -> quantize=16 (same compute; `half` still works but logs a
    # deprecation warning on EVERY frame, ~70k lines per game). Older versions only know `half`.
    if half:
        import torch
        half = torch.cuda.is_available()          # FP16 only on a GPU (CUDA or ROCm); see docstring
    if half:
        try:
            from ultralytics.cfg import DEFAULT_CFG_DICT
            kw.update({"quantize": 16} if "quantize" in DEFAULT_CFG_DICT else {"half": True})
        except ImportError:
            kw["half"] = True
    fi = -1
    while True:
        fi += 1
        if fi % step:
            if not cap.grab():
                break
            continue
        ok, fr = cap.read()
        if not ok:
            break
        r = model.predict(fr, imgsz=imgsz, conf=conf, verbose=False, **kw)[0]
        if len(r.boxes):
            i = int(np.argmax(r.boxes.conf.cpu().numpy()))
            x, y = r.boxes.xywh[i][:2].cpu().numpy()
            track.append((fi, float(x), float(y), float(r.boxes.conf[i])))
        if stop is not None and stop(track, fi):
            break
    cap.release()
    return track, n


def _stable_runs(track, radius: float, max_gap_frames: int):
    """Split the track into maximal runs where each point stays within `radius`
    of the run's running centroid and consecutive frames are <= max_gap apart."""
    runs, cur = [], []
    for pt in track:
        if not cur:
            cur = [pt]
            continue
        cx = np.mean([p[1] for p in cur])
        cy = np.mean([p[2] for p in cur])
        gap = pt[0] - cur[-1][0]
        if gap <= max_gap_frames and np.hypot(pt[1] - cx, pt[2] - cy) <= radius:
            cur.append(pt)
        else:
            runs.append(cur)
            cur = [pt]
    if cur:
        runs.append(cur)
    return runs


def _run_center(run):
    return (float(np.median([p[1] for p in run])),
            float(np.median([p[2] for p in run])),
            int(run[len(run) // 2][0]))  # (x, y, mid_frame)


def _setup_candidates(runs, n, min_setup_len):
    cutoff = SETUP_CUTOFF_FRAC * n
    return [r for r in runs if len(r) >= min_setup_len and r[len(r) // 2][0] <= cutoff]


def _select_setup(track, n, radius, max_gap_frames, min_setup_len):
    """The SET-UP half of detect_setup_catch: (fail_reason or None, setup_run, after)."""
    if len(track) < 6:
        return f"only {len(track)} glove frames", None, None
    runs = _stable_runs(track, radius, max_gap_frames)
    # SETUP: longest stable run whose CENTER frame is in the first 65% of the clip.
    setup_cands = _setup_candidates(runs, n, min_setup_len)
    if not setup_cands:
        return "no stable setup run", None, None
    setup_run = max(setup_cands, key=len)
    after = [p for p in track if p[0] > setup_run[-1][0]]
    if len(after) < 2:
        return "no detections after setup", None, None
    return None, setup_run, after


SETUP_END_WINDOW_FR = 60    # the set-up hold must still be going in the last ~1 s (60 fps) before the ball


def _select_setup_anchored(track, anchor, radius, max_gap_frames, min_setup_len,
                           end_window=SETUP_END_WINDOW_FR, min_lead=0):
    """Release-anchored SET-UP: (fail_reason or None, setup_run).

    `anchor` = the first frame the pitched ball is seen in flight (a detect_tracks chain's
    first frame, so already after release). The set-up is the LONGEST glove hold that is
    still going within `end_window` frames before it. Points after the anchor are dropped
    before the runs are built, so the catch and the post-catch hold can never be picked.

    Why not the longest hold in the first 65% (_select_setup): Zac's target review
    (2026-10-15, 117 charted targets) found 27 from the WRONG MOMENT. Timed against the ball,
    7 of 25 sat at or after the ball's first frame (the catch) and 11 sat 130+ frames before
    it (an earlier hold, e.g. while signs are given); 82 of 86 correct ones sat 40-130 before.

    Why not the LAST hold before the ball (tried first, 2026-10-16): on correct pitches the
    catcher holds the target, then drops the glove 20-40 frames before the ball is seen, and
    the drop breaks into short "holds" of its own (rid 8: target f33-147, then f150-159 and
    f162-186 each ~45 px lower). The last hold picked the drop on 4 of the first 7 correct
    targets, by a median 77 px. The real hold ends 17-41 frames before the ball on those 7.

    min_lead > 0 also refuses a hold that STARTS fewer than min_lead frames before the ball:
    when the broadcast cuts to the pitching view at the delivery, the only hold is the glove
    during the flight, and there is no set-up to read (rids 26/29/30/33/34, all graded
    wrong-moment, holds starting 9-31 frames before the ball).
    """
    pre = [p for p in track if p[0] <= anchor]
    if len(pre) < 6:
        return f"only {len(pre)} glove frames before the ball", None
    cands = [r for r in _stable_runs(pre, radius, max_gap_frames)
             if len(r) >= min_setup_len and r[-1][0] >= anchor - end_window]
    if not cands:
        return f"no hold within {end_window} frames before the ball", None
    best = max(cands, key=len)
    if anchor - best[0][0] < min_lead:
        return f"set-up hold starts only {anchor - best[0][0]} frames before the ball", None
    return None, best


def _setup_settled(track, fi, n, radius, max_gap_frames, min_setup_len):
    """True once no glove detection after frame `fi` can change _select_setup's verdict or
    set-up. The miss chain reads only those, and the rest of a clip (the catch, the
    throw-back, the between-pitch shot) was ~35%+ of every glove pass.

    Why each condition is enough, given that a later point can only APPEND to the track:
      * fi past the cutoff, and no run still OPEN (last point within max_gap of fi) that
        began at or before the cutoff: every future run starts after the cutoff, so its
        centre does too, and the candidate list is final.
      * no candidates: the verdict is a failure whatever follows (ok=False both ways).
      * candidates: the chosen run is final; ok needs >= 6 track points and >= 2 points
        after the run's end, and once true both stay true.
    """
    cutoff = SETUP_CUTOFF_FRAC * n
    if fi <= cutoff or not track:
        return False
    runs = _stable_runs(track, radius, max_gap_frames)
    last = runs[-1]
    if fi - last[-1][0] <= max_gap_frames and last[0][0] <= cutoff:
        return False
    cands = _setup_candidates(runs, n, min_setup_len)
    if not cands:
        return True
    best = max(cands, key=len)
    return len(track) >= 6 and sum(p[0] > best[-1][0] for p in track) >= 2


def detect_setup_catch(clip_path: str, model, radius: float = 22.0,
                       max_gap_frames: int = 8, min_setup_len: int = 4,
                       setup_only: bool = False, anchor_frame: int | None = None):
    """Return dict(setup=(x,y,frame), catch=(x,y,frame), ok=bool, reason=str).

    ok=False when the trajectory is too sparse/ambiguous to trust — caller drops
    that pitch from the calibration set (we have plenty of pitches).

    setup_only=True stops reading the clip once the set-up is settled (_setup_settled) and
    returns catch=None: same ok and set-up as a full read, fewer frames. For callers that
    read only ok + setup (stabilize_park, command_miss); the catch scripts keep the default.

    anchor_frame (sequential frame index, the ball's first frame in flight) switches the
    set-up to the release-anchored rule (_select_setup_anchored). None = the longest-hold
    rule, unchanged. With setup_only, the read stops at the anchor.
    """
    if anchor_frame is not None:
        return _detect_setup_anchored(clip_path, model, int(anchor_frame), radius,
                                      max_gap_frames, min_setup_len, setup_only)
    stop, last_read = None, [0]
    if setup_only:
        def stop(track, fi):
            last_read[0] = fi
            return _setup_settled(track, fi, n_frames[0], radius, max_gap_frames, min_setup_len)
        import cv2
        cap = cv2.VideoCapture(clip_path)
        n_frames = [int(cap.get(cv2.CAP_PROP_FRAME_COUNT))]
        cap.release()
    track, n = _glove_track(clip_path, model, stop=stop)
    reason, setup_run, after = _select_setup(track, n, radius, max_gap_frames, min_setup_len)
    if reason:
        return {"ok": False, "reason": reason}
    sx, sy, sframe = _run_center(setup_run)
    if setup_only:
        return {"ok": True, "reason": "ok (setup only)", "setup": (sx, sy, sframe),
                "catch": None, "frames_read": last_read[0] + 1, "n_frames": n}

    # First inter-detection gap >= pitch_gap after the setup hold = the pitch flight.
    pitch_gap = 15
    flight_idx = None
    for i in range(1, len(after)):
        if after[i][0] - after[i - 1][0] >= pitch_gap:
            flight_idx = i
            break
    post = after[flight_idx:] if flight_idx is not None else after

    # CATCH = first stable run (>=2 pts within radius) in the post-flight tail;
    # the glove reappears at the receive point and holds briefly before throw-back.
    post_runs = _stable_runs(post, radius, max_gap_frames)
    catch_run = next((r for r in post_runs if len(r) >= 2), None)
    if catch_run is None:
        cx, cy, cframe = post[0][1], post[0][2], post[0][0]
        n_catch = 1
    else:
        cx, cy, cframe = _run_center(catch_run)
        n_catch = len(catch_run)
    disp = float(np.hypot(cx - sx, cy - sy))
    # Confidence: a clean detection has a real flight gap, a multi-frame settle,
    # and a plausible glove travel (a real command miss won't move the mitt 300px).
    conf = (flight_idx is not None) and (n_catch >= 3) and (disp <= 220)
    return {"ok": True, "reason": "ok",
            "setup": (sx, sy, sframe), "catch": (cx, cy, cframe),
            "flight": flight_idx is not None, "n_catch": n_catch,
            "disp": disp, "confident": bool(conf)}


def anchored_setup_point(run):
    """(x, y, frame) of a release-anchored hold: the median spot over the hold, at the hold's
    LAST frame, i.e. the glove just before the pitch. The longest-hold rule reports the middle
    frame, which on a long hold can be while signs are still being given (round 1, rid 52:
    one spot f0-162, shown at f81, graded wrong-moment)."""
    return (float(np.median([p[1] for p in run])), float(np.median([p[2] for p in run])),
            int(run[-1][0]))


def _detect_setup_anchored(clip_path, model, anchor, radius, max_gap_frames, min_setup_len,
                           setup_only):
    """detect_setup_catch with the release-anchored set-up. Set-up only: the catch half is the
    experimental motion read above and is not ported to this rule."""
    if not setup_only:
        raise ValueError("anchor_frame needs setup_only=True (the catch half is not ported)")
    last_read = [0]

    def stop(track, fi):
        last_read[0] = fi
        return fi >= anchor          # nothing after the anchor can change the anchored pick

    track, n = _glove_track(clip_path, model, stop=stop)
    reason, setup_run = _select_setup_anchored(track, anchor, radius, max_gap_frames,
                                               min_setup_len)
    if reason:
        return {"ok": False, "reason": reason}
    sx, sy, sframe = anchored_setup_point(setup_run)
    return {"ok": True, "reason": "ok (setup only, release-anchored)", "setup": (sx, sy, sframe),
            "catch": None, "frames_read": last_read[0] + 1, "n_frames": n}
