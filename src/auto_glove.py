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


def _glove_track(clip_path: str, model, step: int = 3, conf: float = 0.25,
                 imgsz: int = 640, half: bool = True):
    """[(frame, x, y, conf)] — highest-conf glove per sampled frame.

    Frames are read STRAIGHT THROUGH (grab the skipped ones, decode every `step`-th), so
    `frame` is the sequential index counted from 0 -- the project convention
    (index-and-time-origins.md #2). The earlier `cap.set(POS_FRAMES)` seek cost 7.0 s of
    a 16.8 s clip (662 frames, Asheville) and lands 0-2 frames early, so a set-up `frame`
    from before 2026-10-11 can sit up to 2 frames off this one; the set-up PIXEL is a
    median over the hold and moved <= 0.85 px on 10 clips.

    half=True runs the detector in FP16 on the GPU: 0.018 vs 0.041 s/frame on the AMD
    laptop, box centres a median 0.19 px / max 0.50 px from FP32 over 60 frames
    (2026-10-11). Ultralytics ignores it on CPU. Batching frames was tried and is SLOWER
    on this GPU (batch 16: 0.051 s/frame), so frames go one at a time.
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


def detect_setup_catch(clip_path: str, model, radius: float = 22.0,
                       max_gap_frames: int = 8, min_setup_len: int = 4):
    """Return dict(setup=(x,y,frame), catch=(x,y,frame), ok=bool, reason=str).

    ok=False when the trajectory is too sparse/ambiguous to trust — caller drops
    that pitch from the calibration set (we have plenty of pitches).
    """
    track, n = _glove_track(clip_path, model)
    if len(track) < 6:
        return {"ok": False, "reason": f"only {len(track)} glove frames"}

    runs = _stable_runs(track, radius, max_gap_frames)

    # SETUP: longest stable run whose CENTER frame is in the first 65% of the clip.
    cutoff = 0.65 * n
    setup_cands = [r for r in runs if len(r) >= min_setup_len
                   and r[len(r) // 2][0] <= cutoff]
    if not setup_cands:
        return {"ok": False, "reason": "no stable setup run"}
    setup_run = max(setup_cands, key=len)
    sx, sy, sframe = _run_center(setup_run)
    setup_end = setup_run[-1][0]

    # The pitch flight blurs/occludes the glove -> a big detection GAP right after
    # the setup hold. The CATCH is the first place the glove SETTLES after it.
    after = [p for p in track if p[0] > setup_end]
    if len(after) < 2:
        return {"ok": False, "reason": "no detections after setup"}

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
