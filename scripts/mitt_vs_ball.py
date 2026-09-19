"""Is the annotated MITT centroid the BALL? Measure the offset before anything fits it.

THE QUESTION. data/annotations/*.json carries 55 hand annotations, each with a `setup` box
(the catcher's target) and a `catch` box (where the pitch ended up). Both boxes are the
GLOVE. Root B of the two-root tree (docs/2026-09-18-two-roots-glove-and-plate.md) wants the
BALL at the receive instant, and a 46 x 27 px mitt box on footage where the ball is 8-11 px
can put those two several inches apart -- systematically, in the direction the mitt travels
to receive.

WHY IT MATTERS THAT THIS IS MEASURED FIRST. A systematic mitt-vs-ball offset is a constant.
Every affine, every per-game fit, every pose refinement in this project absorbs a constant
without complaint, so the number would vanish into a parameter and reappear as "the receive
plane offset" -- a physical-sounding quantity with a labelling error baked into it. It has
to be measured before anything fits it.

THE DECOMPOSITION IS THE POINT, NOT THE MAGNITUDE. The offset is reported ALONG and ACROSS
the ball's direction of travel, because those separate two different faults that a single
distance collapses (metric-must-separate-failure-modes.md):
    ALONG  -> the annotated frame is not the receive instant, or the mitt is closing. Timing.
    ACROSS -> the mitt sits off the ball's line. Geometry.
And each is reported as a BIAS (median) beside its SCATTER (MAD), because a systematic
offset and random annotation noise look identical on a contact sheet. This is the same
instrument that caught the half-frame differencing bias on 09-17.

CONTACT ROWS ARE NOT ROOT B ROWS. 14 of the 55 are `endpoint_kind: contact` -- the ball was
hit, so no receive point exists. They are excluded, counted, and named.

SEVEN PARKS, ONE PITCHER. The `ogando_` prefix reads like one park and is not: joined to
ogando_2026_manifest.csv the 55 annotations span HomeTrust/Asheville 27, Greensboro 7,
Winston-Salem 7, Frawley 6, Fluor 4, Bowling Green 2, Rome 2. That makes this experiment
STRONGER -- a real receiving geometry should replicate across seven cameras and a broadcast
artifact should not -- so the offset is reported PER PARK as well as pooled. What genuinely
does not vary is the pitcher: all 55 are Ogando.

RESULT OF THE FIRST RUN, 2026-09-18: **0 of 41 measured, and that is the finding.**
15 clips produced no chain and 26 produced a chain that ENDS 1-20 frames BEFORE the
annotated catch frame. At ~60 fps a 10-frame gap is ~0.17 s, during which a 90 mph pitch
travels ~22 ft -- far too large to be "the detector lost it on the last frame".

A 10-clip probe settled why. AT the annotated catch frame the detector either fires not at
all (5 of 10) or its nearest candidate is 65-345 px from the mitt (the other 5). **The ball
is inside the glove by then.** There is nothing to detect, and no amount of widening the
window or retraining changes that.

SO THE DESIGN BELOW IS WRONG AND IS KEPT ONLY AS THE RECORD OF HOW THAT WAS ESTABLISHED.
Root B's "actual" cannot be "the detected ball at the catch frame". The two options are:
  (a) use the mitt, which is what the annotations already are -- circular, answers nothing;
  (b) fit the ball's pixel path over the frames where it IS clean and EVALUATE it at the
      receive time.
(b) is the right one and it needs no new machinery: `plate_scale_check.py` already
interpolates an observed pixel path to a named time, which is the identical operation one
plane further downstream. It also turns the original question into a measurable one --
compare the EXTRAPOLATED ball position at the catch frame against the mitt box.

Note what the guard did here, because it is the reason this is a finding and not a wrong
number: with no rows measured the script REFUSES to print an offset and says the failure is
about the window or the weights, not about the mitt. A version that fell back to the nearest
available detection would have reported a confident 65-345 px "mitt offset" built entirely
from detections that are not the ball.

    python scripts/mitt_vs_ball.py --weights data/yolo_ball_runs/milb4park/weights/best.pt
    python scripts/mitt_vs_ball.py --weights ... --render output/mitt_vs_ball.png
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from detect_ball_tracks import best_chain  # noqa: E402

CLIP_ROOTS = ["data/clips", "data/clips/ogando_hardcases", "data/clips/ogando_2026"]

# The stem prefix says PITCHER, never park. Resolving it is a manifest lookup, not a guess --
# asserting the park from the `ogando_` prefix is exactly how this file's first version came
# to claim all 55 annotations were one ballpark when they are seven.
_PARK_BY_GAME = {
    821780: "Truist (Winston-Salem)", 821847: "Frawley (Wilmington)",
    821922: "AdventHealth (Rome)", 822162: "Fluor Field (Greenville)",
    822164: "Fluor Field (Greenville)", 822246: "FNB Field (Greensboro)",
    822428: "Bowling Green", 822429: "Bowling Green",
    822498: "HomeTrust (Asheville)", 822504: "HomeTrust (Asheville)",
    822507: "HomeTrust (Asheville)", 822512: "HomeTrust (Asheville)",
    822515: "HomeTrust (Asheville)",
}


def park_lookup():
    """{stem: park name} from the pitcher dir's manifest. Unknown stems map to '?'."""
    import pandas as pd
    p = os.path.join(ROOT, "data", "clips", "ogando_2026", "ogando_2026_manifest.csv")
    if not os.path.exists(p):
        return {}
    m = pd.read_csv(p)
    return {str(s): _PARK_BY_GAME.get(int(g), "game %s" % g)
            for s, g in zip(m["stem"], m["game_pk"])}


def find_clip(name):
    for r in CLIP_ROOTS:
        p = os.path.join(ROOT, r, name)
        if os.path.exists(p):
            return p
    return None


def detect_window(model, path, lo, hi, conf, imgsz, device, batch):
    """Per-frame top-k detections over [lo, hi] of one clip."""
    cap = cv2.VideoCapture(path)
    dets, i, buf = {}, 0, []

    def flush():
        if not buf:
            return
        res = model.predict([b[1] for b in buf], conf=conf, imgsz=imgsz,
                            verbose=False, device=device)
        for (fi, _), r in zip(buf, res):
            if r.boxes is None or not len(r.boxes):
                continue
            xy = r.boxes.xywh.cpu().numpy()
            cf = r.boxes.conf.cpu().numpy()
            order = np.argsort(-cf)[:5]
            dets[fi] = [(float(xy[k, 0]), float(xy[k, 1]), float(cf[k])) for k in order]
        buf.clear()

    while True:
        ok, fr = cap.read()
        if not ok or i > hi:
            break
        if i >= lo:
            buf.append((i, fr))
            if len(buf) >= batch:
                flush()
        i += 1
    cap.release()
    flush()
    return dets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--before", type=int, default=24, help="frames before the catch to scan")
    ap.add_argument("--after", type=int, default=4, help="frames after the catch to scan")
    ap.add_argument("--conf", type=float, default=0.15)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="0")
    ap.add_argument("--extrapolate", action="store_true",
                    help="when the chain stops short of the catch frame, fit the pixel path "
                         "and evaluate it there. Required for any result at all -- the ball "
                         "is inside the glove at the annotated frame.")
    ap.add_argument("--render", default="")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    anns = []
    for f in sorted(glob.glob(os.path.join(ROOT, "data", "annotations", "*.json"))):
        anns.append(json.load(open(f, encoding="utf-8")))
    catch = [d for d in anns if d.get("endpoint_kind") == "catch" and d.get("catch")]
    other = len(anns) - len(catch)
    print("%d annotations; %d catch rows used, %d excluded (contact / no endpoint)"
          % (len(anns), len(catch), other))
    if a.limit:
        catch = catch[:a.limit]

    parks = park_lookup()
    from ultralytics import YOLO
    model = YOLO(a.weights)

    rows, no_clip, no_chain, no_frame = [], 0, 0, 0
    for d in catch:
        path = find_clip(d["clip"])
        if not path:
            no_clip += 1
            continue
        cf = int(d["catch"]["frame"])
        dets = detect_window(model, path, max(0, cf - a.before), cf + a.after,
                             a.conf, a.imgsz, a.device, a.batch)
        chain = best_chain(dets)
        if len(chain) < 4:
            no_chain += 1
            print("  %-22s no chain (%d detected frames)" % (d["stem"], len(dets)))
            continue
        # The ball AT the annotated catch frame. No interpolation and no nearest-frame
        # substitution: a chain that does not reach the annotated frame is a different
        # measurement (the detector lost it early) and is counted separately rather than
        # silently answered with the closest thing available.
        at = [c for c in chain if c[0] == cf]
        gap = 0
        if not at and a.extrapolate:
            # ROOT B v2: the ball is INSIDE THE GLOVE at the annotated catch frame, so there
            # is nothing to detect there (measured: no detection at all in 5 of 10 clips, and
            # the nearest candidate 65-345 px away in the other 5). The receive-plane position
            # therefore has to come from the ball's PATH, fitted where the ball is clean and
            # evaluated at the receive time. Same operation plate_scale_check.py already does
            # one plane upstream.
            #
            # A QUADRATIC IN FRAME INDEX, not a physical model. Over the ~15-25 frames of a
            # chain the pixel path of a pitch is smooth and very nearly parabolic, and fitting
            # anything richer to 6-25 noisy points invents structure.
            fr = np.array([c[0] for c in chain], float)
            if len(fr) >= 4:
                gap = cf - fr[-1]
                bx = float(np.polyval(np.polyfit(fr, [c[1] for c in chain], 2), cf))
                by = float(np.polyval(np.polyfit(fr, [c[2] for c in chain], 2), cf))
                bconf = float(np.mean([c[3] for c in chain]))
                at = [(cf, bx, by, bconf)]
        if not at:
            no_frame += 1
            print("  %-22s chain ends f%d, annotation says f%d"
                  % (d["stem"], chain[-1][0], cf))
            continue
        _f, bx, by, bconf = at[0]
        gx, gy = float(d["catch"]["cx"]), float(d["catch"]["cy"])
        # Direction of travel from the last real step before the catch frame.
        prev = [c for c in chain if c[0] < cf]
        if not prev:
            no_chain += 1
            continue
        px, py = prev[-1][1], prev[-1][2]
        vx, vy = bx - px, by - py
        n = float(np.hypot(vx, vy))
        if n < 1e-6:
            no_chain += 1
            continue
        ux, uy = vx / n, vy / n
        dx, dy = bx - gx, by - gy
        rows.append({
            "stem": d["stem"], "park": parks.get(str(d["stem"]), "?"),
            "pitch_type": d.get("pitch_type", ""),
            "frame": cf, "ball_x": bx, "ball_y": by, "glove_x": gx, "glove_y": gy,
            "conf": bconf, "step_px": n, "gap_frames": gap,
            "along": dx * ux + dy * uy,          # + means the ball is PAST the mitt centroid
            "across": -dx * uy + dy * ux,        # + means the ball is left of the line of travel
            "dist": float(np.hypot(dx, dy)),
            "box_w": d["catch"]["w"], "box_h": d["catch"]["h"],
        })

    print("\nmeasured %d of %d catch rows"
          "  (%d clip missing, %d no chain, %d chain never reached the catch frame)"
          % (len(rows), len(catch), no_clip, no_chain, no_frame))
    if not rows:
        print("NOTHING MEASURED. That is a finding about the window or the weights, not a "
              "result about the mitt -- widen --before or check the detector first.")
        return 1

    def stat(k):
        v = np.array([r[k] for r in rows], float)
        med = float(np.median(v))
        mad = float(np.median(np.abs(v - med)))
        return med, mad, float(v.min()), float(v.max())

    print("\n%-8s %8s %8s %8s %8s" % ("", "median", "MAD", "min", "max"))
    for k in ("along", "across", "dist", "step_px"):
        m, d_, lo, hi = stat(k)
        print("%-8s %8.2f %8.2f %8.2f %8.2f" % (k, m, d_, lo, hi))
    # THE CHECK THAT DECIDES WHETHER ANY OF THIS MEANS ANYTHING.
    # With --extrapolate, the "ball position" is a fit evaluated `gap_frames` past its last
    # real observation. Extrapolation error grows with that distance; a real mitt-vs-ball
    # geometry does not. So if the offset correlates with the gap, what is being measured is
    # the fit running out of road, not the glove -- and the pooled median would be a
    # confident number describing the wrong thing.
    gaps = np.array([r["gap_frames"] for r in rows], float)
    if gaps.any():
        print("\nEXTRAPOLATION AUDIT  (gap = frames from the last real detection to the "
              "annotated catch)")
        print("  gap frames: median %.0f  p90 %.0f  max %.0f" %
              (np.median(gaps), np.percentile(gaps, 90), gaps.max()))
        for k in ("along", "across", "dist"):
            v = np.array([r[k] for r in rows], float)
            if len(v) > 3 and gaps.std() > 0 and v.std() > 0:
                r_ = float(np.corrcoef(gaps, np.abs(v))[0, 1])
                verdict = ("TRACKS THE GAP -- suspect the fit, not the glove"
                           if abs(r_) > 0.5 else "independent of the gap")
                print("    |%-6s| vs gap: r = %+.2f   %s" % (k, r_, verdict))
        near = [r for r in rows if r["gap_frames"] <= 4]
        if len(near) >= 5:
            print("  SHORT-GAP SUBSET (<=4 frames, n=%d): along %+.2f  across %+.2f px"
                  % (len(near), float(np.median([r["along"] for r in near])),
                     float(np.median([r["across"] for r in near]))))
            print("    That subset is the trustworthy read. Quote it beside the pooled "
                  "number, never instead of it.")

    bw = np.median([r["box_w"] for r in rows])
    bh = np.median([r["box_h"] for r in rows])
    print("\nmedian annotated mitt box: %.0f x %.0f px" % (bw, bh))
    print("READ IT THIS WAY: a median ALONG far from 0 is a TIMING fault (wrong frame, or "
          "the mitt is already closing). A median ACROSS far from 0 is a GEOMETRY fault "
          "(the mitt is off the ball's line). A small median with a large MAD is annotation "
          "scatter and costs precision, not accuracy.")
    print("ONE PITCHER (Ogando), SEVEN PARKS. Do not quote a pooled number without the "
          "per-park table below -- an offset present at one camera only is a camera fact.")
    by = {}
    for r in rows:
        by.setdefault(r["park"], []).append(r)
    print()
    print("%-26s %4s %8s %8s %8s" % ("park", "n", "along", "across", "dist"))
    for park, rs in sorted(by.items(), key=lambda kv: -len(kv[1])):
        print("%-26s %4d %8.2f %8.2f %8.2f"
              % (park[:26], len(rs),
                 float(np.median([x["along"] for x in rs])),
                 float(np.median([x["across"] for x in rs])),
                 float(np.median([x["dist"] for x in rs]))))

    out = os.path.join(ROOT, "output", "mitt_vs_ball.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    import csv as _csv
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = _csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("\nwrote " + out)

    if a.render:
        render(rows, a.render)
    return 0


def render(rows, path):
    """Zoomed crops with the mitt box and the detected ball, because a table of pixel
    offsets cannot show that the detector locked onto the catcher's wrist."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    n = min(len(rows), 16)
    cols, R = 4, 100
    fig, axes = plt.subplots((n + cols - 1) // cols, cols,
                             figsize=(4 * cols, 3.2 * ((n + cols - 1) // cols)))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes:
        ax.axis("off")
    for i, r in enumerate(rows[:n]):
        p = find_clip(r["stem"] + ".mp4")
        # Sequential decode. A seek lands 0-2 frames early on these clips
        # (.claude/rules/index-and-time-origins.md instance #2), which on THIS sheet would
        # shift the ball away from the mitt box and make a fine label look like a bad one --
        # the exact error the sheet exists to detect.
        cap = cv2.VideoCapture(p)
        fr, i, ok = None, 0, False
        while i <= int(r["frame"]):
            ok, f = cap.read()
            if not ok:
                break
            fr, i = f, i + 1
        cap.release()
        if fr is None or i <= int(r["frame"]):
            continue
        cx, cy = int(r["glove_x"]), int(r["glove_y"])
        x0, y0 = max(0, cx - R), max(0, cy - R)
        crop = fr[y0:cy + R, x0:cx + R][:, :, ::-1]
        ax = axes[i]
        ax.imshow(crop)
        ax.axis("off")
        ax.add_patch(Rectangle((r["glove_x"] - r["box_w"] / 2 - x0,
                                r["glove_y"] - r["box_h"] / 2 - y0),
                               r["box_w"], r["box_h"], fill=False, ec="#EB6E1F", lw=1.6))
        ax.plot(r["ball_x"] - x0, r["ball_y"] - y0, "o", mfc="none", mec="#00E5FF", ms=11, mew=1.8)
        ax.set_title("%s f%d  along %+.1f  across %+.1f"
                     % (r["stem"], r["frame"], r["along"], r["across"]), fontsize=8)
    fig.suptitle("orange = annotated mitt box   cyan = detected ball", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print("wrote " + path)


if __name__ == "__main__":
    sys.exit(main())
