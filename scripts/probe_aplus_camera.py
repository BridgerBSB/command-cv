"""Step 1 probe: does the A+ broadcast camera move while the ball is in the air?

Decides which OpenCommand pieces port to A+ (docs/2026-09-01-opencommand-v12-
and-cross-camera-accuracy.md section 4). Their phase 2 refits pan/tilt/roll/f
per clip from the drawn zone box and excludes the ball because "camera often
moves mid-ball flight". If our camera is static during the flight, a ball-only
per-clip refit is open to us even without a box.

Window: hand-labeled catch frame minus 30 (~0.5 s at 60 fps; flight ~0.42 s)
through the catch, decoded IN ORDER (cv2 seeks land up to 2 frames early on
these clips). Motion: LK optical flow on background corners in the CENTER band
of the crowd/wall (x 30-70%, y 5-35%), which excludes every corner scorebug,
then a RANSAC similarity fit per frame against the window's first frame ->
worst translation (px) and worst zoom change (%).

LABEL PROVENANCE GUARD. 12 of the 55 labels in data/annotations share a stem
with a Jun 29 GC2-zip clip in data/clips/. ingest_annotation.py resolves a clip
by bare filename and falls back to data/clips/<name>.mp4, so for those 12 it read
the ZIP clip. Six (cu_3, ff_1..ff_5, saved Jul 2) were drawn on the zip clips
outright; their API twins were not downloaded until Jul 8. The other six (cu_1,
fc_1..fc_3, sl_1, sl_2, saved Jul 5) may be drawn on the API clip with only the
saved training image wrong. Measuring a zip-clip label against the API clip puts
the "flight" window somewhere else entirely (ogando_cu_3's "catch" shows the
pitcher still in his set). A label is used only if the training image ingest
saved for it matches the API clip at that frame. Found Sep 11 2026.

Do NOT swap the motion measure for whole-frame phase correlation: its large
readings all come with a correlation response near zero (the pitcher's body
sweeping the frame), and they do not even reproduce run to run on identical
frames (ogando_si_17: max 199 / 255 / 458 px across three runs).

    python scripts/probe_aplus_camera.py
"""
import glob
import json
import os

import cv2
import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ANN = os.path.join(HERE, "data", "annotations")
CLIPS = os.path.join(HERE, "data", "clips", "ogando_2026")
YOLO_IMAGES = os.path.join(HERE, "data", "yolo", "images")


def _small(frame):
    return cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (320, 180)).astype(np.float32)


def read_range(clip, start, stop):
    """Frames [start, stop] by decoding in order from frame 0. No seeking."""
    c = cv2.VideoCapture(clip)
    out, i = {}, 0
    while i <= stop:
        ok, f = c.read()
        if not ok:
            break
        if i >= start:
            out[i] = f
        i += 1
    return out


def label_on_api_clip(ann, tol=2.0):
    """(ok, reason). ok only if every saved training image for this label matches
    the API clip within +-3 frames (mean abs gray diff < tol at 320x180)."""
    clip = os.path.join(CLIPS, ann["stem"] + ".mp4")
    if not os.path.exists(clip):
        return False, "no API clip"
    for tag in ("setup", "catch"):
        if not ann.get(tag):
            return False, f"no {tag} label"
        fi = ann[tag]["frame"]
        imgs = glob.glob(os.path.join(YOLO_IMAGES, "*", f"{ann['stem']}_{tag}_f{fi}.png"))
        if not imgs:
            return False, f"no saved {tag} image to verify against"
        ref = _small(cv2.imread(imgs[0]))
        frames = read_range(clip, max(0, fi - 3), fi + 3)
        if not frames:
            return False, f"{tag} frame {fi} past end of API clip"
        diff = min(float(np.abs(_small(f) - ref).mean()) for f in frames.values())
        if diff >= tol:
            return False, (f"{tag} training image does not match the API clip (diff {diff:.1f}); "
                           "ingest read a different video with the same name")
    return True, "ok"


def verified_labels():
    """(verified [(stem, ann)], rejected [(stem, reason)]) over data/annotations."""
    assert os.path.isdir(ANN) and os.path.isdir(CLIPS), (ANN, CLIPS)
    good, bad = [], []
    for fp in sorted(glob.glob(os.path.join(ANN, "*.json"))):
        ann = json.load(open(fp))
        ok, why = label_on_api_clip(ann)
        (good if ok else bad).append((ann["stem"], ann if ok else why))
    return good, bad


def flight_motion(clip, catch_frame, lead=30, per_frame=False):
    """Worst background translation (px) and zoom change (%) over [catch-lead, catch]."""
    fr = read_range(clip, catch_frame - lead, catch_frame)
    if len(fr) != lead + 1:
        return None
    g = [cv2.cvtColor(fr[i], cv2.COLOR_BGR2GRAY) for i in sorted(fr)]
    h, w = g[0].shape
    mask = np.zeros_like(g[0])
    mask[int(.05 * h):int(.35 * h), int(.30 * w):int(.70 * w)] = 255
    p0 = cv2.goodFeaturesToTrack(g[0], 300, 0.01, 8, mask=mask)
    if p0 is None or len(p0) < 15:
        return None
    pts, keep = p0.copy(), np.ones(len(p0), bool)
    ts, zs, inl = [0.0], [0.0], None
    for i in range(1, lead + 1):
        p1, st, _ = cv2.calcOpticalFlowPyrLK(g[i - 1], g[i], pts, None,
                                             winSize=(21, 21), maxLevel=3)
        keep &= st.ravel() == 1
        if keep.sum() < 15:
            return None
        m, inl = cv2.estimateAffinePartial2D(p0[keep], p1[keep], method=cv2.RANSAC,
                                             ransacReprojThreshold=2.0)
        if m is None:
            return None
        ts.append(float(np.hypot(m[0, 2], m[1, 2])))
        zs.append(abs(float(np.hypot(m[0, 0], m[1, 0])) - 1) * 100)
        pts = p1
    out = dict(t_px=max(ts), zoom_pct=max(zs), n_feat=len(p0))
    if per_frame:
        out.update(t_series=ts, frames=[fr[i] for i in sorted(fr)], p0=p0, p1=pts,
                   keep=keep, inliers=inl.ravel().astype(bool),
                   band=(int(.30 * w), int(.05 * h), int(.70 * w), int(.35 * h)))
    return out


def main():
    good, bad = verified_labels()
    print(f"labels {len(good) + len(bad)} -> verified on API clip {len(good)}, rejected {len(bad)}")
    for s, why in bad:
        print(f"  REJECTED {s:14} {why}")
    rows = []
    for s, ann in good:
        r = flight_motion(os.path.join(CLIPS, s + ".mp4"), ann["catch"]["frame"])
        if r is None:
            print(f"  SKIPPED  {s:14} too few trackable background corners")
            continue
        gw = (ann["setup"]["w"] + ann["catch"]["w"]) / 2
        rows.append((s, r["t_px"], r["zoom_pct"], r["t_px"] * 12.0 / gw))
    t = np.array([r[1] for r in rows]); z = np.array([r[2] for r in rows])
    i = np.array([r[3] for r in rows])
    print(f"measured {len(rows)}")
    print(f"translation px   median {np.median(t):.2f}  p90 {np.percentile(t, 90):.2f}  max {t.max():.2f}")
    print(f"~inches @ glove  median {np.median(i):.2f}  p90 {np.percentile(i, 90):.2f}  max {i.max():.2f}")
    print(f"zoom change %    median {np.median(z):.3f}  p90 {np.percentile(z, 90):.3f}  max {z.max():.3f}")
    for s, tp, zp, ip in sorted(rows, key=lambda r: -r[1])[:6]:
        print(f"  {s:16} {tp:.2f}px  {zp:.3f}%  ~{ip:.2f}in")


if __name__ == "__main__":
    main()
