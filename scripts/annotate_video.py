"""Render a watchable annotated clip: the mitt detector's box on EVERY frame +
the auto-detected SETUP target (persistent) + the command-miss readout.

Shows exactly what the model sees: the glove held at the target, the pitch, the
receive. Writes an .mp4 you can scrub.

  python scripts/annotate_video.py ogando_sl_5 [ogando_si_1 ...]   # by stem
Outputs -> output/annotated/<stem>_annot.mp4
"""
import csv
import glob
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.auto_glove import detect_setup_catch
from src.calibration import solve_affine_ransac

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NAVY, ORANGE, CYAN, GREEN = (98, 45, 0), (31, 110, 235), (255, 200, 0), (60, 200, 60)


def _game_affine(pitcher, game_pk):
    """RANSAC pixel->feet affine for the game (from cached auto-detections)."""
    cache_p = os.path.join(ROOT, "output", "auto_detections.jsonl")
    if not os.path.exists(cache_p):
        return None
    cache = {json.loads(l)["stem"]: json.loads(l) for l in open(cache_p)}
    man = {r["stem"]: r for r in csv.DictReader(
        open(os.path.join(ROOT, "data", "clips", f"{pitcher}_2026", f"{pitcher}_2026_manifest.csv")))
        if r["game_pk"] == game_pk}
    catch, wld = [], []
    for stem, r in man.items():
        c = cache.get(stem)
        if not c or not c.get("ok"):
            continue
        try:
            catch.append(c["catch"]); wld.append([float(r["plate_x"]), float(r["plate_z"])])
        except (TypeError, ValueError):
            pass
    if len(catch) < 5:
        return None
    aff, _ = solve_affine_ransac(np.array(catch, float), np.array(wld, float))
    return aff


def annotate(stem, model):
    man = {r["stem"]: r for r in csv.DictReader(
        open(glob.glob(os.path.join(ROOT, "data", "clips", "*", "*_manifest.csv"))[0]))}
    # find which pitcher dir holds it
    hits = glob.glob(os.path.join(ROOT, "data", "clips", "*", f"{stem}.mp4"))
    if not hits:
        print(f"{stem}: clip not found"); return
    clip = hits[0]
    pitcher = os.path.basename(os.path.dirname(clip)).replace("_2026", "")
    row = man.get(stem, {})
    ptype = row.get("pitch_type", "")

    det = detect_setup_catch(clip, model)
    setup = det.get("setup") if det.get("ok") else None
    aff = _game_affine(pitcher, row.get("game_pk", "")) if row else None
    miss_txt = ""
    if aff and setup and row.get("plate_x"):
        try:
            tx, tz = aff.pixel_to_world(setup[0], setup[1])
            dx, dz = (tx - float(row["plate_x"])) * 12, (tz - float(row["plate_z"])) * 12
            miss_txt = f"command miss {np.hypot(dx, dz):.0f}in"
        except (TypeError, ValueError):
            pass

    outdir = os.path.join(ROOT, "output", "annotated"); os.makedirs(outdir, exist_ok=True)
    outp = os.path.join(outdir, f"{stem}_annot.mp4")
    if os.path.exists(outp):
        print(f"{stem}: already rendered, skip"); return outp
    cap = cv2.VideoCapture(clip)
    w, h = int(cap.get(3)), int(cap.get(4)); n = int(cap.get(7))
    vw = cv2.VideoWriter(outp, cv2.VideoWriter_fourcc(*"mp4v"), 30, (w, h))
    for fi in range(n):
        ok, fr = cap.read()
        if not ok:
            break
        # persistent SETUP target
        if setup:
            cv2.drawMarker(fr, (int(setup[0]), int(setup[1])), ORANGE, cv2.MARKER_CROSS, 26, 2)
            cv2.putText(fr, "TARGET", (int(setup[0]) + 14, int(setup[1]) - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, ORANGE, 2)
        # live glove box
        r = model.predict(fr, imgsz=640, conf=0.25, verbose=False)[0]
        if len(r.boxes):
            i = int(np.argmax(r.boxes.conf.cpu().numpy()))
            x0, y0, x1, y1 = r.boxes.xyxy[i].cpu().numpy().astype(int)
            cv2.rectangle(fr, (x0, y0), (x1, y1), CYAN, 2)
            cv2.putText(fr, f"glove {float(r.boxes.conf[i]):.2f}", (x0, y0 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, CYAN, 2)
        cv2.rectangle(fr, (0, 0), (w, 34), NAVY, -1)
        cv2.putText(fr, f"Command CV  |  {pitcher.title()} {ptype}  |  {miss_txt}",
                    (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        vw.write(fr)
    cap.release(); vw.release()
    print(f"{stem}: -> {outp}")
    return outp


def main():
    from ultralytics import YOLO
    model = YOLO(os.path.join(ROOT, "data", "yolo", "runs", "mitt", "weights", "best.pt"))
    for stem in sys.argv[1:]:
        annotate(stem, model)


if __name__ == "__main__":
    main()
