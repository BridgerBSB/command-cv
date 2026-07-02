"""Ingest browser-annotated JSON (from tools/annotate.html) into the pipeline.

Accepts EITHER shape the annotator can produce:
  * a single annotation object  (one clip)
  * a JSON array of them         (batch: "Download batch JSON (all clips)")

For each entry it does the headless half of annotate_tool's `write()`: extracts
the tagged setup + catch frames via cv2 and writes

  data/annotations/<stem>.json          (canonical, read by methodb_batch.py)
  data/yolo/images|labels/train/<...>   (YOLO training samples, read by train_yolo.py)

No GUI, so nothing here can crash the way the matplotlib tool did.

Run (batch, one command for all clips):
  python scripts/ingest_annotation.py ~/Downloads/command_cv_batch.json \
      --clips-dir "C:/Users/Owner/Downloads/cv_clips/CV Miss Distance Tracking"

Run (single clip):
  python scripts/ingest_annotation.py ~/Downloads/ogando_ff_1.json \
      --clip "C:/Users/Owner/Downloads/.../ogando_ff_1.mp4"

Clip resolution order per entry: --clip (single only) -> --clips-dir (recursive)
-> next to the JSON -> data/clips/ -> current dir.
"""
import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dataset import Box, save_sample, write_data_yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _find_in_dir(fname, clips_dir):
    if not clips_dir or not os.path.isdir(clips_dir):
        return None
    direct = os.path.join(clips_dir, fname)
    if os.path.isfile(direct):
        return direct
    hits = glob.glob(os.path.join(clips_dir, "**", fname), recursive=True)
    return hits[0] if hits else None


def _resolve_clip(ann, json_path, clip_arg, clips_dir):
    stem = ann.get("stem") or "clip"
    fname = ann.get("clip") or f"{stem}.mp4"
    for cand in (
        clip_arg if clip_arg and os.path.isfile(clip_arg) else None,
        _find_in_dir(fname, clips_dir),
        os.path.join(os.path.dirname(os.path.abspath(json_path)), fname),
        os.path.join(ROOT, "data", "clips", fname),
        os.path.join(os.getcwd(), fname),
    ):
        if cand and os.path.isfile(cand):
            return cand
    return None


def _read_frame(clip, idx):
    import cv2

    cap = cv2.VideoCapture(clip)
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, fr = cap.read()
    if not ok:
        cap.release()
        cap = cv2.VideoCapture(clip)
        i, fr = 0, None
        while i <= idx:
            ok, f = cap.read()
            if not ok:
                break
            fr, i = f, i + 1
    cap.release()
    if fr is None:
        raise RuntimeError(f"could not read frame {idx} from {clip}")
    return fr


def _box(d):
    return Box(cx=float(d["cx"]), cy=float(d["cy"]), w=float(d["w"]), h=float(d["h"]))


def _ingest_one(ann, json_path, clip_arg, clips_dir):
    stem = ann.get("stem") or "clip"
    for k in ("setup", "catch"):
        if not ann.get(k) or "frame" not in ann[k]:
            return f"SKIP {stem}: missing '{k}' box"
    clip = _resolve_clip(ann, json_path, clip_arg, clips_dir)
    if not clip:
        return (f"SKIP {stem}: clip '{ann.get('clip', stem + '.mp4')}' not found "
                f"(pass --clips-dir or --clip)")

    ann_dir = os.path.join(ROOT, "data", "annotations")
    os.makedirs(ann_dir, exist_ok=True)
    out = {
        "clip": os.path.basename(clip), "stem": stem,
        "pitcher": ann.get("pitcher", stem), "pitch_type": ann.get("pitch_type", ""),
        "hand": ann.get("hand", "R"),
        "setup": {k: ann["setup"][k] for k in ("frame", "cx", "cy", "w", "h")},
        "catch": {k: ann["catch"][k] for k in ("frame", "cx", "cy", "w", "h")},
        "endpoint_kind": ann.get("endpoint_kind", "catch"),  # "catch" (received) | "contact" (hit)
        "trackman": ann.get("trackman", {}),
    }
    with open(os.path.join(ann_dir, f"{stem}.json"), "w") as f:
        json.dump(out, f, indent=2)

    write_data_yaml()
    for tag in ("setup", "catch"):
        fi = int(ann[tag]["frame"])
        save_sample(_read_frame(clip, fi), _box(ann[tag]), f"{stem}_{tag}_f{fi}")
    return f"OK   {stem}: setup f{out['setup']['frame']}  catch f{out['catch']['frame']}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("json", help="single annotation OR a batch array from tools/annotate.html")
    p.add_argument("--clip", default=None, help="explicit .mp4 path (single-clip JSON only)")
    p.add_argument("--clips-dir", default=None, help="folder holding the clips (searched recursively)")
    a = p.parse_args()

    with open(a.json) as f:
        data = json.load(f)
    entries = data if isinstance(data, list) else [data]
    if len(entries) > 1 and a.clip:
        print("note: --clip ignored for a batch; using --clips-dir / auto-resolve")
        a.clip = None

    results = [_ingest_one(e, a.json, a.clip, a.clips_dir) for e in entries]
    for r in results:
        print(r)
    n_ok = sum(1 for r in results if r.startswith("OK"))
    print(f"\n{n_ok}/{len(entries)} ingested -> data/annotations/ + YOLO samples")
    if n_ok:
        print("next: python scripts/methodb_batch.py")


if __name__ == "__main__":
    main()
