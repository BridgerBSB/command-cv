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
      --clips-dir data/clips/ogando_2026

Run (single clip):
  python scripts/ingest_annotation.py ~/Downloads/ogando_ff_1.json \
      --clip data/clips/ogando_2026/ogando_ff_1.mp4

CLIP RESOLUTION IS AMBIGUITY-SAFE (fixed Sep 11 2026). `--clip` / `--clips-dir`
win outright. Without them, every place a clip of that name could live is searched
(next to the JSON, data/clips and everything under it, the cwd) and a name matching
MORE THAN ONE video is REFUSED, naming the candidates.

Why: 12 of 55 Ogando labels were ingested against the wrong video. The Jun 29 GC2
zip clips in data/clips/ share stems (ogando_ff_1, ogando_cu_3, ...) with the Jul
API pull in data/clips/ogando_2026/, the old resolver tried `data/clips/<name>.mp4`
before looking deeper, and it silently won. The saved training images came from a
different game than the boxes were drawn on. See docs/BACKLOG.md.

Every annotation now records WHICH video it was read from (`clip_path`,
`clip_bytes`, `clip_frames`), so provenance is checkable later without guessing.
"""
import argparse
import glob
import json
import os
import sys
from datetime import datetime
from zoneinfo import ZoneInfo

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


def resolve_clip(ann, json_path, clip_arg=None, clips_dir=None):
    """(path, None) on success, (None, reason) otherwise.

    An explicit --clip / --clips-dir is taken as given. Otherwise the search is
    exhaustive and a name that matches more than one distinct video is refused:
    picking one silently is the bug this function exists to prevent."""
    stem = ann.get("stem") or "clip"
    fname = ann.get("clip") or f"{stem}.mp4"
    if clip_arg:
        return (clip_arg, None) if os.path.isfile(clip_arg) else (None, f"--clip not a file: {clip_arg}")
    if clips_dir:
        hit = _find_in_dir(fname, clips_dir)
        return (hit, None) if hit else (None, f"'{fname}' not found under --clips-dir {clips_dir}")
    seen, cands = set(), []
    for cand in ([os.path.join(os.path.dirname(os.path.abspath(json_path)), fname)]
                 + glob.glob(os.path.join(ROOT, "data", "clips", "**", fname), recursive=True)
                 + [os.path.join(os.getcwd(), fname)]):
        if cand and os.path.isfile(cand):
            key = os.path.realpath(cand)
            if key not in seen:
                seen.add(key)
                cands.append(cand)
    if not cands:
        return None, f"clip '{fname}' not found (pass --clips-dir or --clip)"
    if len(cands) > 1:
        rel = [os.path.relpath(c, ROOT) for c in cands]
        return None, ("AMBIGUOUS: '%s' matches %d videos (%s). Pass --clips-dir to say which."
                      % (fname, len(cands), ", ".join(rel)))
    return cands[0], None


def _read_frame(clip, idx):
    """Frame `idx` by decoding IN ORDER from the start.

    Never `cap.set(CAP_PROP_POS_FRAMES)`: measured on these clips, a seek lands 0 to 2
    frames EARLY, which silently saves a training image that is not the frame the box
    was drawn on."""
    import cv2

    cap = cv2.VideoCapture(clip)
    fr, i = None, 0
    while i <= idx:
        ok, f = cap.read()
        if not ok:
            break
        fr, i = f, i + 1
    cap.release()
    if fr is None or i <= idx:
        raise RuntimeError(f"could not read frame {idx} from {clip} (decoded {i} frames)")
    return fr


def _clip_meta(clip):
    import cv2

    cap = cv2.VideoCapture(clip)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return dict(clip_path=os.path.relpath(clip, ROOT).replace("\\", "/"),
                clip_bytes=os.path.getsize(clip), clip_frames=n)


def _box(d):
    return Box(cx=float(d["cx"]), cy=float(d["cy"]), w=float(d["w"]), h=float(d["h"]))


def _ingest_one(ann, json_path, clip_arg, clips_dir):
    stem = ann.get("stem") or "clip"
    for k in ("setup", "catch"):
        if not ann.get(k) or "frame" not in ann[k]:
            return f"SKIP {stem}: missing '{k}' box"
    clip, why = resolve_clip(ann, json_path, clip_arg, clips_dir)
    if not clip:
        return f"SKIP {stem}: {why}"

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
        "ingested_ct": datetime.now(ZoneInfo("America/Chicago")).strftime("%Y-%m-%d %H:%M:%S"),
        **_clip_meta(clip),
    }
    with open(os.path.join(ann_dir, f"{stem}.json"), "w") as f:
        json.dump(out, f, indent=2)

    write_data_yaml()
    for tag in ("setup", "catch"):
        fi = int(ann[tag]["frame"])
        save_sample(_read_frame(clip, fi), _box(ann[tag]), f"{stem}_{tag}_f{fi}")
    return f"OK   {stem}: setup f{out['setup']['frame']}  catch f{out['catch']['frame']}  <- {out['clip_path']}"


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
    if n_ok < len(entries):
        print("refused entries above are NOT a failure to find the clip: an ambiguous name means "
              "two videos share it. Re-run those with --clips-dir pointing at the right pull.")
    if n_ok:
        print("next: python scripts/methodb_batch.py")


if __name__ == "__main__":
    main()
