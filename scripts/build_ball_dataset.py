"""Turn the free ball labels into a YOLO dataset.

`ball_clock_align.py` writes where the ball is on every flight frame, derived from the camera
and the trajectory rather than from anyone clicking. This turns that into images and labels a
detector can train on.

Two decisions worth knowing:

CROPS, NOT FULL FRAMES. The ball is about 10 px across in a 1280x720 broadcast frame. Resized
to a 640 px training image it is 5 px and mostly gone, and this laptop's torch is CPU-only and
single-threaded, so training at 1280 is not realistic. So each sample is a CROP_PX window at
native resolution: the ball keeps its true size and the image is small enough to train on.

THE BALL IS NOT CENTRED. A crop always centred on the ball teaches "the ball is in the middle".
Each crop is jittered by up to JITTER_PX, and a NEGATIVE crop with no ball is emitted every
NEG_EVERY samples, taken far enough away that the ball cannot be inside it.

SPLIT BY GAME, not by frame. Frames from one pitch are nearly identical, so a random split
would put near-duplicates on both sides and report a score that means nothing. With one game
the split falls back to play_id and says so: same camera on both sides is still optimistic.

    python scripts/build_ball_dataset.py --labels output/ball_labels_824821.csv \
        --clips data/clips/mlb_824821 --out data/yolo_ball [--val-frac 0.2]
"""
import argparse
import os
import random

import cv2
import numpy as np
import pandas as pd

CROP_PX = 320
JITTER_PX = 110
NEG_EVERY = 6
BOX_SCALE = 1.8          # box side / ball diameter, so a small detection error still overlaps
MIN_BOX_PX = 6


def _write(split, name, img, boxes, out):
    """boxes: list of (cx, cy, w, h) in crop pixels. YOLO txt is normalised xywh, class 0."""
    ip = os.path.join(out, "images", split, name + ".png")
    lp = os.path.join(out, "labels", split, name + ".txt")
    cv2.imwrite(ip, img)
    h, w = img.shape[:2]
    with open(lp, "w") as f:
        for cx, cy, bw, bh in boxes:
            f.write(f"0 {cx / w:.6f} {cy / h:.6f} {bw / w:.6f} {bh / h:.6f}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True, action="append",
                    help="labels CSV; repeat with a matching --clips per game")
    ap.add_argument("--clips", required=True, action="append",
                    help="clips dir for the preceding --labels")
    ap.add_argument("--out", default="data/yolo_ball")
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--angle", default="BROADCAST")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    assert len(a.labels) == len(a.clips), "one --clips per --labels"
    rng = random.Random(a.seed)
    parts = []
    for lp, cp in zip(a.labels, a.clips):
        d = pd.read_csv(lp)
        # TWO LABEL SCHEMAS REACH HERE, and they name the clip file differently.
        #   MLB  (ball_clock_align)   play_id, frame_idx, x, y, diameter_px
        #                             -> data/clips/mlb_<pk>/<play_id>_<ANGLE>.mp4
        #   MiLB (refine_park_pose)   stem, game_pk, frame, x, y, diameter_px
        #                             -> data/clips/<level>_<pk>/<stem>.mp4
        # Normalising to (play_id, frame_idx, _file) here rather than writing a second
        # builder keeps ONE crop convention, one jitter, one negative rate and one box
        # scale across both. Two builders would drift and the drift would be invisible in
        # any aggregate computed per dataset.
        milb = "stem" in d.columns
        if milb:
            d = d.rename(columns={"stem": "play_id", "frame": "frame_idx"})
            d["game_pk"] = d["game_pk"].astype(str)
            d["_file"] = d.play_id.map(lambda s: os.path.join(cp, f"{s}.mp4"))
        else:
            d["game_pk"] = os.path.basename(os.path.normpath(cp)).replace("mlb_", "")
            d["_file"] = d.play_id.map(lambda s: os.path.join(cp, f"{s}_{a.angle}.mp4"))
        assert "diameter_px" in d.columns, (
            f"{lp} has no diameter_px -- a box sized from a guess is not a label. "
            "Re-run refine_park_pose.py (it writes the column) or ball_clock_align.py.")
        d["_clips"] = cp
        parts.append(d)
    lab = pd.concat(parts, ignore_index=True)
    key = "game_pk" if lab.game_pk.nunique() > 1 else "play_id"
    if key == "play_id":
        print("WARNING: one game only, so train/val split is by play_id. Both sides share a camera "
              "and a stadium, so the val score will be optimistic. Re-split once more games land.")
    else:
        print(f"splitting by GAME across {lab.game_pk.nunique()} games: "
              f"{', '.join(sorted(lab.game_pk.unique()))} - val is a camera and a park the model "
              f"never trained on, which is the honest test")
    groups = sorted(lab[key].unique())
    rng.shuffle(groups)
    n_val = max(1, int(len(groups) * a.val_frac))
    val = set(groups[:n_val])
    for s in ("train", "val"):
        for k in ("images", "labels"):
            os.makedirs(os.path.join(a.out, k, s), exist_ok=True)
    counts = {"train": 0, "val": 0, "neg": 0, "skipped": 0}
    n = 0
    for pid, g in lab.groupby("play_id"):
        # SAMPLE NAME MUST BE UNIQUE PER CLIP. An MLB play_id is a UUID, so its first 8
        # characters identify it; a MiLB stem is `aaa_815419_0010`, whose first 8 are the
        # LEVEL AND GAME -- shared by all 50 clips. Truncating it would have written every
        # clip's frame-190 crop to the same filename, so a 40-clip park would silently
        # yield one clip's worth of images with no error anywhere. Use the full stem, with
        # only path-hostile characters replaced.
        tag = pid[:8] if len(pid) >= 32 and "-" in pid else "".join(
            ch if ch.isalnum() or ch in "-_" else "_" for ch in str(pid))
        clip = g["_file"].iloc[0]
        if not os.path.exists(clip):
            counts["skipped"] += len(g)
            continue
        want = {int(r.frame_idx): r for r in g.itertuples()}
        cap = cv2.VideoCapture(clip)
        i, hi = 0, max(want)
        while i <= hi:
            ok, fr = cap.read()
            if not ok:
                break
            r = want.get(i)
            i += 1
            if r is None:
                continue
            H, W = fr.shape[:2]
            if not (0 <= r.x < W and 0 <= r.y < H):
                counts["skipped"] += 1
                continue
            n += 1
            split = "val" if getattr(r, key, pid) in val or pid in val else "train"
            jx, jy = rng.randint(-JITTER_PX, JITTER_PX), rng.randint(-JITTER_PX, JITTER_PX)
            x0 = int(np.clip(r.x - CROP_PX // 2 + jx, 0, W - CROP_PX))
            y0 = int(np.clip(r.y - CROP_PX // 2 + jy, 0, H - CROP_PX))
            crop = fr[y0:y0 + CROP_PX, x0:x0 + CROP_PX]
            side = max(MIN_BOX_PX, r.diameter_px * BOX_SCALE)
            cx, cy = r.x - x0, r.y - y0
            if side / 2 <= cx <= CROP_PX - side / 2 and side / 2 <= cy <= CROP_PX - side / 2:
                _write(split, f"{tag}_f{i - 1}", crop, [(cx, cy, side, side)], a.out)
                counts[split] += 1
            else:
                counts["skipped"] += 1
                continue
            if n % NEG_EVERY == 0:      # a crop that cannot contain the ball
                for _ in range(8):
                    nx, ny = rng.randint(0, W - CROP_PX), rng.randint(0, H - CROP_PX)
                    if abs(nx - x0) > CROP_PX or abs(ny - y0) > CROP_PX:
                        _write(split, f"{tag}_f{i - 1}_neg", fr[ny:ny + CROP_PX, nx:nx + CROP_PX], [], a.out)
                        counts["neg"] += 1
                        break
        cap.release()
    with open(os.path.join(a.out, "data.yaml"), "w") as f:
        f.write(f"path: {os.path.abspath(a.out)}\ntrain: images/train\nval: images/val\n"
                f"names:\n  0: ball\n")
    print(counts)
    # THE REQUESTED FRACTION IS OVER GROUPS, AND WHAT YOU GET IS OVER IMAGES. Splitting by
    # park is the honest test, but parks are wildly unequal -- a 4-park set at --val-frac
    # 0.25 holds out ONE park, and if the shuffle draws the smallest you get 5.6% of the
    # labels in val, not 25%. Stating only the requested fraction would let a reader believe
    # a number the split never produced.
    tot = counts["train"] + counts["val"]
    got = counts["val"] / max(1, tot)
    print(f"val groups ({key}): {n_val} of {len(groups)} -> {counts['val']} of {tot} "
          f"labelled images = {100 * got:.1f}% (requested {100 * a.val_frac:.0f}% of GROUPS)")
    if abs(got - a.val_frac) > 0.1:
        print(f"  NOTE: the val share of IMAGES is {100 * got:.1f}%, far from the "
              f"requested group share.")
        print("  That is what an unequal park split gives you. It is not a bug, but do "
              "not quote the requested number.")
    print(os.path.join(a.out, "data.yaml"))


if __name__ == "__main__":
    main()
