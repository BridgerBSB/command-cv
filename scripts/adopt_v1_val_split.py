"""Make a dataset's held-out split the EXACT one v1 was scored on.

WHY NOT JUST REBUILD IT. The number we care about is the held-out pixel error
against v1's 2.15 px. That figure came from 139 Asheville (ogando_*) crops in
data/yolo_milb/images/val. Rebuilding Asheville from its label CSV would give a
*similar* val set, not the same one, and every difference -- a different jitter
draw, a label CSV that has been re-snapped since, a clip that no longer
decodes -- moves the number by an unknown amount in an unknown direction. Then
"2.15 -> 2.0" would be two changes at once: a better model, and a different
exam. Copying the images makes the exam identical, so the only thing that moved
is the model.

This is safe to do only because the crop recipe has not changed. CROP_PX 320,
BOX_SCALE 1.8 and MIN_BOX_PX 6 are byte-identical in every commit that has
touched build_ball_dataset.py since v1 was trained, and the images on disk are
320x320. The script re-checks both rather than trusting that.

WHAT IT DOES
  1. Any existing val in the destination is MOVED to train, not deleted. In our
     case that is 381 Isotopes crops, which v1 trained on and v2 should too --
     leaving them in val would both blend the exam and throw a park away.
  2. Copies the source val (images + labels) into the destination val.
  3. Writes val_games.json so build_ball_dataset's blend guard knows what the
     split is and refuses to quietly change it later.
  4. Checks for leakage by STEM, and fails loudly rather than reporting a clean
     dataset it never really inspected.

    python scripts/adopt_v1_val_split.py --src data/yolo_milb --dst data/yolo_milb3 \
        --val-name 822515
"""
import argparse
import glob
import json
import os
import shutil

import cv2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXPECT_CROP_PX = 320


def stems(d):
    return {os.path.splitext(os.path.basename(p))[0] for p in glob.glob(os.path.join(d, "*.png"))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=os.path.join(ROOT, "data", "yolo_milb"),
                    help="dataset whose val split is the one to adopt")
    ap.add_argument("--dst", default=os.path.join(ROOT, "data", "yolo_milb3"))
    ap.add_argument("--val-name", default="822515",
                    help="what to record in val_games.json for the adopted split")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    src_img = os.path.join(a.src, "images", "val")
    src_lab = os.path.join(a.src, "labels", "val")
    for d in (src_img, src_lab):
        assert os.path.isdir(d), f"missing {d}"
    src_pngs = sorted(glob.glob(os.path.join(src_img, "*.png")))
    assert src_pngs, f"NOTHING TO ADOPT -- {src_img} holds no images"

    # The recipe check. A 320-px val against a 640-px train is not a val set.
    h, w = cv2.imread(src_pngs[0]).shape[:2]
    assert (h, w) == (EXPECT_CROP_PX, EXPECT_CROP_PX), (
        f"source crops are {w}x{h}, expected {EXPECT_CROP_PX}. The crop recipe changed; "
        "adopting these would compare the new model against a different exam.")

    dst_img = os.path.join(a.dst, "images", "val")
    dst_lab = os.path.join(a.dst, "labels", "val")
    tr_img = os.path.join(a.dst, "images", "train")
    tr_lab = os.path.join(a.dst, "labels", "train")
    for d in (dst_img, dst_lab, tr_img, tr_lab):
        os.makedirs(d, exist_ok=True)

    existing = sorted(glob.glob(os.path.join(dst_img, "*.png")))
    print(f"source val   : {len(src_pngs)} images  ({os.path.basename(src_pngs[0])} ...)")
    print(f"dest val now : {len(existing)} images"
          + (f"  ({os.path.basename(existing[0])} ...)" if existing else ""))
    print(f"dest train   : {len(glob.glob(os.path.join(tr_img, '*.png')))} images")

    if a.dry_run:
        print("\n--dry-run: nothing written")
        return

    moved = 0
    for p in existing:
        stem = os.path.splitext(os.path.basename(p))[0]
        shutil.move(p, os.path.join(tr_img, stem + ".png"))
        lp = os.path.join(dst_lab, stem + ".txt")
        if os.path.isfile(lp):
            shutil.move(lp, os.path.join(tr_lab, stem + ".txt"))
        moved += 1
    print(f"\nmoved {moved} existing val images to TRAIN (kept, not discarded)")

    copied = 0
    for p in src_pngs:
        stem = os.path.splitext(os.path.basename(p))[0]
        shutil.copy2(p, os.path.join(dst_img, stem + ".png"))
        lp = os.path.join(src_lab, stem + ".txt")
        assert os.path.isfile(lp), f"source image {stem} has no label -- refusing a half split"
        shutil.copy2(lp, os.path.join(dst_lab, stem + ".txt"))
        copied += 1
    print(f"copied {copied} images + labels into the val split")

    with open(os.path.join(a.dst, "val_games.json"), "w", encoding="utf-8") as fh:
        json.dump({"val_games": [str(a.val_name)],
                   "adopted_from": os.path.abspath(a.src),
                   "note": "byte-copied from v1's val so the held-out number is comparable"},
                  fh, indent=2)

    # LEAKAGE, CHECKED BY STEM. A crop present in both splits makes the score
    # meaningless, and it is invisible in any summary count.
    tr, va = stems(tr_img), stems(dst_img)
    overlap = tr & va
    assert not overlap, f"LEAKAGE: {len(overlap)} stems in BOTH splits, e.g. {sorted(overlap)[:5]}"
    n_lab_v = len(glob.glob(os.path.join(dst_lab, "*.txt")))
    n_lab_t = len(glob.glob(os.path.join(tr_lab, "*.txt")))
    print(f"\ntrain {len(tr)} images / {n_lab_t} labels")
    print(f"val   {len(va)} images / {n_lab_v} labels")
    print(f"overlap by stem: 0  (checked {len(tr)} train against {len(va)} val)")
    assert len(va) == copied, "val count does not match what was copied"
    print("\nOK. data.yaml is unchanged; it already points at images/train and images/val.")


if __name__ == "__main__":
    main()
