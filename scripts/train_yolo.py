"""Train the mitt detector on the annotated frames (Ultralytics YOLO).

Consumes data/yolo (images/ + labels/ + data.yaml) produced by annotate_tool.py.
Once trained, point detection at the weights to auto-locate the mitt instead of
clicking — drop-in via src/detect.py YoloDetector.

Install:  pip install ultralytics
Run:      python scripts/train_yolo.py --epochs 100 --model yolov8n.pt

Small dataset tips: start from a pretrained yolov8n, heavy augmentation, and add
more clips/labels — the busy orange background (fans, wall, mask cage) means the
model needs variety to beat the classical color detector.
"""
import argparse
import glob
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dataset import write_data_yaml, yolo_root


def _ensure_val_split():
    """Ultralytics needs a val set. If none, copy ~20% of train into val."""
    root = yolo_root()
    tr_img = os.path.join(root, "images", "train")
    val_img = os.path.join(root, "images", "val")
    tr_lbl = os.path.join(root, "labels", "train")
    val_lbl = os.path.join(root, "labels", "val")
    imgs = sorted(glob.glob(os.path.join(tr_img, "*.png")))
    if not imgs:
        raise SystemExit("no training images — run annotate_tool.py first")
    have_val = glob.glob(os.path.join(val_img, "*.png"))
    if have_val:
        return
    os.makedirs(val_img, exist_ok=True)
    os.makedirs(val_lbl, exist_ok=True)
    k = max(1, len(imgs) // 5)
    for img in imgs[:k]:
        stem = os.path.splitext(os.path.basename(img))[0]
        shutil.copy(img, os.path.join(val_img, os.path.basename(img)))
        lbl = os.path.join(tr_lbl, stem + ".txt")
        if os.path.exists(lbl):
            shutil.copy(lbl, os.path.join(val_lbl, stem + ".txt"))
    print(f"created val split: {k} image(s)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--model", default="yolov8n.pt")
    p.add_argument("--imgsz", type=int, default=960)
    a = p.parse_args()

    data_yaml = write_data_yaml()
    _ensure_val_split()

    try:
        from ultralytics import YOLO
    except ImportError:
        raise SystemExit("ultralytics not installed — run: pip install ultralytics")

    model = YOLO(a.model)
    model.train(data=data_yaml, epochs=a.epochs, imgsz=a.imgsz,
                project=os.path.join(yolo_root(), "runs"), name="mitt")
    print("done — weights under data/yolo/runs/mitt/weights/best.pt")


if __name__ == "__main__":
    main()
