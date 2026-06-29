"""YOLO dataset helpers for the trained mitt detector.

The click tool saves, per labeled frame, a full-frame image + a YOLO label
(class 0 = "mitt", normalized xc yc w h). `train_yolo.py` then trains an
Ultralytics model on this set. One class for now (the catcher's mitt = the
intended-target locator); plate/zone keypoint classes can be added later.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

CLASSES = ["mitt"]


@dataclass
class Box:
    """Pixel bounding box (top-left origin)."""
    cx: float
    cy: float
    w: float
    h: float

    @classmethod
    def from_center_radius(cls, cx: float, cy: float, r: float) -> "Box":
        return cls(cx=cx, cy=cy, w=2 * r, h=2 * r)

    def to_yolo(self, img_w: int, img_h: int) -> tuple[float, float, float, float]:
        return (self.cx / img_w, self.cy / img_h, self.w / img_w, self.h / img_h)


def yolo_root() -> str:
    return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "data", "yolo")


def save_sample(frame_bgr, box: Box, name: str, split: str = "train") -> tuple[str, str]:
    """Write one training image + its YOLO label. `name` is a unique stem.

    Returns (image_path, label_path).
    """
    import cv2

    root = yolo_root()
    img_dir = os.path.join(root, "images", split)
    lbl_dir = os.path.join(root, "labels", split)
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lbl_dir, exist_ok=True)

    h, w = frame_bgr.shape[:2]
    img_path = os.path.join(img_dir, f"{name}.png")
    lbl_path = os.path.join(lbl_dir, f"{name}.txt")
    cv2.imwrite(img_path, frame_bgr)
    xc, yc, bw, bh = box.to_yolo(w, h)
    with open(lbl_path, "w") as f:
        f.write(f"0 {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")
    return img_path, lbl_path


def write_data_yaml() -> str:
    """Write data/yolo/data.yaml for Ultralytics. Returns its path."""
    root = yolo_root()
    os.makedirs(root, exist_ok=True)
    path = os.path.join(root, "data.yaml")
    lines = [
        f"path: {root}",
        "train: images/train",
        "val: images/val",
        f"nc: {len(CLASSES)}",
        f"names: {CLASSES}",
        "",
    ]
    with open(path, "w") as f:
        f.write("\n".join(lines))
    return path
