"""Orange catcher's-mitt detector (color + motion), the Method-B proposer.

Classical detector used to PROPOSE the mitt center so a human only nudges it in
the annotation tool, and as a fallback before the trained YOLO model exists. The
mitt is a distinct tan/orange against a busy background (orange fans, wall
padding, Astros gear), so we gate the color mask by motion vs a quiet reference
frame and restrict to a catcher region of interest.

cv2 + numpy are imported lazily so importing this module stays light.
"""
from __future__ import annotations

from dataclasses import dataclass

from .miss_methodb import GlovePx

# HSV range for the tan/orange leather mitt (OpenCV H in 0-179).
HSV_LO = (8, 90, 90)
HSV_HI = (30, 255, 255)


@dataclass
class ROI:
    """Catcher region as fractions of frame width/height."""
    x0: float = 0.55
    x1: float = 0.82
    y0: float = 0.45
    y1: float = 0.90

    def px(self, w: int, h: int) -> tuple[int, int, int, int]:
        return (int(w * self.x0), int(w * self.x1),
                int(h * self.y0), int(h * self.y1))


def detect_mitt(frame, ref_frame=None, roi: ROI | None = None,
                motion_thresh: int = 40, min_area: int = 40) -> GlovePx | None:
    """Return the proposed mitt GlovePx in a BGR (or RGB) frame, or None.

    If `ref_frame` is given, the color mask is AND-ed with a motion mask
    (abs-diff > motion_thresh) to suppress static orange background. `diameter_px`
    is 2*minEnclosingCircle radius of the largest surviving blob.
    """
    import cv2
    import numpy as np

    roi = roi or ROI()
    h, w = frame.shape[:2]
    x0, x1, y0, y1 = roi.px(w, h)
    sub = frame[y0:y1, x0:x1]
    hsv = cv2.cvtColor(sub, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, HSV_LO, HSV_HI)
    if ref_frame is not None:
        ref = ref_frame[y0:y1, x0:x1]
        diff = cv2.absdiff(sub, ref).sum(2).astype("uint8")
        motion = (diff > motion_thresh).astype("uint8") * 255
        mask = cv2.bitwise_and(mask, motion)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(c) < min_area:
        return None
    (cx, cy), r = cv2.minEnclosingCircle(c)
    return GlovePx(cx=x0 + cx, cy=y0 + cy, diameter_px=2 * r)


def quiet_reference(frames: list, index: int = 60):
    """Pick a low-motion reference frame (catcher set, pre-delivery)."""
    return frames[min(index, len(frames) - 1)]
