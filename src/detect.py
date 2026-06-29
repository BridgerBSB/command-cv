"""Detection interface: locate home plate, strike zone, and the catcher's glove
in a single glove-flash frame.

Two backends behind one interface so the engine is usable before a model exists:
  - ManualDetections : caller supplies pixel coords (click-tool / annotation).
  - YoloDetector     : trained Ultralytics model (drop-in; lazy-imports cv2+ultralytics).

The "train our model" deliverable swaps ManualDetections for YoloDetector with
zero change to calibration / report / pipeline.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class FrameDetections:
    """Pixel-space detections in one frame.

    plate_pts : list of (px, py) for the home-plate reference corners
                (order must match calibration.home_plate_world_points()).
    zone_pts  : optional list of 4 (px, py) strike-zone corners
                (order: TL, TR, BR, BL) to match strikezone_world_points().
    glove_px  : (px, py) center of the catcher's glove (the intended target).
    """
    plate_pts: list[tuple[float, float]]
    glove_px: tuple[float, float]
    zone_pts: list[tuple[float, float]] | None = None
    confidence: float | None = None


class Detector:
    """Backend interface."""

    def detect(self, frame) -> FrameDetections:  # pragma: no cover - interface
        raise NotImplementedError


@dataclass
class ManualDetections(Detector):
    """Pass-through backend: detections provided by a human / annotation tool."""
    det: FrameDetections

    def detect(self, frame=None) -> FrameDetections:
        return self.det


class YoloDetector(Detector):
    """Trained-model backend (planned). Lazy-imports heavy deps so importing this
    module never requires cv2/ultralytics."""

    def __init__(self, weights_path: str):
        self.weights_path = weights_path
        self._model = None

    def _load(self):
        if self._model is None:
            from ultralytics import YOLO  # lazy
            self._model = YOLO(self.weights_path)
        return self._model

    def detect(self, frame) -> FrameDetections:
        raise NotImplementedError(
            "YoloDetector.detect is a stub until the model is trained on "
            "labeled MiLB/MLB frames (glove / plate / zone keypoints)."
        )
